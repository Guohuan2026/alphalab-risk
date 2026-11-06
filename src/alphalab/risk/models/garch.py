"""GARCH-family risk models: GARCH(1,1)-t, GJR-GARCH(1,1)-t and filtered historical simulation.

Model (o = 0 gives plain GARCH, o = 1 adds the GJR leverage term):

    r_t  = μ + ε_t,   ε_t = σ_t z_t,   z_t ~ unit-variance Student-t(ν)
    σ²_t = ω + (α + γ·1[ε_{t−1} < 0]) ε²_{t−1} + β σ²_{t−1}

Estimation: maximum likelihood with the ``arch`` package (Sheppard), on returns in PERCENT
(the optimiser is better conditioned there); parameters are converted back to decimals.
Refits are warm-started from the previous estimates.

Forecasting: the daily variance filter is ``gjr_variance_filter`` from ``risk/simulate.py``,
the same code that generates the simulated ground truth, so timing is identical (σ_{t+1}
uses returns up to t). Then

    GARCH-t / GJR-t :  VaR = −μ + σ_{t+1} q_a,     ES = −μ + σ_{t+1} s_a     (Student-t tails)
    FHS             :  VaR = −μ + σ_{t+1} Q̂_a,     ES = −μ + σ_{t+1} Ŝ_a     (empirical tails)

where Q̂_a and Ŝ_a are the empirical VaR/ES of the standardised residuals ẑ = (r − μ)/σ over
the training window (Barone-Adesi, Giannopoulos & Vosper 1999). FHS keeps GARCH's fast
reaction but lets the data, not the t distribution, shape the tail, including skew.

Failed fits: if the optimiser does not converge (or raises), the previous parameters are
kept and the failure is counted in ``fit_info``. The very first fit must succeed.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np

from alphalab.risk.models.base import Forecast, RiskModel
from alphalab.risk.simulate import GJRParams, gjr_variance_filter, standardized_t_tail

log = logging.getLogger(__name__)


def fit_gjr_t(
    returns: np.ndarray, asymmetric: bool = True, start: GJRParams | None = None
) -> tuple[GJRParams, bool]:
    """ML estimates (decimal units) and a convergence flag."""
    from arch import arch_model  # imported lazily: heavy import, only needed for fitting

    am = arch_model(
        np.asarray(returns) * 100,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=1 if asymmetric else 0,
        q=1,
        dist="t",
        rescale=False,
    )
    sv = None
    if start is not None:
        sv = [start.mu * 100, start.omega * 1e4, start.alpha]
        sv += [start.gamma] if asymmetric else []
        sv += [start.beta, start.nu]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence is checked explicitly below
        res = am.fit(disp="off", starting_values=sv)
    p = res.params
    est = GJRParams(
        mu=p["mu"] / 100,
        omega=p["omega"] / 1e4,
        alpha=p["alpha[1]"],
        gamma=p["gamma[1]"] if asymmetric else 0.0,
        beta=p["beta[1]"],
        nu=p["nu"],
    )
    ok = res.convergence_flag == 0 and est.persistence < 1 and est.nu > 2 and est.omega > 0
    return est, bool(ok)


@dataclass
class GarchT(RiskModel):
    """GARCH(1,1)-t (``asymmetric=False``) or GJR-GARCH(1,1)-t (``asymmetric=True``)."""

    asymmetric: bool = True
    tails: str = "t"  # "t": Student-t quantiles; "fhs": empirical residual quantiles
    min_obs: int = 500
    params: GJRParams | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.tails not in ("t", "fhs"):
            raise ValueError("tails must be 't' or 'fhs'")
        default = {
            ("t", False): "garch_t",
            ("t", True): "gjr_t",
            ("fhs", False): "fhs_garch",
            ("fhs", True): "fhs_gjr",
        }[(self.tails, self.asymmetric)]
        if self.name == "garcht":
            self.name = default
        self.n_fits = 0
        self.n_failed = 0
        self._resid_tail: dict[float, tuple[float, float]] = {}
        self._cache_len = -1
        self._cache_s2 = np.nan  # σ² for the day after history[cache_len - 1]
        self._cache_last = np.nan

    @property
    def min_history(self) -> int:
        return self.min_obs

    # ---------- estimation ----------
    def fit(self, train: np.ndarray) -> GarchT:
        train = np.asarray(train, dtype=float)
        try:
            est, ok = fit_gjr_t(train, self.asymmetric, self.params)
        except Exception as exc:  # optimiser crash: treat like non-convergence
            log.warning("%s fit raised %s", self.name, exc)
            est, ok = None, False
        self.n_fits += 1
        if not ok:
            self.n_failed += 1
            if self.params is None:
                raise RuntimeError(f"{self.name}: first fit failed; cannot forecast")
            log.warning("%s: fit %d failed, keeping previous parameters", self.name, self.n_fits)
        else:
            self.params = est
        if self.tails == "fhs":
            s2 = gjr_variance_filter(train, self.params, self._init_s2(train))
            z = (train - self.params.mu) / np.sqrt(s2)
            loss_z = -z[50:]  # drop the filter's burn-in, where σ² still depends on σ²_0
            self._resid_tail = {}
            for a in self.alphas:
                q = float(np.quantile(loss_z, 1 - a))
                self._resid_tail[a] = (q, float(loss_z[loss_z >= q].mean()))
        self._cache_len = -1  # parameters changed: recompute the filter from scratch
        return self

    def fit_info(self) -> dict:
        p = self.params
        return {
            "mu": p.mu,
            "omega": p.omega,
            "alpha": p.alpha,
            "gamma": p.gamma,
            "beta": p.beta,
            "nu": p.nu,
            "persistence": p.persistence,
            "uncond_vol_annual": float(np.sqrt(p.unconditional_variance * 252)),
            "n_fits": self.n_fits,
            "n_failed": self.n_failed,
        }

    # ---------- daily filter ----------
    def _init_s2(self, r: np.ndarray) -> float:
        return float(np.var(r[: min(len(r), 250)]))  # sample variance of the first year

    def next_variance(self, history: np.ndarray) -> float:
        """σ²_{t+1} from returns up to t, updated incrementally when history grows by one."""
        p = self.params
        n = len(history)
        if n == self._cache_len + 1 and self._cache_len > 0 and history[-2] == self._cache_last:
            e = history[-1] - p.mu
            s2 = p.omega + (p.alpha + p.gamma * (e < 0)) * e * e + p.beta * self._cache_s2
        else:
            path = gjr_variance_filter(history, p, self._init_s2(history))
            e = history[-1] - p.mu
            s2 = p.omega + (p.alpha + p.gamma * (e < 0)) * e * e + p.beta * path[-1]
        self._cache_len, self._cache_s2, self._cache_last = n, s2, history[-1]
        return float(s2)

    def predict(self, history: np.ndarray) -> Forecast:
        if self.params is None:
            raise RuntimeError(f"{self.name}: predict called before fit")
        history = np.asarray(history, dtype=float)
        sigma = float(np.sqrt(self.next_variance(history)))
        mu = self.params.mu
        var, es = {}, {}
        for a in self.alphas:
            if self.tails == "t":
                q, s = standardized_t_tail(a, self.params.nu)
            else:
                q, s = self._resid_tail[a]
            var[a] = -mu + sigma * q
            es[a] = -mu + sigma * s
        return Forecast(var, es, sigma)
