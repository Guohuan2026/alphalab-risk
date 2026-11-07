"""Linear benchmarks for the ML comparison.

1. HAR-FHS ("har"): Corsi's (2009) Heterogeneous Autoregressive model of volatility, with
   filtered-HS tails. Daily, weekly and monthly volatility components predict tomorrow's
   variance through a linear regression:

       r²_{s+1} = b0 + b1·RV1_s + b5·RV5_s + b22·RV22_s + u,    RVk_s = mean of r² over k days

   estimated by OLS on the training window. VaR/ES = σ̂_{t+1} × empirical VaR/ES of the
   standardised training residuals z = r_{s+1}/σ̂_{s+1}, exactly as FHS does with GARCH σ.
   It uses returns only, so it applies to every portfolio. Zero mean is assumed (as EWMA).

2. Linear quantile regression ("linqr"): the CONTROL for hypothesis H2b. The same feature
   set the ML models get, but a linear model: for each tail level τ,

       q_τ(r_{s+1} | x_s) = x_s' β_τ      (Koenker & Bassett 1978; statsmodels QuantReg)

   VaR_α = −q_α. ES_α = −mean of q at τ_k = α(k − ½)/K, k = 1..K (K = 5), the discrete
   version of ES_α = (1/α)∫₀^α VaR_u du. Predicted quantiles are sorted across levels
   to stop "quantile crossing". If ML beats this model, the gain comes from non-linearity,
   not from having better inputs.

Both standardise inputs with the TRAINING window's mean/std only (no future information).
Linear QR also clips each feature to the training window's [0.5%, 99.5%] quantiles first
(``clip_q``). Without it, a linear model extrapolates straight lines into stress days that
lie far outside the training range: on SPY, the day after the 1997 crash, features were
4–9 standard deviations out and the unclipped model forecast a NEGATIVE 5% VaR (RESULTS_LOG,
RESEARCH_DESIGN Deviation 2).
Known weakness, stated in the design: at τ = 0.1% only a handful of training points lie
below the quantile, so deep-tail linear quantiles are noisy; the ES average uses them.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

from alphalab.risk.models.base import Forecast, RiskModel

log = logging.getLogger(__name__)


def es_levels(alpha: float, k: int = 5) -> np.ndarray:
    """Tail levels α(k − ½)/K whose average quantile approximates ES_α."""
    return alpha * (np.arange(1, k + 1) - 0.5) / k


# ---------------------------------------------------------------- HAR-FHS
def _har_design(r: np.ndarray) -> np.ndarray:
    """Rows s: [1, RV1_s, RV5_s, RV22_s] (NaN until 22 observations are available)."""
    r2 = pd.Series(r**2)
    return np.column_stack(
        [
            np.ones(len(r)),
            r2.to_numpy(),
            r2.rolling(5, min_periods=5).mean().to_numpy(),
            r2.rolling(22, min_periods=22).mean().to_numpy(),
        ]
    )


@dataclass
class HARFHS(RiskModel):
    min_obs: int = 500
    floor_frac: float = 0.05  # predicted variance floored at 5% of the training variance

    def __post_init__(self):
        super().__post_init__()
        if self.name == "harfhs":
            self.name = "har_fhs"
        self.coef: np.ndarray | None = None
        self._floor = np.nan
        self._tail: dict[float, tuple[float, float]] = {}

    @property
    def min_history(self) -> int:
        return self.min_obs

    def _variance(self, design: np.ndarray) -> np.ndarray:
        return np.maximum(design @ self.coef, self._floor)

    def fit(self, train: np.ndarray) -> HARFHS:
        r = np.asarray(train, dtype=float)
        D = _har_design(r)[:-1]  # regressors at s
        y = r[1:] ** 2  # target r²_{s+1}
        ok = np.isfinite(D).all(axis=1)
        self.coef = np.linalg.lstsq(D[ok], y[ok], rcond=None)[0]
        self._floor = self.floor_frac * float(np.var(r))
        z = r[1:][ok] / np.sqrt(self._variance(D[ok]))
        loss_z = -z
        self._tail = {}
        for a in self.alphas:
            q = float(np.quantile(loss_z, 1 - a))
            self._tail[a] = (q, float(loss_z[loss_z >= q].mean()))
        return self

    def fit_info(self) -> dict:
        b0, b1, b5, b22 = self.coef
        return {"b0": b0, "b_day": b1, "b_week": b5, "b_month": b22}

    def predict(self, history: np.ndarray) -> Forecast:
        r = np.asarray(history[-30:], dtype=float)  # 22 needed; a few spare
        d = _har_design(r)[-1]
        sigma = float(np.sqrt(self._variance(d[None, :])[0]))
        var = {a: sigma * q for a, (q, _) in self._tail.items()}
        es = {a: sigma * s for a, (_, s) in self._tail.items()}
        return Forecast(var, es, sigma)


# ---------------------------------------------------------- linear quantile regression
@dataclass
class LinearQuantileRegression(RiskModel):
    """Linear QR on the feature set; optionally L1-penalised ("lasso QR") with the penalty
    tuned at every refit.

    L1 option (``l1_grid`` set; RESEARCH_DESIGN Deviation 2): collinear features (Parkinson
    vs Garman–Klass variance, rv_5 vs |r|) give unpenalised QR large offsetting coefficients
    that flip the sign of VaR on unusual days. The L1 penalty λ Σ|β_j| (intercept not
    penalised) shrinks such pairs. λ is chosen at each refit on the last ``val_frac`` of the
    training pairs, separated by a ``embargo``-pair gap, minimising the mean tick loss over
    the VaR levels; clipping and standardisation for that choice use only the earlier part.
    The grid includes 0, so "no penalty" can win. Solver: exact linear programming (HiGHS,
    via scikit-learn), slower than the unpenalised IRLS, hence yearly refits.
    """

    features: Sequence[str] | None = None  # None = every feature column
    es_k: int = 5
    clip_q: float | None = 0.005  # clip features to training [q, 1−q] quantiles; None = off
    l1_grid: Sequence[float] | None = None  # None = unpenalised (statsmodels IRLS)
    val_frac: float = 0.2
    embargo: int = 1
    min_obs: int = 500
    max_iter: int = 5000
    uses_features: ClassVar[bool] = True
    _params: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.name == "linearquantileregression":
            self.name = "linqr_l1" if self.l1_grid is not None else "linqr"
        if self.l1_grid is not None:
            self.l1_grid = tuple(float(x) for x in self.l1_grid)
            if not self.l1_grid or min(self.l1_grid) < 0:
                raise ValueError("l1_grid must be non-empty and non-negative")
        self.levels = np.unique(
            np.concatenate([[a, *es_levels(a, self.es_k)] for a in self.alphas])
        )
        self.cols: list[str] = []
        self._mu = self._sd = None
        self._lo = self._hi = None
        self.lam = 0.0
        self.n_failed = 0

    @property
    def min_history(self) -> int:
        return self.min_obs

    # ---------- transform: clip + standardise, statistics from the rows given ----------
    def _prepare(self, Xs: pd.DataFrame) -> None:
        if self.clip_q is not None:
            self._lo = Xs.quantile(self.clip_q).to_numpy()
            self._hi = Xs.quantile(1 - self.clip_q).to_numpy()
            Xs = Xs.clip(self._lo, self._hi, axis=1)
        else:
            self._lo = self._hi = None
        self._mu = Xs.mean().to_numpy()
        self._sd = Xs.std().replace(0, 1).to_numpy()

    def _xmat(self, X: pd.DataFrame) -> np.ndarray:
        x = X[self.cols].to_numpy(dtype=float)
        if self._lo is not None:
            x = np.clip(x, self._lo, self._hi)  # bounds come from the training window
        Z = (x - self._mu) / self._sd
        return np.column_stack([np.ones(len(Z)), Z])

    def _fit_levels(self, y: np.ndarray, design: np.ndarray, levels, lam: float | None) -> dict:
        """Coefficients [const, β] per level. lam None = unpenalised IRLS; else L1 via HiGHS."""
        params = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence warnings: failures counted below
            for tau in levels:
                try:
                    if lam is None:
                        import statsmodels.api as sm

                        params[tau] = (
                            sm.QuantReg(y, design).fit(q=tau, max_iter=self.max_iter).params
                        )
                    else:
                        from sklearn.linear_model import QuantileRegressor

                        qr = QuantileRegressor(quantile=tau, alpha=lam, solver="highs")
                        qr.fit(design[:, 1:], y)
                        params[tau] = np.r_[qr.intercept_, qr.coef_]
                except Exception as exc:  # singular design etc.
                    log.warning("%s: QR failed at tau=%g: %s", self.name, tau, exc)
                    self.n_failed += 1
                    if tau in self._params:
                        params[tau] = self._params[tau]
                    else:
                        raise
        return params

    def _tune(self, Xs: pd.DataFrame, y: np.ndarray) -> float:
        from alphalab.backtest.losses import tick_loss

        n_val = int(len(y) * self.val_frac)
        n_fit = len(y) - n_val - self.embargo
        self._prepare(Xs.iloc[:n_fit])  # tuning statistics from the earlier part only
        d_fit, d_val = self._xmat(Xs.iloc[:n_fit]), self._xmat(Xs.iloc[-n_val:])
        best, best_loss = self.l1_grid[0], np.inf
        for lam in self.l1_grid:
            p = self._fit_levels(y[:n_fit], d_fit, self.alphas, lam)
            loss = np.mean([tick_loss(y[-n_val:], -(d_val @ p[a]), a).mean() for a in self.alphas])
            if loss < best_loss:
                best, best_loss = lam, loss
        return best

    def fit(self, train: np.ndarray, X: pd.DataFrame) -> LinearQuantileRegression:
        self.cols = list(self.features) if self.features is not None else list(X.columns)
        y = np.asarray(train, dtype=float)[1:]  # r_{s+1}
        Xs = X[self.cols].iloc[:-1]  # x_s
        ok = np.isfinite(Xs.to_numpy(dtype=float)).all(axis=1)
        Xs, y = Xs[ok], y[ok]
        if len(y) < 100:
            raise ValueError(f"{self.name}: only {len(y)} complete training rows")
        lam = None
        if self.l1_grid is not None:
            lam = self.lam = self._tune(Xs, y)
        self._prepare(Xs)  # final model: statistics from the whole training window
        self._params = self._fit_levels(y, self._xmat(Xs), self.levels, lam)
        return self

    def fit_info(self) -> dict:
        a = min(self.alphas)
        beta = self._params[a]
        info = {f"b_{c}@{a:g}": float(b) for c, b in zip(["const", *self.cols], beta, strict=True)}
        if self.l1_grid is not None:
            info["l1_lambda"] = self.lam
            info["n_zero_coef"] = int(np.sum(np.abs(beta[1:]) < 1e-12))
        return info

    def predict(self, history: np.ndarray, X: pd.DataFrame) -> Forecast:
        x = X[self.cols].iloc[[-1]]
        if not np.isfinite(x.to_numpy(dtype=float)).all():
            raise ValueError(f"{self.name}: missing feature values at {X.index[-1]}")
        xm = self._xmat(x)[0]
        q = np.sort([xm @ self._params[tau] for tau in self.levels])  # no crossing
        qmap = dict(zip(self.levels, q, strict=True))
        var = {a: -qmap[a] for a in self.alphas}
        es = {a: -float(np.mean([qmap[t] for t in es_levels(a, self.es_k)])) for a in self.alphas}
        return Forecast(var, es)
