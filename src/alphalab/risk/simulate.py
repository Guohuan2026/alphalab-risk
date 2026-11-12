"""Simulated returns with KNOWN conditional VaR and ES (ground truth for model tests).

Process: GJR-GARCH(1,1) with standardized Student-t innovations (Glosten, Jagannathan &
Runkle 1993), a standard description of daily equity returns: volatility clusters,
falls raise volatility more than rises (leverage), and tails are fat.

    r_t   = μ + ε_t,      ε_t = σ_t z_t,      z_t ~ t_ν scaled to unit variance
    σ²_t  = ω + (α + γ·1[ε_{t−1} < 0]) ε²_{t−1} + β σ²_{t−1}

σ_t depends only on information up to t−1, so the TRUE one-day-ahead risk of r_t is known
at the previous close:

    VaR_a,t = −μ + σ_t · q_a          ES_a,t = −μ + σ_t · s_a

where q_a and s_a are the lower-tail VaR and ES of the unit-variance t (as positive losses):

    T ~ t_ν,  t_a = F⁻¹_ν(a) < 0,  c = √((ν−2)/ν)
    q_a = −c · t_a
    s_a =  c · f_ν(t_a)/a · (ν + t_a²)/(ν − 1)

Row alignment: ``sigma``, ``true_var_*`` and ``true_es_*`` on row t describe ``ret`` on the
SAME row; they were known at the close of t−1. A model forecast stamped at origin t−1
should be compared with row t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_ALPHAS = (0.01, 0.025, 0.05)


def standardized_t_tail(alpha: float, nu: float) -> tuple[float, float]:
    """(VaR, ES) of a unit-variance Student-t at tail probability ``alpha``, as positive losses."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if nu <= 2:
        raise ValueError("nu must be > 2 for a finite variance")
    c = np.sqrt((nu - 2) / nu)
    t_a = stats.t.ppf(alpha, nu)
    var = -c * t_a
    es = c * stats.t.pdf(t_a, nu) / alpha * (nu + t_a**2) / (nu - 1)
    return float(var), float(es)


@dataclass(frozen=True)
class GJRParams:
    """Defaults resemble SPY daily returns: ~1% daily (~16% annual) unconditional volatility,
    persistence 0.98, strong leverage, ν = 6 (fat tails)."""

    mu: float = 0.0004
    omega: float = 2e-6
    alpha: float = 0.02
    gamma: float = 0.12
    beta: float = 0.90
    nu: float = 6.0

    @property
    def persistence(self) -> float:
        return self.alpha + self.gamma / 2 + self.beta  # γ/2: P(ε<0) = ½ for symmetric z

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1 - self.persistence)

    def check(self) -> None:
        if self.omega <= 0 or self.alpha < 0 or self.beta < 0 or self.alpha + self.gamma < 0:
            raise ValueError("need omega > 0, alpha >= 0, beta >= 0, alpha + gamma >= 0")
        if self.persistence >= 1:
            raise ValueError(f"not covariance-stationary: persistence {self.persistence:.4f} >= 1")
        if self.nu <= 2:
            raise ValueError("nu must be > 2")


def gjr_variance_filter(ret: np.ndarray, p: GJRParams, sigma2_0: float | None = None) -> np.ndarray:
    """σ²_t for each t from returns up to t−1 (row t uses r_{t−1}). Shared by the simulator
    and, later, by models, so both use identical timing."""
    n = len(ret)
    s2 = np.empty(n)
    s2[0] = p.unconditional_variance if sigma2_0 is None else sigma2_0
    for t in range(1, n):
        e = ret[t - 1] - p.mu
        s2[t] = p.omega + (p.alpha + p.gamma * (e < 0)) * e * e + p.beta * s2[t - 1]
    return s2


def simulate_gjr_garch_t(
    n: int,
    params: GJRParams | None = None,
    seed: int = 0,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    burn_in: int = 1000,
    start: str | None = None,
    calendar: str = "XNYS",
) -> pd.DataFrame:
    """Simulate ``n`` daily returns plus their true conditional sigma, VaR and ES.

    Index: 0..n−1, or real trading sessions beginning at ``start``.
    The first ``burn_in`` draws are discarded so the start does not depend on σ²_0.
    """
    p = params or GJRParams()
    p.check()
    rng = np.random.default_rng(seed)
    m = n + burn_in
    z = rng.standard_t(p.nu, m) * np.sqrt((p.nu - 2) / p.nu)  # unit variance
    s2 = np.empty(m)
    r = np.empty(m)
    s2[0] = p.unconditional_variance
    r[0] = p.mu + np.sqrt(s2[0]) * z[0]
    for t in range(1, m):
        e = r[t - 1] - p.mu
        s2[t] = p.omega + (p.alpha + p.gamma * (e < 0)) * e * e + p.beta * s2[t - 1]
        r[t] = p.mu + np.sqrt(s2[t]) * z[t]
    r, sigma = r[burn_in:], np.sqrt(s2[burn_in:])

    out = {"ret": r, "sigma": sigma}
    for a in alphas:
        q, s = standardized_t_tail(a, p.nu)
        out[f"true_var_{a:g}"] = -p.mu + sigma * q
        out[f"true_es_{a:g}"] = -p.mu + sigma * s
    if start is None:
        index = pd.RangeIndex(n)
    else:
        from alphalab.core.calendar import get_calendar

        sess = get_calendar(calendar).sessions
        sess = sess[sess >= pd.Timestamp(start)]
        if len(sess) < n:
            raise ValueError(
                f"only {len(sess)} sessions after {start}; use start=None for long runs"
            )
        index = pd.DatetimeIndex(sess[:n], name="date")
    return pd.DataFrame(out, index=index)
