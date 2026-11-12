"""EWMA volatility with normal quantiles: J.P. Morgan RiskMetrics (1996).

    σ²_{t+1} = λ σ²_t + (1 − λ) r_t²,          λ = 0.94 (RiskMetrics daily value)
    VaR_a = σ_{t+1} · z_a,   ES_a = σ_{t+1} · φ(z_a) / a,   z_a = Φ⁻¹(1 − a)

Zero mean is assumed (standard for daily risk; the daily mean is ~0.04%, much smaller
than σ ≈ 1%). λ is fixed, so there is nothing to estimate and no estimation risk. That
makes it a strong, transparent benchmark. Its known weakness: normal tails. With fat-tailed
returns it tends to under-cover at 1% even when volatility is tracked well, and to
understate ES more than VaR.

Unrolled, the variance is an exponentially weighted average of past squared returns:

    σ²_{n} = λ^n σ²_0 + (1 − λ) Σ_{k=0}^{n−1} λ^{n−1−k} r_k²

σ²_0 is the mean of r² over the first ``init_obs`` returns of the history. The history
always starts at the same date, so this does not depend on the origin (point-in-time).
With λ = 0.94 its weight after 1,000 days is 0.94^1000 ≈ 1e-27, i.e. irrelevant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from alphalab.risk.models.base import Forecast, RiskModel


def ewma_variance(returns: np.ndarray, lam: float, init_obs: int = 30) -> float:
    """σ² forecast for the day after ``returns[-1]`` (vectorised form of the recursion)."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < init_obs:
        raise ValueError(f"EWMA needs at least {init_obs} observations")
    s2_0 = float(np.mean(r[:init_obs] ** 2))
    weights = lam ** np.arange(n - 1, -1, -1)  # λ^{n−1−k}
    return float(lam**n * s2_0 + (1 - lam) * np.dot(weights, r**2))


@dataclass
class EWMANormal(RiskModel):
    lam: float = 0.94
    init_obs: int = 30

    def __post_init__(self):
        super().__post_init__()
        if not 0 < self.lam < 1:
            raise ValueError("lambda must be in (0, 1)")
        self.name = self.name if self.name != "ewmanormal" else "ewma"

    @property
    def min_history(self) -> int:
        return self.init_obs

    def predict(self, history: np.ndarray) -> Forecast:
        sigma = np.sqrt(ewma_variance(history, self.lam, self.init_obs))
        var, es = {}, {}
        for a in self.alphas:
            z = stats.norm.ppf(1 - a)
            var[a] = float(sigma * z)
            es[a] = float(sigma * stats.norm.pdf(z) / a)
        return Forecast(var, es, float(sigma))
