"""Expected Shortfall backtest: McNeil & Frey (2000) exceedance-residual test.

Idea: ES is the mean loss GIVEN that VaR is exceeded. So on the days where L_{t+1} > VaR_t,
the residual ξ = L_{t+1} − ES_t should average zero if ES is right. If ES is too small
(the dangerous direction), ξ averages above zero.

    H0: E[ξ | hit] = 0      H1: E[ξ | hit] > 0   (one-sided: ES understated)

Standardised version: ξ/σ_t, where the model provides σ (GARCH family, EWMA). This removes
the scale differences between calm and turbulent days, which makes the test sharper.

p-value by bootstrap (no normality assumption on the few, fat-tailed residuals): resample
the CENTRED residuals (mean 0, i.e. as under H0) and count how often their mean reaches
the observed mean.

Caveats: the test conditions on the model's OWN hits, so a model with too few hits is
judged on very few points. With fewer than ``min_hits`` exceedances it returns NaN rather
than a misleading p-value. It is an absolute test (acceptable or not); ranking models
uses FZ0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ESTestResult:
    alpha: float
    n_hits: int
    mean_resid: float  # mean of (L − ES) on hit days, in return units
    severity_ratio: float  # mean realised loss / mean ES forecast, on hit days (1 = right)
    p_raw: float  # bootstrap p-value, unstandardised residuals
    p_std: float  # bootstrap p-value, residuals / sigma (NaN if no sigma)

    def to_dict(self) -> dict:
        return asdict(self)


def _boot_p(x: np.ndarray, reps: int, rng: np.random.Generator) -> float:
    obs = x.mean()
    centred = x - obs
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    boot = centred[idx].mean(axis=1)
    return float((1 + np.sum(boot >= obs)) / (reps + 1))


def mcneil_frey(
    loss,
    var,
    es,
    alpha: float,
    sigma=None,
    reps: int = 10_000,
    seed: int = 0,
    min_hits: int = 5,
) -> ESTestResult:
    loss = np.asarray(loss, dtype=float)
    var = np.asarray(var, dtype=float)
    es = np.asarray(es, dtype=float)
    hit = loss > var
    m = int(hit.sum())
    if m < min_hits:
        return ESTestResult(alpha, m, np.nan, np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    resid = loss[hit] - es[hit]
    p_raw = _boot_p(resid, reps, rng)
    p_std = np.nan
    if sigma is not None:
        s = np.asarray(sigma, dtype=float)[hit]
        if np.all(np.isfinite(s)) and np.all(s > 0):
            p_std = _boot_p(resid / s, reps, rng)
    return ESTestResult(
        alpha=alpha,
        n_hits=m,
        mean_resid=float(resid.mean()),
        severity_ratio=float(loss[hit].mean() / es[hit].mean()),
        p_raw=p_raw,
        p_std=float(p_std),
    )
