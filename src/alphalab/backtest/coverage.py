"""VaR coverage backtests: is a VaR forecast calibrated?

A "hit" (exceedance) on day t+1 is L_{t+1} > VaR_t. If VaR is correct at level a, hits are
i.i.d. Bernoulli(a): right FREQUENCY (unconditional coverage) and no CLUSTERING
(independence). Each test is a likelihood ratio (LR): how much better does an unrestricted
model of the hits fit than the null?

Kupiec (1995), proportion of failures. n days, x hits, π̂ = x/n:
    LR_POF = −2 [ (n−x) ln(1−a) + x ln a ] + 2 [ (n−x) ln(1−π̂) + x ln π̂ ]   ~ χ²(1)

Christoffersen (1998), independence. A first-order Markov chain for the hits; n_ij counts
day pairs (state i yesterday → state j today):
    π01 = n01/(n00+n01)   P(hit | no hit yesterday)
    π11 = n11/(n10+n11)   P(hit | hit yesterday)
    π   = (n01+n11)/(n00+n01+n10+n11)
    LR_IND = −2 ln[(1−π)^(n00+n10) π^(n01+n11)]
             + 2 ln[(1−π01)^n00 π01^n01 (1−π11)^n10 π11^n11]          ~ χ²(1)
Conditional coverage: LR_CC = LR_POF + LR_IND ~ χ²(2) (the usual approximation, with
LR_POF computed on the full sample).

Power warning: at a = 1% and 250 days only 2.5 hits are expected. Kupiec then cannot tell
a 1% model from a 2% model reliably; use long samples and also a = 5%.
Convention: 0·ln 0 = 0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * np.log(y)


@dataclass(frozen=True)
class CoverageResult:
    alpha: float
    n: int
    hits: int
    hit_rate: float
    lr_pof: float
    p_pof: float
    lr_ind: float
    p_ind: float
    lr_cc: float
    p_cc: float

    def to_dict(self) -> dict:
        return asdict(self)


def kupiec_pof(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    h = np.asarray(hits, dtype=bool)
    n, x = len(h), int(h.sum())
    if n == 0:
        raise ValueError("no observations")
    pi = x / n
    ll0 = _xlogy(n - x, 1 - alpha) + _xlogy(x, alpha)
    ll1 = _xlogy(n - x, 1 - pi) + _xlogy(x, pi)
    lr = max(0.0, -2 * (ll0 - ll1))
    return lr, float(stats.chi2.sf(lr, 1))


def christoffersen_ind(hits: np.ndarray) -> tuple[float, float]:
    h = np.asarray(hits, dtype=int)
    if len(h) < 2:
        raise ValueError("need at least 2 observations")
    prev, cur = h[:-1], h[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll1 = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    lr = max(0.0, -2 * (ll0 - ll1))
    return lr, float(stats.chi2.sf(lr, 1))


def coverage_tests(hits: np.ndarray, alpha: float) -> CoverageResult:
    h = np.asarray(hits, dtype=bool)
    lr_pof, p_pof = kupiec_pof(h, alpha)
    lr_ind, p_ind = christoffersen_ind(h)
    lr_cc = lr_pof + lr_ind
    return CoverageResult(
        alpha=alpha,
        n=len(h),
        hits=int(h.sum()),
        hit_rate=float(h.mean()),
        lr_pof=lr_pof,
        p_pof=p_pof,
        lr_ind=lr_ind,
        p_ind=p_ind,
        lr_cc=lr_cc,
        p_cc=float(stats.chi2.sf(lr_cc, 2)),
    )


def traffic_light(hits_99: np.ndarray, window: int = 250) -> dict[str, float]:
    """Basel traffic light for 99% VaR: exceedances in each rolling 250-day window.

    Green 0–4, yellow 5–9, red 10+ (Basel Committee, 1996 backtesting framework).
    Returns the share of windows in each zone and the worst window's count.
    """
    h = np.asarray(hits_99, dtype=int)
    if len(h) < window:
        return {"green": np.nan, "yellow": np.nan, "red": np.nan, "max_hits": np.nan, "windows": 0}
    counts = np.convolve(h, np.ones(window, dtype=int), mode="valid")
    return {
        "green": float(np.mean(counts <= 4)),
        "yellow": float(np.mean((counts >= 5) & (counts <= 9))),
        "red": float(np.mean(counts >= 10)),
        "max_hits": int(counts.max()),
        "windows": len(counts),
    }


@dataclass(frozen=True)
class DQResult:
    alpha: float
    n: int
    stat: float
    df: int
    p_value: float

    def to_dict(self) -> dict:
        return asdict(self)


def dq_test(hits: np.ndarray, var: np.ndarray, alpha: float, lags: int = 4) -> DQResult:
    """Engle & Manganelli (2004) Dynamic Quantile test.

    Demeaned hit Hit_t = 1[L_t > VaR_t] − α. Under a correct VaR, Hit_t is unpredictable
    from anything known when the forecast was made. Regress it on a constant, its own
    ``lags`` lags and the VaR forecast itself:

        Hit_t = β0 + Σ_k βk Hit_{t−k} + γ VaR_t + u_t
        DQ = β̂' X'X β̂ / (α(1 − α))  ~  χ²(1 + lags + 1)  under H0

    It nests Kupiec (the constant) and Christoffersen (lag 1), and adds the VaR regressor:
    it catches breaches that become more likely when VaR is LOW, which neither of the
    other tests sees.
    """
    h = np.asarray(hits, dtype=float) - alpha
    v = np.asarray(var, dtype=float)
    if len(h) != len(v) or len(h) <= lags + 10:
        raise ValueError("hits and var must be aligned, with more than lags + 10 observations")
    y = h[lags:]
    cols = [np.ones(len(y))] + [h[lags - k : len(h) - k] for k in range(1, lags + 1)] + [v[lags:]]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    stat = float(beta @ (X.T @ X) @ beta / (alpha * (1 - alpha)))
    df = X.shape[1]
    return DQResult(alpha=alpha, n=len(y), stat=stat, df=df, p_value=float(stats.chi2.sf(stat, df)))
