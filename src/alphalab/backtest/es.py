"""Expected Shortfall backtests: McNeil & Frey (2000) exceedance residuals; Acerbi & Szekely
(2014) Z2 (bottom of the file).

McNeil–Frey:

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


# ---------- Acerbi & Szekely (2014) Z2 (spec: RESEARCH_DESIGN Clarification 7) ----------
@dataclass(frozen=True)
class Z2Result:
    alpha: float
    n: int
    n_hits: int
    z2: float  # 0 if VaR and ES are right; < 0 = risk understated
    null_nu: float  # Student-t dof of the shape-matched null (inf = normal)
    p_value: float  # one-sided: P_null(Z2 <= observed)

    def to_dict(self) -> dict:
        return asdict(self)


def _t_tail(alpha: float, nu: float) -> tuple[float, float]:
    """(VaR, ES) of a standard Student-t (nu = inf: normal) for losses at tail prob alpha."""
    from scipy import stats

    if np.isinf(nu):
        q = stats.norm.isf(alpha)
        return q, stats.norm.pdf(q) / alpha
    q = stats.t.isf(alpha, nu)
    return q, (nu + q**2) / (nu - 1) * stats.t.pdf(q, nu) / alpha


NU_MIN = 2.05


def shape_matched_nu(ratio: float, alpha: float) -> float:
    """Student-t dof whose ES/VaR ratio at alpha equals ``ratio`` (inf if normal or thinner)."""
    from scipy.optimize import brentq

    def r(nu):
        v, e = _t_tail(alpha, nu)
        return e / v

    if ratio <= r(np.inf):
        return np.inf
    if ratio >= r(NU_MIN):
        return NU_MIN
    if ratio <= r(1e4):
        return 1e4
    return float(brentq(lambda nu: r(nu) - ratio, NU_MIN, 1e4, xtol=1e-6))


def z2_null(n: int, alpha: float, nu: float, reps: int = 10_000, seed: int = 0) -> np.ndarray:
    """Draws of Z2 under H0 for i.i.d. losses from a (scaled) Student-t with correct VaR/ES.

    Only the hits matter: their number is Binomial(n, alpha), and given a hit the loss is
    the t quantile at a uniform tail probability in (0, alpha). Scale cancels in L/ES.
    """
    from scipy import stats

    rng = np.random.default_rng(seed)
    _, es = _t_tail(alpha, nu)
    k = rng.binomial(n, alpha, size=reps)
    u = rng.uniform(0, alpha, size=int(k.sum()))
    tail = stats.norm.isf(u) if np.isinf(nu) else stats.t.isf(u, nu)
    sums = np.bincount(np.repeat(np.arange(reps), k), weights=tail / es, minlength=reps)
    return 1 - sums / (n * alpha)


def acerbi_szekely_z2(loss, var, es, alpha: float, reps: int = 10_000, seed: int = 0) -> Z2Result:
    """Z2 = 1 − (1/(Tα)) Σ L_t·1[L_t > VaR_t] / ES_t, with a shape-matched Monte Carlo p-value."""
    loss = np.asarray(loss, dtype=float)
    var = np.asarray(var, dtype=float)
    es = np.asarray(es, dtype=float)
    if not (len(loss) == len(var) == len(es)) or len(loss) == 0:
        raise ValueError("loss, var and es must be aligned and non-empty")
    if np.any(var <= 0) or np.any(es < var):
        raise ValueError("need 0 < VaR <= ES")
    hit = loss > var
    n = len(loss)
    z2 = float(1 - np.sum(loss[hit] / es[hit]) / (n * alpha))
    nu = shape_matched_nu(float(np.median(es / var)), alpha)
    null = z2_null(n, alpha, nu, reps, seed)
    p = float((1 + np.sum(null <= z2)) / (reps + 1))
    return Z2Result(alpha, n, int(hit.sum()), z2, nu, p)
