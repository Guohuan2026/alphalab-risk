"""Expected Shortfall backtests: McNeil & Frey (2000) exceedance residuals; Acerbi & Szekely
(2014) Z2; Bayer & Dimitriadis (2020) ESR regression backtest (bottom of the file).

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


# ---------- Bayer & Dimitriadis (2020) ESR backtest (spec: RESEARCH_DESIGN Addition 9) ----------
@dataclass(frozen=True)
class ESRResult:
    alpha: float
    n: int
    n_tail: int  # observations at or below the fitted quantile
    strict_g0: float  # ES-equation intercept (return units); H0 = 0
    strict_g1: float  # ES-equation slope on the ES forecast; H0 = 1
    strict_wald: float
    strict_p: float  # chi2(2), two-sided
    int_g0: float  # one-sided intercept ESR: tail mean of Y - ES_hat (< 0 = ES understated)
    int_t: float
    int_p: float  # one-sided: Phi(t)

    def to_dict(self) -> dict:
        return asdict(self)


def quantile_regression(y: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
    """Exact linear quantile regression at level alpha (Koenker–Bassett LP, HiGHS)."""
    from scipy import sparse
    from scipy.optimize import linprog

    n, k = x.shape
    eye = sparse.identity(n, format="csc")
    a_eq = sparse.hstack([sparse.csc_matrix(x), eye, -eye], format="csc")
    c = np.concatenate([np.zeros(k), np.full(n, alpha), np.full(n, 1 - alpha)])
    bounds = [(None, None)] * k + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=a_eq, b_eq=y, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"quantile regression failed: {res.message}")
    return res.x[:k]


def _es_equation(
    y: np.ndarray, q: np.ndarray, xe: np.ndarray, alpha: float, e0: np.ndarray, max_iter: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """ES coefficients from the FZ0 first-order condition given fitted quantiles q.

    FOC: sum X_e (e - S)/e^2 = 0 with S = q + (y - q) 1[y <= q] / alpha, solved by iterated
    weighted least squares (weights 1/e^2, starting from e0 < 0). Returns (gamma, cov).
    """
    s = q + (y - q) * (y <= q) / alpha
    e = e0
    gamma = np.full(xe.shape[1], np.nan)
    for _ in range(max_iter):
        if np.any(e >= 0):
            raise RuntimeError("fitted ES must be negative (return units) for FZ0 weights")
        w = 1.0 / e**2
        new = np.linalg.solve((xe * w[:, None]).T @ xe, (xe * w[:, None]).T @ s)
        converged = np.allclose(new, gamma, rtol=1e-12, atol=1e-14)
        gamma, e = new, xe @ new
        if converged:
            break
    else:
        raise RuntimeError("ES equation did not converge")
    if np.any(e >= 0):
        raise RuntimeError("fitted ES must be negative (return units) for FZ0 weights")
    n = len(y)
    w = 1.0 / e**2
    lam = (xe * w[:, None]).T @ xe / n
    psi = xe * (w * (e - s))[:, None]
    c = psi.T @ psi / n
    lam_inv = np.linalg.inv(lam)
    return gamma, lam_inv @ c @ lam_inv / n


def esr_backtest(loss, var, es, alpha: float, min_tail: int = 10) -> ESRResult:
    """Strict and one-sided intercept ESR tests (Bayer & Dimitriadis 2020), asymptotic p-values.

    loss = realised L_{t+1}; var, es = forecasts made at t (positive = loss). ``var`` is only
    validated: the ESR regressions use the ES forecast alone, as in the paper.
    """
    from scipy import stats

    loss = np.asarray(loss, dtype=float)
    var = np.asarray(var, dtype=float)
    es = np.asarray(es, dtype=float)
    if not (len(loss) == len(var) == len(es)) or len(loss) == 0:
        raise ValueError("loss, var and es must be aligned and non-empty")
    if not (np.all(np.isfinite(loss)) and np.all(np.isfinite(es))):
        raise ValueError("loss and es must be finite")
    if np.any(es <= 0):
        raise ValueError("need ES > 0 (loss units)")
    n = len(loss)
    y, ehat = -loss, -es
    x = np.column_stack([np.ones(n), ehat])
    if np.ptp(ehat) == 0:
        # Constant forecast: the strict slope is not identified (strict = NaN); the intercept
        # test's quantile equation reduces to a constant.
        g0, t, p, n_tail = _intercept_esr(y - ehat, np.ones((n, 1)), alpha, min_tail)
        return ESRResult(alpha, n, n_tail, *([np.nan] * 4), g0, t, p)

    # Strict ESR: quantile and ES on (1, ehat); H0 gamma = (0, 1).
    # Location shift as in esreg: fit on y - c with c = max(y, 0) so every fitted ES is
    # negative (needed by G2(z) = -1/z); only the intercept moves, and it is shifted back.
    c = max(float(y.max()), 0.0)
    q = x @ quantile_regression(y - c, x, alpha)
    n_tail = int(np.sum(y - c <= q))
    nan = ESRResult(alpha, n, n_tail, *([np.nan] * 7))
    if n_tail < min_tail:
        return nan
    g, cov = _es_equation(y - c, q, x, alpha, e0=ehat - c)
    g = g + np.array([c, 0.0])
    d = g - np.array([0.0, 1.0])
    wald = float(d @ np.linalg.solve(cov, d))
    p_strict = float(stats.chi2.sf(wald, 2))

    # One-sided intercept ESR: Z = Y - ehat; quantile on (1, ehat), ES on 1; H0 g0 = 0 vs < 0.
    g0, t, p, _ = _intercept_esr(y - ehat, x, alpha, min_tail)
    return ESRResult(alpha, n, n_tail, float(g[0]), float(g[1]), wald, p_strict, g0, t, p)


def _intercept_esr(
    z: np.ndarray, xq: np.ndarray, alpha: float, min_tail: int
) -> tuple[float, float, float, int]:
    """(gamma0, t, one-sided p, n_tail): with an intercept-only ES equation the FZ0 estimate
    is mean(S), and its sandwich standard error is sd(S)/sqrt(T)."""
    from scipy import stats

    qz = xq @ quantile_regression(z, xq, alpha)
    n_tail = int(np.sum(z <= qz))
    if n_tail < min_tail:
        return np.nan, np.nan, np.nan, n_tail
    s = qz + (z - qz) * (z <= qz) / alpha
    g0 = float(s.mean())
    se = float(s.std(ddof=0) / np.sqrt(len(z)))
    if se == 0:  # degenerate sample (e.g. no variation in Z): no test
        return g0, np.nan, np.nan, n_tail
    t = g0 / se
    return g0, t, float(stats.norm.cdf(t)), n_tail
