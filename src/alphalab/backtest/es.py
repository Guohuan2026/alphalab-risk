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
# Returns convention: Y = -L, e = ES in return units (< 0). FZ0 loss (G1 = 0, G2(z) = -1/z):
#   rho(q, e; y) = (1/(-e)) [ (q - y) 1[y <= q] / alpha - q ] + log(-e) - 1.
# Given e, the q-part is a pinball loss weighted by 1/(-e_t); given q, the e-part is smooth
# with first-order condition sum X_e (e - S)/e^2 = 0, S = q + (y - q) 1[y <= q] / alpha.


class ESRFitError(RuntimeError):
    """The joint quantile/ES regression could not be fitted on this sample."""


@dataclass(frozen=True)
class ESRResult:
    alpha: float
    n: int
    n_tail: int  # observations at or below the fitted quantile (intercept-test fit)
    strict_g0: float  # ES-equation intercept (return units); H0 = 0
    strict_g1: float  # ES-equation slope on the ES forecast; H0 = 1
    strict_wald: float
    strict_p: float  # chi2(2), two-sided; over-sized in finite samples (LIMITATIONS #17)
    strict_status: str  # "ok" or why the strict test is unavailable
    int_g0: float  # one-sided intercept ESR: ES of Y - ES_hat (< 0 = ES understated)
    int_t: float
    int_p: float  # one-sided: Phi(t)
    int_status: str

    def to_dict(self) -> dict:
        return asdict(self)


def quantile_regression(
    y: np.ndarray, x: np.ndarray, alpha: float, weights: np.ndarray | None = None
) -> np.ndarray:
    """Exact (optionally weighted) linear quantile regression at level alpha (LP, HiGHS):
    argmin_b sum_t w_t rho_alpha(y_t - x_t'b)."""
    from scipy import sparse
    from scipy.optimize import linprog

    n, k = x.shape
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    eye = sparse.identity(n, format="csc")
    a_eq = sparse.hstack([sparse.csc_matrix(x), eye, -eye], format="csc")
    c = np.concatenate([np.zeros(k), alpha * w, (1 - alpha) * w])
    bounds = [(None, None)] * k + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=a_eq, b_eq=y, bounds=bounds, method="highs")
    if not res.success:
        raise ESRFitError(f"quantile regression failed: {res.message}")
    return res.x[:k]


def _fz0(y: np.ndarray, q: np.ndarray, e: np.ndarray, alpha: float) -> float:
    return float(np.mean(((q - y) * (y <= q) / alpha - q) / (-e) + np.log(-e) - 1))


def _es_step(
    y: np.ndarray, q: np.ndarray, xe: np.ndarray, alpha: float, gamma: np.ndarray
) -> np.ndarray:
    """Minimise the FZ0 e-part over gamma for fixed q: IRLS (weights 1/e^2) with step halving
    that keeps every fitted e < 0 and never increases the objective."""
    s = q + (y - q) * (y <= q) / alpha
    e = xe @ gamma
    obj = _fz0(y, q, e, alpha)
    for _ in range(200):
        w = 1.0 / e**2
        target = np.linalg.lstsq(xe * np.sqrt(w)[:, None], s * np.sqrt(w), rcond=None)[0]
        step, t = target - gamma, 1.0
        while True:
            cand = gamma + t * step
            ec = xe @ cand
            if np.all(ec < 0):
                oc = _fz0(y, q, ec, alpha)
                if oc <= obj + 1e-15 * (1 + abs(obj)):
                    break
            t /= 2
            if t < 1e-10:
                raise ESRFitError("ES step found no feasible descent (fitted ES must stay < 0)")
        done = np.max(np.abs(t * step)) <= 1e-12 * (1 + np.max(np.abs(gamma)))
        gamma, e, obj = cand, ec, oc
        if done:
            return gamma
    raise ESRFitError("ES step did not converge")


def _fit_joint(
    y: np.ndarray, xq: np.ndarray, xe: np.ndarray, alpha: float, gamma0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Joint FZ0 M-estimator by block coordinate descent: weighted quantile regression given
    e (exact), then the ES step given q; each block lowers the joint loss, stop at no gain."""
    gamma = np.asarray(gamma0, dtype=float)
    e = xe @ gamma
    if np.any(e >= 0):
        raise ESRFitError("starting ES must be negative")
    obj = np.inf
    for _ in range(100):
        beta = quantile_regression(y, xq, alpha, weights=1.0 / (-e))
        q = xq @ beta
        gamma = _es_step(y, q, xe, alpha, gamma)
        e = xe @ gamma
        new = _fz0(y, q, e, alpha)
        if obj - new <= 1e-12 * (1 + abs(new)):
            return beta, gamma
        obj = new
    raise ESRFitError("joint FZ0 estimation did not converge")


def _below(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """1[y < q] with observations ON the fitted quantile counted as not below.

    An LP quantile fit interpolates a few observations (residual ~1e-15); whether they land
    at, above or below q then depends on the last bits, which made the covariance differ
    across platforms (2026-11-10). The tolerance is far above the LP's feasibility error
    and far below the spacing of continuous data in normalised units.
    """
    tol = 1e-7 * (1.0 + float(np.max(np.abs(y))))
    return y < q - tol


def _hall_sheather(n: int, alpha: float) -> float:
    from scipy import stats

    z = stats.norm.ppf(alpha)
    return float(
        n ** (-1 / 3)
        * stats.norm.ppf(0.975) ** (2 / 3)
        * (1.5 * stats.norm.pdf(z) ** 2 / (2 * z**2 + 1)) ** (1 / 3)
    )


def _joint_cov(
    y: np.ndarray,
    xq: np.ndarray,
    xe: np.ndarray,
    alpha: float,
    beta: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    """Misspecification-robust sandwich Lambda^-1 C Lambda^-T / T for (beta, gamma) of the joint
    FZ0 estimator (Dimitriadis & Bayer 2019; esreg vcovA with misspecification terms). The
    conditional density at q_t uses Hendricks–Koenker 'nid' difference quotients."""
    n, kq = xq.shape
    q, e = xq @ beta, xe @ gamma
    ind = _below(y, q).astype(float)
    s = q + (y - q) * ind / alpha
    psi = np.hstack([xq * ((ind - alpha) / (alpha * -e))[:, None], xe * ((e - s) / e**2)[:, None]])
    c = psi.T @ psi / n
    h = _hall_sheather(n, alpha)
    lo, hi = max(alpha - h, alpha / 2), min(alpha + h, (1 + alpha) / 2)  # ordered, in (0, 1)
    dq = xq @ (quantile_regression(y, xq, hi) - quantile_regression(y, xq, lo))
    f = np.where(dq > 0, (hi - lo) / np.where(dq > 0, dq, 1.0), 0.0)
    l11 = (xq * (f / (alpha * -e))[:, None]).T @ xq / n
    l12 = (xq * ((ind - alpha) / (alpha * e**2))[:, None]).T @ xe / n
    l21 = (xe * ((ind / alpha - 1) / e**2)[:, None]).T @ xq / n
    l22 = (xe * (1 / e**2 - 2 * (e - s) / e**3)[:, None]).T @ xe / n
    lam = np.block([[l11, l12], [l21, l22]])
    if not np.all(np.isfinite(lam)) or np.linalg.cond(lam) > 1e12:
        raise ESRFitError("sandwich bread is singular or ill-conditioned")
    lam_inv = np.linalg.inv(lam)
    return (lam_inv @ c @ lam_inv.T / n)[kq:, kq:]


def esr_backtest(loss, var, es, alpha: float, min_tail: int = 10) -> ESRResult:
    """Strict and one-sided intercept ESR tests (Bayer & Dimitriadis 2020), asymptotic p-values.

    loss = realised L_{t+1}; var, es = forecasts made at t (positive = loss). ``var`` is only
    checked (finite, aligned): the ESR regressions use the ES forecast alone, as in the
    paper. A test that cannot be computed on the sample returns NaN with the reason in its
    ``*_status`` field; the other test is still computed. Inputs are normalised by sd(loss)
    internally, so results do not depend on units.
    """
    from scipy import stats

    if not (np.isfinite(alpha) and 0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    loss = np.asarray(loss, dtype=float)
    var = np.asarray(var, dtype=float)
    es = np.asarray(es, dtype=float)
    if loss.ndim != 1 or var.ndim != 1 or es.ndim != 1:
        raise ValueError("loss, var and es must be one-dimensional")
    if not (len(loss) == len(var) == len(es)) or len(loss) == 0:
        raise ValueError("loss, var and es must be aligned and non-empty")
    if not (np.all(np.isfinite(loss)) and np.all(np.isfinite(var)) and np.all(np.isfinite(es))):
        raise ValueError("loss, var and es must be finite")
    if np.any(es <= 0):
        raise ValueError("need ES > 0 (loss units)")
    n = len(loss)
    # Work in normalised units (divide by ``unit``) so tolerances and the conditioning check do not
    # depend on whether returns are in decimals, percent or basis points. Slopes, Wald and t
    # statistics are invariant; intercepts are multiplied back by ``unit``.
    unit = float(np.std(loss)) or float(np.mean(es))
    y, ehat = -loss / unit, -es / unit
    # The slope is identified only if the forecast varies (relative to its own size).
    identified = np.std(ehat) > 1e-8 * np.mean(np.abs(ehat))
    x = np.column_stack([np.ones(n), ehat]) if identified else np.ones((n, 1))

    # Strict ESR: quantile and ES on (1, ehat); H0 gamma = (0, 1). Location shift as in
    # esreg: fit on y - c, c = max(y, 0), starting at gamma = (-c, 1) so every e < 0; only the
    # intercept moves, and it is shifted back.
    sg0 = sg1 = wald = p_strict = np.nan
    if not identified:
        s_status = "ES forecast (nearly) constant: slope not identified"
    else:
        c = max(float(y.max()), 0.0)
        try:
            beta, g = _fit_joint(y - c, x, x, alpha, np.array([-c, 1.0]))
            n_q = int(np.sum(~_below(x @ beta, y - c)))  # at or below the fitted quantile
            if n_q < min_tail:
                raise ESRFitError(f"only {n_q} observations at or below the fitted quantile")
            cov = _joint_cov(y - c, x, x, alpha, beta, g)
            d = g + np.array([c, 0.0]) - np.array([0.0, 1.0])
            wald = float(d @ np.linalg.solve(cov, d))
            if not np.isfinite(wald) or wald < 0:
                raise ESRFitError("covariance is not positive definite")
            sg0, sg1 = float((g[0] + c) * unit), float(g[1])
            p_strict, s_status = float(stats.chi2.sf(wald, 2)), "ok"
        except (ESRFitError, np.linalg.LinAlgError) as err:
            sg0 = sg1 = wald = np.nan
            s_status = str(err)

    # One-sided intercept ESR: Z = Y - ehat; quantile on (1, ehat), ES on a constant;
    # H0 g0 = 0 vs g0 < 0. Shifted by cz = max(z, 0); start below every z so e < 0.
    z = y - ehat
    cz = max(float(z.max()), 0.0)
    ones = np.ones((n, 1))
    g0 = t = p_int = np.nan
    n_tail = 0
    try:
        start = float(np.min(z - cz)) - float(np.std(z)) - 1e-12
        beta, g = _fit_joint(z - cz, x, ones, alpha, np.array([start]))
        n_tail = int(np.sum(~_below(x @ beta, z - cz)))  # at or below the fitted quantile
        if n_tail < min_tail:
            raise ESRFitError(f"only {n_tail} observations at or below the fitted quantile")
        se = float(np.sqrt(_joint_cov(z - cz, x, ones, alpha, beta, g)[0, 0]))
        if not (np.isfinite(se) and se > 0):
            raise ESRFitError("zero or undefined standard error")
        t = float(g[0] + cz) / se
        g0 = float(g[0] + cz) * unit
        p_int, i_status = float(stats.norm.cdf(t)), "ok"
    except (ESRFitError, np.linalg.LinAlgError) as err:
        g0 = t = p_int = np.nan
        i_status = str(err)
    return ESRResult(alpha, n, n_tail, sg0, sg1, wald, p_strict, s_status, g0, t, p_int, i_status)
