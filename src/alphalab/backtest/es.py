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

import warnings
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
    strict_p: float  # chi2(2), two-sided (secondary test; Addition 9)
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


TIE_RTOL = 1e-7  # tie band for "at or below the fitted quantile", in normalised units


def _at_or_below(resid: np.ndarray, scale: float) -> np.ndarray:
    """1[resid <= 0] with a tie band: resid <= TIE_RTOL * (1 + scale).

    Why: an LP quantile fit interpolates a few observations exactly (y = q in exact
    arithmetic), but their computed residuals are ~±1e-15, so a raw ``<= 0`` flips with the
    last bits (it moved the Wald by up to 5% between macOS and Linux, 2026-11-10). Every
    definition involved (S_t, esreg's tail selection, the empirical CDF) uses 1[Y <= q], so
    exact arithmetic counts these points as "at or below"; the band reproduces that.

    Approximation (documented, not a guarantee): a genuine observation within the band above
    q is also counted. The band is ~5 orders above the LP's feasibility error and, for
    continuous data with density f, catches a genuine point with probability about
    2·n·f·tol (~1e-4 at n = 6,500); it changes counts by at most the few points inside it.
    Other hard boundaries (bandwidth bins, the identification threshold) remain, so results
    are not invariant to the last bits in every possible sample.
    """
    return resid <= TIE_RTOL * (1.0 + scale)


def _hall_sheather(n: int, alpha: float) -> float:
    from scipy import stats

    z = stats.norm.ppf(alpha)
    return float(
        n ** (-1 / 3)
        * stats.norm.ppf(0.975) ** (2 / 3)
        * (1.5 * stats.norm.pdf(z) ** 2 / (2 * z**2 + 1)) ** (1 / 3)
    )


# --- Covariance: a port of esreg::vcovA(sparsity="nid", sigma_est="scl_sp", misspec=TRUE)
# (esreg 0.6.3, Bayer & Dimitriadis; sandwich_loops.cpp, covariance helpers in R). Ported
# 2026-11-10 after a cross-check against esreg showed that the earlier empirical
# outer-product meat over-rejected (21% at T = 1,000) where esreg's model-based meat has
# close to nominal size. Inputs are the SHIFTED data (y - max(y)) and coefficients, as in
# esreg, so the misspecification terms that involve q in levels match.


def _cond_mean_sigma(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """esreg conditional_mean_sigma: Gaussian ML of y ~ N(x b1, (x b2)^2), started from OLS
    of y and of |residual| on x; BFGS, then Nelder–Mead, else the starting values.

    Unlike esreg (R optim defaults, which stop early: up to ~9,500 log-likelihood units short
    on one P1 series), BFGS is restarted from its own solution until the likelihood stops
    improving, so the result is the actual maximum and does not depend on the last bits of
    the input (2026-11-10 cross-check)."""
    from scipy import stats
    from scipy.optimize import minimize

    k = x.shape[1]
    b1 = np.linalg.lstsq(x, y, rcond=None)[0]
    b2 = np.linalg.lstsq(x, np.abs(y - x @ b1), rcond=None)[0]
    b2[0] -= min(0.001, float(np.min(x @ b2)))
    b0 = np.concatenate([b1, b2])

    def nll(par):
        sig = x @ par[k:]
        if not np.all(sig > 0):
            return np.inf
        return -float(np.sum(stats.norm.logpdf(y, loc=x @ par[:k], scale=sig)))

    def nll_grad(par):  # analytic gradient: exact convergence without finite differences
        mu, sig = x @ par[:k], x @ par[k:]
        if not np.all(sig > 0):
            return np.inf, np.zeros_like(par)
        r = (y - mu) / sig
        val = -float(np.sum(stats.norm.logpdf(r) - np.log(sig)))
        g_mu = -(r / sig)
        g_sig = 1 / sig - r**2 / sig
        return val, np.concatenate([x.T @ g_mu, x.T @ g_sig])

    b = b0
    if not np.isfinite(nll(b0)):  # R's optim errors out on a non-finite start: keep b0
        return x @ b[:k], x @ b[k:]
    best = nll(b0)
    with np.errstate(invalid="ignore", over="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # infeasible trial points
        for _ in range(20):
            res = minimize(nll_grad, b, jac=True, method="BFGS", options={"gtol": 1e-10})
            if not (np.isfinite(res.fun) and res.fun < best - 1e-12 * (1 + abs(best))):
                break
            b, best = res.x, float(res.fun)
        if b is b0:  # BFGS made no progress: Nelder–Mead, as esreg's fallback
            res = minimize(nll, b0, method="Nelder-Mead", options={"maxiter": 10_000})
            if res.success and np.isfinite(res.fun) and res.fun < best:
                b = res.x
    return x @ b[:k], x @ b[k:]


def _bw_sj(v: np.ndarray, nb: int = 1000) -> float:
    """R's bw.SJ(method = "ste"): Sheather–Jones solve-the-equation bandwidth with R's binned
    pair counts (n > nb/2) or exact pair counts, and the same bracket and tolerance."""
    from scipy.optimize import brentq

    v = np.asarray(v, dtype=float)
    n = len(v)
    rng = float(v.max() - v.min())
    if rng == 0:
        raise ESRFitError("data are constant in bandwidth calculation")
    if n > nb / 2:
        d = rng * 1.01 / nb
        xx = (np.trunc(np.abs(v) / d) * np.sign(v)).astype(np.int64)
        xx = xx - xx.min()
        w = np.bincount(xx[xx < nb], minlength=nb).astype(float)
        full = np.correlate(w, w, mode="full")[nb - 1 :]  # sum_i w_i w_{i+k}
        cnt = full.copy()
        cnt[0] = float(np.sum(w * (w - 1))) / 2
    else:
        d = rng * 1.01 / nb
        ii = np.trunc(v / d).astype(np.int64)
        diff = np.abs(ii[:, None] - ii[None, :])[np.triu_indices(n, 1)]
        cnt = np.bincount(diff, minlength=nb)[:nb].astype(float)
    idx = np.arange(nb)

    def _sum(h, poly, diag):
        delta = (idx * d / h) ** 2
        keep = delta < 1000
        return 2 * float(np.sum(np.exp(-delta[keep] / 2) * poly(delta[keep]) * cnt[keep])) + diag

    def phi4(h):
        s = _sum(h, lambda t: t * t - 6 * t + 3, 3.0 * n)
        return s / (n * (n - 1) * h**5) / np.sqrt(2 * np.pi)

    def phi6(h):
        s = _sum(h, lambda t: t**3 - 15 * t**2 + 45 * t - 15, -15.0 * n)
        return s / (n * (n - 1) * h**7) / np.sqrt(2 * np.pi)

    q75, q25 = np.percentile(v, [75, 25])
    scale = min(float(np.std(v, ddof=1)), float(q75 - q25) / 1.349)
    a = 1.24 * scale * n ** (-1 / 7)
    b = 1.23 * scale * n ** (-1 / 9)
    c1 = 1 / (2 * np.sqrt(np.pi) * n)
    td = -phi6(b)
    if not np.isfinite(td) or td <= 0:
        raise ESRFitError("sample is too sparse to find TD")
    alph2 = 1.357 * (phi4(a) / td) ** (1 / 7)
    if not np.isfinite(alph2):
        raise ESRFitError("sample is too sparse to find alph2")

    def fsd(h):
        return (c1 / phi4(alph2 * h ** (5 / 7))) ** (1 / 5) - h

    hmax = 1.144 * scale * n ** (-1 / 5)
    lower, upper = 0.1 * hmax, hmax
    tol = 0.1 * lower
    for itry in range(1, 101):
        if fsd(lower) * fsd(upper) <= 0:
            break
        if itry > 99:
            raise ESRFitError("no solution in the specified range of bandwidths")
        if itry % 2:
            upper *= 1.2
        else:
            lower /= 1.2
    return float(brentq(fsd, lower, upper, xtol=tol))


def _cond_trunc_var(u: np.ndarray, x: np.ndarray, approach: str) -> tuple[np.ndarray, str]:
    """esreg conditional_truncated_variance of the quantile residuals u (tail: u <= 0).
    "scl_sp": location-scale model + Sheather–Jones kernel density of the standardised
    residuals, truncated moments by the trapezoid rule; falls back to "ind" (the sample
    variance of u[u <= 0]) exactly when esreg does. Returns (cv, approach actually used)."""
    from scipy import stats

    tail = _at_or_below(u, float(np.max(np.abs(u))))  # same tie policy as everywhere else
    if np.sum(tail) <= 2:
        raise ESRFitError("not enough negative quantile residuals")
    ind = np.full(len(u), float(np.var(u[tail], ddof=1)))
    if approach == "ind":
        return ind, "ind"
    try:
        mu, sig = _cond_mean_sigma(u, x)
        z = (u - mu) / sig
        bw = _bw_sj(z)
        grid = np.linspace(z.min() - 3 * bw, z.max() + 3 * bw, 512)  # R density(): n=512, cut=3
        dens = stats.norm.pdf((grid[:, None] - z[None, :]) / bw).mean(axis=1) / bw
        beta = -mu / sig
        div = 1000
        h = (float(beta.max()) - grid[0]) / (div - 1)
        b_approx = grid[0] + h * np.arange(div + 1)  # R seq(lower, max(beta) + h, h)
        y0 = np.interp(b_approx, grid, dens, left=0.0, right=0.0)
        y1, y2 = b_approx * y0, b_approx**2 * y0
        drop = div - 1  # R's x[-div]: remove the div-th element (1-based)
        cb = np.cumsum(y0[1:] + np.delete(y0, drop)) / 2 * h
        m1 = np.cumsum(y1[1:] + np.delete(y1, drop)) / 2 * h
        m2 = np.cumsum(y2[1:] + np.delete(y2, drop)) / 2 * h
        with np.errstate(divide="ignore", invalid="ignore"):
            cv_approx = m2 / cb - (m1 / cb) ** 2
        mid = np.delete(b_approx, drop) + h / 2
        ok = np.isfinite(cv_approx) & (cv_approx >= 0)  # R: negative -> NA, NA pairs dropped
        if ok.sum() < 2:
            raise ESRFitError("truncated variance grid is empty")
        mo, co = mid[ok], cv_approx[ok]
        inside = (beta >= mo[0]) & (beta <= mo[-1])  # R approx(rule = 1): NA outside
        cv = sig**2 * np.interp(beta, mo, co)
        if not inside.all() or not np.all(np.isfinite(cv)) or np.any(cv < 0):
            raise ESRFitError("scl_sp failed")
        return cv, "scl_sp"
    except (ESRFitError, ValueError, FloatingPointError):
        return ind, "ind"


def _cdf_at_quantile(y: np.ndarray, x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """esreg cdf_at_quantile: empirical CDF of location-scale standardised y at q."""
    mu, sig = _cond_mean_sigma(y, x)
    v = np.sort((y - mu) / sig)
    # cdf_i = share of j with v_j - z_i <= 0, with the tie band of _at_or_below (points the
    # LP fit interpolates sit exactly at q; R's ecdf counts them when equality is exact).
    z = (q - mu) / sig
    tol = TIE_RTOL * (1.0 + float(np.max(np.abs(v))))
    return np.searchsorted(v, z + tol, side="right") / len(v)


def _density_at_quantile(
    y: np.ndarray, x: np.ndarray, u: np.ndarray, alpha: float, sparsity: str
) -> np.ndarray:
    """esreg density_quantile_function, Hall–Sheather bandwidth: 'nid' difference quotients
    of quantile regressions at alpha +/- h, or the 'iid' order-statistic estimate."""
    n, k = x.shape
    h = _hall_sheather(n, alpha)
    if sparsity == "iid":
        m = max(k + 1, int(np.ceil(n * h)))
        ir = np.arange(k + 1, m + k + 2)  # R (k+1):(h+k+1), 1-based
        ord_resid = np.sort(u[np.argsort(np.abs(u), kind="stable")][ir - 1])
        xt = ir / (n - k)
        slope = quantile_regression(ord_resid, np.column_stack([np.ones(len(xt)), xt]), 0.5)[1]
        return np.full(n, 1.0 / slope)
    if not (alpha - h > 0 and alpha + h < 1):
        raise ESRFitError("density bandwidth leaves (0, 1)")
    eps = np.finfo(float).eps ** (2 / 3)
    dq = x @ (quantile_regression(y, x, alpha + h) - quantile_regression(y, x, alpha - h))
    return np.maximum(0.0, 2 * h / (dq - eps))


def _joint_cov(
    y: np.ndarray,
    xq: np.ndarray,
    xe: np.ndarray,
    alpha: float,
    beta: np.ndarray,
    gamma: np.ndarray,
) -> tuple[np.ndarray, str]:
    """ES block of esreg's vcovA (nid / scl_sp / misspec = TRUE) for G1 = 0, G2(z) = -1/z.
    ``y`` must already be shifted so that max(y) = 0 (esreg's shift). Returns (cov, approach
    used for the truncated variance)."""
    n, kq = xq.shape
    ke = xe.shape[1]
    q, e = xq @ beta, xe @ gamma
    if np.any(e >= 0):
        raise ESRFitError("fitted ES must be negative")
    simple = kq == 1 and ke == 1  # esreg switches to iid sparsity and ind variance here
    f = _density_at_quantile(y, xq, y - q, alpha, "iid" if simple else "nid")
    cdf = _cdf_at_quantile(y, xq, q)
    cv, approach = _cond_trunc_var(y - q, xq, "ind" if simple else "scl_sp")
    g2, g2p, g2pp = -1 / e, 1 / e**2, -2 / e**3
    m = (cdf - alpha) / alpha

    def outer(a_, b_, w):
        return (a_ * w[:, None]).T @ b_ / n

    l11 = outer(xq, xq, g2 / alpha * f)
    l12 = outer(xq, xe, g2p * m)
    l22 = outer(xe, xe, g2p + g2pp * q * m)
    lam = np.block([[l11, l12], [l12.T, l22]])
    r = (1 - alpha) / alpha
    s11 = outer(xq, xq, g2**2 * (r + (1 - 2 * alpha) * (cdf - alpha) / alpha**2))
    s12 = outer(xe, xq, g2 * g2p * (r * (q - e) + (r * q * m - m * (q - e))))
    s22 = outer(xe, xe, g2p**2 * (cv / alpha + r * (q - e) ** 2 + 2 * (q - e) * q * (-m)))
    sig = np.block([[s11, s12.T], [s12, s22]])
    if not (np.all(np.isfinite(lam)) and np.all(np.isfinite(sig))):
        raise ESRFitError("sandwich is not finite")
    if np.linalg.cond(lam) > 1e12:
        raise ESRFitError("sandwich bread is singular or ill-conditioned")
    lam_inv = np.linalg.inv(lam)
    return (lam_inv @ sig @ lam_inv / n)[kq:, kq:], approach


def _strict_fit(
    y: np.ndarray, x: np.ndarray, alpha: float, min_tail: int
) -> tuple[np.ndarray, np.ndarray, str]:
    """Strict ESR fit in normalised units: quantile and ES on x = (1, ehat). Location shift
    as in esreg (fit on y - max(y)); the start gamma = (-max(c, 0), 1) keeps every e < 0;
    only the intercept moves and it is shifted back. Returns (gamma, ES-block cov, variance
    approach)."""
    c = float(y.max())
    beta, g = _fit_joint(y - c, x, x, alpha, np.array([-max(c, 0.0), 1.0]))
    ys = y - c
    n_q = int(np.sum(_at_or_below(ys - x @ beta, float(np.max(np.abs(ys))))))
    if n_q < min_tail:
        raise ESRFitError(f"only {n_q} observations at or below the fitted quantile")
    cov, approach = _joint_cov(y - c, x, x, alpha, beta, g)
    return g + np.array([c, 0.0]), cov, approach


def _intercept_fit(
    z: np.ndarray, x: np.ndarray, alpha: float, min_tail: int
) -> tuple[float, float, int, str]:
    """One-sided intercept ESR fit on z = y - ehat in normalised units: quantile on x, ES on a
    constant; shifted by max(z) as in esreg, started below every z. Returns (gamma0, se,
    n_tail, variance approach)."""
    n = len(z)
    cz = float(z.max())
    ones = np.ones((n, 1))
    start = float(np.min(z - cz)) - float(np.std(z)) - 1e-12
    beta, g = _fit_joint(z - cz, x, ones, alpha, np.array([start]))
    zs = z - cz
    n_tail = int(np.sum(_at_or_below(zs - x @ beta, float(np.max(np.abs(zs))))))
    if n_tail < min_tail:
        raise ESRFitError(f"only {n_tail} observations at or below the fitted quantile")
    cov, approach = _joint_cov(z - cz, x, ones, alpha, beta, g)
    se = float(np.sqrt(cov[0, 0]))
    if not (np.isfinite(se) and se > 0):
        raise ESRFitError("zero or undefined standard error")
    return float(g[0] + cz), se, n_tail, approach


def esr_bootstrap_draws(
    y: np.ndarray,
    ehat: np.ndarray,
    alpha: float,
    idx: np.ndarray,
    g_hat,
    g0_hat,
    min_tail: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Studentised pairs-bootstrap statistics, as esback::esr_backtest(B > 0): for each row of
    ``idx`` (resampled positions) refit both versions and return
    W* = (g* - g_hat)' cov*^-1 (g* - g_hat) (strict) and t* = (g0* - g0_hat)/se* (intercept),
    NaN where a resample cannot be fitted. ``y``, ``ehat``, ``g_hat`` and ``g0_hat`` are in
    the normalised units of :func:`esr_backtest` (statistics are unit-free)."""
    w_star = np.full(len(idx), np.nan)
    t_star = np.full(len(idx), np.nan)
    for i, ix in enumerate(idx):
        yb, eb = y[ix], ehat[ix]
        if not np.std(eb) > 1e-8 * np.mean(np.abs(eb)):
            continue
        x = np.column_stack([np.ones(len(ix)), eb])
        try:
            g, cov, _ = _strict_fit(yb, x, alpha, min_tail)
            d = g - np.asarray(g_hat, dtype=float)
            w = float(d @ np.linalg.solve(cov, d))
            w_star[i] = w if np.isfinite(w) and w >= 0 else np.nan
        except (ESRFitError, np.linalg.LinAlgError):
            pass
        try:
            g0, se, _, _ = _intercept_fit(yb - eb, x, alpha, min_tail)
            t_star[i] = (g0 - g0_hat) / se
        except (ESRFitError, np.linalg.LinAlgError):
            pass
    return w_star, t_star


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
    with np.errstate(over="ignore", invalid="ignore"):
        unit = float(np.std(loss)) or float(np.mean(es))
        y, ehat = -loss / unit, -es / unit
    if not (
        np.isfinite(unit) and unit > 0 and np.all(np.isfinite(y)) and np.all(np.isfinite(ehat))
    ):
        raise ValueError("inputs too large to normalise (sd or mean overflows)")
    # The slope is identified only if the forecast varies (relative to its own size).
    identified = np.std(ehat) > 1e-8 * np.mean(np.abs(ehat))
    x = np.column_stack([np.ones(n), ehat]) if identified else np.ones((n, 1))

    sg0 = sg1 = wald = p_strict = np.nan
    if not identified:
        s_status = "ES forecast (nearly) constant: slope not identified"
    else:
        try:
            g, cov, s_var = _strict_fit(y, x, alpha, min_tail)
            d = g - np.array([0.0, 1.0])
            wald = float(d @ np.linalg.solve(cov, d))
            if not np.isfinite(wald) or wald < 0:
                raise ESRFitError("covariance is not positive definite")
            sg0, sg1 = float(g[0] * unit), float(g[1])
            p_strict = float(stats.chi2.sf(wald, 2))
            s_status = "ok" if s_var == "scl_sp" else f"ok ({s_var} variance fallback)"
        except (ESRFitError, np.linalg.LinAlgError) as err:
            sg0 = sg1 = wald = np.nan
            s_status = str(err)

    g0 = t = p_int = np.nan
    n_tail = 0
    try:
        g0n, se, n_tail, i_var = _intercept_fit(y - ehat, x, alpha, min_tail)
        t = g0n / se
        g0 = g0n * unit
        p_int = float(stats.norm.cdf(t))
        expected = "scl_sp" if x.shape[1] > 1 else "ind"  # esreg's own choice for kq = ke = 1
        i_status = "ok" if i_var == expected else f"ok ({i_var} variance fallback)"
    except (ESRFitError, np.linalg.LinAlgError) as err:
        g0 = t = p_int = np.nan
        i_status = str(err)
    return ESRResult(alpha, n, n_tail, sg0, sg1, wald, p_strict, s_status, g0, t, p_int, i_status)
