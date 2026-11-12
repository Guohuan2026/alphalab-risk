"""Milestone 16: finite-sample calibration of the backtests (RESEARCH_DESIGN Addition 10).

EXPLORATORY: designed and run after the lockbox was opened; simulated data only (the real
development series are used only to set the simulation parameters; the lockbox is not used).

Two families of null designs, each run as resumable, checkpointed batches:

* ``diff``: loss-differential tests. Diebold–Mariano (two-sided, as in ``compare.py``), the
  H3 stress-dummy HAC regression (two-sided t on the dummy, as in ``stress/analysis.py``)
  and the Model Confidence Set (11 models, 90%). The null is "equal expected loss"; the
  differential is d_t = mu_t + s_t * sigma_t * eps_t with
    - eps_t   standardised Student-t(3) (finite variance, infinite kurtosis; real FZ0
              differentials have sample kurtosis 60 to > 6,000),
    - sigma_t log-AR(1) stochastic volatility (phi 0.98, tau 0.15),
    - s_t     = STRESS_MULT on the real episode calendar (Clarification 4), else 1
              (real sd ratio stress/calm: median 1.65, 90th percentile 2.37),
    - mu_t    a zero-mean AR(1) component (phi_mu, variance share w) matched to the real
              differentials' autocorrelation and long-run-variance ratios (design "mid":
              median pair; "high": 90th percentile, the ML-vs-FHS-GJR range).
  Designs D0n / D0t (i.i.d. normal / t3) are sanity designs for the fixed-b fallback.
* ``es``: ES and coverage tests on exact forecasts. Designs A, B1, B2 of the ESR size
  simulation (``esr_sim``) and C (Gaussian scale-mixture shocks, a tail shape that is not
  Student-t). Tests: McNeil–Frey (raw and standardised), Z2, Kupiec POF, Christoffersen
  IND and CC, DQ. Under every design the forecasts are the exact VaR/ES given the
  forecaster's information.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

SEED = 20261112
STRESS_MULT = 2.0
NU_D = 3.0  # Student-t dof of the differential shocks


# ------------------------------------------------------------------ differentials
@dataclass(frozen=True)
class DiffDesign:
    name: str
    phi_mu: float = 0.0  # persistence of the zero-mean component
    w: float = 0.0  # its share of the differential's variance (calm days)
    sv: bool = True  # stochastic volatility
    stress: bool = True  # stress multiplier on the real episode calendar
    nu: float = NU_D  # inf = normal shocks
    phi2: float = 0.0  # optional second (medium-memory) zero-mean component
    w2: float = 0.0


def t_std(nu: float, size, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance Student-t (nu = inf: standard normal)."""
    if np.isinf(nu):
        return rng.standard_normal(size)
    return rng.standard_t(nu, size) * np.sqrt((nu - 2) / nu)


def _ar1(phi: float, n: int, rng: np.random.Generator, burn: int = 500) -> np.ndarray:
    """Stationary unit-variance AR(1) path."""
    from scipy.signal import lfilter

    e = rng.standard_normal(n + burn) * np.sqrt(1 - phi**2)
    x = lfilter([1.0], [1.0, -phi], e)
    return x[burn:]


def sim_diff(d: DiffDesign, stress: np.ndarray, rng: np.random.Generator, k: int = 1) -> np.ndarray:
    """k differential paths (n x k), sharing sigma_t (common market volatility); eps and mu
    are independent across columns. Zero expected value by construction."""
    n = len(stress)
    sig = np.ones(n)
    if d.sv:  # log-AR(1), phi 0.98, innovation sd 0.15 (stationary sd 0.15 / sqrt(1 - phi^2))
        sig = np.exp(0.15 / np.sqrt(1 - 0.98**2) * _ar1(0.98, n, rng))
        sig = sig / np.sqrt(np.mean(sig**2))
    s = np.where(stress, STRESS_MULT, 1.0) if d.stress else np.ones(n)
    out = np.empty((n, k))
    for j in range(k):
        e = t_std(d.nu, n, rng) * np.sqrt(1 - d.w - d.w2)
        mu = np.sqrt(d.w) * _ar1(d.phi_mu, n, rng) if d.w > 0 else 0.0
        if d.w2 > 0:
            mu = mu + np.sqrt(d.w2) * _ar1(d.phi2, n, rng)
        out[:, j] = mu + s * sig * e
    return out


def diff_stats(x: np.ndarray) -> dict:
    """Statistics used to match a design to the real differentials (no test is run)."""
    from alphalab.backtest.compare import default_lag, newey_west_lrv

    x = np.asarray(x, float)
    xc = x - x.mean()
    v = xc @ xc

    def acf(k):
        return float(xc[k:] @ xc[:-k] / v)

    var = v / len(x)
    return {
        "acf1": acf(1),
        "acf5": acf(5),
        "lrv_q": newey_west_lrv(x, default_lag(len(x))) / var,
        "lrv_100": newey_west_lrv(x, 100) / var,
    }


# Frozen designs (Addition 10; parameters matched by ``diff_stats`` to the real P1 FZ0
# differentials vs FHS-GJR, 2026-11-12; the least persistent of near-equal fits was taken).
DIFF_DESIGNS = (
    DiffDesign("D0n_iid_normal", sv=False, stress=False, nu=float("inf")),
    DiffDesign("D0t_iid_t3", sv=False, stress=False),
    DiffDesign("Dmid", phi_mu=0.95, w=0.05),
    DiffDesign("Dhigh", phi_mu=0.99, w=0.08),
    # Codex pre-run review: Dhigh under-matches the upper default-lag ratio; Dhigh2 matches
    # both upper targets (default lag 1.83 vs 1.87; lag 100 4.84 vs 4.54; acf1 0.10 > 0.045)
    DiffDesign("Dhigh2", phi_mu=0.99, w=0.06, phi2=0.9, w2=0.10),
)
MCS_DESIGNS = ("Dmid", "Dhigh", "Dhigh2")
K_MODELS = 11
FIXED_B = 0.1


def kv_cv_bartlett(b: float) -> float:
    """Kiefer & Vogelsang (2005) fixed-b 97.5% critical value, Bartlett kernel (two-sided 5%)."""
    return 1.9600 + 2.9694 * b + 0.4160 * b**2 - 0.5324 * b**3


def dm_fixed_b(d: np.ndarray, b: float = FIXED_B) -> float:
    """|t| of the DM mean with Bartlett bandwidth M = b * n, to compare with kv_cv_bartlett."""
    from alphalab.backtest.compare import newey_west_lrv

    n = len(d)
    lrv = newey_west_lrv(d, max(1, int(b * n)))
    return float(abs(np.mean(d)) / np.sqrt(lrv / n)) if lrv > 0 else 0.0


def _rep_seed(*parts) -> np.random.SeedSequence:
    h = int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)
    return np.random.SeedSequence([SEED, h])


def diff_replication(d: DiffDesign, stress: np.ndarray, r: int) -> dict:
    """One replication: DM and H3 (primary and fixed-b fallback) on one differential; MCS
    (primary block and block 40) on 11 models' losses in the persistent designs."""
    from alphalab.backtest.compare import diebold_mariano, hac_regression, model_confidence_set

    n = len(stress)
    rng = np.random.default_rng(_rep_seed("diff", d.name, n, r))
    x = sim_diff(d, stress, rng, k=K_MODELS if d.name in MCS_DESIGNS else 1)
    d0 = x[:, 0]
    X = np.column_stack([np.ones(n), stress.astype(float)])
    cv = kv_cv_bartlett(FIXED_B)
    reg = hac_regression(d0, X)
    reg_b = hac_regression(d0, X, lag=max(1, int(FIXED_B * n)))
    out = {
        "rep": r,
        "dm_p": diebold_mariano(d0, np.zeros(n)).p_value,
        "dm_fb_rej": bool(dm_fixed_b(d0) > cv),
        "h3_p": reg.p[1],
        "h3_fb_rej": bool(abs(reg_b.t[1]) > cv),
    }
    if d.name in MCS_DESIGNS:
        L = pd.DataFrame(x, columns=[f"m{j}" for j in range(K_MODELS)])
        for tag, blk in (("mcs", None), ("mcs_b40", 40)):
            m = model_confidence_set(L, reps=2000, block_size=blk, seed=r)
            out[f"{tag}_excl"] = int((~m.in_mcs).sum())  # all equal: any exclusion is an error
    return out


# ------------------------------------------------------------------ ES / coverage
@dataclass(frozen=True)
class ESDesign:
    name: str
    s_eta: float = 0.0  # missed persistent volatility component (esr_sim designs B)
    phi_eta: float = 0.99
    mixture: bool = False  # Gaussian scale-mixture shocks instead of Student-t(5)


ES_DESIGNS = (
    ESDesign("A_correct"),
    ESDesign("B1_missed_vol_phi095", 0.8, 0.95),
    ESDesign("B2_missed_vol_phi099", 0.7, 0.99),
    ESDesign("C_mixture_tails", mixture=True),
)
ALPHAS = (0.01, 0.025, 0.05)
MIX_P, MIX_K = 0.1, 3.0  # 10% of shocks with 3x the sd of the rest
MIX_S1 = 1.0 / np.sqrt(1 - MIX_P + MIX_P * MIX_K**2)


def mixture_var_es(alpha: float) -> tuple[float, float]:
    """alpha-quantile and ES (negative, return units of a unit-variance shock) of the mixture."""
    from scipy.optimize import brentq

    s1, s2 = MIX_S1, MIX_K * MIX_S1
    q = brentq(
        lambda v: (1 - MIX_P) * stats.norm.cdf(v / s1) + MIX_P * stats.norm.cdf(v / s2) - alpha,
        -50,
        0,
        xtol=1e-14,
    )
    pe = (1 - MIX_P) * -s1 * stats.norm.pdf(q / s1) + MIX_P * -s2 * stats.norm.pdf(q / s2)
    return float(q), float(pe / alpha)


def sim_es(d: ESDesign, n: int, rng: np.random.Generator, burn: int = 500):
    """Returns y (returns) and sigma_t (the forecaster's scale); same equations as esr_sim."""
    from alphalab.backtest.esr_sim import NU, T_SCALE

    m = n + burn
    zs, ze = rng.normal(size=m), rng.normal(size=m)
    ls, le = np.empty(m), np.empty(m)
    ls[0] = rng.normal(0, 0.15 / np.sqrt(1 - 0.98**2))
    le[0] = rng.normal(0, d.s_eta)
    ie = d.s_eta * np.sqrt(1 - d.phi_eta**2)
    for t in range(1, m):
        ls[t] = 0.98 * ls[t - 1] + 0.15 * zs[t]
        le[t] = d.phi_eta * le[t - 1] + ie * ze[t]
    sig = np.exp(ls[burn:])
    eta = np.exp(le[burn:] - d.s_eta**2 / 2)
    if d.mixture:
        big = rng.random(n) < MIX_P
        eps = rng.standard_normal(n) * np.where(big, MIX_K * MIX_S1, MIX_S1)
    else:
        eps = T_SCALE * rng.standard_t(NU, size=n)
    return 0.01 * sig * eta * eps, sig


def es_replication(d: ESDesign, n: int, r: int) -> list[dict]:
    """All ES and coverage tests at every alpha on one simulated path (exact forecasts)."""
    from alphalab.backtest.coverage import coverage_tests, dq_test
    from alphalab.backtest.es import acerbi_szekely_z2, mcneil_frey
    from alphalab.backtest.esr_sim import product_var_es

    rng = np.random.default_rng(_rep_seed("es", d.name, n, r))
    y, sig = sim_es(d, n, rng)
    loss = -y
    rows = []
    for a in ALPHAS:
        q, e = mixture_var_es(a) if d.mixture else product_var_es(a, d.s_eta)
        var, es = -0.01 * sig * q, -0.01 * sig * e
        hits = loss > var
        cov = coverage_tests(hits, a)
        try:
            dq = dq_test(hits, var, a).p_value
        except (ValueError, np.linalg.LinAlgError):
            dq = float("nan")
        mf = mcneil_frey(loss, var, es, a, sigma=0.01 * sig, seed=r)
        z2 = acerbi_szekely_z2(loss, var, es, a, seed=r)
        rows.append(
            {
                "rep": r,
                "alpha": a,
                "hits": int(hits.sum()),
                "pof_p": cov.p_pof,
                "ind_p": cov.p_ind,
                "cc_p": cov.p_cc,
                "dq_p": dq,
                "mf_raw_p": mf.p_raw,
                "mf_std_p": mf.p_std,
                "z2_p": z2.p_value,
            }
        )
    return rows


# ------------------------------------------------------------------ running
SIZES = (6535, 961)  # P1 development length; lockbox length


@dataclass(frozen=True)
class CalSettings:
    family: str  # "diff" or "es"
    reps: int = 2000
    batch: int = 25
    workers: int = 4
    nice: int = 10
    sizes: tuple = SIZES
    extra: dict = field(default_factory=dict)


def _init(nice: int) -> None:
    for v in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[v] = "1"
    with contextlib.suppress(OSError):
        os.nice(nice)
    import warnings

    warnings.simplefilter("ignore")


def stress_for(n: int, stress_p1: np.ndarray) -> np.ndarray:
    """Real P1 episode calendar (6,535 days); the lockbox-length designs use its last n days."""
    return np.asarray(stress_p1, bool)[-n:]


def _job(args) -> tuple[str, float]:
    root, s, dname, n, j, stress_p1 = args
    t0 = time.perf_counter()
    lo, hi = j * s.batch, min((j + 1) * s.batch, s.reps)
    rows = []
    if s.family == "diff":
        d = next(x for x in DIFF_DESIGNS if x.name == dname)
        st = stress_for(n, stress_p1)
        rows = [diff_replication(d, st, r) for r in range(lo, hi)]
    else:
        d = next(x for x in ES_DESIGNS if x.name == dname)
        for r in range(lo, hi):
            rows += es_replication(d, n, r)
    from alphalab.backtest.esr_bootstrap import _atomic_write_bytes

    path = Path(root) / f"{dname}_n{n}" / f"batch_{j:04d}.json"
    _atomic_write_bytes(path, json.dumps(rows).encode())
    return f"{dname}_n{n} batch {j}", time.perf_counter() - t0


def run(root: Path, s: CalSettings, stress_p1: np.ndarray, log_path: Path | None = None) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    designs = DIFF_DESIGNS if s.family == "diff" else ES_DESIGNS
    doc = asdict(s) | {
        "seed": SEED,
        "designs": [asdict(x) for x in designs],
        "stress_sha256": hashlib.sha256(np.asarray(stress_p1, bool).tobytes()).hexdigest(),
        "n_stress": len(stress_p1),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    mpath = root / "settings.json"
    if mpath.exists() and json.loads(mpath.read_text()) != json.loads(json.dumps(doc)):
        raise RuntimeError(f"{root}: existing run used other settings or code; use a new folder")
    mpath.write_text(json.dumps(doc, indent=2))
    if log_path is not None:
        h = logging.FileHandler(log_path)
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    names = [x.name for x in (DIFF_DESIGNS if s.family == "diff" else ES_DESIGNS)]
    nb = -(-s.reps // s.batch)
    todo = []
    for name in names:
        for n in s.sizes:
            (root / f"{name}_n{n}").mkdir(exist_ok=True)
            done = {int(p.stem.split("_")[1]) for p in (root / f"{name}_n{n}").glob("batch_*.json")}
            todo += [(str(root), s, name, n, j, stress_p1) for j in range(nb) if j not in done]
    log.info("calibration %s: %d batches to do", s.family, len(todo))
    t0, k = time.time(), 0
    with ProcessPoolExecutor(s.workers, initializer=_init, initargs=(s.nice,)) as ex:
        for f in as_completed([ex.submit(_job, a) for a in todo]):
            name, sec = f.result()
            k += 1
            eta = (time.time() - t0) / k * (len(todo) - k) / 3600
            log.info("%s done in %.0fs | %d/%d | ETA %.2f h", name, sec, k, len(todo), eta)
    return {"batches_done": k, "batches_total": len(todo)}


# ------------------------------------------------------------------ frozen decision rules
LEVEL = 0.05
THRESH = {0.05: 0.064, 0.10: 0.119}  # nominal + 2 binomial sd at R = 1,000 (amendment 6 rule)
STRICT = {0.05: 0.060, 0.10: 0.113}  # informational: nominal + 2 binomial sd at R = 2,000
MAX_NAN = 0.05
GATE_DESIGNS = {  # designs under which each test's null holds and is gated
    "dm": ("D0n_iid_normal", "D0t_iid_t3", "Dmid", "Dhigh", "Dhigh2"),
    "h3": ("D0n_iid_normal", "D0t_iid_t3", "Dmid", "Dhigh", "Dhigh2"),
    "mcs": MCS_DESIGNS,
    "mf_raw": ("A_correct", "B1_missed_vol_phi095", "B2_missed_vol_phi099", "C_mixture_tails"),
    "mf_std": ("A_correct", "B1_missed_vol_phi095", "B2_missed_vol_phi099", "C_mixture_tails"),
    "z2": ("A_correct", "B1_missed_vol_phi095", "B2_missed_vol_phi099", "C_mixture_tails"),
    "pof": ("A_correct", "C_mixture_tails"),
    "ind": ("A_correct", "C_mixture_tails"),
    "cc": ("A_correct", "C_mixture_tails"),
    "dq": ("A_correct", "C_mixture_tails"),
}


def _wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    c = (k + z * z / 2) / (n + z * z)
    h = z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return max(0.0, c - h), min(1.0, c + h)


def _load(root: Path, cell: str) -> pd.DataFrame:
    rows = []
    for p in sorted((Path(root) / cell).glob("batch_*.json")):
        rows += json.loads(p.read_text())
    return pd.DataFrame(rows)


def cell_rates(root: Path, s: CalSettings) -> pd.DataFrame:
    """Rejection rate per (test, design, n, alpha), with Wilson 95% interval, NaN share, the
    threshold and PASS / FAIL / UNAVAILABLE. NaN p-values count as non-rejections in the
    rate and are reported; > 5% NaN makes the cell UNAVAILABLE."""
    out = []
    names = [x.name for x in (DIFF_DESIGNS if s.family == "diff" else ES_DESIGNS)]
    for name in names:
        for n in s.sizes:
            df = _load(root, f"{name}_n{n}")
            if df.empty:
                continue
            groups = [(np.nan, df)] if s.family == "diff" else list(df.groupby("alpha"))
            for a, g in groups:
                cols = {}
                if s.family == "diff":
                    cols = {
                        "dm": g.dm_p < LEVEL,
                        "dm_fb": g.dm_fb_rej,
                        "h3": g.h3_p < LEVEL,
                        "h3_fb": g.h3_fb_rej,
                    }
                    if "mcs_excl" in g:
                        cols |= {"mcs": g.mcs_excl > 0, "mcs_b40": g.mcs_b40_excl > 0}
                    nan = {
                        k: g[{"dm": "dm_p", "h3": "h3_p"}[k]].isna().mean()
                        if k in ("dm", "h3")
                        else 0.0
                        for k in cols
                    }
                else:
                    for t in ("pof", "ind", "cc", "dq", "mf_raw", "mf_std", "z2"):
                        p = pd.to_numeric(g[f"{t}_p"], errors="coerce")
                        cols[t] = p < LEVEL
                    nan = {
                        t: pd.to_numeric(g[f"{t}_p"], errors="coerce").isna().mean() for t in cols
                    }
                for t, rej in cols.items():
                    base = t.replace("_fb", "").replace("_b40", "")
                    nominal = 0.10 if base == "mcs" else LEVEL
                    k, m = float(np.sum(rej.fillna(False))), len(rej)
                    rate = k / m if m else float("nan")
                    ids = g.rep.to_numpy()
                    complete = bool(
                        len(ids) == s.reps
                        and len(set(ids)) == len(ids)
                        and set(ids) == set(range(s.reps))
                    )
                    fin = 1 - nan[t]
                    rate_fin = k / (m * fin) if m and fin > 0 else float("nan")
                    lo, hi = _wilson(k, m)
                    status = (
                        "UNAVAILABLE"
                        if nan[t] > MAX_NAN
                        else "PASS"
                        if rate <= THRESH[nominal]
                        else "FAIL"
                    )
                    out.append(
                        {
                            "test": t,
                            "design": name,
                            "n": n,
                            "alpha": a,
                            "reps": m,
                            "complete": complete,
                            "rate_finite": rate_fin,
                            "above_strict": rate > STRICT[nominal],
                            "rate": rate,
                            "nominal": nominal,
                            "wilson_lo": lo,
                            "wilson_hi": hi,
                            "nan_share": nan[t],
                            "threshold": THRESH[nominal],
                            "gated": name in GATE_DESIGNS[base],
                            "status": status,
                            "mean_hits": float(g.hits.mean()) if "hits" in g else float("nan"),
                        }
                    )
    return pd.DataFrame(out)


def decisions(cells: pd.DataFrame) -> pd.DataFrame:
    """Frozen rule: a test is SUPPORTED in a domain (n) and at a level alpha if every gated
    design passes there and every gated cell is complete; otherwise UNSUPPORTED. A
    pre-declared fallback (DM/H3: fixed-b; MCS: block 40) is adopted only where the primary
    is UNSUPPORTED and the fallback is SUPPORTED."""
    g = cells[cells.gated]
    rows = []
    for (t, n, a), x in g.groupby(["test", "n", "alpha"], dropna=False):
        base = t.replace("_fb", "").replace("_b40", "")
        full_set = set(x.design) == set(GATE_DESIGNS[base]) and len(x) == len(GATE_DESIGNS[base])
        ok = bool(full_set and (x.status == "PASS").all() and x.complete.all())
        rows.append(
            {
                "test": t,
                "n": n,
                "alpha": a,
                "supported": ok,
                "worst_design": x.loc[x.rate.idxmax(), "design"],
                "worst_rate": x.rate.max(),
            }
        )
    d = pd.DataFrame(rows)
    final = []
    for _, r in d[~d.test.str.contains("_fb|_b40")].iterrows():
        # H3's fixed-b result is an empirical diagnostic only (Codex pre-run review: the
        # Kiefer–Vogelsang conditions are not established for a persistent deterministic
        # episode dummy with stress-dependent variance); it is never adopted.
        fb = {"dm": "dm_fb", "mcs": "mcs_b40"}.get(r.test)
        verdict = "SUPPORTED" if r.supported else "UNSUPPORTED"
        if not r.supported and fb:
            f = d[
                (d.test == fb)
                & (d.n == r.n)
                & (d.alpha.fillna(-1) == (-1 if pd.isna(r.alpha) else r.alpha))
            ]
            if len(f) and bool(f.supported.iloc[0]):
                verdict = f"SUPPORTED WITH FALLBACK ({fb})"
        final.append(r.to_dict() | {"decision": verdict})
    return pd.DataFrame(final)


def hit_strata(root: Path, s: CalSettings, n: int = 961, alpha: float = 0.01) -> pd.DataFrame:
    """Rejection rates by number of hits (lockbox length, 1%): where sparse hits bite."""
    out = []
    bins, labels = [-1, 4, 7, 10, 13, 1000], ["0-4", "5-7", "8-10", "11-13", "14+"]
    for d in ES_DESIGNS:
        df = _load(root, f"{d.name}_n{n}")
        if df.empty:
            continue
        g = df[np.isclose(df.alpha, alpha)].copy()
        g["hit_bin"] = pd.cut(g.hits, bins, labels=labels)
        for b, x in g.groupby("hit_bin", observed=True):
            row = {"design": d.name, "n": n, "alpha": alpha, "hits": str(b), "reps": len(x)}
            for t in ("pof", "ind", "cc", "dq", "mf_raw", "mf_std", "z2"):
                p = pd.to_numeric(x[f"{t}_p"], errors="coerce")
                row[f"{t}_rej"] = float((p < LEVEL).mean())
                row[f"{t}_nan"] = float(p.isna().mean())
            out.append(row)
    return pd.DataFrame(out)


def stress_calendar_p1() -> np.ndarray:
    """Real P1 development episode dummy by target date (Clarification 4); no lockbox data."""
    from alphalab.config import load_config
    from alphalab.pipeline import _processed_dir, _results_dir
    from alphalab.risk.engine import attach_outcomes
    from alphalab.stress.analysis import episode_labels

    cfg = load_config("configs/risk_p1_spy.yaml")
    proc = _processed_dir(cfg)
    fc = pd.read_parquet(_results_dir(cfg, proc) / "forecasts.parquet")  # development only
    m0 = fc.model.iloc[0]
    ret = pd.read_parquet(proc / "portfolio_daily.parquet")["ret"]
    ev = attach_outcomes(fc[fc.model == m0].drop(columns="model"), ret)
    if (ev["target_date"] >= pd.Timestamp(cfg.risk.lockbox_start)).any():
        raise RuntimeError("development forecasts reach the lockbox")
    return episode_labels(ev["target_date"]).to_numpy() != ""


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m alphalab.backtest.calibration")
    ap.add_argument("family", choices=["diff", "es"])
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--reps", type=int, default=CalSettings("diff").reps)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    s = CalSettings(a.family, reps=a.reps, workers=a.workers)
    root = Path("data/results/calibration") / f"{a.family}_reps{s.reps}_seed{SEED}"
    if a.summarize:
        cells = cell_rates(root, s)
        cells.to_csv(root / "cells.csv", index=False)
        dec = decisions(cells)
        dec.to_csv(root / "decisions.csv", index=False)
        if a.family == "es":
            hs = hit_strata(root, s)
            hs.to_csv(root / "hit_strata_n961_a1.csv", index=False)
            print(hs.round(3).to_string(index=False))
        pd.set_option("display.width", 250)
        print(cells.round(4).to_string(index=False))
        print(dec.round(4).to_string(index=False))
        return 0
    Path("logs").mkdir(exist_ok=True)
    print(run(root, s, stress_calendar_p1(), Path("logs") / f"calibration_{root.name}.log"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
