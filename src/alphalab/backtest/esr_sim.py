"""Monte Carlo size of the ESR bootstrap tests under dependence (RESEARCH_DESIGN Addition 9).

Why: the block bootstrap's validity for the ESR tests rests on first-order asymptotics. This
module measures null rejection rates in simulated data with volatility clustering and, in
design B, serially correlated ESR scores like those found in the real series.

Designs (returns Y_t; the forecaster observes sigma_t, not eta_t):
    log sigma_t = phi * log sigma_{t-1} + tau * N(0,1)              (stochastic volatility)
    log eta_t   = phi_eta * log eta_{t-1} + N(0, s_eta^2 (1 - phi_eta^2))  - s_eta^2 / 2 (centred)
    Y_t = 0.01 * sigma_t * eta_t * eps_t,   eps_t standardised Student-t(5), all independent.
The VaR/ES forecasts are the exact alpha-quantile and ES of Y_t given sigma_t, i.e. of
0.01 * sigma_t * (eta * eps) with eta at its stationary law (Gauss-Hermite quadrature, no
Monte Carlo error). So the ESR null "the ES forecast is the conditional ES given the
forecast" holds exactly in both designs:
    A (s_eta = 0): forecasts are fully correct; ESR scores are a martingale difference.
    B (s_eta > 0): forecasts miss a persistent volatility component; given the forecast they
       are still correct (eta is independent of sigma), but tail hits cluster, so the ESR
       scores are autocorrelated. The asymptotic tests and the iid bootstrap should
       over-reject here; a valid block bootstrap should not.

Bootstrap size by the warp-speed method (Giacomini, Politis & White 2013, Econometric
Theory 29): each replication r computes the statistic S_r and ONE bootstrap statistic S*_r;
the bootstrap p-value of replication r is the share of S*_1..S*_R at least as extreme as
S_r. This estimates the size of the B -> infinity bootstrap test at the cost of two fits per
replication. All block lengths reuse the same replications (paired comparison).

Results are checkpointed per batch of replications (atomic npz files) and resumable.
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
from scipy import special, stats

from alphalab.backtest.es import esr_bootstrap_draws
from alphalab.backtest.esr_bootstrap import _atomic_write_bytes, chunk_indices, make_cell

log = logging.getLogger(__name__)
NU = 5.0
T_SCALE = np.sqrt((NU - 2) / NU)  # eps = T_SCALE * t_5 has unit variance


@dataclass(frozen=True)
class Design:
    name: str
    s_eta: float  # stationary sd of log eta (0 = forecasts fully correct)
    phi_eta: float = 0.99
    phi: float = 0.98
    tau: float = 0.15
    alpha: float = 0.025


@dataclass(frozen=True)
class SimSettings:
    designs: tuple = (
        Design("A_correct", 0.0),
        Design("B1_missed_vol_phi095", 0.8, phi_eta=0.95),
        Design("B2_missed_vol_phi099", 0.7, phi_eta=0.99),
    )
    n: int = 2500
    reps: int = 1000
    blocks: tuple = (1, 10, 20, 40)
    batch: int = 10
    seed: int = 20261111
    burn: int = 500
    workers: int = 4
    nice: int = 10
    extra: dict = field(default_factory=dict)
    cov: str = "scl_sp"  # covariance variant (es.COV_VARIANTS); amendment 10


# ---------------------------------------------------------------- exact forecasts
def _eps_partial(c: np.ndarray) -> np.ndarray:
    """E[eps 1{eps <= c}] for the standardised t(NU)."""
    u = np.asarray(c, dtype=float) / T_SCALE
    return T_SCALE * (-(NU + u**2) / (NU - 1)) * stats.t.pdf(u, NU)


def product_var_es(alpha: float, s_eta: float, nodes: int = 120) -> tuple[float, float]:
    """alpha-quantile q and ES (both in return units, negative) of eta * eps, with
    log eta ~ N(-s^2/2, s^2) (E eta = 1) independent of eps; Gauss-Hermite in log eta."""
    if s_eta == 0:
        q = T_SCALE * stats.t.ppf(alpha, NU)
        return float(q), float(_eps_partial(q) / alpha)
    x, w = special.roots_hermitenorm(nodes)
    w = w / w.sum()
    eta = np.exp(-(s_eta**2) / 2 + s_eta * x)

    def cdf(v):
        return float(np.sum(w * stats.t.cdf(v / eta / T_SCALE, NU)))

    from scipy.optimize import brentq

    q = brentq(lambda v: cdf(v) - alpha, -100.0, 0.0, xtol=1e-14)
    es = float(np.sum(w * eta * _eps_partial(q / eta))) / alpha
    return float(q), es


def simulate(d: Design, n: int, rng: np.random.Generator, burn: int = 500):
    """One sample: (loss, var_forecast, es_forecast), loss = -Y, forecasts positive."""
    m = n + burn
    ls = np.empty(m)
    le = np.empty(m)
    ls[0] = rng.normal(0, d.tau / np.sqrt(1 - d.phi**2))
    le[0] = rng.normal(0, d.s_eta)
    zs = rng.normal(size=m)
    ze = rng.normal(size=m)
    ie = d.s_eta * np.sqrt(1 - d.phi_eta**2)
    for t in range(1, m):
        ls[t] = d.phi * ls[t - 1] + d.tau * zs[t]
        le[t] = d.phi_eta * le[t - 1] + ie * ze[t]
    sig = np.exp(ls[burn:])
    eta = np.exp(le[burn:] - d.s_eta**2 / 2)
    eps = T_SCALE * rng.standard_t(NU, size=n)
    y = 0.01 * sig * eta * eps
    q, es = product_var_es(d.alpha, d.s_eta)
    return -y, -0.01 * sig * q, -0.01 * sig * es


# ---------------------------------------------------------------- replications
def _rep_seed(s: SimSettings, d: Design, r: int) -> np.random.SeedSequence:
    h = int(hashlib.sha256(f"{d.name}|{s.n}".encode()).hexdigest()[:8], 16)
    return np.random.SeedSequence([s.seed, h, r])


def one_replication(s: SimSettings, d: Design, r: int) -> dict:
    from alphalab.backtest.es import cov_variant

    with cov_variant(s.cov):
        return _one_replication(s, d, r)


def _one_replication(s: SimSettings, d: Design, r: int) -> dict:
    rng = np.random.default_rng(_rep_seed(s, d, r))
    loss, var, es = simulate(d, s.n, rng, s.burn)
    out = {"rep": r, "ok": False}
    try:
        cell = make_cell(f"{d.name}_{r}", loss, var, es, d.alpha)
    except Exception:  # noqa: BLE001 - count any failed fit as unavailable
        cell = None
    if cell is None:
        return out
    from alphalab.backtest import esr_backtest

    a = esr_backtest(loss, var, es, d.alpha)
    out.update(ok=True, wald0=cell.wald0, t0=cell.t0, p_strict_asym=a.strict_p, p_int_asym=a.int_p)
    for b in s.blocks:
        idx = chunk_indices(cell.key, s.seed, b, 1, s.n, b)
        w, t = esr_bootstrap_draws(cell.y, cell.ehat, d.alpha, idx, cell.g_hat, cell.g0_hat)
        out[f"wstar_b{b}"] = float(w[0])
        out[f"tstar_b{b}"] = float(t[0])
    return out


def _batch_job(args) -> tuple[str, int, float]:
    root, s, d, j = args
    t0 = time.perf_counter()
    lo, hi = j * s.batch, min((j + 1) * s.batch, s.reps)
    rows = [one_replication(s, d, r) for r in range(lo, hi)]
    path = Path(root) / d.name / f"batch_{j:04d}.json"
    _atomic_write_bytes(path, json.dumps(rows).encode())
    return d.name, j, time.perf_counter() - t0


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


def _settings_doc(s: SimSettings) -> dict:
    doc = asdict(s)
    doc["designs"] = [asdict(d) for d in s.designs]
    if doc["cov"] == "scl_sp":  # keep pre-amendment settings files byte-compatible
        del doc["cov"]
    return doc


def run(root: Path, s: SimSettings, log_path: Path | None = None) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    mpath = root / "settings.json"
    doc = _settings_doc(s)
    if mpath.exists():
        if json.loads(mpath.read_text()) != json.loads(json.dumps(doc)):
            raise RuntimeError(f"{root}: existing simulation used other settings; use a new folder")
    else:
        _atomic_write_bytes(mpath, json.dumps(doc, indent=2).encode())
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        h = logging.FileHandler(log_path)
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    nb = -(-s.reps // s.batch)
    todo = []
    for d in s.designs:
        (root / d.name).mkdir(exist_ok=True)
        done = {int(p.stem.split("_")[1]) for p in (root / d.name).glob("batch_*.json")}
        todo += [(str(root), s, d, j) for j in range(nb) if j not in done]
    log.info("esr-sim: %d batches to do (%s)", len(todo), json.dumps(doc))
    t0, k = time.time(), 0
    with ProcessPoolExecutor(s.workers, initializer=_init, initargs=(s.nice,)) as ex:
        for f in as_completed([ex.submit(_batch_job, a) for a in todo]):
            name, j, sec = f.result()
            k += 1
            eta = (time.time() - t0) / k * (len(todo) - k) / 3600
            log.info(
                "%s batch %d done in %.0fs | %d/%d | ETA %.2f h", name, j, sec, k, len(todo), eta
            )
    return {"batches_done": k, "batches_total": len(todo)}


# ---------------------------------------------------------------- results
def warp_p(stat: np.ndarray, star: np.ndarray, upper: bool) -> np.ndarray:
    """Warp-speed bootstrap p-values: share of all finite bootstrap draws as extreme as each
    stat. NaN where the stat is not finite or the pool of draws is empty."""
    stat = np.asarray(stat, dtype=float)
    star = np.sort(star[np.isfinite(star)])
    if len(star) == 0:
        return np.full(stat.shape, np.nan)
    if upper:
        p = 1.0 - np.searchsorted(star, stat, side="left") / len(star)
    else:
        p = np.searchsorted(star, stat, side="right") / len(star)
    return np.where(np.isfinite(stat), p, np.nan)


def warp_rejection(
    stat: np.ndarray, star: np.ndarray, upper: bool, level: float, n_boot: int = 500, seed: int = 0
) -> tuple[float, float]:
    """Warp-speed rejection rate and its Monte Carlo sd. The sd resamples whole replication
    rows (stat, star) and recomputes the pooled critical value each time, so it includes the
    uncertainty of the common critical value (the binomial sd does not)."""
    stat, star = np.asarray(stat, dtype=float), np.asarray(star, dtype=float)
    keep = np.isfinite(stat)
    stat, star = stat[keep], star[keep]
    if len(stat) == 0 or not np.isfinite(star).any():
        return float("nan"), float("nan")
    rate = float(np.mean(warp_p(stat, star, upper) < level))
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        i = rng.integers(0, len(stat), len(stat))
        reps.append(np.mean(warp_p(stat[i], star[i], upper) < level))
    return rate, float(np.std(reps, ddof=1))


MAX_BOOT_FAIL = 0.05  # same rule as the real-data bootstrap


def load(root: Path, d: Design) -> pd.DataFrame:
    rows = []
    for p in sorted((Path(root) / d.name).glob("batch_*.json")):
        rows += json.loads(p.read_text())
    return pd.DataFrame(rows)


def degenerate_flags(s: SimSettings, d: Design, r: int) -> dict:
    """Regenerate replication r exactly and flag esreg's degenerate location-scale start
    (Addition 9, amendments 7-8) in the full sample and in each block's bootstrap draw."""
    from alphalab.backtest.esr_bootstrap import _normalise, degenerate_start

    rng = np.random.default_rng(_rep_seed(s, d, r))
    loss, _, es = simulate(d, s.n, rng, s.burn)
    y, eh = _normalise(loss, es)

    def bad(yy, ee):
        x = np.column_stack([np.ones(len(yy)), ee])
        z = yy - ee
        return degenerate_start(yy - yy.max(), x), degenerate_start(z - z.max(), x)

    fs, fi = bad(y, eh)
    out = {"rep": r, "full_strict": fs, "full_int": fi}
    for b in s.blocks:
        ix = chunk_indices(f"{d.name}_{r}", s.seed, b, 1, s.n, b)[0]
        out[f"draw_strict_b{b}"], out[f"draw_int_b{b}"] = bad(y[ix], eh[ix])
    return out


def summarize(root: Path, s: SimSettings, level: float = 0.05) -> pd.DataFrame:
    """Null rejection rates per design and method. ``*_mc_sd`` resamples replication rows
    (including the warp-speed critical value); ``binom_sd`` is the binomial reference only.
    A bootstrap result is ``usable`` only with a non-empty pool and at most 5% failed draws
    among replications whose own fit succeeded; otherwise its rates are NaN."""
    out = []
    for d in s.designs:
        df = load(root, d)
        if df.empty:
            continue
        if s.cov == "scl_sp":  # post-hoc masking only for the pre-amendment estimator
            flags = pd.DataFrame([degenerate_flags(s, d, int(r)) for r in df.rep])
            df = df.join(flags.set_index("rep"), on="rep")
            full_bad = df.full_strict | df.full_int
            df.loc[full_bad, "ok"] = False  # invalid covariance: replication unavailable
            for b in s.blocks:  # degenerate draws are failed draws, never observations
                df.loc[df[f"draw_strict_b{b}"].astype(bool), f"wstar_b{b}"] = np.nan
                df.loc[df[f"draw_int_b{b}"].astype(bool), f"tstar_b{b}"] = np.nan
        else:  # amendment 10: no scale fit (V2) or failures raised inside the fit (V1)
            full_bad = pd.Series(False, index=df.index)
        ok = df[df.ok.astype(bool)]
        base = {
            "degenerate_full_reps": int(full_bad.sum()),
            "design": d.name,
            "n": s.n,
            "reps_done": len(df),
            "reps_target": s.reps,
            "reps_ok": len(ok),
            "binom_sd": float(np.sqrt(level * (1 - level) / max(len(ok), 1))),
        }
        ps, pi = ok.p_strict_asym.to_numpy(), ok.p_int_asym.to_numpy()
        out.append(
            base
            | {
                "method": "asymptotic",
                "block": np.nan,
                "boot_fail_share": np.nan,
                "usable": True,
                "strict_rej": float(np.mean(ps[np.isfinite(ps)] < level)),
                "strict_mc_sd": float(np.sqrt(level * (1 - level) / max(len(ok), 1))),
                "int_rej": float(np.mean(pi[np.isfinite(pi)] < level)),
                "int_mc_sd": float(np.sqrt(level * (1 - level) / max(len(ok), 1))),
            }
        )
        for b in s.blocks:
            w = pd.to_numeric(ok[f"wstar_b{b}"], errors="coerce").to_numpy(float)
            t = pd.to_numeric(ok[f"tstar_b{b}"], errors="coerce").to_numpy(float)
            fail = float(np.mean(~np.isfinite(w) | ~np.isfinite(t))) if len(ok) else np.nan
            usable = bool(len(ok) and fail <= MAX_BOOT_FAIL)
            rs, sds = warp_rejection(ok.wald0.to_numpy(), w, True, level, seed=b)
            ri, sdi = warp_rejection(ok.t0.to_numpy(), t, False, level, seed=b + 1)
            if not usable:
                rs = sds = ri = sdi = np.nan
            out.append(
                base
                | {
                    "method": "iid bootstrap" if b == 1 else "block bootstrap",
                    "block": b,
                    "boot_fail_share": fail,
                    "usable": usable,
                    "strict_rej": rs,
                    "strict_mc_sd": sds,
                    "int_rej": ri,
                    "int_mc_sd": sdi,
                }
            )
    return pd.DataFrame(out)
