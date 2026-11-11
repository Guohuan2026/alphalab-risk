"""Resumable bootstrap calibration of the ESR backtests (RESEARCH_DESIGN Addition 9).

Why: the asymptotic ESR p-values are approximations; Bayer & Dimitriadis (and esback)
calibrate them with a studentised iid pairs bootstrap. On our series the ESR scores are
serially correlated (Ljung-Box rejects in 98% of cells; long-run / iid variance 1.1-4.2),
so iid resampling would understate the variance just as the asymptotic covariance does.
We therefore resample circular blocks of consecutive days (Politis & Romano 1992), block
length ceil(T^(1/3)) by default, keeping the studentisation by the iid (esreg) covariance:
the bootstrap then reproduces the distribution of that statistic, autocorrelation
included, without needing a HAC estimate (first-order validity under weak dependence,
block -> infinity, block/T -> 0). ``block=1`` gives esback's iid scheme. Each resample
refits the joint (quantile, ES) regression, so a full run (all models x levels x
portfolios, B = 999) takes hours. This module makes that an overnight job that can be
stopped and resumed at any time.

Design
- One *cell* = (portfolio run, model, alpha). Its B resamples are split into chunks.
- Every chunk's resample indices come from a seed derived from (seed, cell, chunk), so a
  chunk always draws the same resamples, whatever the order, the worker count or how often
  the job was interrupted: the final p-values are reproducible bit for bit.
- Each chunk is written atomically (temp file + rename) as soon as it finishes. A restart
  skips finished chunks. A per-cell manifest pins the inputs (data hash, estimator version,
  seed, B, chunk size); a mismatch is refused rather than mixed.
- Resource limits: a worker cap (default: half the cores, at most 4), low CPU priority
  (nice), one BLAS/OpenMP thread per worker, an optional wall-clock budget (``max_hours``)
  after which no new chunk starts, and a lock file so two runs cannot write the same cell.
- Progress is logged (chunks done, rate, ETA) to ``logs/esr_bootstrap_<run>.log``.
- Development sample only: the lockbox forecasts are never read; a target date at or after
  the lockbox start aborts the job.

Usage (also ``alphalab esr-bootstrap --help``):
    alphalab esr-bootstrap --config configs/risk_p1_spy.yaml --plan        # estimate only
    alphalab esr-bootstrap --config configs/risk_p1_spy.yaml --max-hours 8 # run / resume
    alphalab esr-bootstrap --config configs/risk_p1_spy.yaml --summarize   # p-values
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alphalab.backtest.es import (
    ESRFitError,
    _intercept_fit,
    _strict_fit,
    esr_backtest,
    esr_bootstrap_draws,
)

log = logging.getLogger(__name__)

# Bump when the estimator or covariance changes: chunks made by another version are refused.
ESTIMATOR_VERSION = "esr-2026-11-10-esreg-vcovA-port-tieband-cbb"
MAX_FAIL_SHARE = 0.05  # esback: more than 5% failed resamples -> no bootstrap p-value


@dataclass(frozen=True)
class BootstrapSettings:
    reps: int = 999
    chunk: int = 25
    seed: int = 20261110
    workers: int = max(1, min(4, (os.cpu_count() or 2) // 2))
    nice: int = 10
    max_hours: float | None = None
    min_tail: int = 10
    block: int | None = None  # circular block length; None -> ceil(T^(1/3)); 1 -> iid

    def block_for(self, n: int) -> int:
        b = self.block if self.block is not None else int(np.ceil(n ** (1 / 3)))
        if not 1 <= b <= n:
            raise ValueError(f"block length {b} outside [1, {n}]")
        return b


@dataclass(frozen=True)
class Cell:
    """One (run, model, alpha) series in normalised units, with its full-sample estimates."""

    key: str
    alpha: float
    y: np.ndarray
    ehat: np.ndarray
    g_hat: tuple[float, float]
    g0_hat: float
    wald0: float
    t0: float
    data_sha256: str


# ---------------------------------------------------------------- building the cells
def _normalise(loss, es):
    """Same normalisation as esr_backtest (which has already validated the inputs)."""
    loss = np.asarray(loss, dtype=float)
    es = np.asarray(es, dtype=float)
    unit = float(np.std(loss)) or float(np.mean(es))
    return -loss / unit, -es / unit


def make_cell(key: str, loss, var, es, alpha: float, min_tail: int = 10) -> Cell | None:
    """Full-sample fit for one series; None if either asymptotic test is unavailable."""
    res = esr_backtest(loss, var, es, alpha, min_tail=min_tail)
    if not (res.strict_status.startswith("ok") and res.int_status.startswith("ok")):
        log.warning(
            "%s: skipped (strict: %s; intercept: %s)", key, res.strict_status, res.int_status
        )
        return None
    y, ehat = _normalise(loss, es)
    x = np.column_stack([np.ones(len(y)), ehat])
    g, _, _ = _strict_fit(y, x, alpha, min_tail)
    g0, _, _, _ = _intercept_fit(y - ehat, x, alpha, min_tail)
    h = hashlib.sha256()
    for arr in (np.asarray(loss, float), np.asarray(var, float), np.asarray(es, float)):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(repr(float(alpha)).encode())
    return Cell(
        key,
        float(alpha),
        y,
        ehat,
        (float(g[0]), float(g[1])),
        float(g0),
        float(res.strict_wald),
        float(res.int_t),
        h.hexdigest(),
    )


def cells_from_config(cfg, min_tail: int = 10) -> tuple[Path, list[Cell]]:
    """Development-sample cells for every model and alpha of a risk config."""
    from alphalab.backtest.report import align
    from alphalab.pipeline import _processed_dir, _results_dir
    from alphalab.risk.engine import attach_outcomes

    proc = _processed_dir(cfg)
    out = _results_dir(cfg, proc)
    path = out / "forecasts.parquet"  # development forecasts only, never *_lockbox
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run forecast first")
    fc = pd.read_parquet(path)
    ret = pd.read_parquet(proc / "portfolio_daily.parquet")["ret"]
    evs = align(
        {
            m: attach_outcomes(f.drop(columns="model"), ret)
            for m, f in fc.groupby("model", sort=False)
        }
    )
    lock = pd.Timestamp(cfg.risk.lockbox_start)
    cells = []
    for name, ev in evs.items():
        if (ev["target_date"] >= lock).any():
            raise RuntimeError(f"{name}: development forecasts reach the lockbox ({lock.date()})")
        for a in cfg.risk.alphas:
            c = make_cell(
                f"{name}_{a:g}", ev["loss_next"], ev[f"var_{a:g}"], ev[f"es_{a:g}"], a, min_tail
            )
            if c is not None:
                cells.append(c)
    return out / "esr_bootstrap", cells


def root_for(root: Path, block: int | None) -> Path:
    """Main run (default block) in ``esr_bootstrap``; any explicit block length in its own
    folder, so sensitivity runs can never touch the main run's checkpoints."""
    return root if block is None else root.with_name(f"{root.name}_block{block}")


# ---------------------------------------------------------------- chunks and storage
def chunk_bounds(reps: int, chunk: int) -> list[tuple[int, int]]:
    return [(s, min(s + chunk, reps)) for s in range(0, reps, chunk)]


def chunk_indices(
    cell_key: str, seed: int, j: int, size: int, n: int, block: int = 1
) -> np.ndarray:
    """Resample positions for chunk j of a cell: depends only on (seed, cell, j, block).

    Circular block bootstrap: ceil(n / block) blocks of ``block`` consecutive days with
    uniform start points, wrapping around the end, truncated to n. Every day is equally
    likely to be drawn, so the bootstrap world's true parameter is the full-sample estimate
    (the centring the studentised statistics use). block = 1 is the iid pairs bootstrap.
    """
    ss = np.random.SeedSequence([seed, zlib.crc32(cell_key.encode()), j])
    rng = np.random.default_rng(ss)
    k = -(-n // block)
    starts = rng.integers(0, n, size=(size, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(size, k * block)[:, :n]


def _manifest(cell: Cell, s: BootstrapSettings) -> dict:
    return {
        "estimator_version": ESTIMATOR_VERSION,
        "cell": cell.key,
        "alpha": cell.alpha,
        "n": len(cell.y),
        "data_sha256": cell.data_sha256,
        "seed": s.seed,
        "reps": s.reps,
        "chunk": s.chunk,
        "min_tail": s.min_tail,
        "scheme": "circular block" if s.block_for(len(cell.y)) > 1 else "iid pairs",
        "block": s.block_for(len(cell.y)),
        "g_hat": list(cell.g_hat),
        "g0_hat": cell.g0_hat,
        "wald0": cell.wald0,
        "t0": cell.t0,
    }


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def prepare_cell_dir(root: Path, cell: Cell, s: BootstrapSettings) -> Path:
    """Create or validate a cell's directory; refuse a manifest made with other inputs."""
    d = root / cell.key
    d.mkdir(parents=True, exist_ok=True)
    m = _manifest(cell, s)
    mpath = d / "manifest.json"
    if mpath.exists():
        old = json.loads(mpath.read_text())
        keys = (
            "estimator_version",
            "data_sha256",
            "seed",
            "reps",
            "chunk",
            "min_tail",
            "n",
            "block",
        )
        diff = [k for k in keys if old.get(k) != m[k]]
        if diff:
            raise RuntimeError(
                f"{d}: existing bootstrap was made with different {diff}; move the folder "
                "away to start this cell afresh (results are never mixed)"
            )
    else:
        _atomic_write_bytes(mpath, json.dumps(m, indent=2).encode())
    return d


def done_chunks(d: Path) -> set[int]:
    return {int(p.stem.split("_")[1]) for p in d.glob("chunk_*.npz")}


def _run_chunk(args) -> tuple[str, int, float]:
    """Worker: compute and save one chunk. Returns (cell key, chunk id, seconds)."""
    d, cell, s, j, lo, hi = args
    t = time.perf_counter()
    idx = chunk_indices(cell.key, s.seed, j, hi - lo, len(cell.y), s.block_for(len(cell.y)))
    w, tt = esr_bootstrap_draws(
        cell.y, cell.ehat, cell.alpha, idx, cell.g_hat, cell.g0_hat, s.min_tail
    )
    import io

    buf = io.BytesIO()
    np.savez(buf, rep_lo=lo, rep_hi=hi, wald_star=w, t_star=tt)
    _atomic_write_bytes(Path(d) / f"chunk_{j:04d}.npz", buf.getvalue())
    return cell.key, j, time.perf_counter() - t


def _init_worker(nice: int) -> None:
    with contextlib.suppress(OSError):
        os.nice(nice)


# ---------------------------------------------------------------- running
class _Lock:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as err:
            self.fh.close()
            raise RuntimeError(f"another esr-bootstrap run holds {self.path}") from err
        self.fh.write(str(os.getpid()))
        self.fh.flush()
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def time_one_resample(cell: Cell, s: BootstrapSettings, k: int = 2) -> float:
    """Seconds per resample (both versions), measured on k resamples of this cell."""
    n = len(cell.y)
    idx = chunk_indices(cell.key, s.seed + 1, 0, k, n, s.block_for(n))  # not a real chunk
    t = time.perf_counter()
    esr_bootstrap_draws(cell.y, cell.ehat, cell.alpha, idx, cell.g_hat, cell.g0_hat, s.min_tail)
    return (time.perf_counter() - t) / k


def plan(root: Path, cells: list[Cell], s: BootstrapSettings) -> dict:
    """Remaining work and a wall-clock estimate (timed on the largest cell)."""
    bounds = chunk_bounds(s.reps, s.chunk)
    remaining = 0
    for c in cells:
        d = root / c.key
        finished = done_chunks(d) if d.exists() else set()
        remaining += sum(hi - lo for j, (lo, hi) in enumerate(bounds) if j not in finished)
    per = time_one_resample(max(cells, key=lambda c: len(c.y)), s) if cells and remaining else 0
    hours = remaining * per / max(1, s.workers) / 3600
    return {
        "cells": len(cells),
        "resamples_remaining": remaining,
        "seconds_per_resample": round(per, 3),
        "workers": s.workers,
        "estimated_hours": round(hours, 2),
    }


def run(root: Path, cells: list[Cell], s: BootstrapSettings, log_path: Path | None = None) -> dict:
    """Run or resume all remaining chunks within the settings' limits."""
    os.environ.update(
        {
            k: "1"
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        }
    )
    handler = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        log.addHandler(handler)
        if log.getEffectiveLevel() > logging.INFO:  # progress must reach the file
            log.setLevel(logging.INFO)
    try:
        with _Lock(root / ".lock"):
            bounds = chunk_bounds(s.reps, s.chunk)
            todo = []
            for c in cells:
                d = prepare_cell_dir(root, c, s)
                finished = done_chunks(d)
                todo += [
                    (str(d), c, s, j, lo, hi)
                    for j, (lo, hi) in enumerate(bounds)
                    if j not in finished
                ]
            total = len(todo)
            log.info(
                "esr-bootstrap: %d cells, %d chunks to do (%s)",
                len(cells),
                total,
                json.dumps(asdict(s)),
            )
            deadline = None if s.max_hours is None else time.time() + s.max_hours * 3600
            done, t0, stopped = 0, time.time(), False
            if total:
                with ProcessPoolExecutor(
                    s.workers, initializer=_init_worker, initargs=(s.nice,)
                ) as ex:
                    queue = iter(todo)
                    running = set()
                    for _ in range(s.workers * 2):  # bounded queue: budget checked per start
                        nxt = next(queue, None)
                        if nxt is not None:
                            running.add(ex.submit(_run_chunk, nxt))
                    while running:
                        fin, running = wait(running, return_when=FIRST_COMPLETED)
                        for f in fin:
                            key, j, sec = f.result()
                            done += 1
                            rate = done / (time.time() - t0)
                            eta = (total - done) / rate / 3600 if rate else float("nan")
                            log.info(
                                "%-22s chunk %4d done in %6.1fs | %d/%d | ETA %.2f h",
                                key,
                                j,
                                sec,
                                done,
                                total,
                                eta,
                            )
                        if deadline is not None and time.time() > deadline:
                            if not stopped:
                                log.info(
                                    "time budget reached: no new chunks; finishing the %d running",
                                    len(running),
                                )
                            stopped = True
                            continue
                        for _ in range(len(fin)):
                            nxt = next(queue, None)
                            if nxt is not None:
                                running.add(ex.submit(_run_chunk, nxt))
            log.info(
                "esr-bootstrap: %d/%d chunks done this session%s",
                done,
                total,
                " (stopped by the time budget; rerun to resume)" if stopped else "",
            )
            return {"chunks_done": done, "chunks_total": total, "stopped_by_budget": stopped}
    finally:
        if handler is not None:
            log.removeHandler(handler)
            handler.close()


# ---------------------------------------------------------------- dependence diagnostic
def _ljung_box_p(x: np.ndarray, lags: int = 10) -> float:
    from scipy import stats

    x = x - x.mean()
    n, d = len(x), float(x @ x)
    if d == 0:
        return float("nan")
    r = np.array([x[k:] @ x[:-k] / d for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(r**2 / (n - np.arange(1, lags + 1)))
    return float(stats.chi2.sf(q, lags))


def _long_run_ratio(x: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett) long-run variance / variance: 1 without autocorrelation."""
    x = x - x.mean()
    g0 = float(x @ x) / len(x)
    lr = g0 + sum(
        2 * (1 - k / (lags + 1)) * float(x[k:] @ x[:-k]) / len(x) for k in range(1, lags + 1)
    )
    return lr / g0 if g0 > 0 else float("nan")


def score_dependence(cell: Cell) -> dict:
    """Serial dependence of the fitted one-sided-intercept ESR scores (normalised units).

    Both the asymptotic covariance and an iid bootstrap assume the scores are serially
    uncorrelated. Reports Ljung-Box p-values (10 lags) for the quantile score 1[Z <= q] - a
    and the ES-intercept score (e - S)/e^2, and their Newey-West long-run / iid variance
    ratios with ceil(T^(1/3)) lags (> 1: iid standard errors too small)."""
    from alphalab.backtest.es import _at_or_below, _fit_joint

    z = cell.y - cell.ehat
    n = len(z)
    zs = z - float(z.max())
    x = np.column_stack([np.ones(n), cell.ehat])
    beta, g = _fit_joint(zs, x, np.ones((n, 1)), cell.alpha, np.array([zs.min() - z.std() - 1e-12]))
    q = x @ beta
    hit = _at_or_below(zs - q, float(np.abs(zs).max())).astype(float)
    s_es = (g[0] - (q + (zs - q) * hit / cell.alpha)) / g[0] ** 2
    s_q = hit - cell.alpha
    lags = int(np.ceil(n ** (1 / 3)))
    return {
        "cell": cell.key,
        "alpha": cell.alpha,
        "n": n,
        "lb10_p_quantile_score": _ljung_box_p(s_q),
        "lb10_p_es_score": _ljung_box_p(s_es),
        "lrv_ratio_quantile_score": _long_run_ratio(s_q, lags),
        "lrv_ratio_es_score": _long_run_ratio(s_es, lags),
    }


# ---------------------------------------------------------------- results
def degenerate_start(y: np.ndarray, x: np.ndarray) -> bool:
    """True if esreg's location-scale starting value has a non-positive fitted sd.

    esreg (and our port) shift the |residual| regression by min(0.001, min fitted), which
    sets the smallest fitted sd to (about) 0 whenever it is below 0.001; the likelihood is
    then undefined at the start, both optimisers fail and the start is kept, so the
    covariance divides by ~0 (Codex review round 4; found 2026-11-10). Such a draw has no
    valid covariance and is counted as failed. The quantile and the ES forecast lie in the
    span of the regressors, so this depends only on the OLS of y on x, i.e. on the resample.
    """
    b1 = np.linalg.lstsq(x, y, rcond=None)[0]
    b2 = np.linalg.lstsq(x, np.abs(y - x @ b1), rcond=None)[0]
    return bool(np.min(x @ b2) < 0.001)


def _degenerate_draws(cell: Cell, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per resample: (strict draw degenerate, intercept draw degenerate)."""
    ds, di = np.zeros(len(idx), bool), np.zeros(len(idx), bool)
    for i, ix in enumerate(idx):
        yb, eb = cell.y[ix], cell.ehat[ix]
        x = np.column_stack([np.ones(len(ix)), eb])
        ds[i] = degenerate_start(yb - yb.max(), x)
        z = yb - eb
        di[i] = degenerate_start(z - z.max(), x)
    return ds, di


def summarize(root: Path, s: BootstrapSettings, cells: list[Cell] | None = None) -> pd.DataFrame:
    """Bootstrap p-values per cell, from finished chunks. p-values only for complete cells
    with at most 5% failed resamples (esback's rule).

    With ``cells`` (the same inputs the run used; checked by data hash), draws whose
    resample hits esreg's degenerate location-scale start are counted as failed, and a cell
    whose full sample hits it gets no p-value (Addition 9, amendment 7)."""
    rows = []
    bounds = chunk_bounds(s.reps, s.chunk)
    by_key = {c.key: c for c in cells} if cells is not None else {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        m = json.loads((d / "manifest.json").read_text())
        if (
            m["estimator_version"] != ESTIMATOR_VERSION
            or m["reps"] != s.reps
            or m.get("block") != s.block_for(m["n"])
        ):
            continue
        files = sorted(d.glob("chunk_*.npz"), key=lambda f: int(f.stem.split("_")[1]))
        w = np.concatenate([np.load(f)["wald_star"] for f in files]) if files else np.array([])
        t = np.concatenate([np.load(f)["t_star"] for f in files]) if files else np.array([])
        complete = len(files) == len(bounds)
        row = {
            "cell": m["cell"],
            "alpha": m["alpha"],
            "reps_done": len(w),
            "reps_target": m["reps"],
            "complete": complete,
            "scheme": m.get("scheme"),
            "block": m.get("block"),
            "wald0": m["wald0"],
            "t0": m["t0"],
        }
        cell = by_key.get(m["cell"])
        full_bad = {"strict": False, "int": False}
        if cell is not None:
            if cell.data_sha256 != m["data_sha256"]:
                raise RuntimeError(f"{d}: cell data differ from the run's (hash mismatch)")
            js = [int(f.stem.split("_")[1]) for f in files]
            n, blk = len(cell.y), m["block"]
            idx = [
                chunk_indices(cell.key, m["seed"], j, hi - lo, n, blk)
                for j in js
                for lo, hi in [bounds[j]]
            ]
            ds, di = _degenerate_draws(cell, np.vstack(idx)) if idx else (np.array([]),) * 2
            w = np.where(ds, np.nan, w)
            t = np.where(di, np.nan, t)
            x = np.column_stack([np.ones(n), cell.ehat])
            z = cell.y - cell.ehat
            full_bad = {
                "strict": degenerate_start(cell.y - cell.y.max(), x),
                "int": degenerate_start(z - z.max(), x),
            }
            row["degenerate_share"] = float(np.mean(ds | di)) if len(ds) else np.nan
        for name, xs, observed, upper in (
            ("strict", w, m["wald0"], True),
            ("int", t, m["t0"], False),
        ):
            fail = float(np.mean(~np.isfinite(xs))) if len(xs) else np.nan
            ok = xs[np.isfinite(xs)]
            usable = complete and len(ok) > 0 and fail <= MAX_FAIL_SHARE and not full_bad[name]
            # strict: share of W* >= W0; intercept (one-sided, lower tail): share of t* <= t0
            p_boot = np.mean(ok >= observed) if upper else np.mean(ok <= observed)
            row[f"{name}_fail_share"] = fail
            row[f"{name}_full_sample_degenerate"] = full_bad[name]
            row[f"{name}_p_boot"] = float(p_boot) if usable else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "BootstrapSettings",
    "Cell",
    "ESRFitError",
    "cells_from_config",
    "degenerate_start",
    "chunk_bounds",
    "chunk_indices",
    "make_cell",
    "plan",
    "run",
    "summarize",
]
