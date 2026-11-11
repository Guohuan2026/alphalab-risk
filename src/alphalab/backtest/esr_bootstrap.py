"""Resumable bootstrap calibration of the ESR backtests (RESEARCH_DESIGN Addition 9).

Why: the asymptotic ESR p-values are approximations; Bayer & Dimitriadis (and esback)
calibrate them with a studentised iid pairs bootstrap. Each resample refits the joint
(quantile, ES) regression, so a full run (all models x levels x portfolios, B = 999) takes
hours. This module makes that an overnight job that can be stopped and resumed at any time.

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
ESTIMATOR_VERSION = "esr-2026-11-10-esreg-vcovA-port"
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


# ---------------------------------------------------------------- chunks and storage
def chunk_bounds(reps: int, chunk: int) -> list[tuple[int, int]]:
    return [(s, min(s + chunk, reps)) for s in range(0, reps, chunk)]


def chunk_indices(cell_key: str, seed: int, j: int, size: int, n: int) -> np.ndarray:
    """Resample positions for chunk j of a cell: depends only on (seed, cell, j)."""
    ss = np.random.SeedSequence([seed, zlib.crc32(cell_key.encode()), j])
    return np.random.default_rng(ss).integers(0, n, size=(size, n))


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
        keys = ("estimator_version", "data_sha256", "seed", "reps", "chunk", "min_tail", "n")
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
    idx = chunk_indices(cell.key, s.seed, j, hi - lo, len(cell.y))
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
    idx = chunk_indices(cell.key, s.seed + 1, 0, k, len(cell.y))  # not a real chunk
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


# ---------------------------------------------------------------- results
def summarize(root: Path, s: BootstrapSettings) -> pd.DataFrame:
    """Bootstrap p-values per cell, from finished chunks. p-values only for complete cells
    with at most 5% failed resamples (esback's rule)."""
    rows = []
    bounds = chunk_bounds(s.reps, s.chunk)
    for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        m = json.loads((d / "manifest.json").read_text())
        if m["estimator_version"] != ESTIMATOR_VERSION or m["reps"] != s.reps:
            continue
        files = sorted(d.glob("chunk_*.npz"))
        w = np.concatenate([np.load(f)["wald_star"] for f in files]) if files else np.array([])
        t = np.concatenate([np.load(f)["t_star"] for f in files]) if files else np.array([])
        complete = len(files) == len(bounds)
        row = {
            "cell": m["cell"],
            "alpha": m["alpha"],
            "reps_done": len(w),
            "reps_target": m["reps"],
            "complete": complete,
            "wald0": m["wald0"],
            "t0": m["t0"],
        }
        for name, x, observed, upper in (
            ("strict", w, m["wald0"], True),
            ("int", t, m["t0"], False),
        ):
            fail = float(np.mean(~np.isfinite(x))) if len(x) else np.nan
            ok = x[np.isfinite(x)]
            usable = complete and len(ok) > 0 and fail <= MAX_FAIL_SHARE
            # strict: share of W* >= W0; intercept (one-sided, lower tail): share of t* <= t0
            p_boot = np.mean(ok >= observed) if upper else np.mean(ok <= observed)
            row[f"{name}_fail_share"] = fail
            row[f"{name}_p_boot"] = float(p_boot) if usable else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "BootstrapSettings",
    "Cell",
    "ESRFitError",
    "cells_from_config",
    "chunk_bounds",
    "chunk_indices",
    "make_cell",
    "plan",
    "run",
    "summarize",
]
