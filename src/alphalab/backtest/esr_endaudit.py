"""End-solution audit of the ESR auxiliary scale fit (EXPLORATORY, post-lockbox).

The Gaussian location-scale likelihood behind esreg's "scl_sp" covariance has sd linear in
the regressors, so it is unbounded as a fitted sd -> 0: an optimiser can *end* on that
boundary even from a valid start. The degenerate-*start* detector does not see this. Here a
fit is flagged when min/median fitted sd < THRESHOLD (0.05, the interior criterion
pre-specified in RESEARCH_DESIGN Addition 9 amendment 8(d)(2)); 0.01 and 0.10 are reported
as sensitivity. A flagged draw is a failed draw, never a valid observation.

Covers (1) every saved legacy draw (esreg's start, as run) and (2) every pilot sample
(robust start). Run: ``python -m alphalab.backtest.esr_endaudit``.
"""

from __future__ import annotations

import contextlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from alphalab.backtest import esr_bootstrap as eb
from alphalab.backtest import esr_pilot as ep
from alphalab.backtest import esr_sim as es_
from alphalab.backtest.es import ESRFitError, _cond_mean_sigma, robust_scale

THRESHOLD = 0.05
SENSITIVITY = (0.01, 0.05, 0.10)


def end_ratio(y: np.ndarray, eh: np.ndarray, b2=None, robust: bool = False) -> dict:
    """min/median fitted sd of the two auxiliary fits (strict: y; intercept: z = y - eh),
    NaN if the fit raises (robust mode only)."""
    x = np.column_stack([np.ones(len(y)), eh])
    z = y - eh
    out = {}
    for lab, v in (("strict", y - y.max()), ("int", z - z.max())):
        ctx = robust_scale(b2) if robust else contextlib.nullcontext()
        try:
            with ctx:
                _, sig = _cond_mean_sigma(v, x)
            out[f"ratio_{lab}"] = float(np.min(sig) / np.median(sig))
        except ESRFitError:
            out[f"ratio_{lab}"] = float("nan")
    return out


def _legacy_job(args) -> list[dict]:
    cfgname, cell_dir = args
    ep._init()
    from alphalab.config import load_config

    d = Path(cell_dir)
    m = json.loads((d / "manifest.json").read_text())
    loss, var, es, alpha = eb.series_from_config(load_config(f"configs/{cfgname}.yaml"))[m["cell"]]
    if eb._series_hash(loss, var, es, alpha) != m["data_sha256"]:
        raise RuntimeError(f"{d}: input data differ from the run's (hash mismatch)")
    y, eh = eb._normalise(loss, es)
    n = len(y)
    bounds = eb.chunk_bounds(m["reps"], m["chunk"])
    rows = []
    for f in sorted(d.glob("chunk_*.npz"), key=lambda f: int(f.stem.split("_")[1])):
        j = int(f.stem.split("_")[1])
        lo, hi = bounds[j]
        idx = eb.chunk_indices(m["cell"], m["seed"], j, hi - lo, n, m["block"])
        x = np.column_stack([np.ones(n), eh])
        for i, ix in enumerate(idx):
            r = end_ratio(y[ix], eh[ix])
            r["start_degenerate"] = eb.degenerate_start(
                y[ix] - y[ix].max(), x[ix]
            ) or eb.degenerate_start((y - eh)[ix] - (y - eh)[ix].max(), x[ix])
            rows.append({"source": f"legacy {cfgname}", "cell": m["cell"], "draw": lo + i, **r})
    return rows


def _pilot_real_job(args) -> list[dict]:
    cfgname, key = args
    ep._init()
    from alphalab.config import load_config

    loss, var, es, alpha = eb.series_from_config(load_config(f"configs/{cfgname}.yaml"))[key]
    y, eh = eb._normalise(loss, es)
    n = len(y)
    b2 = ep.full_sample_b2(y, eh)
    rows = [
        {"source": f"pilot {cfgname}", "cell": key, "draw": "full", **end_ratio(y, eh, b2, True)}
    ]
    idx = eb.chunk_indices(key, ep.SEED, 0, ep.DRAWS, n, int(np.ceil(n ** (1 / 3))))
    for i, ix in enumerate(idx):
        rows.append(
            {
                "source": f"pilot {cfgname}",
                "cell": key,
                "draw": i,
                **end_ratio(y[ix], eh[ix], b2, True),
            }
        )
    return rows


def _pilot_sim_job(args) -> list[dict]:
    name, r0, r1 = args
    ep._init()
    s = es_.SimSettings()
    d = next(x for x in s.designs if x.name == name)
    rows = []
    for r in range(r0, r1):
        L, V, E = es_.simulate(d, s.n, np.random.default_rng(es_._rep_seed(s, d, r)), s.burn)
        y, eh = eb._normalise(L, E)
        b2 = ep.full_sample_b2(y, eh)
        ix = eb.chunk_indices(f"{d.name}_{r}", s.seed, 20, 1, s.n, 20)[0]
        for lab, yy, ee in (("full", y, eh), ("block20", y[ix], eh[ix])):
            rows.append(
                {
                    "source": f"pilot sim {name}",
                    "cell": f"rep{r}",
                    "draw": lab,
                    **end_ratio(yy, ee, b2, True),
                }
            )
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per source and cell: draws, failed fits, and flagged draws at each threshold
    (a draw is flagged if either auxiliary fit is flagged or failed)."""
    worst = df[["ratio_strict", "ratio_int"]].min(axis=1, skipna=False)
    out = df.assign(worst=worst).groupby(["source", "cell"], sort=False)
    t = out.agg(
        draws=("worst", "size"),
        failed_fit=("worst", lambda w: int(w.isna().sum())),
        min_ratio=("worst", "min"),
    )
    for c in SENSITIVITY:
        t[f"flag_{c:g}"] = out["worst"].apply(lambda w, c=c: int((w.isna() | (w < c)).sum()))
    if "start_degenerate" in df:
        t["start_degenerate"] = out["start_degenerate"].sum().astype(int)
        t["flag_0.05_not_start"] = out.apply(
            lambda g: int(
                ((g.worst.isna() | (g.worst < THRESHOLD)) & ~g.start_degenerate.astype(bool)).sum()
            )
        )
    return t.reset_index()


def main(out_dir: Path = Path("data/results/esr_pilot"), workers: int = 4) -> None:
    jobs = []
    legacy_root = Path("data/results/p1_spy/20261106T222941Z_e325d363/esr_bootstrap")
    for d in sorted(p for p in legacy_root.iterdir() if p.is_dir()):
        if any(d.glob("chunk_*.npz")):
            jobs.append((_legacy_job, ("risk_p1_spy", str(d))))
    for cfgname, keys in ep.REAL.items():
        jobs += [(_pilot_real_job, (cfgname, k)) for k in keys]
    for dz in es_.SimSettings().designs:
        jobs += [(_pilot_sim_job, (dz.name, r, r + 50)) for r in range(0, 200, 50)]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(f, a) for f, a in jobs]
        for fu in futs:
            rows += fu.result()
    df = pd.DataFrame(rows)
    df["draw"] = df["draw"].astype(str)
    df.to_csv(out_dir / "end_solution_draws.csv", index=False)
    t = summarize(df)
    t.to_csv(out_dir / "end_solution_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
