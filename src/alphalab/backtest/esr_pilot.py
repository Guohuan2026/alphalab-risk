"""Pilot of the robust location-scale starts for the ESR bootstrap (Addition 9, amendment 9).

Why: with esreg's start the location-scale fit inside the covariance is degenerate in up to
26% of simulated samples and draws, so bootstrap results were unusable. The robust mode
(`es.robust_scale`) warm-starts each draw from the full-sample scale estimate and falls back
to a corrected and random starts. This pilot measures, per draw, whether the fit now fails,
which start succeeded, the draw's tail exceedances n_exceed (next to T*alpha) and whether
esreg's start would have failed on the same draw. Failed draws are kept and counted; they
are never dropped.

Pilot set (200 draws each): real development cells where failures occurred plus a control,
and 200 replications of each simulation design (full-sample fit + one block-20 draw).
Run: python -m alphalab.backtest.esr_pilot  (writes data/results/esr_pilot/draws.csv)
"""

from __future__ import annotations

import contextlib
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from alphalab.backtest import esr_bootstrap as eb
from alphalab.backtest import esr_sim as es_
from alphalab.backtest.es import (
    ESRFitError,
    _cond_mean_sigma,
    _intercept_fit,
    _strict_fit,
    robust_scale,
)

REAL = {
    "risk_p1_spy": ["fhs_gjr_0.025", "linqr_l1_0.025"],
    "risk_p1_spy_rolling": ["linqr_l1_0.05"],
    "risk_p3_xiu": ["qrf_0.025", "qrf_0.05", "gbq_0.025", "mlvol_fhs_0.05"],
}
DRAWS = 200
SEED = 20261111


def _init() -> None:
    for v in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[v] = "1"
    with contextlib.suppress(OSError):
        os.nice(10)
    warnings.simplefilter("ignore")


def full_sample_b2(y: np.ndarray, ehat: np.ndarray) -> np.ndarray | None:
    """Robust full-sample scale coefficients (the warm start); None if even that fails."""
    x = np.column_stack([np.ones(len(y)), ehat])
    ys = y - y.max()
    with robust_scale(None):
        try:
            mu, sig = _cond_mean_sigma(ys, x)
        except ESRFitError:
            return None
    return np.linalg.lstsq(x, sig, rcond=None)[0]  # sig = x @ b2 exactly


def _old_start_fails(y, eh) -> bool:
    x = np.column_stack([np.ones(len(y)), eh])
    z = y - eh
    return eb.degenerate_start(y - y.max(), x) or eb.degenerate_start(z - z.max(), x)


def robust_stats(y, eh, alpha, b2, g_hat=None, g0_hat=None) -> dict:
    """Strict W (vs H0, or vs g_hat if given) and intercept t, robust mode, with failures."""
    x = np.column_stack([np.ones(len(y)), eh])
    trace: list = []
    out = {
        "strict_ok": False,
        "int_ok": False,
        "stat_w": np.nan,
        "stat_t": np.nan,
        "g1": np.nan,
        "g0": np.nan,
    }
    with robust_scale(b2, trace):
        try:
            g, cov, _ = _strict_fit(y, x, alpha, 10)
            d = g - (np.array([0.0, 1.0]) if g_hat is None else np.asarray(g_hat))
            w = float(d @ np.linalg.solve(cov, d))
            if np.isfinite(w) and w >= 0:
                out.update(strict_ok=True, stat_w=w, g1=float(g[1]), g_strict=g)
        except (ESRFitError, np.linalg.LinAlgError) as ex:
            out["strict_err"] = str(ex)[:60]
        try:
            g0, se, _, _ = _intercept_fit(y - eh, x, alpha, 10)
            out.update(
                int_ok=True, stat_t=(g0 - (0.0 if g0_hat is None else g0_hat)) / se, g0=float(g0)
            )
        except (ESRFitError, np.linalg.LinAlgError) as ex:
            out["int_err"] = str(ex)[:60]
    out["starts_used"] = ",".join(trace)
    return out


def _real_job(args) -> list[dict]:
    cfgname, key, j0, j1 = args
    from alphalab.backtest.report import align
    from alphalab.config import load_config
    from alphalab.pipeline import _processed_dir, _results_dir
    from alphalab.risk.engine import attach_outcomes

    cfg = load_config(f"configs/{cfgname}.yaml")
    proc = _processed_dir(cfg)
    out = _results_dir(cfg, proc)
    fc = pd.read_parquet(out / "forecasts.parquet")
    m, a = key.rsplit("_", 1)
    ev = align(
        {
            m: attach_outcomes(
                fc[fc.model == m].drop(columns="model"),
                pd.read_parquet(proc / "portfolio_daily.parquet")["ret"],
            )
        }
    )[m]
    L, V, E = ev["loss_next"].to_numpy(), ev[f"var_{a}"].to_numpy(), ev[f"es_{a}"].to_numpy()
    y, eh = eb._normalise(L, E)
    alpha, n = float(a), len(y)
    b2 = full_sample_b2(y, eh)
    full = robust_stats(y, eh, alpha, b2)
    blk = int(np.ceil(n ** (1 / 3)))
    idx = eb.chunk_indices(key, SEED, 0, DRAWS, n, blk)[j0:j1]
    rows = []
    for i, ix in zip(range(j0, j1), idx, strict=True):
        r = robust_stats(y[ix], eh[ix], alpha, b2, full.get("g_strict"), full["g0"])
        r.pop("g_strict", None)
        rows.append(
            {
                "source": f"real {cfgname.replace('risk_', '')}",
                "cell": key,
                "draw": i,
                "T": n,
                "T_alpha": n * alpha,
                "n_exceed": int(np.sum(L[ix] > V[ix])),
                "old_start_fails": _old_start_fails(y[ix], eh[ix]),
                "full_sample_ok": full["strict_ok"] and full["int_ok"],
                **r,
            }
        )
    return rows


def _sim_job(args) -> list[dict]:
    name, r0, r1 = args
    s = es_.SimSettings()
    d = next(x for x in s.designs if x.name == name)
    rows = []
    for r in range(r0, r1):
        rng = np.random.default_rng(es_._rep_seed(s, d, r))
        L, V, E = es_.simulate(d, s.n, rng, s.burn)
        y, eh = eb._normalise(L, E)
        b2 = full_sample_b2(y, eh)
        full = robust_stats(y, eh, d.alpha, b2)
        ix = eb.chunk_indices(f"{d.name}_{r}", s.seed, 20, 1, s.n, 20)[0]
        base = {"source": f"sim {d.name}", "cell": f"rep{r}", "T": s.n, "T_alpha": s.n * d.alpha}
        rows.append(
            base
            | {
                "draw": "full",
                "n_exceed": int(np.sum(L > V)),
                "old_start_fails": _old_start_fails(y, eh),
                "full_sample_ok": True,
                **{k: v for k, v in full.items() if k != "g_strict"},
            }
        )
        if full["strict_ok"] and full["int_ok"]:
            bd = robust_stats(y[ix], eh[ix], d.alpha, b2, full["g_strict"], full["g0"])
            bd.pop("g_strict", None)
        else:  # no full-sample estimate to centre on: the draw cannot be used
            bd = {"strict_ok": False, "int_ok": False, "starts_used": "no full-sample fit"}
        rows.append(
            base
            | {
                "draw": "block20",
                "n_exceed": int(np.sum(L[ix] > V[ix])),
                "old_start_fails": _old_start_fails(y[ix], eh[ix]),
                "full_sample_ok": full["strict_ok"] and full["int_ok"],
                **bd,
            }
        )
    return rows


def main(out_dir: Path = Path("data/results/esr_pilot"), workers: int = 4) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (_real_job, (c, k, j, j + 50))
        for c, ks in REAL.items()
        for k in ks
        for j in range(0, DRAWS, 50)
    ]
    jobs += [
        (_sim_job, (d.name, r, r + 25))
        for d in es_.SimSettings().designs
        for r in range(0, DRAWS, 25)
    ]
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        futs = [ex.submit(f, a) for f, a in jobs]
        parts = [r for fu in futs for r in fu.result()]
    df = pd.DataFrame(parts)
    df.to_csv(out_dir / "draws.csv", index=False)
    return df


def report(df: pd.DataFrame) -> pd.DataFrame:
    """Failure rate per group (a result in its own right), with n_exceed of failed vs
    successful draws next to T*alpha, and esreg's-start failures on the same draws."""
    df = df.assign(
        failed=~(df.strict_ok.astype(bool) & df.int_ok.astype(bool)),
        group=np.where(
            df.source.str.startswith("real"),
            df.source + " " + df.cell,
            df.source + " " + df.draw.astype(str),
        ),
    )
    rows = []
    for gname, x in df.groupby("group"):
        n, k = len(x), int(x.failed.sum())
        lo, hi = _wilson(k, n)
        rows.append(
            {
                "group": gname,
                "draws": n,
                "failed": k,
                "fail_rate": k / n,
                "fail_ci95": f"[{lo:.3f}, {hi:.3f}]",
                "strict_failed": int((~x.strict_ok.astype(bool)).sum()),
                "int_failed": int((~x.int_ok.astype(bool)).sum()),
                "old_start_fail_rate": float(x.old_start_fails.mean()),
                "T_alpha": float(x.T_alpha.iloc[0]),
                "n_exceed_failed_med": float(x.n_exceed[x.failed].median()) if k else np.nan,
                "n_exceed_ok_med": float(x.n_exceed[~x.failed].median()),
                "warm_start_share": float(x.starts_used.fillna("").str.contains("warm").mean()),
                "fallback_used": int(
                    x.starts_used.fillna("").str.contains("corrected|random").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------- start invariance
TOL = 1e-6  # Codex decision 2026-11-11 (docs/DECISIONS.md): |dx| <= TOL * max(1, |x|), |dp| <= TOL


def _both_modes(y, eh, alpha, b2) -> dict:
    """W (vs H0), t, p-values with esreg's start and with the warm start, on one sample."""
    from scipy import stats

    x = np.column_stack([np.ones(len(y)), eh])
    out = {}
    for mode in ("esreg", "warm"):
        ctx = contextlib.nullcontext() if mode == "esreg" else robust_scale(b2)
        r = {"ok": False}
        with ctx, np.errstate(all="ignore"):
            try:
                g, cov, _ = _strict_fit(y, x, alpha, 10)
                d = g - np.array([0.0, 1.0])
                w = float(d @ np.linalg.solve(cov, d))
                g0, se, _, _ = _intercept_fit(y - eh, x, alpha, 10)
                t = g0 / se
                r = {
                    "ok": bool(np.isfinite(w) and np.isfinite(t)),
                    "w": w,
                    "t": t,
                    "p_w": float(stats.chi2.sf(w, 2)),
                    "p_t": float(stats.norm.cdf(t)),
                }
            except (ESRFitError, np.linalg.LinAlgError):
                pass
        out[mode] = r
    return out


def _inv_real(args) -> list[dict]:
    cfgname, key, n_draws = args
    from alphalab.backtest.report import align
    from alphalab.config import load_config
    from alphalab.pipeline import _processed_dir, _results_dir
    from alphalab.risk.engine import attach_outcomes

    cfg = load_config(f"configs/{cfgname}.yaml")
    proc = _processed_dir(cfg)
    fc = pd.read_parquet(_results_dir(cfg, proc) / "forecasts.parquet")
    m, a = key.rsplit("_", 1)
    ev = align(
        {
            m: attach_outcomes(
                fc[fc.model == m].drop(columns="model"),
                pd.read_parquet(proc / "portfolio_daily.parquet")["ret"],
            )
        }
    )[m]
    L, E = ev["loss_next"].to_numpy(), ev[f"es_{a}"].to_numpy()
    y, eh = eb._normalise(L, E)
    n, alpha = len(y), float(a)
    b2 = full_sample_b2(y, eh)
    rows = [
        {
            "source": cfgname,
            "cell": key,
            "draw": "full",
            "old_start_fails": _old_start_fails(y, eh),
            **_flat(_both_modes(y, eh, alpha, b2)),
        }
    ]
    idx = eb.chunk_indices(key, SEED, 0, DRAWS, n, int(np.ceil(n ** (1 / 3))))[:n_draws]
    for i, ix in enumerate(idx):
        rows.append(
            {
                "source": cfgname,
                "cell": key,
                "draw": i,
                "old_start_fails": _old_start_fails(y[ix], eh[ix]),
                **_flat(_both_modes(y[ix], eh[ix], alpha, b2)),
            }
        )
    return rows


def _inv_sim(args) -> list[dict]:
    name, r0, r1 = args
    s = es_.SimSettings()
    d = next(x for x in s.designs if x.name == name)
    rows = []
    for r in range(r0, r1):
        L, V, E = es_.simulate(d, s.n, np.random.default_rng(es_._rep_seed(s, d, r)), s.burn)
        y, eh = eb._normalise(L, E)
        b2 = full_sample_b2(y, eh)
        ix = eb.chunk_indices(f"{d.name}_{r}", s.seed, 20, 1, s.n, 20)[0]
        for lab, yy, ee in (("full", y, eh), ("block20", y[ix], eh[ix])):
            rows.append(
                {
                    "source": f"sim {name}",
                    "cell": f"rep{r}",
                    "draw": lab,
                    "old_start_fails": _old_start_fails(yy, ee),
                    **_flat(_both_modes(yy, ee, d.alpha, b2)),
                }
            )
    return rows


def _flat(res: dict) -> dict:
    return {f"{m}_{k}": v for m, r in res.items() for k, v in r.items()}


def invariance(
    out_dir: Path = Path("data/results/esr_pilot"),
    workers: int = 4,
    real_draws: int = 50,
    sim_reps: int = 100,
) -> pd.DataFrame:
    jobs = [(_inv_real, (c, k, real_draws)) for c, ks in REAL.items() for k in ks]
    jobs += [
        (_inv_sim, (d.name, r, r + 25))
        for d in es_.SimSettings().designs
        for r in range(0, sim_reps, 25)
    ]
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        futs = [ex.submit(f, a) for f, a in jobs]
        df = pd.DataFrame([r for fu in futs for r in fu.result()])
    df.to_csv(out_dir / "start_invariance.csv", index=False)
    return df


def invariance_report(df: pd.DataFrame) -> dict:
    # esreg's start "succeeds" only if it is not degenerate: a degenerate start still returns a
    # finite but meaningless statistic (Codex round 4), which must not count as success.
    eok = df.esreg_ok.fillna(False).astype(bool)
    wok = df.warm_ok.fillna(False).astype(bool)
    old_bad = df.old_start_fails.astype(bool)
    evalid = eok & ~old_bad
    both = df[evalid & wok]
    rel = lambda a, b: np.abs(a - b) / np.maximum(1.0, np.abs(a))  # noqa: E731
    dw, dt = rel(both.esreg_w, both.warm_w), rel(both.esreg_t, both.warm_t)
    dpw, dpt = np.abs(both.esreg_p_w - both.warm_p_w), np.abs(both.esreg_p_t - both.warm_p_t)
    worst = both.assign(d=np.maximum(dw, dt)).sort_values("d").tail(3)
    return {
        "samples": len(df),
        "both_succeeded": len(both),
        "esreg_degenerate_finite_garbage": int((eok & old_bad).sum()),
        "esreg_failed_warm_ok": int((~evalid & wok).sum()),
        "warm_failed_esreg_ok": int((evalid & ~wok).sum()),
        "max_rel_dW": float(dw.max()),
        "max_rel_dt": float(dt.max()),
        "max_abs_dp_strict": float(dpw.max()),
        "max_abs_dp_int": float(dpt.max()),
        "within_tolerance": bool(max(dw.max(), dt.max(), dpw.max(), dpt.max()) <= TOL),
        "n_beyond_tol": int(((dw > TOL) | (dt > TOL) | (dpw > TOL) | (dpt > TOL)).sum()),
        "worst": worst[
            ["source", "cell", "draw", "esreg_w", "warm_w", "esreg_t", "warm_t"]
        ].to_dict("records"),
    }


if __name__ == "__main__":
    _init()
    d = main()
    print(f"wrote {len(d)} draws")
