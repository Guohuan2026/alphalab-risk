"""H3: does relative forecasting performance change in stress periods?

Rules (RESEARCH_DESIGN Clarification 4, fixed before any stress number was computed):
- A forecast belongs to an episode if its TARGET date (the day whose loss it forecasts)
  falls inside the episode. Calm = in no episode.
- VIX regime: VIX at the forecast ORIGIN > 30. It is known when the forecast is made, so
  conditioning on it is legitimate (unlike conditioning on the outcome).
- Benchmark: FHS-GJR. Loss differential d = FZ0_model − FZ0_FHS regressed on a stress
  dummy with Newey–West errors: b > 0 means the model loses ground in stress.
- Power: every regime table shows EXPECTED hits. A cell with 0.5 expected exceedances
  cannot reject anything and is reported for completeness only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphalab.backtest.compare import diebold_mariano, hac_regression, model_confidence_set
from alphalab.backtest.coverage import kupiec_pof
from alphalab.backtest.es import mcneil_frey
from alphalab.backtest.losses import fz0_loss
from alphalab.backtest.report import align

EPISODES: dict[str, tuple[str, str]] = {
    "ltcm_1998": ("1998-08-01", "1998-10-31"),
    "dotcom_2000_02": ("2000-03-01", "2002-10-31"),
    "gfc_2007_09": ("2007-07-01", "2009-03-31"),
    "euro_2011": ("2011-07-01", "2011-10-31"),
    "aug_2015": ("2015-08-01", "2015-08-31"),
    "q4_2018": ("2018-10-01", "2018-12-31"),
    "covid_2020": ("2020-02-20", "2020-04-30"),
    "rates_2022": ("2022-01-01", "2022-10-31"),
}
VIX_THRESHOLD = 30.0
MIN_TEST_OBS = 30  # DM / regression / MCS need enough days in (and out of) the regime


def episode_labels(target_dates) -> pd.Series:
    """Episode name per target date, "" if calm."""
    d = pd.DatetimeIndex(target_dates)
    out = pd.Series("", index=d, dtype=object)
    for name, (a, b) in EPISODES.items():
        out[(d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))] = name
    return out


def _regimes(ev: pd.DataFrame, vix: pd.Series | None) -> dict[str, np.ndarray]:
    labels = episode_labels(ev["target_date"]).to_numpy()
    masks = {"all": np.ones(len(ev), bool), "calm": labels == "", "episodes": labels != ""}
    masks |= {name: labels == name for name in EPISODES}
    if vix is not None:
        v = vix.reindex(ev.index).to_numpy()  # VIX at the ORIGIN (index = origin date)
        masks["vix_gt30"] = v > VIX_THRESHOLD
        masks["vix_le30"] = v <= VIX_THRESHOLD
    return masks


def stress_tables(
    evs: dict[str, pd.DataFrame],
    alphas,
    vix: pd.Series | None = None,
    benchmark: str = "fhs_gjr",
    mcs_reps: int = 2000,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    evs = align(evs)
    if benchmark not in evs:
        raise ValueError(f"benchmark {benchmark!r} not among models {sorted(evs)}")
    ref = next(iter(evs.values()))
    masks = _regimes(ref, vix)
    reg_rows, dm_rows, ols_rows, mcs_rows = [], [], [], []
    for a in alphas:
        fz = {
            name: fz0_loss(-ev["loss_next"], ev[f"var_{a:g}"], ev[f"es_{a:g}"], a)
            for name, ev in evs.items()
        }
        for regime, m in masks.items():
            if m.sum() == 0:
                continue
            for name, ev in evs.items():
                loss, var, es = (
                    ev[c].to_numpy()[m] for c in ("loss_next", f"var_{a:g}", f"es_{a:g}")
                )
                hits = loss > var
                mf = mcneil_frey(loss, var, es, a, seed=seed)
                reg_rows.append(
                    {
                        "regime": regime,
                        "model": name,
                        "alpha": a,
                        "n": int(m.sum()),
                        "expected_hits": float(a * m.sum()),
                        "hits": int(hits.sum()),
                        "hit_rate": float(hits.mean()),
                        "p_pof": kupiec_pof(hits, a)[1],
                        "mean_fz0": float(fz[name][m].mean()),
                        "es_severity": mf.severity_ratio,
                    }
                )
        # Under-powered or degenerate regimes (e.g. no episode day in a short sample) stay in
        # the regime table but get no DM / regression / MCS.
        testable = {r for r, m in masks.items() if MIN_TEST_OBS <= m.sum() <= len(m) - MIN_TEST_OBS}
        for regime in [r for r in ("episodes", "calm", "vix_gt30", "vix_le30") if r in testable]:
            m = masks[regime]
            for name in evs:
                if name == benchmark:
                    continue
                res = diebold_mariano(fz[name][m], fz[benchmark][m])
                dm_rows.append(
                    {"regime": regime, "alpha": a, "model": name, "vs": benchmark, **res.to_dict()}
                )
        for dummy in [r for r in ("episodes", "vix_gt30") if r in testable]:
            D = masks[dummy].astype(float)
            X = np.column_stack([np.ones(len(D)), D])
            for name in evs:
                if name == benchmark:
                    continue
                res = hac_regression(fz[name] - fz[benchmark], X)
                ols_rows.append(
                    {
                        "dummy": dummy,
                        "alpha": a,
                        "model": name,
                        "vs": benchmark,
                        "calm_diff": res.coef[0],  # a: mean differential outside stress
                        "t_calm": res.t[0],
                        "p_calm": res.p[0],
                        "stress_shift": res.coef[1],  # b: extra differential in stress
                        "t_shift": res.t[1],
                        "p_shift": res.p[1],
                        "n_stress": int(D.sum()),
                    }
                )
        for regime in [r for r in ("episodes", "vix_gt30") if r in testable]:
            m = masks[regime]
            L = pd.DataFrame({name: f[m] for name, f in fz.items()})
            mcs = model_confidence_set(L, reps=mcs_reps, seed=seed)
            mcs_rows += [{"regime": regime, "alpha": a, **row} for row in mcs.to_dict("records")]
    return {
        "regimes": pd.DataFrame(reg_rows),
        "dm": pd.DataFrame(dm_rows),
        "regression": pd.DataFrame(ols_rows),
        "mcs": pd.DataFrame(mcs_rows),
    }


def event_study(
    evs: dict[str, pd.DataFrame],
    alpha: float = 0.01,
    top: float = 0.01,
    pre: int = 5,
    post: int = 20,
    rise: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How fast does each model's VaR react to a large loss?

    Events: target days with the ``top`` share of largest realised losses. For an event at
    target T (forecast origin position e = T−1), the path is VaR at origins e+k,
    k = −pre..post, divided by VaR at origin e−pre. Returns (mean path per model indexed by
    k, summary with days until the mean path first reaches ``rise`` × its starting value).
    k = 0 is the forecast FOR the crash day (made before it); k = 1 is the first forecast
    that can react to it.
    """
    evs = align(evs)
    ref = next(iter(evs.values()))
    loss = ref["loss_next"].to_numpy()
    cutoff = np.quantile(loss, 1 - top)
    events = [e for e in np.flatnonzero(loss >= cutoff) if e - pre >= 0 and e + post < len(loss)]
    ks = np.arange(-pre, post + 1)
    paths, summary = {}, []
    for name, ev in evs.items():
        v = ev[f"var_{alpha:g}"].to_numpy()
        rel = np.array([v[e + ks] / v[e - pre] for e in events])
        mean_path = rel.mean(axis=0)
        paths[name] = mean_path
        after = np.flatnonzero((ks >= 1) & (mean_path >= rise))
        summary.append(
            {
                "model": name,
                "n_events": len(events),
                "var_ratio_k1": float(mean_path[ks == 1][0]),
                "var_ratio_k5": float(mean_path[ks == 5][0]),
                "peak_ratio": float(mean_path.max()),
                f"days_to_{rise:g}x": int(ks[after[0]]) if len(after) else np.nan,
            }
        )
    return pd.DataFrame(paths, index=pd.Index(ks, name="k")), pd.DataFrame(summary)
