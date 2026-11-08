"""Descriptive 10-day risk: does √h scaling of each model's 1-day forecast hold?

Specification: RESEARCH_DESIGN Clarification 8 (committed before any 10-day number).

For a forecast origin t (data up to the close of t), the 1-day forecasts VaR₁, ES₁ are
scaled to h days by the square-root-of-time rule used in Basel market-risk rules:

    VaR_h = √h · VaR₁        ES_h = √h · ES₁

and compared with the compounded loss over the next h sessions,

    L_h = −( Π_{i=1..h} (1 + R_{t+i}) − 1 )

Windows do not overlap (every h-th origin), so each outcome is used once. No model is
refitted: the forecasts are exactly the saved 1-day forecasts, so this step adds no
estimation and no new look-ahead risk. L_h is a LABEL (future data), for evaluation only.

√h is exact only for i.i.d. returns with zero mean; volatility clustering, mean reversion
of volatility and fat tails all bend it. With ~650 windows on P1, only ~6.5 breaches are
expected at 1%, so the table is descriptive: no hypothesis tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphalab.backtest.losses import fz0_loss


def h_day_loss(returns: pd.Series, h: int) -> pd.DataFrame:
    """LABEL stamped at origin t: compounded loss over t+1..t+h and the window's end date.

    NaN where fewer than h sessions follow t.
    """
    r = returns.sort_index()
    log_growth = np.log1p(r.to_numpy())
    csum = np.concatenate([[0.0], np.cumsum(log_growth)])
    n = len(r)
    loss = np.full(n, np.nan)
    end = pd.Series(pd.NaT, index=r.index, dtype="datetime64[ns]")
    ok = np.arange(n) + h < n
    pos = np.flatnonzero(ok)
    loss[pos] = -np.expm1(csum[pos + h + 1] - csum[pos + 1])
    end.iloc[pos] = r.index[pos + h]
    return pd.DataFrame({"loss_h": loss, "end_date": end}, index=r.index)


def h_day_windows(
    forecasts: dict[str, pd.DataFrame], returns: pd.Series, h: int = 10, before=None
) -> pd.DatetimeIndex:
    """Non-overlapping origins: every h-th trading day from the first origin all models share.

    Only windows whose last day is before ``before`` (the lockbox start) are kept.
    """
    common = None
    for f in forecasts.values():
        common = f.index if common is None else common.intersection(f.index)
    if common is None or len(common) == 0:
        raise ValueError("no forecast origins shared by all models")
    r = returns.sort_index()
    start = r.index.get_loc(common.min())
    origins = r.index[start::h]
    lab = h_day_loss(r, h).loc[origins].dropna(subset=["loss_h"])
    if before is not None:
        lab = lab[lab["end_date"] < pd.Timestamp(before)]
    keep = lab.index.intersection(common)
    if len(keep) != len(lab):
        raise ValueError("a non-overlapping origin is missing for some model")
    return keep


def h_day_table(
    forecasts: dict[str, pd.DataFrame],
    returns: pd.Series,
    alphas,
    h: int = 10,
    before=None,
) -> pd.DataFrame:
    """One row per model and α: windows, breaches vs expected, ES severity, mean FZ0."""
    origins = h_day_windows(forecasts, returns, h, before)
    lab = h_day_loss(returns, h).loc[origins]
    loss = lab["loss_h"]
    k = np.sqrt(h)
    rows = []
    for name, f in forecasts.items():
        f = f.loc[origins]
        for a in alphas:
            var, es = k * f[f"var_{a:g}"], k * f[f"es_{a:g}"]
            hit = loss > var
            n, m = len(loss), int(hit.sum())
            rows.append(
                {
                    "model": name,
                    "alpha": a,
                    "h": h,
                    "windows": n,
                    "first_origin": str(origins[0].date()),
                    "last_end": str(lab["end_date"].iloc[-1].date()),
                    "breaches": m,
                    "expected": n * a,
                    "breach_rate": m / n,
                    "breach_ratio": m / (n * a),
                    "severity_ratio": (float(loss[hit].mean() / es[hit].mean()) if m else np.nan),
                    "mean_fz0": float(np.mean(fz0_loss(-loss, var, es, a))),
                }
            )
    out = pd.DataFrame(rows)
    out["rank_fz0"] = out.groupby("alpha")["mean_fz0"].rank().astype(int)
    return out
