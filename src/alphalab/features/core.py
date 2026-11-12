"""Feature construction for 1-day-ahead VaR/ES models. See docs/METHODOLOGY.md §7.

Information set: a feature stamped at session t may use anything published by the end of
day t (after the close, before the t+1 open). The forecast it feeds is for the loss on t+1.

Feature families
----------------
1. Return-based (HAR structure, Corsi 2009): r_t, |r_t|, and realized variance as the mean
   of r^2 over 1, 5, 22 and 66 sessions (day / week / month / quarter). Volatility is
   persistent at several horizons; HAR captures that with a few averages.
2. Range-based variance from one asset's daily high/low/open/close (more efficient than
   r^2, because the intraday range carries information that the close-to-close return loses):
       Parkinson      σ²_PK = (ln H/L)² / (4 ln 2)
       Garman–Klass   σ²_GK = ½ (ln H/L)² − (2 ln 2 − 1)(ln C/O)²
   Both ignore the overnight gap (close t−1 → open t), so they understate total variance;
   models learn the scale, but this is a stated limitation.
3. Drawdown: value relative to its trailing 252-session high, using the portfolio's
   FORWARD-built wealth index (never a vendor's backward-adjusted level).
4. Market-stress levels: VIX (level, 5-session log change) and FRED spreads (level, 22-session
   change). Spreads use differences, not log changes, because they can be negative.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from alphalab.core.calendar import DEFAULT_CALENDAR, get_calendar

RV_WINDOWS = (5, 22, 66)
RANGE_WINDOWS = (5, 22)
DRAWDOWN_WINDOW = 252


def series_name(raw: str) -> str:
    """Column name for a source series: '^VIX' -> 'vix', 'BAA10Y' -> 'baa10y'."""
    return raw.strip().lstrip("^").lower().replace(".", "_").replace("-", "_")


def align_asof(
    series: pd.Series,
    sessions: pd.DatetimeIndex,
    lag_sessions: int = 0,
    max_stale_sessions: int = 5,
    calendar: str = DEFAULT_CALENDAR,
) -> pd.Series:
    """Put an irregularly dated series onto trading sessions without look-ahead.

    The value at session t is the latest observation dated on or before the session
    ``lag_sessions`` trading days earlier (lag 1 = "yesterday's FRED value, published this
    morning"). If that observation is more than ``max_stale_sessions`` sessions old, the
    result is NaN instead of an endlessly carried-forward number (a discontinued series
    must not look like a calm market).
    """
    obs = series.dropna().sort_index()
    cal = get_calendar(calendar).sessions
    sessions = pd.DatetimeIndex(sessions)
    pos = cal.searchsorted(sessions, side="right") - 1  # each date's session position (as-of)
    ref_pos = pos - lag_sessions
    out = np.full(len(sessions), np.nan)
    valid = ref_pos >= 0
    ref = cal[np.clip(ref_pos, 0, None)]
    j = obs.index.searchsorted(ref, side="right") - 1  # latest observation <= reference date
    valid &= j >= 0
    if valid.any() and len(obs):
        jj = np.clip(j, 0, None)
        obs_pos = cal.searchsorted(obs.index[jj], side="right") - 1
        fresh = (ref_pos - obs_pos) <= max_stale_sessions
        keep = valid & fresh
        out[keep] = obs.to_numpy()[jj[keep]]
    return pd.Series(out, index=sessions, name=series.name)


def build_features(
    inputs: pd.DataFrame,
    log_change_levels: Sequence[str] = (),
    diff_levels: Sequence[str] = (),
) -> pd.DataFrame:
    """Feature matrix from a session-indexed ``inputs`` frame.

    ``inputs`` must have ``ret`` (portfolio daily simple return). Optional columns:
    ``open, high, low, close`` (one asset's OHLC, for range-based variance) and the level
    columns named in ``log_change_levels`` (e.g. ``vix``) and ``diff_levels`` (e.g. spreads),
    already aligned with ``align_asof``. Rolling windows require a full window
    (``min_periods=window``), so early rows are NaN rather than estimated on 3 days of data.
    """
    if "ret" not in inputs:
        raise ValueError("inputs must contain a 'ret' column")
    x = inputs.sort_index()
    r = x["ret"]
    f = pd.DataFrame(index=x.index)
    f["ret"] = r
    f["abs_ret"] = r.abs()
    f["rv_1"] = r**2
    for w in RV_WINDOWS:
        f[f"rv_{w}"] = (r**2).rolling(w, min_periods=w).mean()

    if {"open", "high", "low", "close"} <= set(x.columns):
        hl = np.log(x["high"] / x["low"])
        co = np.log(x["close"] / x["open"])
        f["pk_1"] = hl**2 / (4 * np.log(2))
        gk = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
        f["gk_1"] = gk
        for w in RANGE_WINDOWS:
            f[f"gk_{w}"] = gk.rolling(w, min_periods=w).mean()

    wealth = (1.0 + r.fillna(0.0)).cumprod()  # forward-built: level at t uses returns <= t
    f["dd_252"] = wealth / wealth.rolling(DRAWDOWN_WINDOW, min_periods=1).max() - 1.0

    for c in log_change_levels:
        f[c] = x[c]
        f[f"{c}_chg_5"] = np.log(x[c] / x[c].shift(5))
    for c in diff_levels:
        f[c] = x[c]
        f[f"{c}_chg_22"] = x[c] - x[c].shift(22)
    f.index.name = "date"
    return f


def assemble_inputs(
    ret: pd.Series,
    series_long: pd.DataFrame,
    yahoo_levels: Sequence[str] = (),
    fred: Sequence[str] = (),
    fred_lag_sessions: int = 1,
    ohlc: pd.DataFrame | None = None,
    calendar: str = DEFAULT_CALENDAR,
) -> pd.DataFrame:
    """Session-indexed inputs for ``build_features``.

    - ``ret``: portfolio returns; its index defines the sessions.
    - ``series_long``: long [date, security_id, value] frame (a feature snapshot).
    - Yahoo levels (VIX) are same-day (lag 0): published at the close of t.
    - FRED series are lagged ``fred_lag_sessions`` (published the next morning).
    - ``ohlc``: optional date-indexed open/high/low/close of one asset (exact-date match;
      a missing day stays NaN rather than borrowing another day's range).
    """
    idx = pd.DatetimeIndex(ret.index)
    x = pd.DataFrame({"ret": ret.to_numpy()}, index=idx)
    by_id = {sid: g.set_index("date")["value"] for sid, g in series_long.groupby("security_id")}
    for raw, lag in [*((s, 0) for s in yahoo_levels), *((s, fred_lag_sessions) for s in fred)]:
        if raw not in by_id:
            raise ValueError(f"feature series {raw!r} not in snapshot (have {sorted(by_id)})")
        x[series_name(raw)] = align_asof(
            by_id[raw], idx, lag_sessions=lag, calendar=calendar
        ).to_numpy()
    if ohlc is not None:
        x = x.join(ohlc[["open", "high", "low", "close"]], how="left")
    return x
