"""Return calculations. See docs/METHODOLOGY.md for derivations.

Core identity (holding 1 share from close t-1 to close t):

    R_t = s_t * (P_t + D_t) / P_{t-1} - 1          total return
    RX_t = s_t * P_t / P_{t-1} - 1                  price-only return
    R_t  = RX_t + s_t * D_t / P_{t-1}               income component

where s_t is the split ratio on day t (new shares per old share), P is the RAW
close and D is the dividend per post-split share. For SPLIT_ADJUSTED prices the
vendor has already rescaled history, so s_t is treated as 1.

Point-in-time rule used throughout: a value stamped at date t may only use
information available at the close of t. Anything that looks forward
(``forward_returns``) is a LABEL for evaluation, never an input to a signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphalab.core.calendar import DEFAULT_CALENDAR, last_session_on_or_before
from alphalab.data.schema import KEY, PriceBasis


def compute_daily_returns(prices: pd.DataFrame, price_basis: PriceBasis | str) -> pd.DataFrame:
    """Long frame [date, security_id, ret, retx, div_yield] from canonical prices.

    The first observation of each security is NaN (no prior close). A gap in the
    data (missing days) produces a multi-day return on the next available row;
    validation flags such gaps.
    """
    basis = PriceBasis(price_basis)
    df = prices.sort_values(KEY, kind="mergesort")
    prev_close = df.groupby("security_id", sort=False)["close"].shift(1)
    s = df["split_ratio"] if basis is PriceBasis.RAW else 1.0
    out = pd.DataFrame(
        {
            "date": df["date"],
            "security_id": df["security_id"],
            "ret": s * (df["close"] + df["dividend"]) / prev_close - 1.0,
            "retx": s * df["close"] / prev_close - 1.0,
            "div_yield": s * df["dividend"] / prev_close,
        }
    )
    return out.reset_index(drop=True)


def to_wide(long: pd.DataFrame, value: str = "ret") -> pd.DataFrame:
    """Pivot long [date, security_id, value] to a date x security_id matrix."""
    wide = long.pivot(index="date", columns="security_id", values=value).sort_index()
    wide.columns.name = None
    return wide


def log_returns(ret: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Continuously compounded returns: ln(1 + R). Additive over time."""
    return np.log1p(ret)


def total_return_index(daily_long: pd.DataFrame, value: str = "ret") -> pd.DataFrame:
    """Total-return index (wide), built FORWARD from 1.0 on each security's first price date.

    Takes the LONG output of ``compute_daily_returns``: a row exists exactly when a
    price was observed, so the base date is known without peeking ahead. (An earlier
    version anchored on "the row before the first non-NaN return", which needs the
    NEXT row - the truncation test in tests/test_lookahead.py caught that leak.)

    Unlike a vendor's backward-adjusted close, the level at t depends only on
    returns up to t. NaN before a security's first observation.
    """
    df = daily_long.sort_values(KEY, kind="mergesort")
    growth = 1.0 + df[value].fillna(0.0)  # first row per security -> growth 1 -> base 1.0
    level = growth.groupby(df["security_id"], sort=False).cumprod()
    return to_wide(df.assign(tri=level), "tri")


def compound_returns(
    ret_wide: pd.DataFrame,
    freq: str = "ME",
    min_obs: int = 1,
    drop_incomplete_last: bool = True,
    calendar: str | None = DEFAULT_CALENDAR,
) -> pd.DataFrame:
    """Compound daily returns into periods (``"ME"`` month-end, ``"W-FRI"`` weekly).

    - Periods with fewer than ``min_obs`` daily returns are set to NaN.
    - Missing days inside a period are skipped (implicitly 0% return); validation
      flags the gaps.
    - ``drop_incomplete_last`` removes the final period if the data stop before that
      period's last trading session on ``calendar`` (e.g. a month still in progress).
      Using the exchange calendar means a month whose last weekday is a holiday
      (Good Friday 2018-03-30) is correctly treated as complete. ``calendar=None``
      falls back to Mon-Fri business days, for data with no exchange calendar.
    """
    log_r = np.log1p(ret_wide)
    out = np.expm1(log_r.resample(freq).sum(min_count=1))
    counts = ret_wide.resample(freq).count()
    out = out.where(counts >= min_obs)
    if drop_incomplete_last and len(out):
        last_label = out.index[-1]
        if calendar is None:
            last_day = pd.offsets.BDay().rollback(last_label)
        else:
            last_day = last_session_on_or_before(last_label, calendar)
        if ret_wide.index.max() < last_day:
            out = out.iloc[:-1]
    return out


def forward_returns(ret_wide: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """LABEL: compounded return over periods t+1 .. t+horizon, stamped at t.

    This deliberately looks ahead. Use it only as the target when evaluating a
    signal observed at t. Never feed it into a signal.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    log_r = np.log1p(ret_wide)
    summed = log_r.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    return np.expm1(summed)


def trailing_returns(ret_wide: pd.DataFrame, window: int, skip: int = 0) -> pd.DataFrame:
    """Compounded return over periods t-skip-window+1 .. t-skip, stamped at t.

    Point-in-time safe. With monthly data, ``window=11, skip=1`` gives classic
    "12-1" momentum: months t-11..t-1, skipping the most recent month t.
    """
    if window < 1 or skip < 0:
        raise ValueError("window must be >= 1 and skip >= 0")
    log_r = np.log1p(ret_wide)
    summed = log_r.rolling(window, min_periods=window).sum().shift(skip)
    return np.expm1(summed)
