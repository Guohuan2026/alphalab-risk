"""Exchange trading calendars (thin wrapper around ``exchange_calendars``).

Why not Mon-Fri business days? US exchanges close ~9 days a year (Good Friday,
Thanksgiving, ...) plus rare unscheduled closures (Hurricane Sandy 2012-10-29, the
national day of mourning 2025-01-09). Treating those as trading days makes us
"see" missing data that never existed and mis-judge whether a month is complete.

Sessions are returned as tz-naive midnight timestamps, matching the canonical
``date`` column.

Point-in-time note: the calendar is a fact about *when trading happened*, not market
information, so using it is not look-ahead. Unscheduled closures are only known in
hindsight, but they can only affect bookkeeping (gap counts, "is this month done?"),
never a signal value.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import exchange_calendars as xcals
import numpy as np
import pandas as pd

DEFAULT_CALENDAR = "XNYS"  # NYSE; NASDAQ (XNAS) shares its holiday schedule
_CALENDAR_START = "1925-01-01"  # CRSP daily history begins 1925-12-31


@lru_cache(maxsize=8)
def get_calendar(name: str = DEFAULT_CALENDAR) -> xcals.ExchangeCalendar:
    """Cached calendar from 1925 to ~1 year ahead (the library's default end)."""
    with warnings.catch_warnings():
        # exchange_calendars 4.13 uses Series.values on tz-aware data, deprecated in
        # pandas 3.1. Harmless for us (sessions are tz-naive); silence only that message.
        warnings.filterwarnings("ignore", message="Series.values returning an ndarray")
        return xcals.get_calendar(name, start=_CALENDAR_START)


def trading_sessions(start, end, calendar: str = DEFAULT_CALENDAR) -> pd.DatetimeIndex:
    """All sessions in [start, end], inclusive. Bounds need not be sessions."""
    sessions = get_calendar(calendar).sessions
    return sessions[(sessions >= pd.Timestamp(start)) & (sessions <= pd.Timestamp(end))]


def is_session(dates, calendar: str = DEFAULT_CALENDAR) -> np.ndarray:
    """Boolean array: is each date a trading session?"""
    sessions = get_calendar(calendar).sessions
    return pd.DatetimeIndex(dates).isin(sessions)


def sessions_between(start_dates, end_dates, calendar: str = DEFAULT_CALENDAR) -> np.ndarray:
    """Number of sessions in (start, end] for each pair: consecutive sessions give 1.

    The session-based counterpart of ``np.busday_count``. A non-session endpoint is
    treated as the last session before it (so a holiday row does not add a day).
    """
    sessions = get_calendar(calendar).sessions.values
    a = np.searchsorted(sessions, pd.DatetimeIndex(start_dates).values, side="right")
    b = np.searchsorted(sessions, pd.DatetimeIndex(end_dates).values, side="right")
    return b - a


def last_session_on_or_before(date, calendar: str = DEFAULT_CALENDAR) -> pd.Timestamp:
    """The latest session <= ``date`` (e.g. a month-end label -> its last trading day)."""
    sessions = get_calendar(calendar).sessions
    i = sessions.searchsorted(pd.Timestamp(date), side="right") - 1
    if i < 0:
        raise ValueError(f"{date} is before the calendar starts ({sessions[0].date()})")
    return sessions[i]
