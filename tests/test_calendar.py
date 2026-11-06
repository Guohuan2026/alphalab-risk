"""Exchange-calendar behaviour (M2 part 1). Each test pins one real-world holiday case."""

from __future__ import annotations

import pandas as pd
import pytest

from alphalab.config import load_config
from alphalab.core.calendar import (
    is_session,
    last_session_on_or_before,
    sessions_between,
    trading_sessions,
)
from alphalab.data.providers.synthetic import SyntheticProvider
from alphalab.data.schema import PriceBasis
from alphalab.data.validation import validate_prices
from alphalab.returns.core import compound_returns
from conftest import make_prices

GOOD_FRIDAY_2018 = "2018-03-30"  # a weekday, but NYSE closed; last weekday of March 2018
THANKSGIVING_2024 = "2024-11-28"
CARTER_MOURNING = "2025-01-09"  # unscheduled closure, announced ~2 weeks ahead
SANDY = "2012-10-29"  # unscheduled closure (Hurricane Sandy), known only that weekend


# ---------- the calendar itself ----------
@pytest.mark.parametrize("day", [GOOD_FRIDAY_2018, THANKSGIVING_2024, CARTER_MOURNING, SANDY])
def test_holidays_are_not_sessions(day):
    assert not is_session([day])[0]
    assert pd.Timestamp(day) not in trading_sessions("2012-01-01", "2025-12-31")


def test_trading_sessions_are_tz_naive_and_inclusive():
    s = trading_sessions("2024-12-23", "2024-12-27")
    assert s.tz is None
    # Christmas (25th) is skipped; both endpoints are included.
    assert list(s.strftime("%m-%d")) == ["12-23", "12-24", "12-26", "12-27"]


def test_sessions_between():
    starts = pd.to_datetime(["2024-12-23", "2024-12-24", "2024-12-23"])
    ends = pd.to_datetime(["2024-12-24", "2024-12-26", "2024-12-31"])
    # Consecutive sessions -> 1; across Christmas -> still 1; Dec 23 -> Dec 31 -> 5.
    assert list(sessions_between(starts, ends)) == [1, 1, 5]


def test_last_session_on_or_before():
    assert last_session_on_or_before("2018-03-31") == pd.Timestamp("2018-03-29")
    assert last_session_on_or_before("2024-08-31") == pd.Timestamp("2024-08-30")
    with pytest.raises(ValueError, match="before the calendar starts"):
        last_session_on_or_before("1900-01-01")


# ---------- returns: month completeness (fixes LIMITATIONS #7) ----------
def test_month_ending_on_good_friday_is_complete():
    # Data end Thu 2018-03-29, the real last trading day of March 2018.
    idx = trading_sessions("2018-03-01", "2018-03-29")
    ret = pd.DataFrame({"A": 0.001}, index=idx)
    m = compound_returns(ret, "ME")
    assert m.index[-1] == pd.Timestamp("2018-03-31")
    # The old Mon-Fri rule thinks Fri 30th is missing and wrongly drops March.
    assert compound_returns(ret, "ME", calendar=None).empty


def test_month_still_in_progress_is_dropped():
    idx = trading_sessions("2018-03-01", "2018-03-28")  # stops one session early
    assert compound_returns(pd.DataFrame({"A": 0.001}, index=idx), "ME").empty


# ---------- validation ----------
def _prices_on(dates) -> pd.DataFrame:
    return make_prices([(d, "A", 100.0 + i, 0.0, 1.0) for i, d in enumerate(dates)])


def test_holiday_week_is_not_a_gap():
    # Missing Dec 24, 26, 27, 30 = 4 missing sessions (5 sessions since Dec 23).
    # Mon-Fri counting would say 6 (it counts Christmas) and falsely warn.
    rep = validate_prices(_prices_on(["2024-12-20", "2024-12-23", "2024-12-31"]), "raw")
    assert not any(i.check == "date_gap" for i in rep.issues), rep.summary()


def test_real_gap_is_flagged_in_sessions():
    rep = validate_prices(_prices_on(["2024-12-20", "2024-12-23", "2025-01-02"]), "raw")
    gaps = [i for i in rep.warnings if i.check == "date_gap"]
    assert len(gaps) == 1 and "6 XNYS sessions" in gaps[0].detail


def test_row_on_holiday_is_flagged():
    rep = validate_prices(_prices_on(["2024-12-24", "2024-12-25", "2024-12-26"]), "raw")
    bad = [i for i in rep.warnings if i.check == "non_session_date"]
    assert [i.date for i in bad] == [pd.Timestamp("2024-12-25")]


# ---------- synthetic provider ----------
def test_synthetic_uses_exchange_sessions(synthetic_prices):
    assert is_session(synthetic_prices["date"]).all()
    assert pd.Timestamp("2020-11-26") not in set(synthetic_prices["date"])  # Thanksgiving
    rep = validate_prices(synthetic_prices, PriceBasis.RAW)
    assert not rep.warnings, rep.summary()


def test_synthetic_on_another_calendar():
    # Toronto: open on US Thanksgiving, closed Canada Day (Jul 1, 2024 was a Monday).
    df = SyntheticProvider(calendar="XTSE").fetch(["A"], "2024-06-28", "2024-11-29")
    dates = set(df["date"])
    assert pd.Timestamp(THANKSGIVING_2024) in dates
    assert pd.Timestamp("2024-07-01") not in dates


# ---------- config ----------
def test_old_gap_key_gives_clear_error(tmp_path):
    p = tmp_path / "old.yaml"
    p.write_text(
        "provider: synthetic\nuniverse: [A]\nstart: '2020-01-01'\nend: '2020-12-31'\n"
        "validation: {max_gap_bdays: 5}\n"
    )
    with pytest.raises(ValueError, match="renamed to max_gap_sessions"):
        load_config(p)
