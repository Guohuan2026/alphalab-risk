"""The truncation test must pass for point-in-time code and FAIL for known leaks.

Testing that the detector catches bugs is as important as testing that clean code
passes - otherwise a broken detector would give false comfort.
"""

import pytest

from alphalab.core.lookahead import assert_point_in_time, check_point_in_time
from alphalab.data.schema import PriceBasis
from alphalab.returns.core import (
    compound_returns,
    compute_daily_returns,
    forward_returns,
    to_wide,
    total_return_index,
    trailing_returns,
)


@pytest.fixture
def monthly(synthetic_prices):
    daily = to_wide(compute_daily_returns(synthetic_prices, PriceBasis.RAW))
    return compound_returns(daily, "ME")


@pytest.fixture
def daily(synthetic_prices):
    return to_wide(compute_daily_returns(synthetic_prices, PriceBasis.RAW))


def test_trailing_momentum_is_point_in_time(monthly):
    assert_point_in_time(lambda m: trailing_returns(m, window=11, skip=1), monthly)


def test_total_return_index_is_point_in_time(synthetic_prices):
    long = compute_daily_returns(synthetic_prices, PriceBasis.RAW).set_index("date").sort_index()
    # Include the very first date: that is where the original leak lived.
    dates = [long.index[0], long.index[1], long.index[100], long.index[-2]]
    assert_point_in_time(lambda x: total_return_index(x.reset_index()), long, dates=dates)


def test_forward_returns_are_flagged(monthly):
    assert check_point_in_time(lambda m: forward_returns(m, 1), monthly)


def test_centered_window_is_flagged(daily):
    leak = lambda d: d.rolling(21, center=True).mean()  # noqa: E731
    assert check_point_in_time(leak, daily)


def test_full_sample_zscore_is_flagged(daily):
    leak = lambda d: (d - d.mean()) / d.std()  # noqa: E731  uses future mean/std
    assert check_point_in_time(leak, daily)


def test_backward_adjusted_level_is_flagged(daily):
    # A vendor-style adjusted close: scaled so the LAST value is the anchor.
    def leak(d):
        level = (1 + d.fillna(0)).cumprod()
        return level / level.iloc[-1]

    assert check_point_in_time(leak, daily)


def test_assert_raises_with_message(monthly):
    with pytest.raises(AssertionError, match="look-ahead detected"):
        assert_point_in_time(lambda m: m.shift(-1), monthly)
