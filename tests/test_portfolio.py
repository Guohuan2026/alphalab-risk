"""Portfolio return series: hand-checked math, drift, start alignment, PIT."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphalab.core.calendar import trading_sessions
from alphalab.core.lookahead import assert_point_in_time, check_point_in_time
from alphalab.portfolio import portfolio_returns
from alphalab.returns.core import compute_daily_returns, to_wide


@pytest.fixture
def asset_returns(synthetic_prices) -> pd.DataFrame:
    return to_wide(compute_daily_returns(synthetic_prices, "raw")).dropna()


def test_single_asset_is_identity(asset_returns):
    p = portfolio_returns(asset_returns, {"AAA": 1.0}, "daily")
    pd.testing.assert_series_equal(p["ret"], asset_returns["AAA"], check_names=False)


def test_daily_rebalance_is_weighted_sum(asset_returns):
    w = {"AAA": 0.6, "BBB": 0.3, "CCC": 0.1}
    p = portfolio_returns(asset_returns, w, "daily")
    expected = sum(asset_returns[a] * v for a, v in w.items())
    np.testing.assert_allclose(p["ret"], expected, rtol=1e-12)
    assert (p["w_AAA"] == 0.6).all()


def test_monthly_drift_by_hand():
    # 50/50. Day 1: A +10%, B 0% -> portfolio +5%; A's weight drifts to 0.55/1.05.
    idx = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01"])
    R = pd.DataFrame({"A": [0.10, 0.00, 0.00], "B": [0.00, 0.10, 0.00]}, index=idx)
    p = portfolio_returns(R, {"A": 0.5, "B": 0.5}, "monthly")
    wa = 0.55 / 1.05
    assert p["ret"].iloc[0] == pytest.approx(0.05)
    assert p["w_A"].iloc[1] == pytest.approx(wa)
    assert p["ret"].iloc[1] == pytest.approx((1 - wa) * 0.10)
    assert p["w_A"].iloc[2] == 0.5  # first session of February: back to target


def test_monthly_equals_buy_and_hold_within_month(asset_returns):
    # Inside a month, drifting weights == holding the initial dollar amounts.
    w = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    p = portfolio_returns(asset_returns, w, "monthly")
    month = asset_returns.loc["2021-03"]
    hold = sum(v * (1 + month[a]).prod() for a, v in w.items()) - 1
    assert (1 + p.loc["2021-03", "ret"]).prod() - 1 == pytest.approx(hold, rel=1e-12)


def test_starts_when_all_assets_have_returns():
    idx = trading_sessions("2024-03-01", "2024-03-15")
    R = pd.DataFrame({"A": 0.01, "B": 0.02}, index=idx)
    R.loc[: idx[3], "B"] = np.nan  # B lists later
    p = portfolio_returns(R, {"A": 0.5, "B": 0.5})
    assert p.index[0] == idx[4]


@pytest.mark.parametrize(
    "weights, rule, msg",
    [
        ({"AAA": 0.5, "BBB": 0.4}, "daily", "sum to 1"),
        ({"AAA": 0.5, "ZZZ": 0.5}, "daily", "no returns"),
        ({"AAA": 1.0}, "weekly", "rebalance must be"),
        ({}, "daily", "empty"),
    ],
)
def test_input_errors(asset_returns, weights, rule, msg):
    with pytest.raises(ValueError, match=msg):
        portfolio_returns(asset_returns, weights, rule)


def test_hole_after_start_raises(asset_returns):
    R = asset_returns.copy()
    R.iloc[50, 1] = np.nan
    with pytest.raises(ValueError, match="missing asset returns"):
        portfolio_returns(R, {"AAA": 0.5, "BBB": 0.5})


# ---------- point-in-time ----------
W = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}


@pytest.mark.parametrize("rule", ["daily", "monthly"])
def test_portfolio_is_point_in_time(asset_returns, rule):
    assert_point_in_time(lambda R: portfolio_returns(R, W, rule), asset_returns, n_checks=25)


def test_rebalance_on_last_day_rule_would_leak(asset_returns):
    """Teaching test: deciding 'rebalance at the close of the last day of the month' by
    looking at the NEXT row's month is look-ahead. Where does it show? Not at true month
    ends (there "no next row" and "next row is a new month" agree), but MID-month: data
    truncated on the 14th have no next row, so the rule wrongly treats the 14th as a month
    end. A value that depends on whether tomorrow's row exists is a leak."""

    def leaky(R):
        R = R.sort_index()
        last_day = R.index.to_series().dt.month.shift(-1) != R.index.month  # needs t+1
        out = portfolio_returns(R, W, "daily").copy()
        out["ret"] = out["ret"].where(~last_day.to_numpy(), out["ret"] * 1.0001)
        return out

    by_month = asset_returns.groupby(asset_returns.index.to_period("M"))
    month_ends = list(by_month.tail(1).index[:-1])
    mid_month = list(by_month.nth(10).index[:6])
    assert not check_point_in_time(leaky, asset_returns, dates=month_ends)  # hides here...
    assert check_point_in_time(leaky, asset_returns, dates=mid_month)  # ...caught here
    assert_point_in_time(  # the real implementation passes on both kinds of date
        lambda R: portfolio_returns(R, W, "monthly"), asset_returns, dates=month_ends + mid_month
    )


def test_portfolio_config_guards():
    from alphalab.config import PortfolioConfig

    assert PortfolioConfig("p", {" spy ": 1.0}).weights == {"SPY": 1.0}  # normalised like tickers
    with pytest.raises(ValueError, match="sum to 1"):
        PortfolioConfig("p", {"SPY": 0.6, "IEF": 0.3})
    with pytest.raises(ValueError, match="rebalance"):
        PortfolioConfig("p", {"SPY": 1.0}, rebalance="quarterly")
