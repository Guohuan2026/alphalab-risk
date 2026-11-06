import numpy as np
import pandas as pd
import pytest

from alphalab.data.schema import PriceBasis
from alphalab.returns.core import (
    compound_returns,
    compute_daily_returns,
    forward_returns,
    to_wide,
    total_return_index,
    trailing_returns,
)


def _ret(df, sid, date, col="ret"):
    return df.set_index(["security_id", "date"]).loc[(sid, pd.Timestamp(date)), col]


class TestDailyReturns:
    def test_split_raw_basis(self, split_and_dividend_prices):
        r = compute_daily_returns(split_and_dividend_prices, PriceBasis.RAW)
        # 2-for-1: 1 share at 102 becomes 2 shares at 51.5 => 103 / 102 - 1
        assert _ret(r, "X", "2024-01-04") == pytest.approx(2 * 51.5 / 102 - 1)

    def test_split_ignored_when_split_adjusted(self, split_and_dividend_prices):
        # Same numbers declared as split-adjusted => the ratio must NOT be applied again.
        r = compute_daily_returns(split_and_dividend_prices, PriceBasis.SPLIT_ADJUSTED)
        assert _ret(r, "X", "2024-01-04") == pytest.approx(51.5 / 102 - 1)

    def test_dividend_total_vs_price_return(self, split_and_dividend_prices):
        r = compute_daily_returns(split_and_dividend_prices, PriceBasis.RAW)
        assert _ret(r, "Y", "2024-01-03") == pytest.approx((99 + 1.5) / 100 - 1)
        assert _ret(r, "Y", "2024-01-03", "retx") == pytest.approx(99 / 100 - 1)

    def test_identity_ret_equals_retx_plus_income(self, synthetic_prices):
        r = compute_daily_returns(synthetic_prices, PriceBasis.RAW).dropna()
        np.testing.assert_allclose(r["ret"], r["retx"] + r["div_yield"], atol=1e-14)

    def test_first_obs_is_nan_and_no_cross_security_leak(self, split_and_dividend_prices):
        r = compute_daily_returns(split_and_dividend_prices, PriceBasis.RAW)
        firsts = r.groupby("security_id").head(1)
        assert firsts["ret"].isna().all()  # Y's first day must not use X's last close

    def test_recovers_synthetic_ground_truth(self, synthetic_prices):
        r = compute_daily_returns(synthetic_prices, PriceBasis.RAW)
        truth = synthetic_prices.sort_values(["security_id", "date"])["true_total_return"]
        np.testing.assert_allclose(r["ret"].to_numpy(), truth.to_numpy(), rtol=1e-10)


class TestIndexAndCompounding:
    def test_total_return_index(self):
        idx = pd.bdate_range("2024-01-01", periods=4)
        long = pd.DataFrame(
            {
                "date": list(idx) + list(idx[2:]),
                "security_id": ["A"] * 4 + ["B"] * 2,  # B lists on day 3
                "ret": [np.nan, 0.10, -0.05, 0.02, np.nan, 0.03],
            }
        )
        tri = total_return_index(long)
        np.testing.assert_allclose(tri["A"], [1.0, 1.1, 1.1 * 0.95, 1.1 * 0.95 * 1.02])
        assert tri["B"].iloc[:2].isna().all()  # before listing
        assert tri["B"].iloc[2] == 1.0 and tri["B"].iloc[3] == pytest.approx(1.03)

    def test_monthly_compounding(self):
        idx = pd.to_datetime(["2024-01-29", "2024-01-30", "2024-01-31", "2024-02-29"])
        ret = pd.DataFrame({"A": [0.01, 0.01, 0.01, 0.02]}, index=idx)
        m = compound_returns(ret, "ME")
        assert m.loc["2024-01-31", "A"] == pytest.approx(1.01**3 - 1)
        assert m.loc["2024-02-29", "A"] == pytest.approx(0.02)

    def test_incomplete_last_month_dropped(self):
        idx = pd.bdate_range("2024-01-02", "2024-02-15")
        ret = pd.DataFrame({"A": 0.001}, index=idx)
        m = compound_returns(ret, "ME")
        assert list(m.index) == [pd.Timestamp("2024-01-31")]
        m_keep = compound_returns(ret, "ME", drop_incomplete_last=False)
        assert len(m_keep) == 2

    def test_month_ending_on_weekend_is_complete(self):
        # Aug 2024 ends Sat 31st; last business day is Fri 30th.
        idx = pd.bdate_range("2024-08-01", "2024-08-30")
        m = compound_returns(pd.DataFrame({"A": 0.0}, index=idx), "ME")
        assert pd.Timestamp("2024-08-31") in m.index

    def test_min_obs(self):
        idx = pd.bdate_range("2024-01-25", "2024-02-29")  # only 5 days in Jan
        m = compound_returns(pd.DataFrame({"A": 0.001}, index=idx), "ME", min_obs=15)
        assert np.isnan(m.loc["2024-01-31", "A"]) and not np.isnan(m.loc["2024-02-29", "A"])


class TestForwardAndTrailing:
    ret = pd.DataFrame(
        {"A": [np.nan, 0.1, 0.2, 0.3]}, index=pd.date_range("2024-01-31", periods=4, freq="ME")
    )

    def test_forward_alignment(self):
        f1 = forward_returns(self.ret, 1)["A"]
        assert f1.iloc[0] == pytest.approx(0.1)  # stamped at t, realised over t+1
        assert f1.iloc[2] == pytest.approx(0.3) and np.isnan(f1.iloc[3])
        f2 = forward_returns(self.ret, 2)["A"]
        assert f2.iloc[0] == pytest.approx(1.1 * 1.2 - 1)

    def test_trailing_with_skip(self):
        t = trailing_returns(self.ret, window=2, skip=1)["A"]
        # at t=3: months 1..2 (skip month 3) => 1.1*1.2-1
        assert t.iloc[3] == pytest.approx(1.1 * 1.2 - 1)
        assert np.isnan(t.iloc[2])  # window would include the NaN first month

    def test_to_wide(self, split_and_dividend_prices):
        w = to_wide(compute_daily_returns(split_and_dividend_prices, "raw"))
        assert list(w.columns) == ["X", "Y"] and w.index.is_monotonic_increasing
