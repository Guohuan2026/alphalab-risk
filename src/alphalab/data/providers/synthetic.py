"""Deterministic synthetic provider for offline development and tests.

Generates RAW (as-traded) prices with quarterly dividends and one 2-for-1 split,
plus a ``true_total_return`` column so tests can check that the return engine
recovers the ground truth exactly. Same seed + tickers + dates => identical data.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from alphalab.core.calendar import DEFAULT_CALENDAR, trading_sessions
from alphalab.data.providers.base import DataProvider
from alphalab.data.schema import PriceBasis, coerce_schema, empty_frame


class SyntheticProvider(DataProvider):
    name = "synthetic"
    price_basis = PriceBasis.RAW

    def __init__(
        self,
        seed: int = 42,
        annual_drift: float = 0.08,
        annual_vol: float = 0.25,
        dividend_yield_per_payment: float = 0.005,
        dividend_every_n_days: int = 63,
        calendar: str = DEFAULT_CALENDAR,
    ):
        self.seed = seed
        self.annual_drift = annual_drift
        self.annual_vol = annual_vol
        self.div_yield = dividend_yield_per_payment
        self.div_every = dividend_every_n_days
        self.calendar = calendar

    def _one(self, ticker: str, dates: pd.DatetimeIndex, with_split: bool) -> pd.DataFrame:
        # Seed from (global seed, ticker) so each ticker is stable and independent.
        rng = np.random.default_rng([self.seed, zlib.crc32(ticker.encode())])
        n = len(dates)
        mu, sig = self.annual_drift / 252, self.annual_vol / np.sqrt(252)
        log_tr = rng.normal(mu - 0.5 * sig**2, sig, n)  # log total return per day
        log_tr[0] = 0.0
        split_day = n // 2 if with_split else -1

        close = np.empty(n)
        div = np.zeros(n)
        split = np.ones(n)
        close[0] = 100.0
        for t in range(1, n):
            wealth = close[t - 1] * np.exp(log_tr[t])  # value of 1 old share incl. dividend
            d_old = self.div_yield * close[t - 1] if t % self.div_every == 0 else 0.0
            s = 2.0 if t == split_day else 1.0
            close[t] = (wealth - d_old) / s
            div[t] = d_old / s  # per POST-split share, matching the schema convention
            split[t] = s

        noise = np.abs(rng.normal(0, 0.004, (3, n)))
        open_ = close * np.exp(rng.normal(0, 0.003, n))
        high = np.maximum(open_, close) * (1 + noise[0])
        low = np.minimum(open_, close) * (1 - noise[1])
        true_ret = np.expm1(log_tr)
        true_ret[0] = np.nan
        return pd.DataFrame(
            {
                "date": dates,
                "security_id": ticker,
                "ticker": ticker,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": rng.integers(1e5, 5e6, n).astype(float),
                "dividend": div,
                "split_ratio": split,
                "true_total_return": true_ret,
            }
        )

    def fetch(self, tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
        dates = trading_sessions(start, end, self.calendar)  # real exchange days, no holidays
        if len(dates) < 2 or not tickers:
            return empty_frame()
        frames = [self._one(t, dates, with_split=(i == 0)) for i, t in enumerate(tickers)]
        return coerce_schema(pd.concat(frames, ignore_index=True))

    def metadata(self) -> dict:
        return {
            **super().metadata(),
            "seed": self.seed,
            "calendar": self.calendar,
            "note": "synthetic, not market data",
        }
