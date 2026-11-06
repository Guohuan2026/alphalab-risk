"""Yahoo Finance provider (via the unofficial ``yfinance`` library).

What Yahoo actually gives you with ``auto_adjust=False``:
- ``Close``     : split-adjusted, NOT dividend-adjusted. So the basis is SPLIT_ADJUSTED.
- ``Adj Close`` : split- and dividend-adjusted, *backward* from today. Its LEVEL on a
                  past date depends on dividends paid after that date, so it embeds
                  future information. We keep it only as ``vendor_adj_close`` for
                  cross-checking; we build our own total returns from Close + Dividends.
- ``Dividends`` : per share, split-adjusted to today's share count.
- ``Stock Splits``: ratio on the ex-date, 0 on other days.

Caveats: unofficial API, can change or rate-limit; history is restated after each new
split; delisted tickers are mostly unavailable (survivorship bias). See docs/LIMITATIONS.md.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import pandas as pd

from alphalab.data.providers.base import DataProvider
from alphalab.data.schema import PriceBasis, coerce_schema, empty_frame

log = logging.getLogger(__name__)


def normalize_yfinance_history(hist: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert one ``yf.Ticker(t).history(auto_adjust=False, actions=True)`` frame
    into the canonical schema. Pure function, so it is unit-testable offline."""
    if hist is None or hist.empty:
        return empty_frame()
    idx = pd.DatetimeIndex(hist.index)
    if idx.tz is not None:
        # Drop the tz but keep exchange-local wall time, so the calendar date is right.
        idx = idx.tz_localize(None)
    n = len(hist)

    def col(name: str, default: float) -> pd.Series:
        if name in hist.columns:
            return pd.Series(hist[name].to_numpy(), dtype="float64")
        return pd.Series([default] * n, dtype="float64")

    splits = col("Stock Splits", 0.0)
    out = pd.DataFrame(
        {
            "date": idx.normalize(),
            "security_id": ticker,
            "ticker": ticker,
            "open": col("Open", float("nan")),
            "high": col("High", float("nan")),
            "low": col("Low", float("nan")),
            "close": col("Close", float("nan")),
            "volume": col("Volume", float("nan")),
            "dividend": col("Dividends", 0.0).fillna(0.0),
            # yfinance reports 0 on non-split days; canonical schema uses 1.0.
            "split_ratio": splits.where(splits > 0, 1.0),
        }
    )
    if "Adj Close" in hist.columns:
        out["vendor_adj_close"] = pd.Series(hist["Adj Close"].to_numpy(), dtype="float64")
    return coerce_schema(out)


class YahooProvider(DataProvider):
    name = "yahoo"
    price_basis = PriceBasis.SPLIT_ADJUSTED

    def __init__(self, pause_seconds: float = 0.5, max_retries: int = 2):
        self.pause_seconds = pause_seconds
        self.max_retries = max_retries
        self.failed: list[str] = []

    def fetch(self, tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
        import yfinance as yf  # lazy import: tests and CRSP users don't need it

        # yfinance's `end` is EXCLUSIVE; our interface is inclusive, so add one day.
        end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        frames, self.failed = [], []
        for t in tickers:
            hist = None
            for attempt in range(self.max_retries + 1):
                try:
                    hist = yf.Ticker(t).history(
                        start=start,
                        end=end_exclusive,
                        interval="1d",
                        auto_adjust=False,  # keep raw Close + Dividends; we adjust ourselves
                        actions=True,
                        repair=False,  # no silent vendor-side "fixes"; we validate instead
                    )
                    break
                except Exception as exc:  # network/API errors from an unofficial endpoint
                    log.warning("yahoo %s attempt %d failed: %s", t, attempt + 1, exc)
                    time.sleep(self.pause_seconds * (attempt + 1))
            df = normalize_yfinance_history(hist, t) if hist is not None else empty_frame()
            if df.empty:
                log.warning("no data returned for %s", t)
                self.failed.append(t)
            else:
                frames.append(df)
            time.sleep(self.pause_seconds)
        if not frames:
            return empty_frame()
        return coerce_schema(pd.concat(frames, ignore_index=True))

    def metadata(self) -> dict:
        import yfinance as yf

        return {
            **super().metadata(),
            "yfinance_version": yf.__version__,
            "auto_adjust": False,
            "failed_tickers": list(self.failed),
        }
