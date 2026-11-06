"""FRED (Federal Reserve Bank of St. Louis) daily series via the public CSV endpoint.

No API key is needed. Used for market-stress features:
    BAA10Y  Moody's Baa corporate yield minus 10-year Treasury (credit spread, %)
    T10Y3M  10-year minus 3-month Treasury (term spread, %)

Timing: FRED dates a value by its *observation* day, but it is usually published the next
business morning. Features therefore use it with a one-session lag
(``features.fred_lag_sessions``); see docs/METHODOLOGY.md §7.

Caveat: this downloads the *latest vintage*. Market-based yields are essentially never
revised, but for macro series (GDP, payrolls) you would need ALFRED vintages to stay
point-in-time.
"""

from __future__ import annotations

import io
import logging
import time
import urllib.request
from collections.abc import Sequence

import pandas as pd

from alphalab.data.schema import SERIES_COLUMNS

log = logging.getLogger(__name__)
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}&coed={end}"


def parse_fred_csv(text: str, series_id: str) -> pd.DataFrame:
    """FRED CSV text -> long series frame. Pure function, so it is testable offline.

    FRED marks missing days (e.g. bond-market holidays) with "." or an empty cell;
    those rows are dropped rather than filled, so nothing is invented here.
    """
    raw = pd.read_csv(io.StringIO(text), na_values=["."], keep_default_na=True)
    date_col = raw.columns[0]  # "observation_date" (older files: "DATE")
    if series_id not in raw.columns:
        raise ValueError(f"FRED response has no column {series_id!r}: {list(raw.columns)}")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col]).astype("datetime64[ns]"),
            "security_id": pd.Series(series_id, index=raw.index, dtype="string"),
            "value": pd.to_numeric(raw[series_id], errors="coerce").astype("float64"),
        }
    )
    return out.dropna(subset=["value"]).reset_index(drop=True)[SERIES_COLUMNS]


class FredSource:
    name = "fred"

    def __init__(self, pause_seconds: float = 0.5, timeout: float = 30.0, max_retries: int = 2):
        self.pause_seconds = pause_seconds
        self.timeout = timeout
        self.max_retries = max_retries

    def fetch(self, series_ids: Sequence[str], start: str, end: str) -> pd.DataFrame:
        frames = []
        for sid in series_ids:
            url = FRED_CSV_URL.format(sid=sid, start=start, end=end)
            for attempt in range(self.max_retries + 1):
                try:
                    with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                        frames.append(parse_fred_csv(resp.read().decode(), sid))
                    break
                except Exception as exc:  # network errors; parse errors re-raise at the end
                    log.warning("fred %s attempt %d failed: %s", sid, attempt + 1, exc)
                    if attempt == self.max_retries:
                        raise
                    time.sleep(self.pause_seconds * (attempt + 1))
            time.sleep(self.pause_seconds)
        return (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SERIES_COLUMNS)
        )

    def metadata(self) -> dict:
        return {"source": "fred", "endpoint": FRED_CSV_URL.split("?")[0], "vintage": "latest"}
