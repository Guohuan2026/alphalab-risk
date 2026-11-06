from __future__ import annotations

import pandas as pd
import pytest

from alphalab.data.providers.synthetic import SyntheticProvider
from alphalab.data.schema import coerce_schema


def make_prices(rows: list[tuple]) -> pd.DataFrame:
    """rows: (date, security_id, close, dividend, split_ratio). OHLC set from close."""
    df = pd.DataFrame(rows, columns=["date", "security_id", "close", "dividend", "split_ratio"])
    df["ticker"] = df["security_id"]
    df["open"] = df["high"] = df["low"] = df["close"]
    df["volume"] = 1_000.0
    return coerce_schema(df)


@pytest.fixture
def split_and_dividend_prices() -> pd.DataFrame:
    # X: 2-for-1 split on day 3.  Y: $1.50 dividend on day 2.
    return make_prices(
        [
            ("2024-01-02", "X", 100.0, 0.0, 1.0),
            ("2024-01-03", "X", 102.0, 0.0, 1.0),
            ("2024-01-04", "X", 51.5, 0.0, 2.0),
            ("2024-01-02", "Y", 100.0, 0.0, 1.0),
            ("2024-01-03", "Y", 99.0, 1.5, 1.0),
            ("2024-01-04", "Y", 99.0, 0.0, 1.0),
        ]
    )


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    return SyntheticProvider(seed=7).fetch(["AAA", "BBB", "CCC"], "2020-01-01", "2021-12-31")


@pytest.fixture
def config_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "provider: synthetic\n"
        "universe: [AAA, BBB]\n"
        "start: '2020-01-01'\n"
        "end: '2021-06-30'\n"
        f"data_dir: '{tmp_path / 'data'}'\n"
        "monthly_min_obs: 15\n"
        "provider_options: {seed: 1}\n"
    )
    return p
