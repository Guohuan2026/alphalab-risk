"""Schema, providers, snapshot store and validation."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from alphalab.data.providers import get_provider
from alphalab.data.providers.synthetic import SyntheticProvider
from alphalab.data.providers.yahoo import normalize_yfinance_history
from alphalab.data.schema import PriceBasis, SchemaError, check_schema, coerce_schema
from alphalab.data.store import RawSnapshotStore, content_hash
from alphalab.data.validation import validate_prices


# ---------- schema ----------
def test_coerce_and_check(split_and_dividend_prices):
    check_schema(split_and_dividend_prices)  # should not raise


def test_missing_column_rejected(split_and_dividend_prices):
    with pytest.raises(SchemaError, match="missing"):
        coerce_schema(split_and_dividend_prices.drop(columns="dividend"))


def test_duplicate_key_rejected(split_and_dividend_prices):
    dup = pd.concat([split_and_dividend_prices, split_and_dividend_prices.head(1)])
    with pytest.raises(SchemaError, match="duplicate"):
        check_schema(dup)


# ---------- providers ----------
def test_synthetic_is_deterministic():
    a = SyntheticProvider(seed=3).fetch(["AAA", "BBB"], "2020-01-01", "2020-12-31")
    b = SyntheticProvider(seed=3).fetch(["AAA", "BBB"], "2020-01-01", "2020-12-31")
    pd.testing.assert_frame_equal(a, b)
    c = SyntheticProvider(seed=4).fetch(["AAA", "BBB"], "2020-01-01", "2020-12-31")
    assert not a["close"].equals(c["close"])


def test_synthetic_has_one_split_and_dividends(synthetic_prices):
    assert (synthetic_prices["split_ratio"] == 2.0).sum() == 1
    assert (synthetic_prices["dividend"] > 0).sum() > 0


def test_yfinance_normalization_offline():
    # Mimics yf.Ticker(t).history(auto_adjust=False, actions=True) output.
    idx = pd.DatetimeIndex(["2020-08-27", "2020-08-28", "2020-08-31"]).tz_localize(
        "America/New_York"
    )
    hist = pd.DataFrame(
        {
            "Open": [126.0, 126.0, 127.6],
            "High": [127.5, 126.4, 131.0],
            "Low": [123.8, 124.6, 126.0],
            "Close": [125.0, 124.8, 129.0],
            "Adj Close": [122.0, 121.8, 126.0],
            "Volume": [1e8, 1e8, 2e8],
            "Dividends": [0.0, 0.0, 0.0],
            "Stock Splits": [0.0, 0.0, 4.0],
        },
        index=idx,
    )
    out = normalize_yfinance_history(hist, "AAPL")
    check_schema(out)
    assert out["date"].dt.tz is None
    assert list(out["date"].dt.day) == [27, 28, 31]  # tz drop must not shift dates
    assert list(out["split_ratio"]) == [1.0, 1.0, 4.0]
    assert "vendor_adj_close" in out.columns


def test_yfinance_empty_history():
    assert normalize_yfinance_history(pd.DataFrame(), "ZZZZ").empty


def test_crsp_stub_and_unknown_provider():
    with pytest.raises(NotImplementedError):
        get_provider("crsp").fetch(["AAPL"], "2020-01-01", "2020-12-31")
    with pytest.raises(ValueError):
        get_provider("bloomberg")


# ---------- store ----------
def test_snapshot_roundtrip_and_hash(tmp_path, synthetic_prices):
    store = RawSnapshotStore(tmp_path)
    meta = {"provider": "synthetic", "price_basis": "raw"}
    info = store.save(synthetic_prices, meta, {"tickers": ["AAA", "BBB", "CCC", "ZZZ"]})
    df, man = store.load("synthetic", info.snapshot_id)
    pd.testing.assert_frame_equal(df, synthetic_prices)
    assert man["content_sha256"] == content_hash(synthetic_prices)
    assert man["securities_missing"] == ["ZZZ"]
    assert store.latest("synthetic") == info.snapshot_id


def test_latest_matches_request(tmp_path, synthetic_prices):
    # Two configs share a provider; the newer snapshot belongs to the OTHER config.
    store = RawSnapshotStore(tmp_path)
    meta = {"provider": "synthetic", "price_basis": "raw"}
    req_a = {"tickers": ["AAA"], "start": "2020-01-01", "end": "2021-12-31"}
    req_b = {"tickers": ["BBB"], "start": "2020-01-01", "end": "2021-12-31"}
    a = store.save(synthetic_prices, meta, req_a, now=datetime(2026, 1, 1, tzinfo=UTC))
    b = store.save(synthetic_prices, meta, req_b, now=datetime(2026, 1, 2, tzinfo=UTC))
    assert store.latest("synthetic") == b.snapshot_id  # unfiltered: newest overall
    assert store.latest("synthetic", req_a) == a.snapshot_id
    with pytest.raises(FileNotFoundError, match="run ingest first"):
        store.latest("synthetic", {"tickers": ["CCC"]})


def test_snapshot_is_immutable(tmp_path, synthetic_prices):
    store = RawSnapshotStore(tmp_path)
    meta = {"provider": "synthetic", "price_basis": "raw"}
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store.save(synthetic_prices, meta, {}, now=now)
    with pytest.raises(FileExistsError):
        store.save(synthetic_prices, meta, {}, now=now)


def test_tampering_detected(tmp_path, synthetic_prices):
    store = RawSnapshotStore(tmp_path)
    info = store.save(synthetic_prices, {"provider": "synthetic", "price_basis": "raw"}, {})
    tampered = synthetic_prices.copy()
    tampered.loc[0, "close"] *= 1.01
    tampered.to_parquet(info.path / "prices.parquet", index=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        store.load("synthetic", info.snapshot_id)


# ---------- validation ----------
def test_clean_data_has_no_errors(synthetic_prices):
    rep = validate_prices(synthetic_prices, PriceBasis.RAW)
    assert rep.ok, rep.summary()


def test_validation_catches_problems(synthetic_prices):
    bad = synthetic_prices.copy()
    bad.loc[5, "close"] = -1.0
    bad.loc[6, "dividend"] = -0.1
    rep = validate_prices(bad, PriceBasis.RAW)
    checks = {i.check for i in rep.errors}
    assert {"nonpositive_or_missing_close", "negative_dividend"} <= checks


def test_unhandled_split_is_flagged(synthetic_prices):
    # Declaring raw prices as split-adjusted => a fake -50% day on the split date.
    rep = validate_prices(synthetic_prices, PriceBasis.SPLIT_ADJUSTED)
    assert any(i.check == "extreme_daily_return" for i in rep.warnings)


def test_gap_and_vendor_mismatch_flagged(synthetic_prices):
    df = synthetic_prices[~synthetic_prices["date"].between("2020-03-02", "2020-03-20")].copy()
    df["vendor_adj_close"] = df["close"]  # ignores the split => mismatch on split day
    rep = validate_prices(df, PriceBasis.RAW)
    checks = {i.check for i in rep.warnings}
    assert {"date_gap", "vendor_adj_mismatch"} <= checks
    assert np.isfinite(len(rep.to_frame()))
