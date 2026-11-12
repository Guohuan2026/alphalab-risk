"""Explicit data corrections and separate robustness result folders."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from alphalab.data.corrections import apply_corrections


def test_non_cash_dividend_removed_and_logged(split_and_dividend_prices):
    fixed, log = apply_corrections(
        split_and_dividend_prices, {"non_cash_dividends": {"Y": ["2024-01-03"]}}
    )
    y = fixed[(fixed.security_id == "Y") & (fixed.date == "2024-01-03")]
    assert y["dividend"].iloc[0] == 0.0
    assert log == [
        {
            "type": "non_cash_dividend",
            "security_id": "Y",
            "date": "2024-01-03",
            "dividend_removed": 1.5,
        }
    ]
    orig = split_and_dividend_prices
    assert (
        orig.loc[(orig.security_id == "Y") & (orig.date == "2024-01-03"), "dividend"].iloc[0] == 1.5
    )


@pytest.mark.parametrize(
    "corr, msg",
    [
        ({"non_cash_dividends": {"Y": ["2024-01-04"]}}, "no dividend row"),  # date without dividend
        ({"non_cash_dividends": {"ZZZ": ["2024-01-03"]}}, "no dividend row"),  # unknown security
        ({"split_fix": {}}, "unknown correction"),
    ],
)
def test_bad_corrections_stop_the_build(split_and_dividend_prices, corr, msg):
    with pytest.raises(ValueError, match=msg):
        apply_corrections(split_and_dividend_prices, corr)


def test_no_corrections_is_a_no_op(split_and_dividend_prices):
    out, log = apply_corrections(split_and_dividend_prices, {})
    assert out is split_and_dividend_prices and log == []


def test_build_records_corrections(config_yaml):
    from alphalab.config import load_config
    from alphalab.data.store import RawSnapshotStore
    from alphalab.pipeline import run_build, run_ingest

    cfg = load_config(config_yaml)
    snap = run_ingest(cfg)
    raw, _ = RawSnapshotStore(cfg.data_dir).load(cfg.provider, snap.snapshot_id)
    d = raw[(raw.security_id == "AAA") & (raw.dividend > 0)]["date"].iloc[0]
    cfg.corrections = {"non_cash_dividends": {"AAA": [str(d.date())]}}
    out = run_build(cfg, snap.snapshot_id)
    man = json.loads((out / "build_manifest.json").read_text())
    assert man["corrections_applied"][0]["date"] == str(d.date())
    raw_again, _ = RawSnapshotStore(cfg.data_dir).load(cfg.provider, snap.snapshot_id)
    pd.testing.assert_frame_equal(raw, raw_again)  # data/raw untouched (and hash-verified)


def test_run_tag_separates_results(config_yaml):
    from alphalab.config import load_config
    from alphalab.pipeline import run_build, run_forecast, run_ingest

    config_yaml.write_text(
        config_yaml.read_text()
        + "portfolio: {name: p, weights: {AAA: 1.0}}\n"
        + "risk: {min_train: 260, models: [{type: ewma}]}\n"
    )
    cfg = load_config(config_yaml)
    run_build(cfg, run_ingest(cfg).snapshot_id)
    main = run_forecast(cfg)
    cfg.risk.run_tag = "rolling"
    tagged = run_forecast(cfg)
    assert tagged == main / "rolling" and (main / "forecasts.parquet").exists()
