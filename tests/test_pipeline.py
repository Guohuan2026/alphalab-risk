"""End-to-end: config -> ingest -> build, plus config guards and an opt-in live test."""

import json
from pathlib import Path

import pandas as pd
import pytest

from alphalab.cli import main
from alphalab.config import Config, PortfolioConfig, load_config
from alphalab.pipeline import run_build, run_ingest


def test_end_to_end_synthetic(config_yaml):
    cfg = load_config(config_yaml)
    info = run_ingest(cfg)
    out = run_build(cfg, info.snapshot_id)
    monthly = pd.read_parquet(out / "returns_monthly.parquet")
    assert list(monthly.columns) == ["AAA", "BBB"]
    assert monthly.index.max() == pd.Timestamp("2021-06-30")
    man = json.loads((out / "build_manifest.json").read_text())
    assert man["source_snapshot"] == info.snapshot_id and man["validation_errors"] == 0


def test_end_to_end_portfolio(config_yaml):
    text = config_yaml.read_text()
    config_yaml.write_text(text + "portfolio: {name: p, weights: {AAA: 0.7, BBB: 0.3}}\n")
    cfg = load_config(config_yaml)
    out = run_build(cfg, run_ingest(cfg).snapshot_id)
    port = pd.read_parquet(out / "portfolio_daily.parquet")
    daily = pd.read_parquet(out / "returns_daily.parquet").loc[port.index]
    # First day of each month starts at the target weights (monthly is the default rule).
    firsts = port.groupby(port.index.to_period("M")).head(1)
    assert (firsts["w_AAA"] == 0.7).all()
    assert port["ret"].iloc[0] == pytest.approx(
        0.7 * daily["AAA"].iloc[0] + 0.3 * daily["BBB"].iloc[0]
    )
    man = json.loads((out / "build_manifest.json").read_text())
    assert man["portfolio"]["rebalance"] == "monthly" and man["portfolio"]["n_days"] == len(port)


SHIPPED_CONFIGS = sorted((Path(__file__).parents[1] / "configs").glob("*.yaml"))


def test_configs_were_found():
    assert len(SHIPPED_CONFIGS) >= 5  # guard: an empty glob would silently test nothing


@pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_load(path):
    cfg = load_config(path)
    assert cfg.universe


def test_cli_runs(config_yaml, capsys):
    assert main(["ingest", "--config", str(config_yaml)]) == 0
    assert main(["build", "--config", str(config_yaml)]) == 0
    assert main(["snapshots", "--config", str(config_yaml)]) == 0
    assert "processed:" in capsys.readouterr().out


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"end": "2999-01-01"}, "past"),
        ({"start": "2021-01-01", "end": "2020-01-01"}, "before"),
        ({"universe": ["A", "a"]}, "duplicate"),
        ({"calendar": "NOPE"}, "unknown exchange calendar"),
        ({"portfolio": PortfolioConfig("p", {"B": 1.0})}, "not in universe"),
    ],
)
def test_config_guards(kwargs, msg):
    base = {"provider": "synthetic", "universe": ["A"], "start": "2020-01-01", "end": "2020-12-31"}
    with pytest.raises(ValueError, match=msg):
        Config(**{**base, **kwargs})


@pytest.mark.network
def test_yahoo_live_aapl_split(tmp_path):
    """Run with `pytest -m network`. AAPL 4:1 split on 2020-08-31 must not look like -75%."""
    from alphalab.data.providers.yahoo import YahooProvider
    from alphalab.data.validation import validate_prices
    from alphalab.returns.core import compute_daily_returns

    df = YahooProvider().fetch(["AAPL"], "2020-08-20", "2020-09-10")
    assert not df.empty
    r = compute_daily_returns(df, "split_adjusted").set_index("date")["ret"]
    assert abs(r.loc["2020-08-31"]) < 0.15
    rep = validate_prices(df, "split_adjusted")
    assert rep.ok, rep.summary()
