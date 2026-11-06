"""Features: hand-checked formulas, publication lags, staleness, point-in-time."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphalab.core.calendar import trading_sessions
from alphalab.core.lookahead import assert_point_in_time, check_point_in_time
from alphalab.data.providers.fred import parse_fred_csv
from alphalab.features import align_asof, build_features, next_day_loss, series_name
from alphalab.returns.core import compute_daily_returns


@pytest.fixture
def inputs(synthetic_prices) -> pd.DataFrame:
    a = synthetic_prices[synthetic_prices["security_id"] == "AAA"].set_index("date")
    rets = compute_daily_returns(synthetic_prices, "raw")
    ret = rets[rets["security_id"] == "AAA"].set_index("date")["ret"]
    rng = np.random.default_rng(0)
    n = len(a)
    return pd.DataFrame(
        {
            "ret": ret,
            "open": a["open"],
            "high": a["high"],
            "low": a["low"],
            "close": a["close"],
            "vix": 20 * np.exp(np.cumsum(rng.normal(0, 0.05, n))),
            "baa10y": 2 + np.cumsum(rng.normal(0, 0.02, n)),
        },
        index=a.index,
    ).iloc[1:]  # drop the first row (no return yet)


def _build(x: pd.DataFrame) -> pd.DataFrame:
    return build_features(x, log_change_levels=["vix"], diff_levels=["baa10y"])


# ---------- names and parsing ----------
def test_series_name():
    assert series_name("^VIX") == "vix"
    assert series_name("BAA10Y") == "baa10y"
    assert series_name("XIU.TO") == "xiu_to"


def test_parse_fred_csv_drops_missing():
    text = "observation_date,BAA10Y\n2024-03-28,1.70\n2024-03-29,.\n2024-04-01,\n2024-04-02,1.68\n"
    df = parse_fred_csv(text, "BAA10Y")
    assert list(df["date"].dt.strftime("%m-%d")) == ["03-28", "04-02"]
    assert df["value"].tolist() == [1.70, 1.68]
    with pytest.raises(ValueError, match="no column"):
        parse_fred_csv(text, "T10Y3M")


# ---------- as-of alignment and publication lag ----------
def test_fred_lag_one_session():
    # Values observed Thu 2024-03-28 and Mon 04-01 (Fri 03-29 = Good Friday, NYSE closed).
    obs = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-03-28", "2024-04-01"]))
    sess = trading_sessions("2024-03-27", "2024-04-03")  # Wed Thu Mon Tue Wed
    lag0 = align_asof(obs, sess, lag_sessions=0)
    lag1 = align_asof(obs, sess, lag_sessions=1)
    assert lag0.tolist()[1:] == [1.0, 2.0, 2.0, 2.0] and np.isnan(lag0.iloc[0])  # same-day
    # With lag 1, Thursday's value is first usable on Monday (the next session), and
    # Monday's value on Tuesday: you can only use what was published by your close.
    assert np.isnan(lag1.iloc[0]) and np.isnan(lag1.iloc[1])
    assert lag1.tolist()[2:] == [1.0, 2.0, 2.0]


def test_stale_series_becomes_nan():
    obs = pd.Series([5.0], index=pd.to_datetime(["2024-01-02"]))  # series then stops
    sess = trading_sessions("2024-01-02", "2024-01-31")
    out = align_asof(obs, sess, max_stale_sessions=5)
    assert out.iloc[:6].notna().all() and out.iloc[6:].isna().all()


def test_align_asof_is_point_in_time():
    obs = pd.Series(np.arange(40.0), index=trading_sessions("2024-01-02", "2024-03-01")[::1][:40])
    sess = trading_sessions("2024-01-02", "2024-03-01")
    full = align_asof(obs, sess, lag_sessions=1)
    for d in sess[5::7]:
        trunc = align_asof(obs.loc[:d], sess[sess <= d], lag_sessions=1)
        assert trunc.loc[d] == full.loc[d] or (np.isnan(trunc.loc[d]) and np.isnan(full.loc[d]))


# ---------- feature formulas ----------
def test_realized_variance_and_drawdown_by_hand():
    idx = trading_sessions("2024-01-02", "2024-01-31")
    r = pd.Series(0.0, index=idx)
    r.iloc[:5] = [0.01, -0.02, 0.03, -0.01, 0.02]
    r.iloc[5] = -0.10  # a 10% fall from the running peak
    f = build_features(pd.DataFrame({"ret": r}))
    assert np.isnan(f["rv_5"].iloc[3])  # needs a full 5-day window
    assert f["rv_5"].iloc[4] == pytest.approx(np.mean(np.square([0.01, -0.02, 0.03, -0.01, 0.02])))
    peak = np.prod([1.01, 0.98, 1.03, 0.99, 1.02])
    assert f["dd_252"].iloc[5] == pytest.approx(0.9 * peak / peak - 1.0)
    assert "gk_1" not in f  # no OHLC supplied -> no range features


def test_garman_klass_by_hand():
    x = pd.DataFrame(
        {"ret": [0.0], "open": [100.0], "high": [103.0], "low": [99.0], "close": [102.0]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    f = build_features(x)
    hl, co = np.log(103 / 99), np.log(102 / 100)
    assert f["gk_1"].iloc[0] == pytest.approx(0.5 * hl**2 - (2 * np.log(2) - 1) * co**2)
    assert f["pk_1"].iloc[0] == pytest.approx(hl**2 / (4 * np.log(2)))


def test_level_changes(inputs):
    f = _build(inputs)
    i = 30
    assert f["vix_chg_5"].iloc[i] == pytest.approx(
        np.log(inputs["vix"].iloc[i] / inputs["vix"].iloc[i - 5])
    )
    assert f["baa10y_chg_22"].iloc[i] == pytest.approx(
        inputs["baa10y"].iloc[i] - inputs["baa10y"].iloc[i - 22]
    )


def test_missing_ret_raises():
    with pytest.raises(ValueError, match="'ret'"):
        build_features(pd.DataFrame({"x": [1.0]}))


# ---------- point-in-time (CLAUDE.md rule: every new feature gets this test) ----------
def test_all_features_are_point_in_time(inputs):
    f = _build(inputs)
    assert f.shape[1] >= 15  # guard: the test covers every family
    assert_point_in_time(_build, inputs, n_checks=30)


def test_label_is_flagged_as_lookahead(inputs):
    """The target must FAIL the truncation test: it is a label, never an input."""
    loss = lambda x: next_day_loss(x["ret"]).to_frame()  # noqa: E731
    assert check_point_in_time(loss, inputs)
    lab = next_day_loss(inputs["ret"])
    assert lab.iloc[0] == -inputs["ret"].iloc[1] and np.isnan(lab.iloc[-1])


# ---------- pipeline integration (offline: a hand-made feature snapshot) ----------
def _features_cfg(config_yaml):
    from alphalab.config import load_config

    config_yaml.write_text(
        config_yaml.read_text()
        + "portfolio: {name: p, weights: {AAA: 0.5, BBB: 0.5}}\n"
        + "features: {yahoo_levels: ['^VIX'], fred: [BAA10Y], range_asset: AAA}\n"
    )
    return load_config(config_yaml)


def test_build_writes_features_with_fred_lag(config_yaml):
    from alphalab.data.schema import check_series_schema
    from alphalab.data.store import RawSnapshotStore
    from alphalab.pipeline import features_request, run_build, run_ingest

    cfg = _features_cfg(config_yaml)
    snap = run_ingest(cfg)
    days = trading_sessions(cfg.start, cfg.end)
    series = pd.concat(
        [
            pd.DataFrame(
                {"date": days, "security_id": "^VIX", "value": 15.0 + np.arange(len(days)) % 7}
            ),
            pd.DataFrame(
                {"date": days, "security_id": "BAA10Y", "value": np.arange(len(days)) / 100}
            ),
        ]
    ).astype({"security_id": "string", "value": "float64"})
    meta = {"provider": "features", "price_basis": "n/a"}
    RawSnapshotStore(cfg.data_dir).save(
        series, meta, features_request(cfg), validate=check_series_schema
    )

    out = run_build(cfg, snap.snapshot_id)
    f = pd.read_parquet(out / "features_daily.parquet")
    assert {"rv_22", "gk_5", "dd_252", "vix", "vix_chg_5", "baa10y", "baa10y_chg_22"} <= set(
        f.columns
    )
    # BAA10Y value i/100 is dated session i; with a one-session lag, session i sees (i-1)/100.
    pos = days.get_indexer(f.index)
    np.testing.assert_allclose(f["baa10y"].to_numpy()[1:], (pos[1:] - 1) / 100)
    assert (f["vix"].to_numpy() == 15.0 + pos % 7).all()  # VIX: same day
    import json

    man = json.loads((out / "build_manifest.json").read_text())["features"]
    assert man["fred_lag_sessions"] == 1 and man["n_complete_rows"] > 0


def test_build_without_feature_snapshot_says_ingest(config_yaml):
    from alphalab.pipeline import run_build, run_ingest

    cfg = _features_cfg(config_yaml)
    with pytest.raises(FileNotFoundError, match="run ingest first"):
        run_build(cfg, run_ingest(cfg).snapshot_id)


def _cfg(**kw):
    from alphalab.config import Config, PortfolioConfig

    base = {"provider": "synthetic", "universe": ["A"], "start": "2020-01-01", "end": "2020-12-31"}
    return Config(**{**base, "portfolio": PortfolioConfig("p", {"A": 1.0}), **kw})


def test_features_need_portfolio():
    from alphalab.config import FeaturesConfig

    with pytest.raises(ValueError, match="need a portfolio"):
        _cfg(portfolio=None, features=FeaturesConfig())


def test_range_asset_must_be_in_universe():
    from alphalab.config import FeaturesConfig

    with pytest.raises(ValueError, match="not in universe"):
        _cfg(features=FeaturesConfig(range_asset="ZZZ"))


def test_fred_lag_must_be_positive():
    from alphalab.config import FeaturesConfig

    with pytest.raises(ValueError, match=">= 1"):
        FeaturesConfig(fred_lag_sessions=0)


def test_assemble_inputs_missing_series():
    from alphalab.features import assemble_inputs

    ret = pd.Series(0.0, index=trading_sessions("2024-01-02", "2024-01-10"))
    empty = pd.DataFrame({"date": [], "security_id": [], "value": []})
    with pytest.raises(ValueError, match="not in snapshot"):
        assemble_inputs(ret, empty, yahoo_levels=["^VIX"])


def test_series_schema_checks():
    from alphalab.data.schema import SchemaError, check_series_schema

    good = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-02"] * 2), "security_id": ["A", "A"], "value": [1.0, 2.0]}
    )
    with pytest.raises(SchemaError, match="duplicate"):
        check_series_schema(good)
    with pytest.raises(SchemaError, match="missing"):
        check_series_schema(good.drop(columns="value"))


@pytest.mark.network
def test_live_feature_ingest(tmp_path):
    """`pytest -m network`: VIX from Yahoo and BAA10Y from FRED for one month."""
    from alphalab.config import Config, FeaturesConfig, PortfolioConfig
    from alphalab.pipeline import run_ingest_features

    cfg = Config(
        provider="yahoo",
        universe=["SPY"],
        start="2020-03-01",
        end="2020-03-31",
        data_dir=tmp_path,
        portfolio=PortfolioConfig("p", {"SPY": 1.0}),
        features=FeaturesConfig(fred=["BAA10Y"]),
    )
    df = pd.read_parquet(run_ingest_features(cfg).path / "prices.parquet")
    vix = df[df["security_id"] == "^VIX"].set_index("date")["value"]
    assert vix.max() > 70  # VIX closed at 82.69 on 2020-03-16
    assert (df["security_id"] == "BAA10Y").sum() >= 20
