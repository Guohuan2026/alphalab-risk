"""Risk models and the forecast engine: formulas, timing, PIT, and behaviour on simulated
data whose true VaR is known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from alphalab.backtest import coverage_tests
from alphalab.risk.engine import EngineSettings, attach_outcomes, forecast_origins, run_forecasts
from alphalab.risk.models import EWMANormal, HistoricalSimulation, RiskModel, build_model
from alphalab.risk.models.base import Forecast
from alphalab.risk.models.ewma import ewma_variance
from alphalab.risk.simulate import simulate_gjr_garch_t


# ---------- model formulas ----------
def test_hs_quantile_and_es():
    r = -np.arange(1, 101) / 1000  # losses 0.001 .. 0.100
    f = HistoricalSimulation(window=100, alphas=(0.05,)).predict(r)
    assert f.var[0.05] == pytest.approx(np.quantile(np.arange(1, 101) / 1000, 0.95))
    assert f.es[0.05] == pytest.approx(np.mean([x / 1000 for x in range(96, 101)]))
    assert np.isnan(f.sigma)


def test_hs_uses_only_last_window():
    hist = np.r_[np.full(500, -0.5), np.zeros(250)]  # huge old losses outside the window
    assert HistoricalSimulation(window=250).predict(hist).var[0.01] == 0.0


def test_ewma_closed_form_equals_recursion():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 400)
    s2 = np.mean(r[:30] ** 2)
    for x in r:
        s2 = 0.94 * s2 + 0.06 * x * x
    assert ewma_variance(r, 0.94, 30) == pytest.approx(s2, rel=1e-12)


def test_ewma_normal_var_es():
    r = np.random.default_rng(1).normal(0, 0.01, 300)
    f = EWMANormal(alphas=(0.01,)).predict(r)
    z = stats.norm.ppf(0.99)
    assert f.var[0.01] == pytest.approx(f.sigma * z)
    assert f.es[0.01] / f.var[0.01] == pytest.approx(stats.norm.pdf(z) / 0.01 / z)  # ≈ 1.146


def test_registry_and_validation():
    m = build_model({"type": "hs", "window": 500}, alphas=[0.01])
    assert isinstance(m, HistoricalSimulation) and m.name == "hs500" and m.alphas == (0.01,)
    assert build_model({"type": "ewma"}, alphas=[0.05]).name == "ewma"
    with pytest.raises(ValueError, match="unknown model type"):
        build_model({"type": "lstm"}, alphas=[0.01])
    with pytest.raises(ValueError, match="tail probabilities"):
        EWMANormal(alphas=(0.99,))  # 0.99 is a confidence level, not a tail probability


# ---------- engine ----------
@pytest.fixture(scope="module")
def sim():
    return simulate_gjr_garch_t(8000, seed=1, start="1995-01-03")


def test_origins_respect_min_train_oos_start_and_lockbox(sim):
    idx = sim.index
    s = EngineSettings(min_train=1000, oos_start="2000-01-03", lockbox_start="2020-01-02")
    pos = forecast_origins(idx, s)
    assert pos[0] >= 999 and idx[pos[0]] >= pd.Timestamp("2000-01-03")
    last_target = idx[pos[-1] + 1]
    assert last_target < pd.Timestamp("2020-01-02")  # no target inside the lockbox
    assert idx[pos[-1] + 2] >= pd.Timestamp("2020-01-02")  # ...and none wasted before it
    full = forecast_origins(idx, s, include_lockbox=True)
    assert full[-1] == len(idx) - 1 and len(full) > len(pos)


class _SpyModel(RiskModel):
    """Records what the engine passes in."""

    def __init__(self):
        super().__init__(alphas=(0.01,), name="spy")
        self.fit_lengths, self.fit_last = [], []

    def fit(self, train):
        self.fit_lengths.append(len(train))
        self.fit_last.append(train[-1])
        return self

    def predict(self, history):
        return Forecast({0.01: float(history[-1])}, {0.01: 0.0})


def test_refit_schedule_and_rolling_window():
    r = pd.Series(np.arange(300.0), index=pd.RangeIndex(300))
    spy = _SpyModel()
    s = EngineSettings(min_train=100, refit_every=21, window="rolling", window_size=50)
    out = run_forecasts(r, lambda: spy, s)
    assert len(out) == 201  # origins 99..299
    assert spy.fit_lengths == [50] * 10  # ceil(201 / 21) = 10 refits, each 50 long
    assert spy.fit_last == [99.0 + 21 * k for k in range(10)]  # train ends AT the origin
    assert (out["var_0.01"].to_numpy() == np.arange(99.0, 300.0)).all()  # history ends at t


@pytest.mark.parametrize(
    "make", [lambda: HistoricalSimulation(window=250), lambda: EWMANormal()], ids=["hs", "ewma"]
)
def test_engine_is_point_in_time(sim, make):
    """Engine-level truncation test: forecast at origin d from returns[:d] == full run."""
    r = sim["ret"].iloc[:2500]
    s = EngineSettings(min_train=1000, refit_every=21)
    full = run_forecasts(r, make, s)
    for d in full.index[[0, 7, 300, 777, -2]]:
        trunc = run_forecasts(r.loc[:d], make, s)
        pd.testing.assert_series_equal(trunc.loc[d], full.loc[d])


def test_attach_outcomes_alignment(sim):
    r = sim["ret"].iloc[:1300]
    f = attach_outcomes(run_forecasts(r, lambda: EWMANormal(), EngineSettings(min_train=1000)), r)
    t = f.index[5]
    nxt = r.index[r.index.get_loc(t) + 1]
    assert f.loc[t, "loss_next"] == -r.loc[nxt] and f.loc[t, "target_date"] == nxt
    assert f.index[-1] == r.index[-2]  # last origin dropped: its outcome is unknown


def test_nan_returns_rejected():
    r = pd.Series([0.01, np.nan, 0.0] * 20)
    with pytest.raises(ValueError, match="NaN"):
        run_forecasts(r, lambda: EWMANormal(), EngineSettings(min_train=10))


# ---------- behaviour on simulated GJR-GARCH-t data (true VaR known) ----------
def _cov(f: pd.DataFrame, a: float):
    return coverage_tests((f["loss_next"] > f[f"var_{a:g}"]).to_numpy(), a)


@pytest.fixture(scope="module")
def sim_forecasts(sim):
    r = sim["ret"]
    s = EngineSettings(min_train=1000)
    truth = sim.filter(like="true_var").shift(-1)  # truth for t+1, stamped at origin t
    truth.columns = [c.replace("true_", "") for c in truth.columns]
    return {
        "true": attach_outcomes(truth.iloc[999:], r),
        "hs250": attach_outcomes(run_forecasts(r, lambda: HistoricalSimulation(window=250), s), r),
        "ewma": attach_outcomes(run_forecasts(r, lambda: EWMANormal(), s), r),
    }


def test_true_var_passes(sim_forecasts):
    for a in (0.01, 0.05):
        assert _cov(sim_forecasts["true"], a).p_cc > 0.05


def test_ewma_normal_undercovers_deep_tail(sim_forecasts):
    """Fat tails: the normal quantile is too small at 1% (rejected), adequate at 5%."""
    c1, c5 = _cov(sim_forecasts["ewma"], 0.01), _cov(sim_forecasts["ewma"], 0.05)
    assert c1.hit_rate > 0.013 and c1.p_pof < 0.001
    assert c5.p_pof > 0.05


def test_hs_hits_cluster(sim_forecasts):
    """HS reacts slowly to volatility bursts, so its exceedances cluster."""
    c5 = _cov(sim_forecasts["hs250"], 0.05)
    assert c5.p_ind < 0.001
    assert _cov(sim_forecasts["hs250"], 0.01).p_pof < 0.001


# ---------- config + pipeline (offline, synthetic) ----------
def _risk_cfg(config_yaml, extra=""):
    from alphalab.config import load_config

    config_yaml.write_text(
        config_yaml.read_text()
        + "portfolio: {name: p, weights: {AAA: 0.5, BBB: 0.5}}\n"
        + "risk: {min_train: 260, lockbox_start: '2021-04-01',"
        + " models: [{type: hs, window: 250}, {type: ewma}]"
        + extra
        + "}\n"
    )
    return load_config(config_yaml)


def test_forecast_and_backtest_respect_lockbox(config_yaml):
    from alphalab.pipeline import run_backtest, run_build, run_forecast, run_ingest

    cfg = _risk_cfg(config_yaml)
    run_build(cfg, run_ingest(cfg).snapshot_id)
    out = run_forecast(cfg)
    fc = pd.read_parquet(out / "forecasts.parquet")
    assert set(fc["model"]) == {"hs250", "ewma"}
    tables = run_backtest(cfg)
    table = tables["coverage"]
    assert len(table) == 2 * 3 and (table["last_target"] < "2021-04-01").all()
    assert "tl_green" in table.columns
    assert set(tables) == {"coverage", "es", "losses", "dm", "dq"}
    assert len(tables["dm"]) == 3  # one model pair x 3 alphas
    assert (out / "backtest_losses.csv").exists()

    run_forecast(cfg, include_lockbox=True)
    lock = run_backtest(cfg, include_lockbox=True)["coverage"]
    assert (lock["first_target"] >= "2021-04-01").all()
    assert (out / "forecast_manifest_lockbox.json").exists()


def test_backtest_before_forecast_says_so(config_yaml):
    from alphalab.pipeline import run_backtest, run_build, run_ingest

    cfg = _risk_cfg(config_yaml)
    run_build(cfg, run_ingest(cfg).snapshot_id)
    with pytest.raises(FileNotFoundError, match="run forecast first"):
        run_backtest(cfg)


def test_risk_config_guards():
    from alphalab.config import RiskConfig

    with pytest.raises(ValueError, match="needs"):
        RiskConfig(min_train=100, models=[{"type": "hs", "window": 250}])
    with pytest.raises(ValueError, match="duplicate"):
        RiskConfig(models=[{"type": "ewma"}, {"type": "ewma", "lam": 0.97}])
    with pytest.raises(ValueError, match="window_size"):
        RiskConfig(window="rolling")


def test_cli_forecast_backtest(config_yaml, capsys):
    from alphalab.cli import main

    _risk_cfg(config_yaml)
    for cmd in ("ingest", "build", "forecast", "backtest"):
        assert main([cmd, "--config", str(config_yaml)]) == 0
    assert "p_cc" in capsys.readouterr().out
