"""ML models: registry and Deviation-3 defaults, shared tuning, block = day-by-day prediction,
point-in-time, out-of-bag residuals, behaviour on simulated truth. Small settings keep the
tests fast; real runs use the frozen grids."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from alphalab.backtest import fz0_loss
from alphalab.features import build_features
from alphalab.risk.engine import EngineSettings, attach_outcomes, run_forecasts
from alphalab.risk.models import (
    MLVolFHS,
    QuantileRandomForest,
    build_model,
)

ALPHAS = (0.01, 0.025, 0.05)
FAST = {
    "qrf": {"type": "qrf", "n_estimators": 60, "grid": [{"min_samples_leaf": 50}]},
    "gbq": {
        "type": "gbq",
        "max_iter": 40,
        "grid": [{"max_leaf_nodes": 7, "min_samples_leaf": 100}],
    },
    "mlvol": {"type": "mlvol", "n_estimators": 60, "grid": [{"min_samples_leaf": 100}]},
}


def test_registry_and_deviation3_defaults():
    q = build_model({"type": "qrf"}, ALPHAS)
    g = build_model({"type": "gbq"}, ALPHAS)
    v = build_model({"type": "mlvol"}, ALPHAS)
    assert (q.name, g.name, v.name) == ("qrf", "gbq", "mlvol_fhs")
    assert q.refit_every == g.refit_every == v.refit_every == 252  # yearly unless overridden
    assert [c["min_samples_leaf"] for c in q.grid] == [10, 50, 200]
    assert len(g.grid) == 4 and [c["min_samples_leaf"] for c in v.grid] == [50, 200, 500]
    assert build_model({"type": "qrf", "refit_every": 21}, ALPHAS).refit_every == 21
    assert q.uses_features and q.stateless


@pytest.fixture(scope="module")
def hetero():
    """Volatility driven by an observed feature: quantiles must depend on it."""
    rng = np.random.default_rng(1)
    n = 2500
    idx = pd.bdate_range("2000-01-03", periods=n)
    vol = np.exp(rng.normal(0, 0.6, n)) * 0.01
    r = np.r_[0.0, vol[:-1] * rng.standard_normal(n - 1)]  # r_{s+1} ~ N(0, vol_s)
    X = pd.DataFrame({"vol": vol, "noise": rng.normal(size=n)}, index=idx)
    return pd.Series(r, index=idx), X


def test_tuning_records_every_config_and_picks_the_best(hetero):
    r, X = hetero
    grid = [{"min_samples_leaf": 400}, {"min_samples_leaf": 30}]
    m = QuantileRandomForest(alphas=(0.05,), es_k=2, n_estimators=60, grid=grid)
    m.fit(r.to_numpy(), X)
    assert len(m.val_losses) == 2
    assert m.best == grid[int(np.argmin(m.val_losses))]
    assert m.fit_info()["val_loss"] == min(m.val_losses)


def test_ties_go_to_the_first_config(hetero):
    r, X = hetero
    grid = [{"min_samples_leaf": 50}, {"min_samples_leaf": 50}]  # identical -> identical loss
    m = MLVolFHS(alphas=(0.05,), n_estimators=40, grid=grid).fit(r.to_numpy(), X)
    assert m.val_losses[0] == m.val_losses[1] and m.best is m.grid[0]


@pytest.mark.parametrize("kind", ["qrf", "gbq", "mlvol"])
def test_var_rises_with_volatility_feature(hetero, kind):
    r, X = hetero
    m = build_model(FAST[kind], ALPHAS).fit(r.to_numpy(), X)
    lo = X.iloc[[-1]].assign(vol=X["vol"].quantile(0.1))
    hi = X.iloc[[-1]].assign(vol=X["vol"].quantile(0.9))
    assert m.predict_block(hi)[0].var[0.01] > 1.5 * m.predict_block(lo)[0].var[0.01]


def test_qrf_recovers_normal_quantiles():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2000-01-03", periods=3000)
    r = pd.Series(rng.normal(0, 0.01, 3000), index=idx)
    X = pd.DataFrame({"noise": rng.normal(size=3000)}, index=idx)
    m = QuantileRandomForest(alphas=ALPHAS, n_estimators=100, grid=[{"min_samples_leaf": 200}])
    f = m.fit(r.to_numpy(), X).predict(r.to_numpy(), X)
    assert f.var[0.05] == pytest.approx(0.01 * stats.norm.ppf(0.95), rel=0.15)
    assert f.es[0.01] >= f.var[0.01] >= f.var[0.05]


def test_mlvol_uses_out_of_bag_residuals(hetero):
    """In-sample forest fits understate |r| errors, so in-sample residual tails would be
    too thin; the model must use the larger out-of-bag tails."""
    r, X = hetero
    m = MLVolFHS(alphas=(0.01,), n_estimators=100, grid=[{"min_samples_leaf": 5}])
    m.fit(r.to_numpy(), X)
    Xs, y = m._pairs(r.to_numpy(), X)
    in_sample = -(y / np.maximum(m._obj["rf"].predict(Xs), 1e-6))
    q_in = np.quantile(in_sample, 0.99)
    assert m._obj["tail"][0.01][0] > q_in


def test_block_prediction_equals_day_by_day(hetero, monkeypatch):
    r, X = hetero
    s = EngineSettings(min_train=2200, refit_every=21)  # 300 origins, refits at 0/100/200
    make = lambda: build_model({**FAST["qrf"], "refit_every": 100}, ALPHAS)  # noqa: E731
    block = run_forecasts(r, make, s, features=X)
    monkeypatch.setattr(QuantileRandomForest, "stateless", False)  # force the daily path
    daily = run_forecasts(r, make, s, features=X)
    pd.testing.assert_frame_equal(block, daily)


@pytest.mark.parametrize("kind", ["qrf", "mlvol"])
def test_engine_point_in_time(sim6000, kind):
    r = sim6000["ret"].iloc[:1500]
    X = build_features(pd.DataFrame({"ret": r}))
    s = EngineSettings(min_train=1000, refit_every=21)
    make = lambda: build_model({**FAST[kind], "refit_every": 150}, ALPHAS)  # noqa: E731
    full = run_forecasts(r, make, s, features=X)
    for d in full.index[[0, 170, -2]]:
        trunc = run_forecasts(r.loc[:d], make, s, features=X.loc[:d])
        pd.testing.assert_series_equal(trunc.loc[d], full.loc[d])


def test_missing_feature_at_forecast_raises(hetero):
    r, X = hetero
    m = build_model(FAST["mlvol"], ALPHAS).fit(r.to_numpy(), X)
    X2 = X.copy()
    X2.iloc[-1, 0] = np.nan
    with pytest.raises(ValueError, match="missing feature"):
        m.predict(r.to_numpy(), X2)


# ---------- simulated GJR-GARCH-t truth: ML should sit between the truth and HS ----------
@pytest.fixture(scope="module")
def ml_evals(sim6000):
    r = sim6000["ret"]
    X = build_features(pd.DataFrame({"ret": r}))
    s = EngineSettings(min_train=1000)  # tests: refit every 1000 (real runs: 252)
    return {
        k: attach_outcomes(
            run_forecasts(
                r, lambda k=k: build_model({**FAST[k], "refit_every": 1000}, ALPHAS), s, features=X
            ),
            r,
        )
        for k in FAST
    }


@pytest.mark.parametrize("kind", ["qrf", "gbq", "mlvol"])
def test_ml_between_truth_and_hs(ml_evals, sim_evals, kind):
    ev = ml_evals[kind]

    def fz(e):
        e = e.reindex(ev.index)
        return fz0_loss(-e["loss_next"], e["var_0.025"], e["es_0.025"], 0.025).mean()

    assert fz(sim_evals["true"][0]) < fz(ev) < fz(sim_evals["hs"][0])


def test_mlvol_tracks_true_volatility(sim6000, ml_evals):
    ev = ml_evals["mlvol"]
    true_next = sim6000["sigma"].shift(-1).reindex(ev.index)
    assert np.corrcoef(ev["sigma"], true_next)[0, 1] > 0.8


@pytest.mark.parametrize("kind", ["qrf", "gbq", "mlvol"])
def test_refits_are_bit_identical(hetero, kind):
    """Same data + same seed -> identical forecasts, run after run (reproducibility)."""
    r, X = hetero
    a = build_model(FAST[kind], ALPHAS).fit(r.to_numpy(), X).predict_block(X.iloc[-50:])
    b = build_model(FAST[kind], ALPHAS).fit(r.to_numpy(), X).predict_block(X.iloc[-50:])
    assert [f.as_row() for f in a] == [f.as_row() for f in b]
