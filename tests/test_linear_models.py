"""HAR-FHS and linear quantile regression, plus the feature-passing contract of the engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from alphalab.backtest import fz0_loss
from alphalab.features import build_features
from alphalab.risk.engine import EngineSettings, attach_outcomes, run_forecasts
from alphalab.risk.models import HARFHS, LinearQuantileRegression, RiskModel, build_model
from alphalab.risk.models.base import Forecast
from alphalab.risk.models.linear import _har_design, es_levels

ALPHAS = (0.01, 0.025, 0.05)


def test_es_levels():
    np.testing.assert_allclose(es_levels(0.025, 5), [0.0025, 0.0075, 0.0125, 0.0175, 0.0225])
    assert es_levels(0.01, 5).mean() == pytest.approx(0.005)  # midpoint rule on (0, α)


def test_har_design_by_hand():
    r = np.arange(1, 31) / 100
    d = _har_design(r)
    assert d[-1, 1] == pytest.approx(0.30**2)
    assert d[-1, 2] == pytest.approx(np.mean((np.arange(26, 31) / 100) ** 2))
    assert d[-1, 3] == pytest.approx(np.mean((np.arange(9, 31) / 100) ** 2))
    assert np.isnan(d[20, 3]) and np.isfinite(d[21, 3])


def test_registry():
    assert build_model({"type": "har"}, ALPHAS).name == "har_fhs"
    m = build_model({"type": "linqr", "features": ["rv_22", "vix"]}, ALPHAS)
    assert m.name == "linqr" and m.uses_features and list(m.features) == ["rv_22", "vix"]
    assert not HARFHS.uses_features


# ---------- the feature contract ----------
class _SpyFeatureModel(RiskModel):
    uses_features = True

    def __init__(self):
        super().__init__(alphas=(0.01,), name="spyf")
        self.fit_last_x, self.fit_last_r, self.pred_last_x = [], [], []

    def fit(self, train, X):
        assert len(train) == len(X)
        self.fit_last_x.append(X.index[-1])
        self.fit_last_r.append(train[-1])
        return self

    def predict(self, history, X):
        self.pred_last_x.append(X.index[-1])
        return Forecast({0.01: 0.01}, {0.01: 0.02})


def test_engine_passes_features_up_to_origin_only():
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = pd.Series(np.arange(300.0), index=idx)
    X = pd.DataFrame({"f": np.arange(300.0)}, index=idx)
    spy = _SpyFeatureModel()
    out = run_forecasts(r, lambda: spy, EngineSettings(min_train=100, refit_every=50), features=X)
    assert spy.pred_last_x == list(out.index)  # forecast at t sees x_t, never x_{t+1}
    assert spy.fit_last_x[0] == out.index[0] and spy.fit_last_r[0] == 99.0


def test_feature_model_without_features_raises():
    r = pd.Series(np.zeros(200), index=pd.bdate_range("2020-01-01", periods=200))
    with pytest.raises(ValueError, match="needs features"):
        run_forecasts(r, lambda: _SpyFeatureModel(), EngineSettings(min_train=100))


# ---------- linear quantile regression ----------
@pytest.fixture(scope="module")
def iid_normal():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2000-01-03", periods=3000)
    r = pd.Series(rng.normal(0, 0.01, 3000), index=idx)
    X = pd.DataFrame({"noise": rng.normal(size=3000), "noise2": rng.normal(size=3000)}, index=idx)
    return r, X


def test_linqr_recovers_normal_quantiles_and_no_crossing(iid_normal):
    r, X = iid_normal
    m = LinearQuantileRegression(alphas=ALPHAS).fit(r.to_numpy(), X)
    f = m.predict(r.to_numpy(), X)
    for a in ALPHAS:
        z = stats.norm.ppf(1 - a)
        assert f.var[a] == pytest.approx(0.01 * z, rel=0.12)
        assert f.es[a] == pytest.approx(0.01 * stats.norm.pdf(z) / a, rel=0.15)
        assert f.es[a] >= f.var[a]
    assert f.var[0.01] > f.var[0.025] > f.var[0.05]


def test_linqr_uses_train_moments_only(iid_normal):
    r, X = iid_normal
    m = LinearQuantileRegression(alphas=ALPHAS).fit(r.to_numpy()[:1000], X.iloc[:1000])
    train_x = X.iloc[:999]  # pairs use x_0..x_998; nothing after the training window
    lo, hi = train_x.quantile(0.005).to_numpy(), train_x.quantile(0.995).to_numpy()
    np.testing.assert_allclose(m._lo, lo)
    np.testing.assert_allclose(m._hi, hi)
    np.testing.assert_allclose(m._mu, train_x.clip(lo, hi, axis=1).mean().to_numpy())


def test_linqr_missing_feature_raises(iid_normal):
    r, X = iid_normal
    m = LinearQuantileRegression(alphas=ALPHAS).fit(r.to_numpy(), X)
    X2 = X.copy()
    X2.iloc[-1, 0] = np.nan
    with pytest.raises(ValueError, match="missing feature"):
        m.predict(r.to_numpy(), X2)


@pytest.mark.parametrize("kind", ["linqr", "har"])
def test_engine_point_in_time_with_features(sim6000, kind):
    r = sim6000["ret"].iloc[:1400]
    X = build_features(pd.DataFrame({"ret": r}))
    s = EngineSettings(min_train=1000, refit_every=21)
    make = lambda: build_model({"type": kind}, ALPHAS)  # noqa: E731
    full = run_forecasts(r, make, s, features=X)
    for d in full.index[[0, 123, -2]]:
        trunc = run_forecasts(r.loc[:d], make, s, features=X.loc[:d])
        pd.testing.assert_series_equal(trunc.loc[d], full.loc[d])


# ---------- behaviour on simulated GJR-GARCH-t data ----------
@pytest.fixture(scope="module")
def linear_evals(sim6000):
    r = sim6000["ret"]
    X = build_features(pd.DataFrame({"ret": r}))
    s = EngineSettings(min_train=1000, refit_every=63)  # tests only; real runs refit every 21
    out = {}
    for kind in ("har", "linqr"):
        fits: list[dict] = []
        f = run_forecasts(
            r, lambda kind=kind: build_model({"type": kind}, ALPHAS), s, fits=fits, features=X
        )
        out[kind] = (attach_outcomes(f, r), fits)
    return out


def test_har_tracks_true_volatility(sim6000, linear_evals):
    ev, fits = linear_evals["har"]
    true_next = sim6000["sigma"].shift(-1).reindex(ev.index)
    assert np.corrcoef(ev["sigma"], true_next)[0, 1] > 0.85
    assert (ev["sigma"] / true_next - 1).abs().median() < 0.15
    # The true persistence is 0.98, but OLS on daily squared returns gives a much smaller sum
    # (measured ≈ 0.71): the regressors are noisy proxies of σ², and errors-in-variables
    # shrink OLS coefficients toward zero (attenuation bias). Corsi's HAR uses intraday
    # realized variance, which is far less noisy. Only the qualitative facts are asserted.
    last = fits[-1]
    total = last["b_day"] + last["b_week"] + last["b_month"]
    assert 0.4 < total < 0.98 and last["b_month"] > 0


@pytest.mark.parametrize("kind", ["har", "linqr"])
@pytest.mark.parametrize("alpha", [0.01, 0.05])
def test_coverage_close_to_truth(sim6000, linear_evals, sim_evals, kind, alpha):
    ev = linear_evals[kind][0]
    truth = sim_evals["true"][0].reindex(ev.index)
    true_rate = (truth["loss_next"] > truth[f"var_{alpha:g}"]).mean()
    rate = (ev["loss_next"] > ev[f"var_{alpha:g}"]).mean()
    assert abs(rate - true_rate) < 0.2 * alpha


def test_linear_models_beat_hs_but_not_the_true_model(linear_evals, sim_evals):
    def fz(ev):
        return fz0_loss(-ev["loss_next"], ev["var_0.025"], ev["es_0.025"], 0.025).mean()

    common = linear_evals["linqr"][0].index
    hs = fz(sim_evals["hs"][0].reindex(common))
    true = fz(sim_evals["true"][0].reindex(common))
    for kind in ("har", "linqr"):
        m = fz(linear_evals[kind][0])
        assert true < m < hs


def test_pipeline_feeds_features_to_linqr(config_yaml):
    from alphalab.config import load_config
    from alphalab.data.schema import check_series_schema
    from alphalab.data.store import RawSnapshotStore
    from alphalab.pipeline import features_request, run_build, run_forecast, run_ingest

    config_yaml.write_text(
        config_yaml.read_text()
        + "portfolio: {name: p, weights: {AAA: 0.5, BBB: 0.5}}\n"
        + "features: {yahoo_levels: [], fred: [], range_asset: AAA}\n"
        + "risk: {min_train: 260,"
        + " models: [{type: linqr, min_obs: 150}, {type: har, min_obs: 150}]}\n"
    )
    cfg = load_config(config_yaml)
    snap = run_ingest(cfg)
    empty = pd.DataFrame(
        {
            "date": pd.to_datetime([]),
            "security_id": pd.Series([], dtype="string"),
            "value": pd.Series([], dtype="float64"),
        }
    )
    RawSnapshotStore(cfg.data_dir).save(
        empty, {"provider": "features"}, features_request(cfg), validate=check_series_schema
    )
    run_build(cfg, snap.snapshot_id)
    fc = pd.read_parquet(run_forecast(cfg) / "forecasts.parquet")
    assert set(fc["model"]) == {"linqr", "har_fhs"} and fc["var_0.01"].notna().all()


def test_linqr_clips_features_to_training_range(iid_normal):
    """A feature far outside the training range is clipped to the training quantile, so the
    forecast equals the forecast AT that quantile instead of a straight-line extrapolation."""
    r, X = iid_normal
    m = LinearQuantileRegression(alphas=ALPHAS).fit(r.to_numpy(), X)
    hi = X.iloc[:-1]["noise"].quantile(0.995)  # training pairs use rows 0..n-2
    x_far, x_edge = X.copy(), X.copy()
    x_far.iloc[-1, 0], x_edge.iloc[-1, 0] = 50.0, hi
    f_far, f_edge = m.predict(r.to_numpy(), x_far), m.predict(r.to_numpy(), x_edge)
    assert f_far.var[0.05] == pytest.approx(f_edge.var[0.05])
    m_off = LinearQuantileRegression(alphas=ALPHAS, clip_q=None).fit(r.to_numpy(), X)
    assert m_off.predict(r.to_numpy(), x_far).var[0.05] != pytest.approx(f_edge.var[0.05])


def test_invalid_es_is_never_scored():
    from alphalab.backtest.report import evaluate

    idx = pd.bdate_range("2020-01-01", periods=300)
    ev = pd.DataFrame(
        {"loss_next": 0.0, "var_0.05": 0.01, "es_0.05": 0.02, "sigma": np.nan, "target_date": idx},
        index=idx,
    )
    bad = ev.copy()
    bad.iloc[10, bad.columns.get_loc("es_0.05")] = -0.01
    with pytest.raises(ValueError, match="bad: 1 forecasts with ES <= 0"):
        evaluate({"good": ev, "bad": bad}, [0.05], mcs_reps=10)


# ---------- safety floor, per-model refits, L1 control ----------
def test_floor_fixes_invalid_and_leaves_valid_forecasts_alone():
    from alphalab.risk.engine import apply_floor

    bad = Forecast({0.05: -0.003, 0.01: 0.02}, {0.05: -0.001, 0.01: 0.015})
    row = apply_floor(bad, 1e-4)
    assert row["var_0.05"] == 1e-4 and row["es_0.05"] == 1e-4  # ES >= floored VaR
    assert row["es_0.01"] == 0.02  # ES below its own VaR is raised to VaR (coherence)
    assert row["floored"]
    good = Forecast({0.05: 0.012}, {0.05: 0.02}, 0.01)
    assert apply_floor(good, 1e-4) == {**good.as_row(), "floored": False}


def test_floor_changes_nothing_for_econometric_models(sim_evals):
    for name in ("hs", "ewma", "garch", "gjr", "fhs"):
        assert not sim_evals[name][0]["floored"].any()


def test_per_model_refit_override():
    idx = pd.bdate_range("2020-01-01", periods=400)
    r = pd.Series(np.zeros(400), index=idx)
    X = pd.DataFrame({"f": np.zeros(400)}, index=idx)
    spy = _SpyFeatureModel()
    spy.refit_every = 100
    run_forecasts(r, lambda: spy, EngineSettings(min_train=100, refit_every=21), features=X)
    # 301 origins (positions 99..399): refits at origins 0, 100, 200, 300 -> 4 (not 15).
    assert len(spy.fit_last_x) == 4


@pytest.fixture(scope="module")
def collinear():
    """Two nearly identical features (like Parkinson vs Garman–Klass) plus a volatility
    regime, so quantiles truly depend on the features."""
    rng = np.random.default_rng(5)
    n = 1500
    idx = pd.bdate_range("2000-01-03", periods=n)
    vol = np.exp(rng.normal(0, 0.5, n)) * 0.01
    a = vol + rng.normal(0, 0.0005, n)
    X = pd.DataFrame(
        {"a": a, "b": a + rng.normal(0, 1e-5, n), "noise": rng.normal(size=n)}, index=idx
    )
    r = pd.Series(np.r_[0.0, vol[:-1] * rng.standard_t(5, n - 1) * 0.77], index=idx)
    return r, X


def test_l1_shrinks_collinear_pair(collinear):
    r, X = collinear
    kw = dict(alphas=(0.05,), es_k=2)
    plain = LinearQuantileRegression(**kw).fit(r.to_numpy(), X)
    l1 = LinearQuantileRegression(**kw, l1_grid=[1e-3]).fit(r.to_numpy(), X)
    assert l1.name == "linqr_l1" and plain.name == "linqr"
    size = lambda m: np.abs(m._params[0.05][1:3]).sum()  # noqa: E731  (|β_a| + |β_b|)
    assert size(l1) < size(plain)


def test_l1_tuning_uses_only_the_earlier_part(collinear, monkeypatch):
    r, X = collinear
    m = LinearQuantileRegression(alphas=(0.05,), es_k=2, l1_grid=[0.0, 1e-3])
    seen = []
    orig = m._prepare
    monkeypatch.setattr(m, "_prepare", lambda Xs: (seen.append(Xs.index[-1]), orig(Xs)))
    m.fit(r.to_numpy(), X)
    n_pairs = len(X) - 1
    n_val = int(n_pairs * 0.2)
    assert seen[0] == X.index[n_pairs - n_val - 1 - 1]  # tuning stats end before embargo+val
    assert seen[-1] == X.index[n_pairs - 1]  # final model uses the whole training window
    assert m.lam in (0.0, 1e-3)


def test_l1_grid_validation():
    with pytest.raises(ValueError, match="non-negative"):
        LinearQuantileRegression(l1_grid=[-1.0])


def test_engine_point_in_time_l1(collinear):
    r, X = collinear
    s = EngineSettings(min_train=1000, refit_every=21)
    make = lambda: LinearQuantileRegression(  # noqa: E731
        alphas=(0.05,), es_k=2, l1_grid=[0.0, 1e-3], refit_every=200
    )
    full = run_forecasts(r, make, s, features=X)
    for d in full.index[[0, 250, -2]]:
        trunc = run_forecasts(r.loc[:d], make, s, features=X.loc[:d])
        pd.testing.assert_series_equal(trunc.loc[d], full.loc[d])
