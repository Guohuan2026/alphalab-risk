"""GARCH family: parameter recovery, volatility recovery, coverage on simulated truth,
incremental-filter correctness, point-in-time, and failed-fit handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import alphalab.risk.models.garch as garch_mod
from alphalab.backtest import coverage_tests
from alphalab.risk.engine import EngineSettings, attach_outcomes, run_forecasts
from alphalab.risk.models import GarchT, build_model
from alphalab.risk.simulate import GJRParams, gjr_variance_filter, simulate_gjr_garch_t

ALPHAS = (0.01, 0.025, 0.05)


@pytest.fixture(scope="module")
def sim():
    return simulate_gjr_garch_t(6000, seed=1, start="1995-01-03")


def test_registry_names():
    assert build_model({"type": "garch"}, ALPHAS).name == "garch_t"
    assert build_model({"type": "gjr"}, ALPHAS).name == "gjr_t"
    assert build_model({"type": "fhs"}, ALPHAS).name == "fhs_gjr"
    with pytest.raises(ValueError, match="tails"):
        GarchT(tails="normal")


def test_fit_recovers_parameters(sim):
    est, ok = garch_mod.fit_gjr_t(sim["ret"].to_numpy(), asymmetric=True)
    true = GJRParams()
    assert ok
    assert est.gamma == pytest.approx(true.gamma, abs=0.04)
    assert est.beta == pytest.approx(true.beta, abs=0.03)
    assert est.persistence == pytest.approx(true.persistence, abs=0.015)
    assert est.nu == pytest.approx(true.nu, abs=1.5)


def test_incremental_filter_equals_full_recompute(sim):
    r = sim["ret"].to_numpy()[:1500]
    m = GarchT(alphas=ALPHAS).fit(r[:1000])
    inc = [m.next_variance(r[: t + 1]) for t in range(999, 1500)]  # one step per day
    p = m.params
    full = []
    for t in range(999, 1500):
        path = gjr_variance_filter(r[: t + 1], p, m._init_s2(r))
        e = r[t] - p.mu
        full.append(p.omega + (p.alpha + p.gamma * (e < 0)) * e * e + p.beta * path[-1])
    np.testing.assert_allclose(inc, full, rtol=1e-10)


@pytest.fixture(scope="module")
def sim_runs(sim):
    r = sim["ret"]
    s = EngineSettings(min_train=1000, refit_every=21)
    out = {}
    for kind in ("garch", "gjr", "fhs"):
        fits: list[dict] = []
        f = run_forecasts(r, lambda kind=kind: build_model({"type": kind}, ALPHAS), s, fits=fits)
        out[kind] = (attach_outcomes(f, r), fits)
    return out


def _sigma_err(sim, f):
    true_next = sim["sigma"].shift(-1).reindex(f.index)
    return (f["sigma"] / true_next - 1).abs().median()


def test_gjr_recovers_true_volatility(sim, sim_runs):
    """GJR is the true model here: its σ forecast should be within a few % of the truth;
    plain GARCH (no leverage term) is misspecified and visibly worse."""
    err_gjr = _sigma_err(sim, sim_runs["gjr"][0])
    err_garch = _sigma_err(sim, sim_runs["garch"][0])
    assert err_gjr < 0.03
    assert err_garch > 2 * err_gjr


@pytest.mark.parametrize("kind", ["gjr", "fhs"])
@pytest.mark.parametrize("alpha", [0.01, 0.05])
def test_coverage_matches_true_var_on_same_path(sim, sim_runs, kind, alpha):
    """Compare with the TRUE VaR on the same simulated days, not with a fixed p-value.

    On this path the true 5% VaR itself fails conditional coverage (p ≈ 0.03: its hits
    happen to cluster), a Type I error. A correct model cannot be asked to beat the
    truth, so the test asks: are the model's hits as close to α as the truth's, and is
    its frequency not rejected?"""
    ev = sim_runs[kind][0]
    truth = sim[f"true_var_{alpha:g}"].shift(-1).reindex(ev.index)
    true_rate = (ev["loss_next"] > truth).mean()
    res = coverage_tests((ev["loss_next"] > ev[f"var_{alpha:g}"]).to_numpy(), alpha)
    assert abs(res.hit_rate - true_rate) < 0.15 * alpha  # within 15% of α of the truth
    assert res.p_pof > 0.01


def test_fit_info_recorded_every_refit(sim_runs):
    fits = sim_runs["gjr"][1]
    assert len(fits) == fits[-1]["n_fits"] and fits[-1]["n_failed"] == 0
    assert {"origin", "gamma", "persistence", "nu"} <= set(fits[0])
    assert all(0.9 < row["persistence"] < 1 for row in fits)


def test_fhs_tails_close_to_t_and_es_above_var(sim_runs):
    ev = sim_runs["fhs"][0]
    assert (ev["es_0.01"] > ev["var_0.01"]).all()
    gjr = sim_runs["gjr"][0]
    ratio = (ev["var_0.01"] / gjr["var_0.01"]).median()
    assert ratio == pytest.approx(1.0, abs=0.08)  # residuals really are t(6) here


def test_engine_point_in_time(sim):
    r = sim["ret"].iloc[:1300]
    s = EngineSettings(min_train=1000, refit_every=21)
    make = lambda: build_model({"type": "fhs"}, ALPHAS)  # noqa: E731
    full = run_forecasts(r, make, s)
    for d in full.index[[0, 150, -2]]:
        pd.testing.assert_series_equal(run_forecasts(r.loc[:d], make, s).loc[d], full.loc[d])


def test_failed_refit_keeps_previous_params(sim, monkeypatch):
    r = sim["ret"].to_numpy()
    m = GarchT(alphas=ALPHAS).fit(r[:1000])
    good = m.params
    monkeypatch.setattr(garch_mod, "fit_gjr_t", lambda *a, **k: (GJRParams(beta=0.5), False))
    m.fit(r[:1100])
    assert m.params == good and m.n_failed == 1 and m.n_fits == 2
    m.predict(r[:1100])  # still forecasts


def test_first_fit_failure_raises(sim, monkeypatch):
    def boom(*a, **k):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(garch_mod, "fit_gjr_t", boom)
    with pytest.raises(RuntimeError, match="first fit failed"):
        GarchT(alphas=ALPHAS).fit(sim["ret"].to_numpy()[:1000])


def test_pipeline_saves_parameter_paths(config_yaml):
    from alphalab.config import load_config
    from alphalab.pipeline import run_build, run_forecast, run_ingest

    config_yaml.write_text(
        config_yaml.read_text()
        + "portfolio: {name: p, weights: {AAA: 0.5, BBB: 0.5}}\n"
        + "risk: {min_train: 260, models: [{type: gjr, min_obs: 250}]}\n"
    )
    cfg = load_config(config_yaml)
    run_build(cfg, run_ingest(cfg).snapshot_id)
    params = pd.read_parquet(run_forecast(cfg) / "fit_params.parquet")
    assert set(params["model"]) == {"gjr_t"} and params["n_fits"].iloc[-1] == len(params)
