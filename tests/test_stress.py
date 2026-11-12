"""Stress analysis (H3): rules match the committed Clarification 4, HAC regression matches
statsmodels, and the tools detect a known stress failure on simulated data."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphalab.backtest.compare import hac_regression
from alphalab.stress import EPISODES, episode_labels, event_study, stress_tables

DESIGN = Path(__file__).parents[1] / "docs" / "RESEARCH_DESIGN.md"


def test_episodes_match_committed_clarification_4():
    text = DESIGN.read_text()
    clar = text[text.index("Clarification (not a change): exact stress-analysis rules") :]
    for name, (a, b) in EPISODES.items():
        assert re.search(rf"{a}\s*→\s*{b}", clar.replace("\n", " ")), f"{name} dates not in design"


def test_episodes_ordered_disjoint_and_before_lockbox():
    spans = sorted((pd.Timestamp(a), pd.Timestamp(b)) for a, b in EPISODES.values())
    assert all(a <= b for a, b in spans)
    assert all(spans[i][1] < spans[i + 1][0] for i in range(len(spans) - 1))
    assert spans[-1][1] < pd.Timestamp("2023-01-01")  # nothing inside the lockbox


def test_episode_boundaries_inclusive():
    lab = episode_labels(pd.to_datetime(["2020-02-19", "2020-02-20", "2020-04-30", "2020-05-01"]))
    assert lab.tolist() == ["", "covid_2020", "covid_2020", ""]


# ---------- HAC regression ----------
@pytest.mark.parametrize("lag", [0, 3, 9])
def test_hac_regression_matches_statsmodels(lag):
    import statsmodels.api as sm

    rng = np.random.default_rng(0)
    n = 1500
    D = (rng.random(n) < 0.2).astype(float)
    e = np.empty(n)
    e[0] = rng.normal()
    for t in range(1, n):
        e[t] = 0.4 * e[t - 1] + rng.normal()
    y = 0.1 + 0.5 * D + e
    X = np.column_stack([np.ones(n), D])
    ours = hac_regression(y, X, lag=lag)
    ref = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False})
    np.testing.assert_allclose(ours.coef, ref.params, rtol=1e-10)
    np.testing.assert_allclose(ours.se, ref.bse, rtol=1e-8)


# ---------- the tools on simulated truth ----------
@pytest.fixture(scope="module")
def sim_regime(sim6000):
    """A 'VIX' known at the origin: the TRUE next-day sigma, scaled so ~10% of days > 30."""
    s = sim6000["sigma"].shift(-1)
    return s / s.quantile(0.9) * 30


@pytest.fixture(scope="module")
def tables(sim_evals, sim_regime):
    evs = {k: sim_evals[k][0] for k in ("true", "hs", "ewma", "gjr")}
    return stress_tables(evs, [0.01, 0.05], vix=sim_regime, benchmark="true", mcs_reps=300)


def test_true_model_stays_calibrated_in_high_vol_regime(tables):
    t = tables["regimes"].set_index(["regime", "model", "alpha"])
    true_hi = t.loc[("vix_gt30", "true", 0.05)]
    assert true_hi["p_pof"] > 0.05  # conditional calibration: correct even when volatile
    hs_hi = t.loc[("vix_gt30", "hs", 0.05)]
    assert hs_hi["hit_rate"] > 2 * 0.05 and hs_hi["p_pof"] < 1e-6  # HS breaks down


def test_regression_direction_and_its_low_power(tables):
    """HS clearly breaks down in the high-vol regime (hit rate > 2x), and the FZ0 loss
    differential widens there (b > 0). But FZ0 is very noisy on volatile days, so with ~420
    stress days the shift is NOT significant (measured p ≈ 0.2). Lesson for the real
    results: a non-significant H3 regression is not evidence of "no stress effect"."""
    r = tables["regression"].set_index(["dummy", "model", "alpha"])
    hs = r.loc[("vix_gt30", "hs", 0.05)]
    assert hs["stress_shift"] > 0 and hs["calm_diff"] > 0
    assert hs["p_shift"] > 0.05  # documents the low power; flip only if the data change


def test_tables_shapes_and_power_column(tables):
    reg = tables["regimes"]
    assert {"all", "calm", "episodes", "vix_gt30", "vix_le30"} <= set(reg["regime"])
    row = reg[(reg.regime == "all") & (reg.model == "true") & (reg.alpha == 0.01)].iloc[0]
    assert row["expected_hits"] == pytest.approx(0.01 * row["n"])
    assert set(tables["mcs"]["regime"]) == {"episodes", "vix_gt30"}
    assert "true" not in set(tables["dm"]["model"])  # benchmark is never compared with itself


def test_unknown_benchmark_raises(sim_evals):
    with pytest.raises(ValueError, match="benchmark"):
        stress_tables({"hs": sim_evals["hs"][0]}, [0.05], benchmark="fhs_gjr")


def test_event_study_fast_vs_slow(sim_evals):
    paths, summary = event_study({k: sim_evals[k][0] for k in ("ewma", "hs")})
    s = summary.set_index("model")
    assert s.loc["ewma", "var_ratio_k1"] > s.loc["hs", "var_ratio_k1"]  # EWMA reacts at once
    # Large losses mostly strike when VaR is ALREADY elevated, so on average neither model
    # reaches 1.5x its level 5 days before (measured: EWMA 1.20, HS 1.10 at k = 1). The
    # Clarification-4 metric is still reported; k1/k5 ratios carry the information.
    assert s.loc["ewma", "var_ratio_k1"] > 1.1 and s.loc["hs", "var_ratio_k1"] < 1.15
    assert list(paths.index) == list(range(-5, 21)) and paths.loc[-5].eq(1.0).all()


def test_event_study_by_construction():
    idx = pd.bdate_range("2020-01-01", periods=200)
    loss = np.zeros(200)
    loss[100] = 0.1  # one big loss, the only "top 1%" day (top=0.005 -> 1 event)
    base = {"loss_next": loss, "target_date": idx}
    fast = pd.DataFrame({**base, "var_0.01": np.where(np.arange(200) > 100, 0.04, 0.02)}, index=idx)
    slow = pd.DataFrame({**base, "var_0.01": np.full(200, 0.02)}, index=idx)
    _, s = event_study({"fast": fast, "slow": slow}, top=0.005)
    s = s.set_index("model")
    assert s.loc["fast", "days_to_1.5x"] == 1 and np.isnan(s.loc["slow", "days_to_1.5x"])
    assert s.loc["fast", "n_events"] == 1


def test_benchmark_must_be_a_configured_model():
    from alphalab.config import RiskConfig

    with pytest.raises(ValueError, match="benchmark"):
        RiskConfig(models=[{"type": "ewma"}], benchmark="fhs_gjr")


def test_run_stress_pipeline(config_yaml):
    from alphalab.config import load_config
    from alphalab.pipeline import run_build, run_forecast, run_ingest, run_stress

    config_yaml.write_text(
        config_yaml.read_text()
        + "portfolio: {name: p, weights: {AAA: 0.5, BBB: 0.5}}\n"
        + "risk: {min_train: 260, benchmark: ewma,"
        + " models: [{type: hs, window: 250}, {type: ewma}]}\n"
    )
    cfg = load_config(config_yaml)
    run_build(cfg, run_ingest(cfg).snapshot_id)
    out = run_forecast(cfg)
    t = run_stress(cfg)
    assert {"regimes", "dm", "regression", "mcs", "event_paths", "event_summary"} <= set(t)
    assert "vix_gt30" not in set(t["regimes"]["regime"])  # no VIX feature -> regime skipped
    assert (out / "stress_regression.csv").exists()
