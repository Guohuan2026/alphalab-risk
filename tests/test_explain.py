"""Interpretability (H4): families match the committed Clarification 5, SHAP is exactly
additive, permutation importance finds a known driver, the explain pass reproduces the
forecast run, and the criteria can both pass and fail."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphalab.explain import FAMILIES, explain_model, h4_criteria
from alphalab.risk.engine import EngineSettings, run_forecasts
from alphalab.risk.models import build_model

ALPHAS = (0.01, 0.025, 0.05)
FAST = {
    "gbq": {
        "type": "gbq",
        "max_iter": 40,
        "grid": [{"max_leaf_nodes": 7, "min_samples_leaf": 100}],
    },
    "mlvol": {"type": "mlvol", "n_estimators": 60, "grid": [{"min_samples_leaf": 100}]},
    "qrf": {"type": "qrf", "n_estimators": 60, "grid": [{"min_samples_leaf": 50}]},
}
DESIGN = Path(__file__).parents[1] / "docs" / "RESEARCH_DESIGN.md"


def test_families_match_committed_clarification_5():
    text = DESIGN.read_text()
    clar = text[text.index("Clarification (not a change): testable H4 rules") :].replace("\n", " ")
    for fam, label in [
        ("volatility", "volatility ="),
        ("vix", "VIX ="),
        ("credit", "credit ="),
        ("other", "other ="),
    ]:
        seg = clar[clar.index(label) : clar.index("}", clar.index(label))]
        listed = {x.strip(" {") for x in seg.split("{", 1)[1].split(",")}
        assert listed == {f for f, g in FAMILIES.items() if g == fam}, fam


@pytest.fixture(scope="module")
def driven():
    """Volatility drives next-day risk through volatility and VIX features; 'ret' and
    't10y3m' are pure noise."""
    rng = np.random.default_rng(3)
    n = 2600
    idx = pd.bdate_range("2000-01-03", periods=n)
    vol = np.exp(np.cumsum(rng.normal(0, 0.05, n)) * 0.3 + rng.normal(0, 0.3, n)) * 0.01
    r = np.r_[0.0, vol[:-1] * rng.standard_normal(n - 1)]
    X = pd.DataFrame(
        {
            "rv_22": vol**2 * np.exp(rng.normal(0, 0.1, n)),
            "rv_5": vol**2 * np.exp(rng.normal(0, 0.3, n)),
            "vix": 100 * vol * np.exp(rng.normal(0, 0.1, n)),
            "ret": rng.normal(size=n),
            "t10y3m": rng.normal(size=n),
        },
        index=idx,
    )
    return pd.Series(r, index=idx), X


def test_shap_is_exactly_additive_mlvol(driven):
    r, X = driven
    m = build_model(FAST["mlvol"], ALPHAS).fit(r.to_numpy(), X)
    base, contrib = m.shap_var(X.iloc[-100:], 0.01)
    var = np.array([f.var[0.01] for f in m.predict_block(X.iloc[-100:])])
    np.testing.assert_allclose(base + contrib.sum(axis=1), var, rtol=1e-9, atol=1e-12)


def test_shap_explains_the_raw_gbq_quantile_model(driven):
    """For boosting, SHAP is additive for the RAW 1% quantile model. The reported VaR can
    differ where the per-level models cross and are sorted (the crossing fix), so the
    explain run reports how often that happens."""
    r, X = driven
    m = build_model(FAST["gbq"], ALPHAS).fit(r.to_numpy(), X)
    Xb = X.iloc[-100:]
    base, contrib = m.shap_var(Xb, 0.01)
    raw_var = -m._obj[0.01].predict(Xb[m.cols].to_numpy())
    np.testing.assert_allclose(base + contrib.sum(axis=1), raw_var, rtol=1e-9, atol=1e-12)
    # The crossing fix only REASSIGNS the per-level predictions to levels (sorted), so each
    # row's set of quantiles is unchanged; the 1% slot may receive a neighbour's value.
    raw_all = np.sort(
        np.column_stack([m._obj[float(t)].predict(Xb[m.cols].to_numpy()) for t in m.levels]), axis=1
    )
    reported_all = np.column_stack([[-f.var[0.01] for f in m.predict_block(Xb)]])
    pos = list(m.levels).index(0.01)
    np.testing.assert_allclose(reported_all[:, 0], raw_all[:, pos], rtol=1e-12)


def test_qrf_has_no_shap(driven):
    r, X = driven
    m = build_model(FAST["qrf"], ALPHAS).fit(r.to_numpy(), X)
    with pytest.raises(NotImplementedError, match="TreeSHAP"):
        m.shap_var(X.iloc[-5:], 0.01)


def test_permutation_importance_finds_the_driver(driven):
    r, X = driven
    m = build_model(FAST["mlvol"], ALPHAS)
    with pytest.raises(RuntimeError, match="fit"):
        m.permutation_importance(r.to_numpy(), X)
    m.fit(r.to_numpy(), X)
    imp = m.permutation_importance(r.to_numpy(), X).set_index("feature")["importance"]
    drivers = imp[["rv_22", "rv_5", "vix"]].max()
    assert drivers > 0 and drivers > 5 * max(imp["ret"], imp["t10y3m"], 1e-12)


@pytest.fixture(scope="module")
def explained(driven):
    r, X = driven
    s = EngineSettings(min_train=1800, refit_every=21)
    make = lambda: build_model({**FAST["mlvol"], "refit_every": 300}, ALPHAS)  # noqa: E731
    return explain_model(r, X, make, s), run_forecasts(r, make, s, features=X)


def test_explain_pass_reproduces_the_forecasts(explained):
    """The explain pass refits on the engine's schedule; its SHAP-implied VaR must equal the
    forecast run's VaR: proof that the explained models are the forecasting models."""
    ex, fc = explained
    np.testing.assert_allclose(
        ex["shap"]["var_shap"], fc.loc[ex["shap"].index, "var_0.01"], rtol=1e-9
    )
    assert ex["permutation"]["refit_origin"].nunique() == 3  # origins 0, 300, 600 of 800


def test_criteria_pass_when_volatility_drives_risk(driven, explained):
    r, X = driven
    ex, _ = explained
    crit, signs = h4_criteria(ex["permutation"], ex["shap"], X)
    c = crit.set_index("criterion")["passed"]
    assert c["C1_concentration"] and c["C2_economic_sign"]
    assert set(signs["feature"]) == {"rv_22", "rv_5", "vix"}


def test_criteria_can_fail():
    """C1 fails when the top features are not volatility/VIX; C3 fails when rankings jump."""
    rng = np.random.default_rng(0)
    feats = ["ret", "t10y3m", "dd_252", "rv_22", "vix"]
    rows = []
    for i, d in enumerate(pd.bdate_range("2000-01-03", periods=6, freq="252B")):
        imp = (
            rng.permutation([5.0, 4.0, 3.0, 0.1, 0.0])
            if i % 2
            else np.array([5.0, 4.0, 3.0, 0.1, 0.0])
        )
        rows += [
            {"refit_origin": d, "feature": f, "importance": v}
            for f, v in zip(feats, imp, strict=True)
        ]
    crit, _ = h4_criteria(pd.DataFrame(rows), None, pd.DataFrame())
    c = crit.set_index("criterion")["passed"]
    assert not c["C1_concentration"]
    assert "C2_economic_sign" not in c  # no SHAP -> C2 not evaluated, not silently "passed"


def test_run_explain_needs_ml_models(config_yaml):
    from alphalab.config import load_config
    from alphalab.pipeline import run_explain

    config_yaml.write_text(
        config_yaml.read_text()
        + "portfolio: {name: p, weights: {AAA: 1.0}}\n"
        + "risk: {min_train: 260, models: [{type: ewma}]}\n"
    )
    with pytest.raises(ValueError, match="no ML models"):
        run_explain(load_config(config_yaml))
