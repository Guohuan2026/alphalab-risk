"""The simulator's ground truth must itself be correct before models are judged against it."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from alphalab.risk.simulate import (
    GJRParams,
    gjr_variance_filter,
    simulate_gjr_garch_t,
    standardized_t_tail,
)


@pytest.fixture(scope="module")
def long_sim():
    return simulate_gjr_garch_t(300_000, seed=11)


@pytest.mark.parametrize("alpha", [0.01, 0.025, 0.05])
def test_t_tail_formulas_match_monte_carlo(alpha):
    nu = 5.0
    z = stats.t.rvs(nu, size=4_000_000, random_state=1) * np.sqrt((nu - 2) / nu)
    loss = -z
    var, es = standardized_t_tail(alpha, nu)
    assert np.quantile(loss, 1 - alpha) == pytest.approx(var, rel=0.01)
    assert loss[loss >= var].mean() == pytest.approx(es, rel=0.015)


def test_t_tail_tends_to_normal():
    var, es = standardized_t_tail(0.01, 1e7)
    q = stats.norm.ppf(0.99)
    assert var == pytest.approx(q, rel=1e-5)
    assert es == pytest.approx(stats.norm.pdf(q) / 0.01, rel=1e-5)  # normal ES = φ(q)/α


def test_fat_tails_make_es_ratio_larger():
    # ES/VaR is 1.15 for the normal at 1%; fat tails push it up. A key reason the
    # normal-based EWMA can understate ES even when its VaR looks fine.
    v6, e6 = standardized_t_tail(0.01, 6)
    assert e6 / v6 > 1.2


@pytest.mark.parametrize("alpha", [0.01, 0.05])
def test_true_var_has_correct_coverage(long_sim, alpha):
    loss = -long_sim["ret"]
    hits = (loss > long_sim[f"true_var_{alpha:g}"]).to_numpy()
    n = len(hits)
    sd = np.sqrt(alpha * (1 - alpha) / n)
    assert abs(hits.mean() - alpha) < 4 * sd  # within 4 binomial standard errors


def test_true_es_matches_tail_mean(long_sim):
    # McNeil–Frey idea: on VaR exceedance days, (loss − ES)/σ averages zero if ES is right.
    loss = -long_sim["ret"]
    hit = loss > long_sim["true_var_0.025"]
    resid = ((loss - long_sim["true_es_0.025"]) / long_sim["sigma"])[hit]
    assert abs(resid.mean()) < 4 * resid.std() / np.sqrt(len(resid))


def test_stylised_facts(long_sim):
    r = long_sim["ret"].to_numpy()
    s = long_sim["sigma"].to_numpy()
    r2 = (r - r.mean()) ** 2
    assert np.corrcoef(r2[:-1], r2[1:])[0, 1] > 0.05  # volatility clustering
    neg = r[:-1] < GJRParams().mu
    assert s[1:][neg].mean() > s[1:][~neg].mean()  # leverage: falls raise volatility more
    assert stats.kurtosis(r) > 1  # fat tails


def test_sigma_is_known_one_day_ahead(long_sim):
    # Re-filtering the simulated returns reproduces sigma: row t uses only r_{t-1}.
    head = long_sim.iloc[:2000]
    s2 = gjr_variance_filter(head["ret"].to_numpy(), GJRParams(), head["sigma"].iloc[0] ** 2)
    np.testing.assert_allclose(np.sqrt(s2), head["sigma"], rtol=1e-12)


def test_deterministic_and_dated():
    a = simulate_gjr_garch_t(500, seed=3, start="2010-01-04")
    b = simulate_gjr_garch_t(500, seed=3, start="2010-01-04")
    assert a.equals(b) and a.index[0].strftime("%Y-%m-%d") == "2010-01-04"
    assert not a.equals(simulate_gjr_garch_t(500, seed=4, start="2010-01-04"))


@pytest.mark.parametrize(
    "kw, msg",
    [
        ({"beta": 0.99}, "stationary"),
        ({"nu": 2.0}, "nu"),
        ({"omega": 0.0}, "omega"),
    ],
)
def test_bad_params(kw, msg):
    with pytest.raises(ValueError, match=msg):
        simulate_gjr_garch_t(10, GJRParams(**kw))
