"""Losses, ES backtest, Diebold–Mariano and MCS: hand values, consistency, and the ranking
they produce on simulated data whose true VaR/ES are known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from alphalab.backtest import (
    diebold_mariano,
    fz0_loss,
    mcneil_frey,
    model_confidence_set,
    newey_west_lrv,
    tick_loss,
)
from alphalab.risk.simulate import standardized_t_tail


# ---------- loss functions ----------
def test_tick_loss_by_hand():
    # VaR 2% at α = 1%. Return −3% (a hit): (0.01 − 1)(−0.03 + 0.02) = 0.0099.
    assert tick_loss([-0.03], [0.02], 0.01)[0] == pytest.approx(0.0099)
    # Return +1% (no hit): (0.01 − 0)(0.01 + 0.02) = 0.0003.
    assert tick_loss([0.01], [0.02], 0.01)[0] == pytest.approx(0.0003)


def test_fz0_by_hand():
    y, var, es, a = -0.03, 0.02, 0.025, 0.025
    v, e = -var, -es
    expected = -1 * (v - y) / (a * e) + v / e + np.log(-e) - 1
    assert fz0_loss([y], [var], [es], a)[0] == pytest.approx(expected)
    with pytest.raises(ValueError, match="ES > 0"):
        fz0_loss([0.0], [0.01], [-0.01], a)


@pytest.fixture(scope="module")
def t_sample():
    nu = 5.0
    return stats.t.rvs(nu, size=400_000, random_state=3) * np.sqrt((nu - 2) / nu) * 0.01, nu


def test_tick_loss_minimised_at_true_quantile(t_sample):
    r, nu = t_sample
    true_var = 0.01 * standardized_t_tail(0.01, nu)[0]
    grid = true_var * np.linspace(0.8, 1.2, 41)
    mean_loss = [tick_loss(r, np.full_like(r, v), 0.01).mean() for v in grid]
    assert grid[int(np.argmin(mean_loss))] == pytest.approx(true_var, rel=0.03)


@pytest.mark.parametrize("kv, ke", [(0.9, 1), (1.1, 1), (1, 0.9), (1, 1.1), (0.9, 0.9)])
def test_fz0_is_lowest_for_true_pair(t_sample, kv, ke):
    """Strict consistency: misstating VaR and/or ES raises the average FZ0 loss."""
    r, nu = t_sample
    a = 0.025
    v, e = (0.01 * x for x in standardized_t_tail(a, nu))
    ones = np.ones_like(r)
    true_loss = fz0_loss(r, v * ones, e * ones, a).mean()
    wrong = fz0_loss(r, kv * v * ones, ke * e * ones, a).mean()
    assert true_loss < wrong


# ---------- Diebold–Mariano ----------
def test_newey_west_by_hand():
    x = np.array([1.0, -1.0, 2.0, 0.0])  # mean 0.5; demeaned [0.5, -1.5, 1.5, -0.5]
    d = x - x.mean()
    g0, g1 = d @ d / 4, d[1:] @ d[:-1] / 4
    assert newey_west_lrv(x, 1) == pytest.approx(g0 + 2 * 0.5 * g1)
    assert newey_west_lrv(x, 0) == pytest.approx(g0)


def test_dm_identical_and_shifted():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 2000)
    assert diebold_mariano(a, a).p_value == 1.0
    res = diebold_mariano(a, a + 0.2)  # A always better by 0.2
    assert res.stat < 0 and res.p_value < 1e-10
    with pytest.raises(ValueError, match="aligned"):
        diebold_mariano(a, a[:-1])


def test_dm_matches_t_test_without_autocorrelation():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 5000), rng.normal(0.03, 1, 5000)
    d = a - b
    t = d.mean() / (d.std() / np.sqrt(len(d)))
    assert diebold_mariano(a, b, lag=0).stat == pytest.approx(t, rel=1e-3)


def test_dm_size_under_autocorrelation():
    """AR(1) loss differentials (φ = 0.5), true H0. Measured: naive t-test rejects ~30%,
    Newey–West at the default lag ~10%, so HAC removes most but not all of the distortion
    (a known small-bandwidth effect, Kiefer & Vogelsang 2005). DM p-values are therefore
    somewhat liberal under strong autocorrelation; the MCS is the headline comparison."""
    rng = np.random.default_rng(2)
    naive, hac = [], []
    for _ in range(400):
        e = rng.normal(size=1500)
        d = np.empty(1500)
        d[0] = e[0]
        for t in range(1, 1500):
            d[t] = 0.5 * d[t - 1] + e[t]
        naive.append(diebold_mariano(d, np.zeros(1500), lag=0).p_value < 0.05)
        hac.append(diebold_mariano(d, np.zeros(1500)).p_value < 0.05)
    assert np.mean(naive) > 0.2
    assert np.mean(hac) < np.mean(naive) / 2 and np.mean(hac) < 0.15


# ---------- MCS ----------
def test_mcs_excludes_obvious_loser():
    rng = np.random.default_rng(4)
    L = pd.DataFrame({"good": rng.normal(1, 1, 3000), "same": rng.normal(1, 1, 3000)})
    L["bad"] = rng.normal(1.3, 1, 3000)
    m = model_confidence_set(L, reps=500).set_index("model")
    assert not m.loc["bad", "in_mcs"] and m.loc["good", "in_mcs"] and m.loc["same", "in_mcs"]
    with pytest.raises(ValueError, match="NaN"):
        model_confidence_set(L.assign(bad=np.nan), reps=10)


# ---------- McNeil–Frey ----------
def test_mcneil_frey_by_hand_and_few_hits():
    loss = np.array([0.05, 0.01, 0.04, 0.00, 0.06, 0.03, 0.07])
    var = np.full(7, 0.02)
    es = np.full(7, 0.045)
    res = mcneil_frey(loss, var, es, 0.025, min_hits=3)
    hits = loss[loss > 0.02]
    assert res.n_hits == 5 and res.mean_resid == pytest.approx((hits - 0.045).mean())
    assert res.severity_ratio == pytest.approx(hits.mean() / 0.045)
    assert not np.isnan(mcneil_frey(loss, var, es, 0.025).p_raw)  # 5 hits = min_hits: tested
    assert np.isnan(mcneil_frey(loss[:3], var[:3], es[:3], 0.025).p_raw)  # 2 hits < 5


# ---------- the ranking these tools produce on simulated truth ----------
def _fz0(ev, a=0.025):
    return fz0_loss(-ev["loss_next"], ev[f"var_{a:g}"], ev[f"es_{a:g}"], a)


def test_fz0_ranks_truth_and_gjr_above_ewma_and_hs(sim_evals):
    m = {k: _fz0(v[0]).mean() for k, v in sim_evals.items()}
    assert m["true"] < m["ewma"] and m["true"] < m["hs"]
    assert m["gjr"] < m["ewma"] < m["hs"]


def test_mcs_keeps_truth_and_gjr_drops_hs(sim_evals):
    L = pd.DataFrame({k: _fz0(v[0]) for k, v in sim_evals.items()})
    m = model_confidence_set(L, reps=1000).set_index("model")
    assert m.loc["true", "in_mcs"] and m.loc["gjr", "in_mcs"]
    assert not m.loc["hs", "in_mcs"]


def test_dm_gjr_beats_ewma(sim_evals):
    res = diebold_mariano(_fz0(sim_evals["gjr"][0]), _fz0(sim_evals["ewma"][0]))
    assert res.stat < 0 and res.p_value < 0.05


@pytest.mark.parametrize("name, rejected", [("true", False), ("gjr", False), ("ewma", True)])
def test_mcneil_frey_on_simulated_truth(sim_evals, name, rejected):
    """EWMA's normal ES is too small for t(6) tails; the true and GJR ES are not."""
    ev = sim_evals[name][0]
    res = mcneil_frey(ev["loss_next"], ev["var_0.025"], ev["es_0.025"], 0.025, ev["sigma"])
    assert (res.p_std < 0.01) == rejected
    if rejected:
        assert res.severity_ratio > 1.05  # realised tail losses > 5% above forecast ES


# ---------- Acerbi–Szekely Z2 (Clarification 7) ----------
def test_z2_by_hand():
    from alphalab.backtest import acerbi_szekely_z2

    loss = np.array([0.01, 0.05, -0.02, 0.04, 0.0, 0.01, 0.06, -0.01, 0.0, 0.02])
    var, es = np.full(10, 0.03), np.full(10, 0.045)
    res = acerbi_szekely_z2(loss, var, es, 0.1, reps=500)
    assert res.n_hits == 3
    assert res.z2 == pytest.approx(1 - (0.05 + 0.04 + 0.06) / 0.045 / (10 * 0.1))
    with pytest.raises(ValueError, match="VaR <= ES"):
        acerbi_szekely_z2(loss, var, var * 0.9, 0.1)


def test_z2_shape_matching_recovers_student_t():
    from alphalab.backtest.es import _t_tail, shape_matched_nu

    for nu in (3.0, 6.0, 20.0):
        v, e = _t_tail(0.025, nu)
        assert shape_matched_nu(e / v, 0.025) == pytest.approx(nu, rel=1e-4)
    v, e = _t_tail(0.025, np.inf)
    assert np.isinf(shape_matched_nu(e / v, 0.025))  # normal (or thinner) -> normal null


def _t_losses(rng, n, nu, scale):
    """Heteroskedastic Student-t losses with unit-variance shocks and their true VaR/ES."""
    s = scale * np.sqrt((nu - 2) / nu)
    return rng.standard_t(nu, n) * s, s


def test_z2_size_under_null():
    """Correct VaR and ES with time-varying scale: rejected about 5% of the time."""
    from alphalab.backtest import acerbi_szekely_z2
    from alphalab.backtest.es import _t_tail

    rng = np.random.default_rng(11)
    v, e = _t_tail(0.025, 5.0)
    rej = []
    for _ in range(400):
        scale = np.exp(rng.normal(0, 0.4, 2000)) * 0.01
        loss, s = _t_losses(rng, 2000, 5.0, scale)
        res = acerbi_szekely_z2(loss, v * s, e * s, 0.025, reps=2000)
        rej.append(res.p_value < 0.05)
    assert 0.025 < np.mean(rej) < 0.085


def test_z2_power_when_es_understated():
    """VaR exactly right, ES 20% too small, 5,000 days. Measured power ~0.83 at 5% (200
    replications, 2026-11-08; McNeil–Frey: 1.00, it is sharper for pure ES errors). The
    bound below was set after that measurement, with margin."""
    from alphalab.backtest import acerbi_szekely_z2
    from alphalab.backtest.es import _t_tail

    rng = np.random.default_rng(12)
    v, e = _t_tail(0.025, 4.0)
    rej = []
    for _ in range(100):
        loss, s = _t_losses(rng, 5000, 4.0, np.full(5000, 0.01))
        res = acerbi_szekely_z2(loss, v * s, 0.8 * e * s, 0.025, reps=2000)
        rej.append(res.p_value < 0.05)
    assert np.mean(rej) > 0.6


@pytest.mark.parametrize("name, rejected", [("true", False), ("gjr", False), ("ewma", True)])
def test_z2_on_simulated_truth(sim_evals, name, rejected):
    from alphalab.backtest import acerbi_szekely_z2

    ev = sim_evals[name][0]
    res = acerbi_szekely_z2(ev["loss_next"], ev["var_0.025"], ev["es_0.025"], 0.025)
    assert (res.p_value < 0.01) == rejected


# ---------- Bayer–Dimitriadis ESR (Addition 9) ----------
def test_quantile_regression_exact():
    from alphalab.backtest.es import quantile_regression

    rng = np.random.default_rng(31)
    y = rng.normal(size=401)
    b = quantile_regression(y, np.ones((401, 1)), 0.1)
    assert b[0] == pytest.approx(np.sort(y)[40])  # 10% sample quantile (order statistic 41)
    x = np.column_stack([np.ones(4000), rng.uniform(1, 3, 4000)])
    yy = x[:, 1] * rng.normal(size=4000)  # true 5% quantile = -1.645 * x
    b = quantile_regression(yy, x, 0.05)
    assert b[0] == pytest.approx(0, abs=0.1) and b[1] == pytest.approx(-1.645, abs=0.1)


def test_weighted_quantile_regression_equals_replication():
    """Integer weights = repeating observations."""
    from alphalab.backtest.es import quantile_regression

    rng = np.random.default_rng(36)
    x = np.column_stack([np.ones(300), rng.uniform(1, 2, 300)])
    y = x[:, 1] * rng.normal(size=300)
    w = rng.integers(1, 4, 300)
    b_w = quantile_regression(y, x, 0.1, weights=w)
    b_r = quantile_regression(np.repeat(y, w), np.repeat(x, w, axis=0), 0.1)

    def obj(b):  # LP optima can be non-unique, so compare the weighted objective
        r = y - x @ b
        return np.sum(w * r * (0.1 - (r < 0)))

    assert obj(b_w) == pytest.approx(obj(b_r), rel=1e-9)


def test_es_step_intercept_only_is_tail_mean():
    """With one constant ES regressor the FZ0 ES estimate is mean(S): the sample ES."""
    from alphalab.backtest.es import _es_step, quantile_regression

    rng = np.random.default_rng(32)
    y = rng.standard_t(5, 2000) - 1.0
    x = np.ones((2000, 1))
    q = x @ quantile_regression(y, x, 0.025)
    g = _es_step(y, q, x, 0.025, np.array([-1.0]))
    s = q + (y - q) * (y <= q) / 0.025
    assert g[0] == pytest.approx(s.mean(), rel=1e-9)


def test_esr_is_the_joint_fz0_optimum():
    """Code review round 1 case: no local optimiser can lower the joint FZ0 loss further."""
    from scipy.optimize import minimize

    from alphalab.backtest.es import _fit_joint, _fz0

    rng = np.random.default_rng(0)
    n = 1000
    sc = np.exp(rng.normal(0, 0.8, n)) * 0.01
    y = -rng.standard_t(5, n) * sc
    x = np.column_stack([np.ones(n), -3.52 * sc])
    c = max(y.max(), 0.0)
    beta, gamma = _fit_joint(y - c, x, x, 0.025, np.array([-c, 1.0]))

    def f(th):
        e = x @ th[2:]
        return _fz0(y - c, x @ th[:2], e, 0.025) if np.all(e < 0) else np.inf

    th = np.r_[beta, gamma]
    nm = minimize(f, th, method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-15})
    assert f(th) <= nm.fun + 1e-12


def test_esr_recovers_scaled_es():
    """Forecast ES is the truth / 1.5: the strict ES slope should be about 1.5."""
    from alphalab.backtest import esr_backtest
    from alphalab.backtest.es import _t_tail

    rng = np.random.default_rng(33)
    v, e = _t_tail(0.025, 5.0)
    loss, s = _t_losses(rng, 20000, 5.0, np.exp(rng.normal(0, 0.5, 20000)) * 0.01)
    res = esr_backtest(loss, v * s, e * s / 1.5, 0.025)
    assert res.strict_g1 == pytest.approx(1.5, abs=0.2)
    assert res.strict_p < 1e-3 and res.int_p < 1e-3 and res.int_g0 < 0


def test_esr_input_checks_and_degenerate_samples():
    from alphalab.backtest import esr_backtest

    rng = np.random.default_rng(35)
    loss = rng.normal(0, 0.01, 50)
    with pytest.raises(ValueError, match="aligned"):
        esr_backtest(loss, np.ones(49), np.ones(50), 0.05)
    with pytest.raises(ValueError, match="ES > 0"):
        esr_backtest(loss, np.ones(50), np.zeros(50), 0.05)
    for bad in (0.0, 1.0, np.nan):
        with pytest.raises(ValueError, match="alpha"):
            esr_backtest(loss, np.ones(50), np.ones(50), bad)
    with pytest.raises(ValueError, match="finite"):
        esr_backtest(loss, np.full(50, np.nan), np.ones(50), 0.05)
    with pytest.raises(ValueError, match="one-dimensional"):
        esr_backtest(loss[:, None], np.ones((50, 1)), np.ones((50, 1)), 0.05)
    few = esr_backtest(loss, np.full(50, 0.01), np.linspace(0.01, 0.03, 50), 0.05)
    assert few.n_tail < 10 and np.isnan(few.strict_p) and np.isnan(few.int_p)
    assert few.strict_status != "ok" and few.int_status != "ok"
    # Review round 1: a (nearly) constant forecast must not abort; the intercept test runs.
    n = 1000
    loss = np.random.default_rng(1).normal(0, 0.01, n)
    for es in (np.full(n, 0.02), 0.02 + np.linspace(0, 1e-12, n)):
        res = esr_backtest(loss, 0.8 * es, es, 0.025)
        assert "not identified" in res.strict_status and np.isnan(res.strict_p)
        assert res.int_status == "ok" and np.isfinite(res.int_p)
    # All-zero losses: nothing estimable, but no exception either.
    res = esr_backtest(np.zeros(n), np.full(n, 0.01), np.linspace(0.01, 0.03, n), 0.025)
    assert np.isnan(res.strict_p) and np.isnan(res.int_p)


def test_esr_size_under_null():
    """Correct VaR/ES. Measured 2026-11-10 with the final code (rejections at 5%, 200-300
    reps per cell, Addition 9): strict 21-22% (T=1000), 13-18% (2500), 9-10.5% (5000);
    intercept 0.5-2.3% at every T. Here 60 reps of T = 2,000, so each rate has Monte Carlo
    sd <= 0.05; bounds are about 3 sd above the measured rates at T = 2,500."""
    from alphalab.backtest import esr_backtest
    from alphalab.backtest.es import _t_tail

    rng = np.random.default_rng(34)
    v, e = _t_tail(0.025, 5.0)
    rs, ri = [], []
    for _ in range(60):
        loss, s = _t_losses(rng, 2000, 5.0, np.exp(rng.normal(0, 0.4, 2000)) * 0.01)
        res = esr_backtest(loss, v * s, e * s, 0.025)
        assert res.strict_status == "ok" and res.int_status == "ok"  # NaN must not pass
        rs.append(res.strict_p < 0.05)
        ri.append(res.int_p < 0.05)
    assert np.mean(ri) <= 0.10  # conservative in finite samples
    assert np.mean(rs) <= 0.32  # known over-size; guards against it getting worse


@pytest.mark.parametrize("name, rejected", [("true", False), ("gjr", False), ("ewma", True)])
def test_esr_on_simulated_truth(sim_evals, name, rejected):
    from alphalab.backtest import esr_backtest

    ev = sim_evals[name][0]
    res = esr_backtest(ev["loss_next"], ev["var_0.025"], ev["es_0.025"], 0.025)
    assert res.int_status == "ok" and np.isfinite(res.int_p)
    assert (res.int_p < 0.01) == rejected


def test_esr_is_unit_invariant():
    """Review round 2: rescaling losses and forecasts (decimals vs bp) must not change
    availability or test statistics; intercepts scale with the units."""
    from alphalab.backtest import esr_backtest
    from alphalab.backtest.es import _t_tail

    rng = np.random.default_rng(34)
    v, e = _t_tail(0.025, 5.0)
    loss, s = _t_losses(rng, 2000, 5.0, np.exp(rng.normal(0, 0.4, 2000)) * 0.01)
    base = esr_backtest(loss, v * s, e * s, 0.025)
    assert base.strict_status == "ok" and base.int_status == "ok"
    for f in (1e-5, 1e-4, 1e4):
        r = esr_backtest(loss * f, v * s * f, e * s * f, 0.025)
        assert r.strict_status == "ok" and r.int_status == "ok"
        assert r.strict_wald == pytest.approx(base.strict_wald, rel=1e-6)
        assert r.strict_g1 == pytest.approx(base.strict_g1, rel=1e-6)
        assert r.int_t == pytest.approx(base.int_t, rel=1e-6)
        assert r.int_g0 == pytest.approx(base.int_g0 * f, rel=1e-6)


def test_esr_stable_under_last_bit_perturbations():
    """Regression (CI, 2026-11-10): observations interpolated by the LP quantile fit flipped
    between 'at' and 'below' q with the last bits, moving the Wald by up to 5% across
    platforms. They are now classified with a tolerance (es._below)."""
    from alphalab.backtest import esr_backtest
    from alphalab.backtest.es import _t_tail

    rng = np.random.default_rng(34)
    v, e = _t_tail(0.025, 5.0)
    loss, s = _t_losses(rng, 2000, 5.0, np.exp(rng.normal(0, 0.4, 2000)) * 0.01)
    base = esr_backtest(loss, v * s, e * s, 0.025)
    pr = np.random.default_rng(9)
    for _ in range(12):
        r = esr_backtest(loss * (1 + pr.normal(size=2000) * 1e-15), v * s, e * s, 0.025)
        assert r.strict_wald == pytest.approx(base.strict_wald, rel=1e-9)
        assert r.int_t == pytest.approx(base.int_t, rel=1e-9)
