"""Amendment 10 (exploratory): ESR covariance variants."""

import numpy as np
import pytest

from alphalab.backtest import es as es_mod
from alphalab.backtest import esr_backtest
from alphalab.backtest import esr_bootstrap as eb
from alphalab.backtest import esr_sim as sim


def _sim(name, rep):
    s = sim.SimSettings()
    d = next(x for x in s.designs if x.name == name)
    return sim.simulate(d, s.n, np.random.default_rng(sim._rep_seed(s, d, rep)), s.burn)


def test_default_is_preregistered_and_unchanged():
    loss, var, es = _sim("A_correct", 1)
    a = esr_backtest(loss, var, es, 0.025)
    with es_mod.cov_variant("scl_sp"):
        b = esr_backtest(loss, var, es, 0.025)
    assert a == b and a.strict_status == "ok" and a.int_status == "ok"


def test_ind_nomis_runs_no_location_scale_fit(monkeypatch):
    loss, var, es = _sim("B1_missed_vol_phi095", 85)  # scl_sp ends on the boundary here

    def boom(*a, **k):
        raise AssertionError("location-scale fit called")

    monkeypatch.setattr(es_mod, "_cond_mean_sigma", boom)
    with es_mod.cov_variant("ind_nomis"):
        r = esr_backtest(loss, var, es, 0.025)
    assert r.strict_status == "ok" and r.int_status == "ok"
    assert np.isfinite(r.strict_wald) and np.isfinite(r.int_t)


def test_ind_mis_counts_boundary_fit_as_failure():
    loss, var, es = _sim("B1_missed_vol_phi095", 85)
    with es_mod.cov_variant("ind_mis"):
        r = esr_backtest(loss, var, es, 0.025)
    assert "boundary" in r.strict_status and np.isnan(r.strict_wald)
    loss, var, es = _sim("A_correct", 1)
    with es_mod.cov_variant("ind_mis"):
        r = esr_backtest(loss, var, es, 0.025)
    assert r.strict_status == "ok"


def test_variant_is_scoped_and_validated():
    with pytest.raises(ValueError), es_mod.cov_variant("scl_N"):
        pass
    with es_mod.cov_variant("ind_nomis"):
        assert es_mod._COV_VARIANT.get() == "ind_nomis"
    assert es_mod._COV_VARIANT.get() == "scl_sp"


def test_sim_settings_doc_and_replication_use_variant(tmp_path):
    assert "cov" not in sim._settings_doc(sim.SimSettings())  # old folders stay compatible
    s = sim.SimSettings(cov="ind_nomis", n=800, reps=1, blocks=(20,))
    assert sim._settings_doc(s)["cov"] == "ind_nomis"
    d = s.designs[0]
    out = sim.one_replication(s, d, 0)
    loss, var, es = sim.simulate(d, s.n, np.random.default_rng(sim._rep_seed(s, d, 0)), s.burn)
    with es_mod.cov_variant("ind_nomis"):
        ref = eb.make_cell(f"{d.name}_0", loss, var, es, d.alpha)
    assert out["ok"] and out["wald0"] == ref.wald0
    assert es_mod._COV_VARIANT.get() == "scl_sp"
