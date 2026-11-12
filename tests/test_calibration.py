"""Milestone 16 calibration machinery (not the calibration results)."""

import json

import numpy as np
import pandas as pd
from scipy import stats

from alphalab.backtest import calibration as cb


def test_kv_critical_value_and_fixed_b_size_on_iid_normal():
    assert cb.kv_cv_bartlett(0.0) == 1.96
    rng = np.random.default_rng(1)
    rej = [cb.dm_fixed_b(rng.standard_normal(500)) > cb.kv_cv_bartlett(0.1) for _ in range(3000)]
    assert abs(np.mean(rej) - 0.05) < 0.012  # the fixed-b critical value gives ~5% size


def test_sim_diff_zero_mean_shapes_and_stress_scaling():
    st = np.r_[np.zeros(3000, bool), np.ones(3000, bool)]
    d = cb.DiffDesign("x", phi_mu=0.95, w=0.05)
    x = cb.sim_diff(d, st, np.random.default_rng(2), k=3)
    assert x.shape == (6000, 3) and abs(x.mean()) < 0.1
    assert x[st].std() > 1.6 * x[~st].std()  # STRESS_MULT = 2 on stress days
    s = cb.diff_stats(cb.sim_diff(d, st, np.random.default_rng(3))[:, 0])
    assert 0.0 < s["acf1"] < 0.1 and s["lrv_100"] > 1.2


def test_mixture_var_es_exact():
    q, e = cb.mixture_var_es(0.025)
    s1, s2 = cb.MIX_S1, cb.MIX_K * cb.MIX_S1
    cdf = (1 - cb.MIX_P) * stats.norm.cdf(q / s1) + cb.MIX_P * stats.norm.cdf(q / s2)
    assert abs(cdf - 0.025) < 1e-10
    assert abs((1 - cb.MIX_P) * s1**2 + cb.MIX_P * s2**2 - 1) < 1e-12  # unit variance
    rng = np.random.default_rng(4)
    big = rng.random(2_000_000) < cb.MIX_P
    z = rng.standard_normal(2_000_000) * np.where(big, s2, s1)
    assert abs(z[z <= q].mean() - e) < 0.02 * abs(e)


def test_replications_run_and_decisions_apply_frozen_rules(tmp_path):
    r = cb.diff_replication(cb.DIFF_DESIGNS[2], np.r_[np.zeros(300, bool), np.ones(100, bool)], 0)
    assert {"dm_p", "dm_fb_rej", "h3_p", "h3_fb_rej", "mcs_excl", "mcs_b40_excl"} <= set(r)
    rows = cb.es_replication(cb.ES_DESIGNS[3], 961, 0)
    assert [x["alpha"] for x in rows] == list(cb.ALPHAS)
    s = cb.CalSettings("diff", reps=4, batch=4)
    for name in ("Dmid", "Dhigh", "D0n_iid_normal", "D0t_iid_t3"):
        (tmp_path / f"{name}_n961").mkdir()
        rej = name == "Dhigh"
        rows = [
            {
                "rep": i,
                "dm_p": 0.01 if rej else 0.5,
                "dm_fb_rej": False,
                "h3_p": 0.5,
                "h3_fb_rej": False,
                "mcs_excl": 0,
                "mcs_b40_excl": 0,
            }
            for i in range(4)
        ]
        (tmp_path / f"{name}_n961" / "batch_0000.json").write_text(json.dumps(rows))
    s = cb.CalSettings("diff", reps=4, batch=4, sizes=(961,))
    dec = cb.decisions(cb.cell_rates(tmp_path, s)).set_index("test")
    assert dec.loc["dm", "decision"] == "SUPPORTED WITH FALLBACK (dm_fb)"  # Dhigh fails, fb passes
    assert dec.loc["h3", "decision"] == "SUPPORTED"
    assert dec.loc["mcs", "decision"] == "SUPPORTED"
    assert isinstance(cb.cell_rates(tmp_path, s), pd.DataFrame)
