"""Descriptive 10-day table (Clarification 8): labels, windows, scaling and the pipeline step."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from alphalab.backtest.horizon import h_day_loss, h_day_table, h_day_windows


def _returns(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), name="ret")


def test_h_day_loss_by_hand():
    r = _returns([0.01, -0.02, 0.03, 0.00, -0.01])
    lab = h_day_loss(r, 2)
    assert lab["loss_h"].iloc[0] == pytest.approx(-((1 - 0.02) * (1 + 0.03) - 1))
    assert lab["loss_h"].iloc[2] == pytest.approx(-((1 + 0.0) * (1 - 0.01) - 1))
    assert lab["end_date"].iloc[0] == r.index[2]
    assert lab["loss_h"].iloc[3:].isna().all()  # fewer than h sessions follow


def test_label_uses_exactly_the_next_h_sessions():
    """The label at origin t depends on R_{t+1..t+h} only: not on t itself, not on t+h+1."""
    rng = np.random.default_rng(0)
    r = _returns(rng.normal(0, 0.01, 60))
    base = h_day_loss(r, 10)["loss_h"].iloc[20]
    for pos, changes in [(20, False), (21, True), (30, True), (31, False), (19, False)]:
        r2 = r.copy()
        r2.iloc[pos] += 0.05
        moved = h_day_loss(r2, 10)["loss_h"].iloc[20] != pytest.approx(base)
        assert moved == changes, pos


def _forecasts(r, sigma, alphas, start=0):
    """Correct normal 1-day forecasts (zero mean, known sigma) at every origin from ``start``."""
    idx = r.index[start:]
    cols = {}
    for a in alphas:
        z = stats.norm.isf(a)
        cols[f"var_{a:g}"] = z * sigma
        cols[f"es_{a:g}"] = stats.norm.pdf(z) / a * sigma
    return pd.DataFrame(cols, index=idx)


def test_windows_are_non_overlapping_and_stop_before_the_lockbox():
    rng = np.random.default_rng(1)
    r = _returns(rng.normal(0, 0.01, 400))
    fc = {"a": _forecasts(r, 0.01, [0.05], start=7), "b": _forecasts(r, 0.01, [0.05], start=12)}
    cut = r.index[300]
    w = h_day_windows(fc, r, 10, before=cut)
    pos = r.index.get_indexer(w)
    assert w[0] == r.index[12]  # first origin shared by all models
    assert np.all(np.diff(pos) == 10)  # every 10th session: no overlap
    assert r.index[pos[-1] + 10] < cut  # the last window ends before the lockbox
    assert r.index[pos[-1] + 20] >= cut  # ...and the next one would not


def test_forecasts_are_only_scaled_never_refitted():
    rng = np.random.default_rng(2)
    r = _returns(rng.normal(0, 0.01, 300))
    f = _forecasts(r, 0.01, [0.05])
    f.iloc[::2] *= 2  # arbitrary forecasts: the table must use them as given
    t = h_day_table({"m": f}, r, [0.05], h=10)
    w = h_day_windows({"m": f}, r, 10)
    loss = h_day_loss(r, 10).loc[w, "loss_h"]
    assert t.loc[0, "breaches"] == int((loss > np.sqrt(10) * f.loc[w, "var_0.05"]).sum())
    assert t.loc[0, "windows"] == len(w) == 29


def test_sqrt10_is_right_for_iid_normal_and_wrong_for_mean_reverting_vol():
    """i.i.d. normal: √10 scaling gives about the right breach count. Volatility that is high
    today but reverts within days: √10 × today's σ overstates 10-day risk (too few breaches)."""
    rng = np.random.default_rng(3)
    n = 60_000
    r = _returns(rng.normal(0, 0.01, n))
    t = h_day_table({"m": _forecasts(r, 0.01, [0.05])}, r, [0.05])
    assert 0.85 < t.loc[0, "breach_ratio"] < 1.15  # 300 expected breaches

    # σ of R_i is 3% on the first day of each window (i = 1, 11, 21, ...) and 1% otherwise.
    sig_ret = np.where(np.arange(n) % 10 == 1, 0.03, 0.01)
    r2 = _returns(rng.normal(0, 1, n) * sig_ret)
    sig_fc = np.append(sig_ret[1:], 0.01)  # correct 1-day σ forecast at origin t: σ of R_{t+1}
    t2 = h_day_table({"m": _forecasts(r2, sig_fc, [0.05])}, r2, [0.05])
    # √10 × 3% = 9.5% vs a true 10-day σ of √(3%² + 9 × 1%²) = 4.6%: hardly any breaches.
    assert t2.loc[0, "breach_ratio"] < 0.2


def test_run_horizon_on_the_pipeline(tmp_path):
    from alphalab.config import load_config
    from alphalab.pipeline import run_build, run_forecast, run_horizon, run_ingest

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "provider: synthetic\nuniverse: [AAA, BBB]\nstart: '2018-01-01'\nend: '2021-12-31'\n"
        f"data_dir: '{tmp_path / 'data'}'\nprovider_options: {{seed: 1}}\n"
        "portfolio: {name: p1_spy, weights: {AAA: 0.5, BBB: 0.5}}\n"
        "risk: {min_train: 500, lockbox_start: '2021-01-01',"
        " models: [{type: hs, window: 250}, {type: ewma}]}\n"
    )
    cfg = load_config(cfg_path)
    run_build(cfg, run_ingest(cfg).snapshot_id)
    out = run_forecast(cfg)
    t = run_horizon(cfg)
    assert (out / "horizon10.csv").exists()
    assert set(t.model) == {"hs250", "ewma"} and set(t.alpha) == set(cfg.risk.alphas)
    assert (pd.to_datetime(t.last_end) < pd.Timestamp("2021-01-01")).all()
    # ~260 development sessions after the 500-day warm-up -> about 26 ten-day windows
    assert t.windows.nunique() == 1 and 20 < t.windows.iloc[0] < 30
