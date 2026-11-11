"""ESR bootstrap job (Addition 9): determinism, resume, refusal of mixed inputs, budget."""

from __future__ import annotations

import json

import numpy as np
import pytest

from alphalab.backtest import esr_bootstrap as eb
from alphalab.backtest.es import _t_tail


def _series(seed=7, n=800, es_factor=1.0):
    rng = np.random.default_rng(seed)
    v, e = _t_tail(0.05, 5.0)
    sc = np.exp(rng.normal(0, 0.4, n)) * 0.01 * np.sqrt(3 / 5)
    loss = rng.standard_t(5, n) * sc
    return loss, v * sc, es_factor * e * sc


@pytest.fixture(scope="module")
def cell():
    c = eb.make_cell("sim_0.05", *_series(), 0.05)
    assert c is not None
    return c


def test_chunk_indices_depend_only_on_seed_cell_chunk():
    a = eb.chunk_indices("m_0.01", 1, 3, 4, 50)
    assert np.array_equal(a, eb.chunk_indices("m_0.01", 1, 3, 4, 50))
    assert not np.array_equal(a, eb.chunk_indices("m_0.01", 1, 4, 4, 50))
    assert not np.array_equal(a, eb.chunk_indices("m_0.025", 1, 3, 4, 50))
    assert eb.chunk_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)]


def test_circular_blocks_are_consecutive_and_cover_days_evenly():
    idx = eb.chunk_indices("m_0.01", 1, 0, 400, 50, block=7)
    assert idx.shape == (400, 50)
    starts = idx[:, ::7]
    for k in range(1, 7):  # within a block, positions advance by one, wrapping at n
        cols = idx[:, k::7]
        assert np.array_equal(cols, (starts[:, : cols.shape[1]] + k) % 50)
    freq = np.bincount(idx.ravel(), minlength=50) / idx.size
    assert np.abs(freq - 1 / 50).max() < 0.005  # circular: every day equally likely
    assert np.array_equal(
        eb.chunk_indices("m", 1, 0, 3, 50, block=1), eb.chunk_indices("m", 1, 0, 3, 50)
    )


def test_default_block_length_and_bounds():
    s = eb.BootstrapSettings()
    assert s.block_for(6535) == 19 and s.block_for(1000) == 10
    assert eb.BootstrapSettings(block=1).block_for(10) == 1
    with pytest.raises(ValueError):
        eb.BootstrapSettings(block=0).block_for(10)


def test_identity_resample_gives_zero_statistics(cell):
    """Refitting on the original sample reproduces the full-sample estimates exactly."""
    from alphalab.backtest.es import esr_bootstrap_draws

    idx = np.arange(len(cell.y))[None, :]
    w, t = esr_bootstrap_draws(cell.y, cell.ehat, cell.alpha, idx, cell.g_hat, cell.g0_hat)
    assert w[0] == pytest.approx(0.0, abs=1e-12) and t[0] == pytest.approx(0.0, abs=1e-9)


def test_run_resume_and_summary(tmp_path, cell):
    s = eb.BootstrapSettings(reps=6, chunk=2, workers=1, nice=0)
    out = eb.run(tmp_path, [cell], s, tmp_path / "progress.log")
    assert out == {"chunks_done": 3, "chunks_total": 3, "stopped_by_budget": False}
    logged = (tmp_path / "progress.log").read_text()
    assert "3 chunks to do" in logged and "ETA" in logged and "3/3 chunks done" in logged
    d = tmp_path / cell.key
    first = {f.name: np.load(f)["wald_star"].copy() for f in d.glob("chunk_*.npz")}
    (d / "chunk_0001.npz").unlink()  # simulate an interrupted run
    out = eb.run(tmp_path, [cell], s)
    assert out["chunks_total"] == 1  # only the missing chunk is redone
    again = {f.name: np.load(f)["wald_star"] for f in d.glob("chunk_*.npz")}
    assert first.keys() == again.keys()
    for k in first:  # bit-identical regardless of the interruption
        assert np.array_equal(first[k], again[k], equal_nan=True)
    t = eb.summarize(tmp_path, s)
    row = t.iloc[0]
    assert row.complete and row.reps_done == 6
    assert 0 <= row.strict_p_boot <= 1 and 0 <= row.int_p_boot <= 1
    man = json.loads((d / "manifest.json").read_text())
    assert man["estimator_version"] == eb.ESTIMATOR_VERSION
    assert man["scheme"] == "circular block" and man["block"] == 10  # ceil(800^(1/3))
    assert row.scheme == "circular block" and row.block == 10


def test_incomplete_cell_has_no_p_value(tmp_path, cell):
    s = eb.BootstrapSettings(reps=6, chunk=2, workers=1, nice=0)
    eb.run(tmp_path, [cell], s)
    (tmp_path / cell.key / "chunk_0002.npz").unlink()
    row = eb.summarize(tmp_path, s).iloc[0]
    assert not row.complete and np.isnan(row.strict_p_boot) and np.isnan(row.int_p_boot)


def test_mismatched_inputs_are_refused(tmp_path, cell):
    eb.run(tmp_path, [cell], eb.BootstrapSettings(reps=2, chunk=1, workers=1, nice=0))
    with pytest.raises(RuntimeError, match="different"):
        eb.run(tmp_path, [cell], eb.BootstrapSettings(reps=4, chunk=1, workers=1, nice=0))
    with pytest.raises(RuntimeError, match="block"):
        eb.run(tmp_path, [cell], eb.BootstrapSettings(reps=2, chunk=1, workers=1, nice=0, block=1))
    other = eb.make_cell("sim_0.05", *_series(seed=8), 0.05)  # same key, other data
    with pytest.raises(RuntimeError, match="data_sha256"):
        eb.run(tmp_path, [other], eb.BootstrapSettings(reps=2, chunk=1, workers=1, nice=0))


def test_lock_blocks_a_second_run(tmp_path, cell):
    with eb._Lock(tmp_path / ".lock"), pytest.raises(RuntimeError, match="another"):
        eb.run(tmp_path, [cell], eb.BootstrapSettings(reps=2, chunk=1, workers=1, nice=0))


def test_time_budget_stops_new_chunks(tmp_path, cell):
    s = eb.BootstrapSettings(reps=8, chunk=1, workers=1, nice=0, max_hours=0.0)
    out = eb.run(tmp_path, [cell], s)
    assert out["stopped_by_budget"] and out["chunks_done"] < out["chunks_total"]


def test_score_dependence_detects_autocorrelated_tail_errors(cell):
    """Calibrated iid-like forecasts: ratios near 1. Forecasts that ignore rare, persistent
    high-volatility spells: tail hits cluster, so both scores are autocorrelated."""
    d = eb.score_dependence(cell)
    assert 0.6 < d["lrv_ratio_es_score"] < 1.6 and 0.6 < d["lrv_ratio_quantile_score"] < 1.6
    rng = np.random.default_rng(11)
    n = 3000
    regime = np.repeat(rng.choice([1.0, 3.0], size=n // 50, p=[0.9, 0.1]), 50)
    loss = rng.standard_t(5, n) * 0.01 * regime
    var = 0.025 * (1 + 0.05 * rng.standard_normal(n))  # forecasts ignore the spells
    es = 0.035 * (1 + 0.05 * rng.standard_normal(n))
    db = eb.score_dependence(eb.make_cell("bad_0.025", loss, var, es, 0.025))
    assert db["lrv_ratio_quantile_score"] > 1.8 and db["lb10_p_quantile_score"] < 0.01
    assert db["lrv_ratio_es_score"] > 1.5
