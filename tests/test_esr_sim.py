"""ESR bootstrap size simulation (Addition 9): exact forecasts, warp-speed p, resume."""

from __future__ import annotations

import json

import numpy as np
import pytest

from alphalab.backtest import esr_sim as es_
from alphalab.backtest.es import _t_tail


def test_exact_forecasts_match_closed_form_and_monte_carlo():
    v, e = _t_tail(0.025, 5.0)  # loss-space VaR/ES of the (unit-scale) t5
    q, es = es_.product_var_es(0.025, 0.0)
    assert q == pytest.approx(-v * es_.T_SCALE, rel=1e-10)
    assert es == pytest.approx(-e * es_.T_SCALE, rel=1e-10)
    rng = np.random.default_rng(1)
    s = 0.6
    x = np.exp(-(s**2) / 2 + s * rng.standard_normal(2_000_000))
    x = x * es_.T_SCALE * rng.standard_t(5, 2_000_000)
    q, es = es_.product_var_es(0.025, s)
    qm = np.quantile(x, 0.025)
    assert q == pytest.approx(qm, rel=0.01) and es == pytest.approx(x[x <= qm].mean(), rel=0.02)


def test_simulated_forecasts_are_calibrated_on_average():
    d = es_.Design("B", 0.8, phi_eta=0.95)
    hits = [
        np.mean(np.greater(*es_.simulate(d, 4000, np.random.default_rng(k))[:2])) for k in range(10)
    ]
    assert np.mean(hits) == pytest.approx(0.025, abs=0.003)


def test_warp_p_values():
    star = np.array([0.0, 1.0, 2.0, 3.0])
    assert es_.warp_p(np.array([2.0]), star, upper=True)[0] == 0.5  # share >= 2
    assert es_.warp_p(np.array([1.0]), star, upper=False)[0] == 0.5  # share <= 1
    assert es_.warp_p(np.array([9.0]), np.append(star, np.nan), upper=True)[0] == 0.0
    rng = np.random.default_rng(2)  # same law -> p uniform -> 5% rejections
    p = es_.warp_p(rng.chisquare(2, 4000), rng.chisquare(2, 4000), upper=True)
    assert np.mean(p < 0.05) == pytest.approx(0.05, abs=0.012)


def test_run_is_resumable_and_reproducible(tmp_path):
    s = es_.SimSettings(
        designs=(es_.Design("A", 0.0),), n=600, reps=4, blocks=(1, 10), batch=2, workers=1, nice=0
    )
    assert es_.run(tmp_path, s) == {"batches_done": 2, "batches_total": 2}
    f = tmp_path / "A" / "batch_0001.json"
    before = json.loads(f.read_text())
    f.unlink()
    assert es_.run(tmp_path, s)["batches_total"] == 1
    assert json.loads(f.read_text()) == before
    t = es_.summarize(tmp_path, s)
    assert set(t.method) == {"asymptotic", "iid bootstrap", "block bootstrap"}
    with pytest.raises(RuntimeError, match="other settings"):
        es_.run(tmp_path, es_.SimSettings(designs=s.designs, n=600, reps=6, batch=2, workers=1))
