"""Robust location-scale starts (Addition 9, amendment 9)."""

from __future__ import annotations

import numpy as np

from alphalab.backtest.es import ESRFitError, _cond_mean_sigma, robust_scale


def _bad_data(n=400, seed=5):
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(n), rng.uniform(1, 2, n)])
    return rng.normal(size=n) * np.exp(6 * (x[:, 1] - 2)), x  # esreg start degenerate


def test_default_path_is_esreg_and_degenerate_here():
    y, x = _bad_data()
    with np.errstate(all="ignore"):
        _, sig = _cond_mean_sigma(y, x)
    assert not np.all(sig > 0)  # the defect, unchanged outside robust mode


def test_robust_mode_gives_valid_scale_and_reports_start():
    y, x = _bad_data()
    trace: list = []
    with robust_scale(None, trace):
        _, sig = _cond_mean_sigma(y, x)
    assert np.all(sig > 0) and trace == ["corrected"]


def test_warm_start_is_feasible_on_any_resample_and_used():
    y, x = _bad_data()
    with robust_scale(None):
        _, sig = _cond_mean_sigma(y, x)
    b2 = np.linalg.lstsq(x, sig, rcond=None)[0]
    ix = np.random.default_rng(1).integers(0, len(y), len(y))
    assert np.all(x[ix] @ b2 > 0)  # resample rows are full-sample rows
    trace: list = []
    with robust_scale(b2, trace):
        _, s2 = _cond_mean_sigma(y[ix], x[ix])
    assert np.all(s2 > 0) and trace == ["warm"]


def test_robust_mode_raises_instead_of_returning_invalid_scale():
    x = np.column_stack([np.ones(50), np.linspace(1, 2, 50)])
    y = np.random.default_rng(0).normal(size=50)
    y[3] = np.nan  # no start gives a finite likelihood
    trace: list = []
    with robust_scale(None, trace):
        try:
            _cond_mean_sigma(y, x)
            raised = False
        except ESRFitError:
            raised = True
    assert raised and trace[-1] == "failed"
