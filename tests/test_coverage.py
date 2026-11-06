"""Coverage tests: hand-computed values, correct size under H0, power against clustering."""

from __future__ import annotations

import numpy as np
import pytest

from alphalab.backtest import christoffersen_ind, coverage_tests, kupiec_pof, traffic_light


def test_kupiec_by_hand():
    # n = 250, x = 5 hits at a = 1% (expected 2.5): π̂ = 0.02.
    hits = np.zeros(250, dtype=bool)
    hits[:5] = True
    lr, p = kupiec_pof(hits, 0.01)
    ll0 = 245 * np.log(0.99) + 5 * np.log(0.01)
    ll1 = 245 * np.log(0.98) + 5 * np.log(0.02)
    assert lr == pytest.approx(-2 * (ll0 - ll1)) and lr == pytest.approx(1.956, abs=1e-3)
    assert p == pytest.approx(0.162, abs=1e-3)  # not rejected: 5 vs 2.5 is weak evidence


def test_kupiec_zero_hits_is_finite():
    lr, p = kupiec_pof(np.zeros(1000, dtype=bool), 0.01)
    assert lr == pytest.approx(-2 * 1000 * np.log(0.99))  # 0·ln 0 = 0 convention
    assert p < 0.01  # zero hits in 1000 days at 1% is itself suspicious (too conservative)


def test_christoffersen_by_hand():
    # Sequence with every hit followed by another hit: strong clustering.
    hits = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    lr, _ = christoffersen_ind(hits)
    n00, n01, n10, n11 = 13, 2, 2, 2
    pi01, pi11, pi = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / 19
    ll0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll1 = n00 * np.log(1 - pi01) + n01 * np.log(pi01) + n10 * np.log(1 - pi11) + n11 * np.log(pi11)
    assert lr == pytest.approx(-2 * (ll0 - ll1))


def test_christoffersen_no_clustering_is_zero():
    # Transitions: n00=4, n01=2, n10=2, n11=1 -> P(hit|no hit) = P(hit|hit) = 1/3 -> LR = 0.
    lr, p = christoffersen_ind(np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 0]))
    assert lr == pytest.approx(0.0, abs=1e-12) and p == pytest.approx(1.0)
    # Strict alternation is dependence too (a hit is never followed by a hit).
    assert christoffersen_ind(np.array([0, 1] * 20))[0] > 10


@pytest.mark.parametrize("alpha", [0.01, 0.05])
def test_size_under_null(alpha):
    """A correct model is wrongly rejected about 5% of the time at the 5% level."""
    rng = np.random.default_rng(42)
    rej = [coverage_tests(rng.random(1000) < alpha, alpha).p_cc < 0.05 for _ in range(1500)]
    assert 0.02 < np.mean(rej) < 0.09  # χ² approximation is rough at 1% with ~10 hits


def test_power_against_clustering():
    """Hits arriving in bursts (as with a slow model in a crisis) are caught by IND."""
    rng = np.random.default_rng(7)
    hits = np.zeros(2500, dtype=bool)
    for start in rng.choice(2490, size=5, replace=False):
        hits[start : start + 5] = True  # 25 hits in 5 bursts; rate 1% is right
    res = coverage_tests(hits, 0.01)
    assert res.p_pof > 0.05  # frequency alone looks fine...
    assert res.p_ind < 0.001  # ...the clustering is what gives it away


def test_traffic_light():
    h = np.zeros(500, dtype=int)
    h[100:106] = 1  # 6 hits close together -> some windows yellow
    tl = traffic_light(h)
    assert tl["max_hits"] == 6 and tl["red"] == 0 and 0 < tl["yellow"] < 1
    assert tl["green"] + tl["yellow"] + tl["red"] == pytest.approx(1)
    assert traffic_light(h[:100])["windows"] == 0
