import numpy as np

from alphalab.backtest import esr_bootstrap as eb
from alphalab.backtest import esr_endaudit as ea
from alphalab.backtest import esr_sim as es_


def _sim(name, rep):
    s = es_.SimSettings()
    d = next(x for x in s.designs if x.name == name)
    L, V, E = es_.simulate(d, s.n, np.random.default_rng(es_._rep_seed(s, d, rep)), s.burn)
    return eb._normalise(L, E)


def test_end_ratio_flags_boundary_solution_from_valid_start():
    # B1 rep85 (start-invariance check): esreg's start is valid but its fit ends at sd ~ 0
    y, eh = _sim("B1_missed_vol_phi095", 85)
    x = np.column_stack([np.ones(len(y)), eh])
    assert not eb.degenerate_start(y - y.max(), x)
    r = ea.end_ratio(y, eh)
    assert min(r.values()) < ea.THRESHOLD


def test_end_ratio_interior_on_correct_forecasts():
    y, eh = _sim("A_correct", 1)
    r = ea.end_ratio(y, eh)
    assert min(r.values()) > 0.1


def test_summarize_counts_failed_fits_as_flagged():
    import pandas as pd

    df = pd.DataFrame(
        {
            "source": "s",
            "cell": "c",
            "draw": ["0", "1", "2"],
            "ratio_strict": [0.5, np.nan, 0.03],
            "ratio_int": [0.4, 0.5, 0.5],
        }
    )
    t = ea.summarize(df).iloc[0]
    assert (t.draws, t.failed_fit, t["flag_0.05"], t["flag_0.01"]) == (3, 1, 2, 1)
