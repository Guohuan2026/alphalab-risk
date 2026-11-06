from __future__ import annotations

import pandas as pd
import pytest

from alphalab.data.providers.synthetic import SyntheticProvider
from alphalab.data.schema import coerce_schema


def make_prices(rows: list[tuple]) -> pd.DataFrame:
    """rows: (date, security_id, close, dividend, split_ratio). OHLC set from close."""
    df = pd.DataFrame(rows, columns=["date", "security_id", "close", "dividend", "split_ratio"])
    df["ticker"] = df["security_id"]
    df["open"] = df["high"] = df["low"] = df["close"]
    df["volume"] = 1_000.0
    return coerce_schema(df)


@pytest.fixture
def split_and_dividend_prices() -> pd.DataFrame:
    # X: 2-for-1 split on day 3.  Y: $1.50 dividend on day 2.
    return make_prices(
        [
            ("2024-01-02", "X", 100.0, 0.0, 1.0),
            ("2024-01-03", "X", 102.0, 0.0, 1.0),
            ("2024-01-04", "X", 51.5, 0.0, 2.0),
            ("2024-01-02", "Y", 100.0, 0.0, 1.0),
            ("2024-01-03", "Y", 99.0, 1.5, 1.0),
            ("2024-01-04", "Y", 99.0, 0.0, 1.0),
        ]
    )


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    return SyntheticProvider(seed=7).fetch(["AAA", "BBB", "CCC"], "2020-01-01", "2021-12-31")


@pytest.fixture
def config_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "provider: synthetic\n"
        "universe: [AAA, BBB]\n"
        "start: '2020-01-01'\n"
        "end: '2021-06-30'\n"
        f"data_dir: '{tmp_path / 'data'}'\n"
        "monthly_min_obs: 15\n"
        "provider_options: {seed: 1}\n"
    )
    return p


# ---------- shared simulated forecasts (computed once per test session) ----------
SIM_ALPHAS = (0.01, 0.025, 0.05)


@pytest.fixture(scope="session")
def sim6000():
    from alphalab.risk.simulate import simulate_gjr_garch_t

    return simulate_gjr_garch_t(6000, seed=1, start="1995-01-03")


@pytest.fixture(scope="session")
def sim_evals(sim6000):
    """Walk-forward forecasts + outcomes on simulated GJR-GARCH-t data for every model,
    plus the TRUE forecasts. Returns {name: (evaluated frame, fit_info rows)}."""
    from alphalab.risk.engine import EngineSettings, attach_outcomes, run_forecasts
    from alphalab.risk.models import build_model

    r = sim6000["ret"]
    s = EngineSettings(min_train=1000, refit_every=21)
    specs = {
        "hs": {"type": "hs", "window": 250},
        "ewma": {"type": "ewma"},
        "garch": {"type": "garch"},
        "gjr": {"type": "gjr"},
        "fhs": {"type": "fhs"},
    }
    out = {}
    for name, spec in specs.items():
        fits: list[dict] = []
        f = run_forecasts(r, lambda spec=spec: build_model(spec, SIM_ALPHAS), s, fits=fits)
        out[name] = (attach_outcomes(f, r), fits)
    truth = sim6000[[c for c in sim6000 if c.startswith("true_")] + ["sigma"]].shift(-1)
    truth.columns = [c.replace("true_", "") for c in truth.columns]
    out["true"] = (attach_outcomes(truth.reindex(out["gjr"][0].index), r), [])
    return out
