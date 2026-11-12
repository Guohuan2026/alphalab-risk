"""Dashboard: the read-only data layer, fixed colours, and headless runs of the whole app."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from alphalab.dashboard import data as D

APP = str(Path(__file__).parents[1] / "dashboard" / "app.py")


def test_colours_follow_the_model_not_its_rank():
    assert D.model_color("fhs_gjr") == D.PALETTE["light"][0]
    assert D.model_color("mlvol_fhs", "dark") == D.PALETTE["dark"][1]
    assert D.model_color("hs250") == D.model_color("garch_t") == D.OTHER_COLOR["light"]  # folded
    assert len(D.MODEL_ORDER) == len(D.PALETTE["light"]) == 8  # never more series than slots
    assert D.label("linqr_l1") == "Lasso QR (control)" and D.label("unknown") == "unknown"


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """A small real pipeline run (synthetic data) with development and lockbox results."""
    from alphalab.config import load_config
    from alphalab.pipeline import run_backtest, run_build, run_forecast, run_ingest

    tmp = tmp_path_factory.mktemp("dash")
    cfg_path = tmp / "cfg.yaml"
    cfg_path.write_text(
        "provider: synthetic\nuniverse: [AAA, BBB]\nstart: '2018-01-01'\nend: '2021-12-31'\n"
        f"data_dir: '{tmp / 'data'}'\nprovider_options: {{seed: 1}}\n"
        "portfolio: {name: p1_spy, weights: {AAA: 0.5, BBB: 0.5}}\n"
        "risk: {min_train: 500, lockbox_start: '2021-01-01', benchmark: fhs_gjr,"
        " models: [{type: hs, window: 250}, {type: ewma}, {type: fhs}]}\n"
    )
    cfg = load_config(cfg_path)
    run_build(cfg, run_ingest(cfg).snapshot_id)
    for lb in (False, True):
        run_forecast(cfg, include_lockbox=lb)
        run_backtest(cfg, include_lockbox=lb)
    return tmp / "data"


def test_discover_and_load(results):
    runs = D.discover_runs(results)
    assert len(runs) == 1
    run = runs[0]
    assert run.name == "P1 · SPY (headline) · expanding" and run.has_lockbox
    assert run.lockbox_start == "2021-01-01" and run.processed.exists()
    h = D.headline(run)
    assert {"label", "mean_fz0", "rank_fz0", "hit_rate", "p_cc"} <= set(h.columns)
    lb = D.forecasts(run, lockbox=True)
    assert lb.target_date.min() >= D.pd.Timestamp("2021-01-01")
    dev = D.forecasts(run)
    assert dev.target_date.max() < D.pd.Timestamp("2021-01-01")
    assert D.table(run, "stress_regimes") is None  # missing tables -> None, not an error
    mc = D.market_conditions(run, "2019-01-01", "2019-12-31")
    assert mc["days"] > 200 and "mean_vix" not in mc  # synthetic run: no VIX feature


def test_discover_on_empty_folder(tmp_path):
    assert D.discover_runs(tmp_path) == []


def _app(data_dir, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("ALPHALAB_DATA_DIR", str(data_dir))
    # AppTest replaces sys.modules["__main__"] with the app script. Process pools started
    # later in the same session (spawn start method) re-import __main__ in every worker, so
    # they would re-run the dashboard and, without saved results, die. Restore it on teardown.
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    return AppTest.from_file(APP, default_timeout=60)


def test_app_runs_without_errors(results, monkeypatch):
    at = _app(results, monkeypatch).run()
    assert not at.exception, at.exception
    assert any("Forecasts per model" in m.label for m in at.metric)
    assert len(at.tabs) == 7


def test_app_lockbox_view_and_alpha(results, monkeypatch):
    at = _app(results, monkeypatch).run()
    at.sidebar.radio[0].set_value("Lockbox (2023–2026)").run()
    assert not at.exception, at.exception
    at.sidebar.select_slider[0].set_value(0.01).run()
    assert not at.exception, at.exception
    assert any("Not replicated" in e.value for e in at.error)


def test_app_without_results_explains_what_to_do(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch).run()
    assert not at.exception
    assert any("No results found" in i.value for i in at.info)


def test_app_test_does_not_leak_main_module(tmp_path, monkeypatch):
    main = sys.modules["__main__"]
    with monkeypatch.context() as m:
        _app(tmp_path, m).run()
    assert sys.modules["__main__"] is main
