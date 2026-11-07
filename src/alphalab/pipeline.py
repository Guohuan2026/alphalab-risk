"""Two-stage pipeline: ingest (download -> immutable raw snapshot) and
build (snapshot -> validated, derived return datasets).

Separating the stages means the expensive, non-reproducible step (downloading)
happens once, and every derived dataset records exactly which snapshot it came from.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

import alphalab
from alphalab.backtest.report import evaluate
from alphalab.config import Config
from alphalab.data.corrections import apply_corrections
from alphalab.data.providers import get_provider
from alphalab.data.providers.fred import FredSource
from alphalab.data.providers.yahoo import YahooProvider
from alphalab.data.schema import check_series_schema
from alphalab.data.store import RawSnapshotStore, SnapshotInfo
from alphalab.data.validation import validate_prices
from alphalab.features import assemble_inputs, build_features, series_name
from alphalab.portfolio import portfolio_returns
from alphalab.returns.core import (
    compound_returns,
    compute_daily_returns,
    to_wide,
    total_return_index,
)
from alphalab.risk.engine import attach_outcomes, run_forecasts

log = logging.getLogger(__name__)


class ValidationFailed(RuntimeError):
    pass


def snapshot_request(cfg: Config) -> dict:
    """What was asked of the provider; stored in each snapshot and used to find it again."""
    return {"tickers": cfg.universe, "start": cfg.start, "end": cfg.end}


def features_request(cfg: Config) -> dict:
    f = cfg.features
    return {"yahoo_levels": f.yahoo_levels, "fred": f.fred, "start": cfg.start, "end": cfg.end}


def run_ingest_features(cfg: Config) -> SnapshotInfo:
    """Download feature series (VIX from Yahoo, spreads from FRED) into ONE immutable
    snapshot under data/raw/features/, in the long series format."""
    f = cfg.features
    parts, sources = [], {}
    if f.yahoo_levels:
        yahoo = YahooProvider(**(cfg.provider_options if cfg.provider == "yahoo" else {}))
        px = yahoo.fetch(f.yahoo_levels, cfg.start, cfg.end)
        missing = set(f.yahoo_levels) - (set(px["security_id"]) if len(px) else set())
        if missing:
            raise RuntimeError(f"yahoo returned no data for {sorted(missing)}; nothing saved")
        parts.append(px[["date", "security_id", "close"]].rename(columns={"close": "value"}))
        sources["yahoo"] = yahoo.metadata()
    if f.fred:
        fred = FredSource()
        parts.append(fred.fetch(f.fred, cfg.start, cfg.end))
        sources["fred"] = fred.metadata()
    df = pd.concat(parts, ignore_index=True)
    df["security_id"] = df["security_id"].astype("string")
    df = df.sort_values(["security_id", "date"], kind="mergesort").reset_index(drop=True)
    meta = {"provider": "features", "price_basis": "n/a (levels)", "sources": sources}
    store = RawSnapshotStore(cfg.data_dir)
    info = store.save(df, meta, features_request(cfg), validate=check_series_schema)
    log.info("saved feature snapshot %s (%d rows)", info.snapshot_id, len(df))
    return info


def run_ingest(cfg: Config) -> SnapshotInfo:
    provider = get_provider(cfg.provider, **cfg.provider_options)
    log.info(
        "fetching %d tickers from %s (%s to %s)",
        len(cfg.universe),
        cfg.provider,
        cfg.start,
        cfg.end,
    )
    df = provider.fetch(cfg.universe, cfg.start, cfg.end)
    if df.empty:
        raise RuntimeError("provider returned no data; nothing saved")
    info = RawSnapshotStore(cfg.data_dir).save(df, provider.metadata(), snapshot_request(cfg))
    log.info("saved snapshot %s (%d rows)", info.snapshot_id, len(df))
    if info.manifest["securities_missing"]:
        log.warning("missing securities: %s", info.manifest["securities_missing"])
    return info


def run_build(cfg: Config, snapshot_id: str = "latest", force: bool = False) -> Path:
    store = RawSnapshotStore(cfg.data_dir)
    if snapshot_id == "latest":
        snapshot_id = store.latest(cfg.provider, snapshot_request(cfg))
    prices, manifest = store.load(cfg.provider, snapshot_id)
    basis = manifest["provider"]["price_basis"]
    prices, corrections_log = apply_corrections(prices, cfg.corrections)
    for c in corrections_log:
        log.info("correction: %s", c)

    report = validate_prices(prices, basis, calendar=cfg.calendar, **asdict(cfg.validation))
    log.info(report.summary())
    if not report.ok and not force:
        raise ValidationFailed(f"{len(report.errors)} validation error(s); use --force to override")

    out = cfg.data_dir / "processed" / cfg.provider / snapshot_id
    out.mkdir(parents=True, exist_ok=True)

    daily_long = compute_daily_returns(prices, basis)
    daily_wide = to_wide(daily_long, "ret")
    monthly = compound_returns(daily_wide, "ME", min_obs=cfg.monthly_min_obs, calendar=cfg.calendar)

    daily_long.to_parquet(out / "returns_daily_long.parquet", index=False)
    daily_wide.to_parquet(out / "returns_daily.parquet")
    total_return_index(daily_long).to_parquet(out / "tri_daily.parquet")
    monthly.to_parquet(out / "returns_monthly.parquet")
    port_info = None
    if cfg.portfolio is not None:
        port = portfolio_returns(daily_wide, cfg.portfolio.weights, cfg.portfolio.rebalance)
        port.to_parquet(out / "portfolio_daily.parquet")
        port_info = {
            **asdict(cfg.portfolio),
            "n_days": len(port),
            "first_date": str(port.index[0].date()),
            "last_date": str(port.index[-1].date()),
        }
        log.info(
            "portfolio %s: %d days from %s", cfg.portfolio.name, len(port), port_info["first_date"]
        )
    feat_info = None
    if cfg.features is not None:  # config guarantees a portfolio exists
        feat_info = _build_features(cfg, store, prices, port["ret"], out)
    report.to_frame().to_csv(out / "validation_report.csv", index=False)
    (out / "build_manifest.json").write_text(
        json.dumps(
            {
                "built_utc": datetime.now(UTC).isoformat(),
                "alphalab_version": alphalab.__version__,
                "source_snapshot": snapshot_id,
                "source_sha256": manifest["content_sha256"],
                "price_basis": basis,
                "validation_errors": len(report.errors),
                "validation_warnings": len(report.warnings),
                "forced": force,
                "monthly_min_obs": cfg.monthly_min_obs,
                "calendar": cfg.calendar,
                "portfolio": port_info,
                "corrections_applied": corrections_log,
                "features": feat_info,
                "outputs": sorted(p.name for p in out.iterdir()),
            },
            indent=2,
        )
    )
    log.info("wrote %s", out)
    return out


def _build_features(cfg: Config, store: RawSnapshotStore, prices, port_ret, out: Path) -> dict:
    f = cfg.features
    fsnap = store.latest("features", features_request(cfg))
    series, fman = store.load("features", fsnap)
    ohlc = None
    if f.range_asset is not None:
        a = prices[prices["security_id"] == f.range_asset].set_index("date")
        ohlc = a[["open", "high", "low", "close"]]
    inputs = assemble_inputs(
        port_ret, series, f.yahoo_levels, f.fred, f.fred_lag_sessions, ohlc, cfg.calendar
    )
    feats = build_features(
        inputs,
        log_change_levels=[series_name(s) for s in f.yahoo_levels],
        diff_levels=[series_name(s) for s in f.fred],
    )
    feats.to_parquet(out / "features_daily.parquet")
    complete = feats.dropna()
    first = str(complete.index[0].date()) if len(complete) else None
    log.info("features: %d columns, complete rows from %s", feats.shape[1], first)
    return {
        **asdict(f),
        "snapshot": fsnap,
        "snapshot_sha256": fman["content_sha256"],
        "columns": list(feats.columns),
        "first_complete_date": first,
        "n_complete_rows": len(complete),
        "nan_share_by_column": feats.isna().mean().round(4).to_dict(),
    }


# ---------- risk forecasting and backtesting ----------
def _processed_dir(cfg: Config) -> Path:
    snap = RawSnapshotStore(cfg.data_dir).latest(cfg.provider, snapshot_request(cfg))
    proc = cfg.data_dir / "processed" / cfg.provider / snap
    if not (proc / "portfolio_daily.parquet").exists():
        raise FileNotFoundError(f"no portfolio_daily.parquet in {proc}; run build first")
    return proc


def _results_dir(cfg: Config, proc: Path) -> Path:
    out = cfg.data_dir / "results" / cfg.portfolio.name / proc.name
    if cfg.risk is not None and cfg.risk.run_tag:
        out = out / cfg.risk.run_tag  # robustness runs never overwrite the main results
    out.mkdir(parents=True, exist_ok=True)
    return out


def _suffix(include_lockbox: bool) -> str:
    return "_lockbox" if include_lockbox else ""


def run_forecast(cfg: Config, include_lockbox: bool = False) -> Path:
    """Walk-forward forecasts for every configured model -> forecasts[_lockbox].parquet."""
    if cfg.risk is None:
        raise ValueError("config has no 'risk' section")
    proc = _processed_dir(cfg)
    ret = pd.read_parquet(proc / "portfolio_daily.parquet")["ret"]
    if include_lockbox:
        log.warning(
            "LOCKBOX OPENED: forecasting into the held-out period (%s on)", cfg.risk.lockbox_start
        )
    settings = cfg.risk.engine()
    features = None
    if any(cfg.risk.model(spec).uses_features for spec in cfg.risk.models):
        fpath = proc / "features_daily.parquet"
        if not fpath.exists():
            raise FileNotFoundError(f"{fpath} not found; add a 'features' section and rebuild")
        features = pd.read_parquet(fpath)
    frames, per_model, fit_rows = [], {}, []
    for spec in cfg.risk.models:
        name = cfg.risk.model(spec).name
        fits: list[dict] = []
        f = run_forecasts(
            ret,
            lambda spec=spec: cfg.risk.model(spec),
            settings,
            include_lockbox=include_lockbox,
            fits=fits,
            features=features,
        )
        frames.append(f.assign(model=name))
        fit_rows += [{"model": name, **row} for row in fits]
        per_model[name] = {
            "spec": spec,
            "n": len(f),
            "first_origin": str(f.index[0].date()) if len(f) else None,
            "last_origin": str(f.index[-1].date()) if len(f) else None,
        }
    out = _results_dir(cfg, proc)
    sfx = _suffix(include_lockbox)
    pd.concat(frames).to_parquet(out / f"forecasts{sfx}.parquet")
    if fit_rows:  # parameter paths (GARCH family): one row per model per refit
        pd.DataFrame(fit_rows).to_parquet(out / f"fit_params{sfx}.parquet", index=False)
    (out / f"forecast_manifest{sfx}.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "alphalab_version": alphalab.__version__,
                "processed": str(proc),
                "risk": asdict(cfg.risk),
                "lockbox_opened": include_lockbox,
                "models": per_model,
            },
            indent=2,
            default=str,
        )
    )
    log.info("wrote %s", out / f"forecasts{sfx}.parquet")
    return out


def run_backtest(cfg: Config, include_lockbox: bool = False) -> dict[str, pd.DataFrame]:
    """All backtest tables -> backtest_{coverage,es,losses,dm}[_lockbox].csv.

    Without the lockbox flag: the development sample. With it: ONLY the lockbox period,
    reported separately so the two are never blended.
    """
    proc = _processed_dir(cfg)
    out = _results_dir(cfg, proc)
    sfx = _suffix(include_lockbox)
    path = out / f"forecasts{sfx}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run forecast first")
    fc = pd.read_parquet(path)
    ret = pd.read_parquet(proc / "portfolio_daily.parquet")["ret"]
    evs = {}
    for name, f in fc.groupby("model", sort=False):
        ev = attach_outcomes(f.drop(columns="model"), ret)
        if include_lockbox:
            ev = ev[ev["target_date"] >= pd.Timestamp(cfg.risk.lockbox_start)]
        evs[name] = ev
    tables = evaluate(evs, cfg.risk.alphas)
    for key, table in tables.items():
        table.to_csv(out / f"backtest_{key}{sfx}.csv", index=False)
    log.info("wrote backtest tables to %s", out)
    return tables


def run_stress(cfg: Config) -> dict[str, pd.DataFrame]:
    """H3 tables on the development sample -> stress_*.csv (rules: Clarification 4)."""
    from alphalab.stress import event_study, stress_tables

    if cfg.risk is None or cfg.risk.benchmark is None:
        raise ValueError("set risk.benchmark (e.g. fhs_gjr) to run the stress analysis")
    proc = _processed_dir(cfg)
    out = _results_dir(cfg, proc)
    path = out / "forecasts.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run forecast first")
    fc = pd.read_parquet(path)
    ret = pd.read_parquet(proc / "portfolio_daily.parquet")["ret"]
    evs = {
        name: attach_outcomes(f.drop(columns="model"), ret)
        for name, f in fc.groupby("model", sort=False)
    }
    vix = None
    fpath = proc / "features_daily.parquet"
    if fpath.exists():
        feats = pd.read_parquet(fpath)
        vix = feats.get("vix")  # None when the portfolio has no VIX feature
    tables = stress_tables(evs, cfg.risk.alphas, vix=vix, benchmark=cfg.risk.benchmark)
    paths, summary = event_study(evs, alpha=min(cfg.risk.alphas))
    tables["event_paths"] = paths.reset_index()
    tables["event_summary"] = summary
    for key, table in tables.items():
        table.to_csv(out / f"stress_{key}.csv", index=False)
    log.info("wrote stress tables to %s", out)
    return tables


ML_TYPES = ("qrf", "gbq", "mlvol")


def run_explain(cfg: Config) -> dict[str, pd.DataFrame]:
    """H4 interpretability for the ML models -> explain_*.csv / .parquet (Clarification 5)."""
    from alphalab.explain import explain_model, h4_criteria

    if cfg.risk is None:
        raise ValueError("config has no 'risk' section")
    specs = [s for s in cfg.risk.models if s.get("type") in ML_TYPES]
    if not specs:
        raise ValueError(f"no ML models ({', '.join(ML_TYPES)}) in risk.models")
    proc = _processed_dir(cfg)
    out = _results_dir(cfg, proc)
    ret = pd.read_parquet(proc / "portfolio_daily.parquet")["ret"]
    feats = pd.read_parquet(proc / "features_daily.parquet")
    saved = None
    if (out / "forecasts.parquet").exists():
        saved = pd.read_parquet(out / "forecasts.parquet")
    perm_all, crit_all, sign_all, checks = [], [], [], []
    settings = cfg.risk.engine()
    for spec in specs:
        name = cfg.risk.model(spec).name
        ex = explain_model(ret, feats, lambda spec=spec: cfg.risk.model(spec), settings)
        perm_all.append(ex["permutation"].assign(model=name))
        shap = ex.get("shap")
        crit, signs = h4_criteria(ex["permutation"], shap, feats)
        crit_all.append(crit.assign(model=name))
        sign_all.append(signs.assign(model=name))
        if shap is not None:
            shap.to_parquet(out / f"explain_shap_{name}.parquet")
            if saved is not None:  # do the explained models reproduce the forecasts?
                a = min(cfg.risk.alphas)
                fc = saved[saved["model"] == name][f"var_{a:g}"].reindex(shap.index)
                diff = (shap["var_shap"] - fc).abs()
                checks.append(
                    {
                        "model": name,
                        "n": len(diff),
                        "share_identical": float((diff < 1e-10).mean()),
                        "max_abs_diff": float(diff.max()),
                        "note": "differences = quantile-crossing fix (gbq) or floor",
                    }
                )
    tables = {
        "permutation": pd.concat(perm_all, ignore_index=True),
        "criteria": pd.concat(crit_all, ignore_index=True),
        "signs": pd.concat(sign_all, ignore_index=True),
        "check": pd.DataFrame(checks),
    }
    for key, table in tables.items():
        table.to_csv(out / f"explain_{key}.csv", index=False)
    log.info("wrote explain tables to %s", out)
    return tables
