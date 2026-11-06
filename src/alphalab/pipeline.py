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

import numpy as np
import pandas as pd

import alphalab
from alphalab.backtest import coverage_tests, traffic_light
from alphalab.config import Config
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
    frames, per_model = [], {}
    for spec in cfg.risk.models:
        name = cfg.risk.model(spec).name
        f = run_forecasts(ret, lambda spec=spec: cfg.risk.model(spec), settings, include_lockbox)
        frames.append(f.assign(model=name))
        per_model[name] = {
            "spec": spec,
            "n": len(f),
            "first_origin": str(f.index[0].date()) if len(f) else None,
            "last_origin": str(f.index[-1].date()) if len(f) else None,
        }
    out = _results_dir(cfg, proc)
    sfx = _suffix(include_lockbox)
    pd.concat(frames).to_parquet(out / f"forecasts{sfx}.parquet")
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


def run_backtest(cfg: Config, include_lockbox: bool = False) -> pd.DataFrame:
    """Coverage tests per model and alpha -> backtest_coverage[_lockbox].csv.

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
    rows = []
    for name, f in fc.groupby("model", sort=False):
        ev = attach_outcomes(f.drop(columns="model"), ret)
        if include_lockbox:
            ev = ev[ev["target_date"] >= pd.Timestamp(cfg.risk.lockbox_start)]
        for a in cfg.risk.alphas:
            hits = (ev["loss_next"] > ev[f"var_{a:g}"]).to_numpy()
            row = {"model": name, **coverage_tests(hits, a).to_dict()}
            row["first_target"] = str(ev["target_date"].iloc[0].date())
            row["last_target"] = str(ev["target_date"].iloc[-1].date())
            if np.isclose(a, 0.01):
                row |= {f"tl_{k}": v for k, v in traffic_light(hits).items()}
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / f"backtest_coverage{sfx}.csv", index=False)
    log.info("wrote %s", out / f"backtest_coverage{sfx}.csv")
    return table
