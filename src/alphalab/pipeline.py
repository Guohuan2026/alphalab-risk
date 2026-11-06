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

import alphalab
from alphalab.config import Config
from alphalab.data.providers import get_provider
from alphalab.data.store import RawSnapshotStore, SnapshotInfo
from alphalab.data.validation import validate_prices
from alphalab.portfolio import portfolio_returns
from alphalab.returns.core import (
    compound_returns,
    compute_daily_returns,
    to_wide,
    total_return_index,
)

log = logging.getLogger(__name__)


class ValidationFailed(RuntimeError):
    pass


def snapshot_request(cfg: Config) -> dict:
    """What was asked of the provider; stored in each snapshot and used to find it again."""
    return {"tickers": cfg.universe, "start": cfg.start, "end": cfg.end}


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
                "outputs": sorted(p.name for p in out.iterdir()),
            },
            indent=2,
        )
    )
    log.info("wrote %s", out)
    return out
