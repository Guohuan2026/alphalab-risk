"""Command-line entry point: `alphalab <command> --config <yaml>`."""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from alphalab.config import load_config
from alphalab.data.store import RawSnapshotStore
from alphalab.pipeline import (
    ValidationFailed,
    run_backtest,
    run_build,
    run_forecast,
    run_ingest,
    run_ingest_features,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="alphalab", description="AlphaLab research pipeline")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    for name, help_ in [
        ("ingest", "download prices (and feature series) into immutable raw snapshots"),
        ("build", "validate a snapshot and build return datasets"),
        ("snapshots", "list saved snapshots for the configured provider"),
        ("forecast", "walk-forward VaR/ES forecasts for every model in the 'risk' section"),
        ("backtest", "coverage tests (Kupiec, Christoffersen, traffic light) on saved forecasts"),
    ]:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", required=True, help="path to YAML config")
        if name == "build":
            sp.add_argument("--snapshot", default="latest", help="snapshot id or 'latest'")
            sp.add_argument("--force", action="store_true", help="build despite validation errors")
        if name in ("forecast", "backtest"):
            sp.add_argument(
                "--include-lockbox",
                action="store_true",
                help="use the held-out period (do this ONCE, after models are frozen)",
            )

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)

    if args.command == "ingest":
        info = run_ingest(cfg)
        print(f"snapshot: {info.snapshot_id}\npath:     {info.path}")
        if cfg.features is not None:
            finfo = run_ingest_features(cfg)
            print(f"features: {finfo.snapshot_id}\npath:     {finfo.path}")
    elif args.command == "build":
        try:
            out = run_build(cfg, args.snapshot, force=args.force)
        except ValidationFailed as e:
            print(f"build stopped: {e}", file=sys.stderr)
            return 2
        print(f"processed: {out}")
    elif args.command == "forecast":
        out = run_forecast(cfg, include_lockbox=args.include_lockbox)
        print(f"results: {out}")
    elif args.command == "backtest":
        table = run_backtest(cfg, include_lockbox=args.include_lockbox)
        cols = ["model", "alpha", "n", "hits", "hit_rate", "p_pof", "p_ind", "p_cc"]
        with pd.option_context("display.width", 120, "display.float_format", "{:.4g}".format):
            print(table[cols].to_string(index=False))
    elif args.command == "snapshots":
        for s in RawSnapshotStore(cfg.data_dir).list(cfg.provider):
            print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
