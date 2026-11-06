"""Command-line entry point: `alphalab <command> --config <yaml>`."""

from __future__ import annotations

import argparse
import logging
import sys

from alphalab.config import load_config
from alphalab.data.store import RawSnapshotStore
from alphalab.pipeline import ValidationFailed, run_build, run_ingest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="alphalab", description="AlphaLab research pipeline")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    for name, help_ in [
        ("ingest", "download prices and save an immutable raw snapshot"),
        ("build", "validate a snapshot and build return datasets"),
        ("snapshots", "list saved snapshots for the configured provider"),
    ]:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", required=True, help="path to YAML config")
        if name == "build":
            sp.add_argument("--snapshot", default="latest", help="snapshot id or 'latest'")
            sp.add_argument("--force", action="store_true", help="build despite validation errors")

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)

    if args.command == "ingest":
        info = run_ingest(cfg)
        print(f"snapshot: {info.snapshot_id}\npath:     {info.path}")
    elif args.command == "build":
        try:
            out = run_build(cfg, args.snapshot, force=args.force)
        except ValidationFailed as e:
            print(f"build stopped: {e}", file=sys.stderr)
            return 2
        print(f"processed: {out}")
    elif args.command == "snapshots":
        for s in RawSnapshotStore(cfg.data_dir).list(cfg.provider):
            print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
