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
    run_explain,
    run_forecast,
    run_horizon,
    run_ingest,
    run_ingest_features,
    run_stress,
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
        ("stress", "H3: performance by stress regime, DM/HAC tests vs the benchmark, event study"),
        ("horizon", "descriptive 10-day table: sqrt(10)-scaled 1-day forecasts (development)"),
        (
            "explain",
            "H4: permutation importance, SHAP and the Clarification-5 criteria (ML models)",
        ),
        (
            "esr-bootstrap",
            "resumable bootstrap calibration of the ESR tests (development; Addition 9)",
        ),
        ("esr-sim", "Monte Carlo size of the ESR bootstrap tests under dependence (simulated)"),
    ]:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", required=True, help="path to YAML config")
        if name == "build":
            sp.add_argument("--snapshot", default="latest", help="snapshot id or 'latest'")
            sp.add_argument("--force", action="store_true", help="build despite validation errors")
        if name == "esr-bootstrap":
            from alphalab.backtest.esr_bootstrap import BootstrapSettings as _BS

            d = _BS()
            sp.add_argument("--reps", type=int, default=d.reps, help="bootstrap resamples")
            sp.add_argument("--chunk", type=int, default=d.chunk, help="resamples per chunk")
            sp.add_argument("--seed", type=int, default=d.seed)
            sp.add_argument("--workers", type=int, default=d.workers, help="processes")
            sp.add_argument("--nice", type=int, default=d.nice, help="CPU priority increment")
            sp.add_argument("--max-hours", type=float, default=None, help="wall-clock budget")
            sp.add_argument(
                "--block",
                type=int,
                default=None,
                help="circular block length (default ceil(T^(1/3)); 1 = iid, not valid here); "
                "a non-default length writes to its own folder esr_bootstrap_block<b>",
            )
            sp.add_argument(
                "--only", default=None, help="regex on cell keys, e.g. '_0.025$' (sensitivity runs)"
            )
            sp.add_argument(
                "--legacy-esreg-start",
                action="store_true",
                help="esreg's location-scale start (degenerate in some draws): ONLY to resume or "
                "summarise the legacy run; new runs use robust starts (Addition 9, amendment 9)",
            )
            mode = sp.add_mutually_exclusive_group()
            mode.add_argument("--plan", action="store_true", help="estimate only; no work")
            mode.add_argument("--summarize", action="store_true", help="write p-values")
            mode.add_argument(
                "--diagnose", action="store_true", help="serial dependence of the ESR scores"
            )
            mode.add_argument(
                "--screen", action="store_true", help="degenerate-start check and quarantine"
            )
            mode.add_argument(
                "--audit-degenerate",
                action="store_true",
                help="count saved draws that used esreg's degenerate start (no refitting)",
            )
        if name == "esr-sim":
            from alphalab.backtest.esr_sim import SimSettings as _SS

            d = _SS()
            sp.add_argument("--n", type=int, default=d.n, help="sample length")
            sp.add_argument("--reps", type=int, default=d.reps, help="Monte Carlo replications")
            sp.add_argument("--workers", type=int, default=d.workers, help="processes")
            sp.add_argument(
                "--cov",
                default=d.cov,
                choices=["scl_sp", "ind_nomis", "ind_mis"],
                help="ESR covariance variant (amendment 10; default = pre-registered scl_sp)",
            )
            sp.add_argument("--summarize", action="store_true", help="write rejection rates")
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
        t = run_backtest(cfg, include_lockbox=args.include_lockbox)
        views = {
            "coverage": ["model", "alpha", "n", "hits", "hit_rate", "p_pof", "p_ind", "p_cc"],
            "es": ["model", "alpha", "n_hits", "severity_ratio", "p_raw", "p_std"],
            "losses": ["model", "alpha", "mean_fz0", "rank_fz0", "mcs_p_fz0", "in_mcs_fz0"],
        }
        with pd.option_context("display.width", 120, "display.float_format", "{:.4g}".format):
            for key, cols in views.items():
                print(f"\n== {key} ==")
                print(t[key][cols].to_string(index=False))
    elif args.command == "stress":
        t = run_stress(cfg)
        with pd.option_context("display.width", 120, "display.float_format", "{:.4g}".format):
            print(t["regression"].to_string(index=False))
            print(t["event_summary"].to_string(index=False))
    elif args.command == "horizon":
        t = run_horizon(cfg)
        cols = ["model", "alpha", "windows", "breaches", "expected", "severity_ratio", "rank_fz0"]
        with pd.option_context("display.width", 120, "display.float_format", "{:.4g}".format):
            print(t[cols].to_string(index=False))
    elif args.command == "explain":
        t = run_explain(cfg)
        with pd.option_context("display.width", 140, "display.max_colwidth", 80):
            print(t["criteria"].to_string(index=False))
            print(t["check"].to_string(index=False))
    elif args.command == "esr-bootstrap":
        import json
        from pathlib import Path

        from alphalab.backtest import esr_bootstrap as eb

        s = eb.BootstrapSettings(
            reps=args.reps,
            chunk=args.chunk,
            seed=args.seed,
            workers=args.workers,
            nice=args.nice,
            max_hours=args.max_hours,
            block=args.block,
            robust_scale=not args.legacy_esreg_start,
            allow_legacy_start=args.legacy_esreg_start,
        )
        if args.audit_degenerate:  # raw inputs only: no fitting, so it is quick
            t = eb.audit_degenerate(
                eb.root_for(eb.cells_root(cfg), args.block, robust=s.robust_scale),
                eb.series_from_config(cfg),
            )
            out = (
                eb.cells_root(cfg).parent
                / f"esr_degenerate_audit_{'robust' if s.robust_scale else 'legacy'}.csv"
            )
            t.to_csv(out, index=False)
            print(t.to_string(index=False))
            print(f"wrote {out}")
            return 0
        root, cells = eb.cells_from_config(cfg, robust=s.robust_scale)
        root = eb.root_for(root, args.block, robust=s.robust_scale)
        if args.only:
            import re

            cells = [c for c in cells if re.search(args.only, c.key)]
        if args.screen:
            t = eb.screen(cells)
            out = root.parent / "backtest_esr_screen.csv"
            t.to_csv(out, index=False)
            print(t.drop(columns="reason").to_string(index=False))
            print(f"wrote {out}")
        elif args.diagnose:
            t = pd.DataFrame([eb.score_dependence(c) for c in cells])
            out = root.parent / "backtest_esr_dependence.csv"
            t.to_csv(out, index=False)
            print(t.round(4).to_string(index=False))
            print(f"wrote {out}")
        elif args.summarize:
            t = eb.summarize(root, s, cells)
            out = root.parent / f"backtest_{root.name}.csv"
            t.to_csv(out, index=False)
            with pd.option_context("display.width", 140, "display.float_format", "{:.4g}".format):
                print(t.to_string(index=False))
            print(f"wrote {out}")
        elif args.plan:
            print(json.dumps(eb.plan(root, cells, s), indent=2))
        else:
            run_name = Path(args.config).stem
            print(
                json.dumps(eb.run(root, cells, s, Path("logs") / f"esr_bootstrap_{run_name}.log"))
            )
    elif args.command == "esr-sim":
        from pathlib import Path

        from alphalab.backtest import esr_sim as es_

        s = es_.SimSettings(n=args.n, reps=args.reps, workers=args.workers, cov=args.cov)
        tag = "" if s.cov == "scl_sp" else f"_cov-{s.cov}"  # separate folder per variant
        root = Path(cfg.data_dir) / "results" / "esr_sim" / f"n{s.n}_reps{s.reps}_seed{s.seed}{tag}"
        if args.summarize:
            t = es_.summarize(root, s)
            t.to_csv(root / "rejection_rates.csv", index=False)
            print(t.round(4).to_string(index=False))
            print(f"wrote {root / 'rejection_rates.csv'}")
        else:
            print(es_.run(root, s, Path("logs") / f"esr_sim_{root.name}.log"))
    elif args.command == "snapshots":
        for s in RawSnapshotStore(cfg.data_dir).list(cfg.provider):
            print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
