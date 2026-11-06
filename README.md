# AlphaLab

An educational, production-style quantitative investment research platform in Python.
Public data (Yahoo Finance) for the prototype; a provider interface designed so that
WRDS/CRSP can be plugged in later without touching return or factor code.

**Status: Milestone 2 in progress — exchange calendar done; point-in-time universe next.** No factors or backtests yet.

## What Milestone 1 gives you

| Component | File | What it does |
|---|---|---|
| Canonical schema | `src/alphalab/data/schema.py` | One long format for every vendor: `date, security_id, close, dividend, split_ratio, …` |
| Providers | `src/alphalab/data/providers/` | `yahoo` (live), `synthetic` (offline, deterministic), `crsp` (documented stub) |
| Immutable snapshots | `src/alphalab/data/store.py` | Each download saved once with a SHA-256 hash and a provenance manifest |
| Validation | `src/alphalab/data/validation.py` | Errors block the build; warnings flag splits, gaps, vendor mismatches |
| Returns | `src/alphalab/returns/core.py` | Total/price returns from raw corporate actions, forward-built TRI, monthly compounding, forward/trailing returns |
| Look-ahead check | `src/alphalab/core/lookahead.py` | Truncation test: `f(data[:d])[d] == f(data)[d]` |
| Trading calendar (M2) | `src/alphalab/core/calendar.py` | NYSE sessions via `exchange_calendars`: gap checks, month completeness, synthetic dates |
| CLI | `src/alphalab/cli.py` | `alphalab ingest | build | snapshots --config …` |

## Quick start (macOS)

Fastest: `bash scripts/setup_mac.sh` installs everything with `uv` and runs all checks
(log in `logs/setup_mac.log`). Full, explained manual walkthrough:
**[docs/SETUP_MACOS.md](docs/SETUP_MACOS.md)**.

```bash
cd ~/Projects/alphalab
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
make test          # offline unit tests
make synthetic     # pipeline end-to-end on synthetic data
make network       # live Yahoo test (needs internet)
make yahoo         # real data: ingest + build
```

## Pipeline

```
provider.fetch()  ──►  data/raw/<provider>/<snapshot_id>/        (immutable, hashed)
                         prices.parquet, manifest.json
                              │  alphalab build
                              ▼
                       validate ──(errors? stop)──►
                       data/processed/<provider>/<snapshot_id>/
                         returns_daily_long.parquet   long: date, security_id, ret, retx, div_yield
                         returns_daily.parquet        wide: date × security total returns
                         tri_daily.parquet            total-return index, forward-built from 1.0
                         returns_monthly.parquet      complete months only
                         validation_report.csv, build_manifest.json
```

## Read next
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md) — return math, adjusted prices, look-ahead rules
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md) — what this prototype cannot tell you
- [docs/ROADMAP.md](docs/ROADMAP.md) — milestones through factors, backtesting and CRSP
