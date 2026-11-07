# AlphaLab-Risk

**Machine Learning for Portfolio Risk Forecasting and Stress Resilience**

> Can machine-learning models improve the accuracy and robustness of portfolio downside-risk
> forecasts compared with traditional econometric methods, particularly during market stress?

A reproducible research platform in Python. It compares historical simulation, EWMA,
GARCH / GJR-GARCH and filtered HS with quantile-regression forests, gradient boosting and
ML-volatility hybrids on 1-day VaR and Expected Shortfall. Every model is backtested
out-of-sample (Kupiec, Christoffersen, DQ, ES tests, FZ0 loss, Diebold–Mariano, Model
Confidence Set) with a focus on stress periods.

- Research design, hypotheses and leakage safeguards: **[docs/RESEARCH_DESIGN.md](docs/RESEARCH_DESIGN.md)**
- Plan: **[docs/ROADMAP.md](docs/ROADMAP.md)**
- Results, in the order they were seen: **[docs/RESULTS_LOG.md](docs/RESULTS_LOG.md)**

### Dashboard

```bash
make dashboard        # = streamlit run dashboard/app.py
```

A read-only Streamlit app over the saved results (`data/results/`, so run the pipeline
first). Tabs: Overview (FZ0 ranking, MCS), Forecasts (VaR against realised losses, breaches),
Backtests (coverage, ES, Diebold–Mariano heatmap), Stress (breach rates by regime, reaction
speed), Explain (permutation importance, SHAP over time), Lockbox (development vs held-out),
About. It never fits a model: what it shows is exactly what the results log records.

**Status: Risk week 2 — design frozen (`prereg-v1`); HS, EWMA, GARCH-t, GJR-GARCH-t and FHS
backtested on SPY 1997–2022 (see [docs/RESULTS_LOG.md](docs/RESULTS_LOG.md)).** All 11 models
backtested on SPY 1997–2022: no ML model beats FHS-GJR; the RF-volatility + FHS hybrid ties
with it; direct-quantile ML (QRF, boosting) is worse even than the linear control in the
deep tail. Stress analysis (H3): FHS-GJR alone stays calibrated in crises; the ML hybrid
is better in calm markets but worse in stress. Interpretability (H4): ML relies on VIX and
range volatility, but signs and rankings are unstable, and trees saturate beyond their training
range (2008). Robustness (P2 60/40, P3 Canada, rolling window): FHS-GJR first in 11 of 12
cells; ML always loses ground in stress. **Lockbox 2023–26 (opened once): not replicated.**
In that calm period feature-based models had lower average loss (not significant on SPY),
while FHS-GJR stayed calibrated. Conclusion: regime-dependent, with FHS-GJR the robust
choice. Next: dashboard and write-up. The Python package and CLI are still called `alphalab`.

*History:* the project began as an equity stock-selection platform (AlphaLab). Its data,
returns, calendar and look-ahead infrastructure are reused unchanged. The pre-pivot state is
preserved at git tag `v0.1-equity-research`.

## Foundation reused from AlphaLab (M1–M2)

| Component | File | What it does |
|---|---|---|
| Canonical schema | `src/alphalab/data/schema.py` | One long format for every vendor: `date, security_id, close, dividend, split_ratio, …` |
| Providers | `src/alphalab/data/providers/` | `yahoo` (live), `synthetic` (offline, deterministic), `crsp` (documented stub) |
| Immutable snapshots | `src/alphalab/data/store.py` | Each download saved once with a SHA-256 hash and a provenance manifest |
| Validation | `src/alphalab/data/validation.py` | Errors block the build; warnings flag splits, gaps, vendor mismatches |
| Returns | `src/alphalab/returns/core.py` | Total/price returns from raw corporate actions, forward-built TRI, monthly compounding, forward/trailing returns |
| Look-ahead check | `src/alphalab/core/lookahead.py` | Truncation test: `f(data[:d])[d] == f(data)[d]` |
| Trading calendar (M2) | `src/alphalab/core/calendar.py` | NYSE sessions via `exchange_calendars`: gap checks, month completeness, synthetic dates |
| CLI | `src/alphalab/cli.py` | `alphalab ingest | build | forecast | backtest | snapshots --config …` |

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
