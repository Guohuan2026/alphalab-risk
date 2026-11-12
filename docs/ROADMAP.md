# Roadmap — AlphaLab-Risk

Each step ends with passing tests and lint, updated docs, and a short write-up of what you
learned. Four weeks is ambitious. Items marked *(stretch)* are cut first if time runs short;
the **minimum viable study** is P1 (SPY), h = 1 day, VaR 1%/5%, ES 2.5%.

## Done (foundation, from AlphaLab M1–M2)
- Canonical price schema, Yahoo/synthetic providers, immutable hashed snapshots, validation.
- Dividend- and split-correct total returns, forward-built TRI, monthly compounding.
- Truncation test for look-ahead (`assert_point_in_time`).
- NYSE/TSX exchange calendars.

## Week 1 — Data foundation and first baselines
1. ✅ **Portfolio return series** (`portfolio/`): fixed weights, daily or monthly rebalancing,
   PIT-tested; configs for P1 (SPY), P2 (60/40 balanced), P3 (XIU.TO).
2. ✅ **Features** (`features/`, `providers/fred.py`): HAR realized variance, range-based
   variance, VIX, credit and term spreads (FRED, lagged one session); every feature PIT-tested.
   Synthetic GARCH-t generator with known true VaR/ES for model tests.
3. ✅ **Risk-model interface + forecast engine** (`risk/`): expanding/rolling windows, refit
   schedule, engine-level truncation test; HS and EWMA baselines.
4. Freeze the design: tag `prereg-v1`.

## Week 2 — Econometric models and backtesting
5. ✅ GARCH(1,1)-t, GJR-GARCH-t, filtered HS (via `arch`). HAR moves to week 3 with the
   feature-based models, since it needs the feature matrix.
6. ✅ `backtest/`: Kupiec, Christoffersen, Basel traffic light; tick and FZ0 losses; McNeil–Frey;
   Diebold–Mariano; Model Confidence Set; Engle–Manganelli DQ test (added 2026-11-07,
   development samples only).
7. ✅ First full econometric results on P1 (docs/RESULTS_LOG.md).

## Week 3 — Machine learning, stress and interpretability
8. ✅ Linear benchmarks: HAR-FHS and the H2b control `linqr_l1` (lasso QR, tuned yearly;
   Deviation 2). ✅ QRF, gradient boosting (quantile loss; XGBoost unavailable), RF-volatility
   + FHS hybrid, with the same yearly tuning (Deviation 3).
9. ✅ Stress analysis: ex-ante episodes, VIX > 30 regime, loss-differential regressions,
   reaction-speed event study (Clarification 4).
10. ✅ SHAP and permutation importance; importance stability across refits (Clarification 5).

## Week 4 — Robustness, dashboard, write-up
11. ✅ Robustness: rolling vs expanding window, P2 and P3, α = 2.5% (Clarification 6).
    ✅ Acerbi–Szekely Z2 (2026-11-08, Clarification 7; development only). ✅ h = 10,
    descriptive, √10-scaled 1-day forecasts (2026-11-08, Clarification 8; development only;
    `alphalab horizon`). ✅ Bayer–Dimitriadis ESR backtest (2026-11-10, Addition 9;
    exploratory, development only; `backtest_esr.csv`; covariance ported from and checked
    against the authors' esreg). Circular block bootstrap job (`alphalab esr-bootstrap`),
    chosen after a serial-dependence diagnostic; full run (~13 hours, 4 workers) started 2026-11-10.
12. ✅ Lockbox opened once (2026-11-07; protocol pushed first): the development headline was
    NOT replicated in the calm 2023–26 period (docs/RESULTS_LOG.md).
13. ✅ Streamlit dashboard (`make dashboard`; reads saved results; no model fitting in the app).
14. ✅ Research report draft v1 (`docs/REPORT.md`) and clean-clone reproducibility run (bit-identical).

15. ✅ ESR validation closed (2026-11-11, Addition 9, amendments 9–11): robust starts and the
    independence-covariance bootstrap both failed their pre-specified gates; ESR inference
    retired, statistics kept as diagnostics. Evidence classified in `docs/EVIDENCE_STATUS.md`.

## Next milestones (recommended 2026-11-11, in order)
16. **Calibration check for the provisional tests** (simulation only; no new data, no lockbox;
    acceptance rules written first). Size of DM (with Harvey–Leybourne–Newbold correction
    and fixed-b sensitivity), MCS (blocks 10/20/40) and the H3 HAC regression under
    persistent, heavy-tailed loss differentials calibrated to the real FZ0 series; size of
    McNeil–Frey and Z2 at the real numbers of hits and under a wrong tail shape; Monte Carlo
    p-values for 1% coverage/DQ. Outcome: each headline significance claim either moves to
    "supported" or is restated as descriptive.
    **Done 2026-11-12** (Addition 10): DM, MCS, H3, Z2 and most McNeil–Frey cells unsupported;
    CC supported everywhere, POF/IND/DQ in most cells (`docs/EVIDENCE_STATUS.md`). As frozen,
    fixed-b replaced the HLN correction (HLN scales the statistic by ≈ 1 at these T and cannot
    remove the distortion); Monte Carlo coverage/DQ p-values were recommended, not simulated.
17. **Research report v2** (`docs/REPORT.md`): every claim tagged by evidence category, with all
    DM/MCS/H3 "significant" wording restated as descriptive (Milestone 16);
    figures generated from saved results; a methods appendix on the ESR negative result
    (esreg port, the degenerate-start defect, the size failure under dependence); limitations
    and reproducibility sections; supervisor-ready PDF.
18. **Portfolio release:** README rewritten for a recruiter reader (question, answer, one
    figure, architecture, how to reproduce, CI badge); dashboard "evidence status" labels
    next to every p-value; clean-clone run; tagged release `v1.0`; owner decides on
    merging to `main` and on making the repository public.

## Later / partner-dependent
- CRSP or partner P&L via the provider interface; multivariate models (DCC-GARCH, copulas);
  CAViaR; intraday realized volatility.
- ✅ Dependency locking (`requirements-lock.txt`, 2026-11-07).
- ✅ GitHub Actions CI (`.github/workflows/tests.yml`: lint + offline tests with locked versions;
  `make ci` runs the same locally). Committed; runs on GitHub once pushed.

## Archived (equity stock-selection plan, superseded 2026-11-06)
Not needed for the risk study: index/ETF portfolios replace stock-level universes. Code and
docs remain available at tag `v0.1-equity-research`.
- M2 parts 2–3: point-in-time stock universe, liquidity filters, S&P 500 constituents.
- M3 factor framework (momentum, reversal, low-vol, size); M4 factor evaluation (IC, quintiles);
  M5 long–short backtester; M6 factor tear sheets.
