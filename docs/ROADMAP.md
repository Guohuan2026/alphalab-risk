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
5. GARCH(1,1)-t, GJR-GARCH-t, filtered HS, HAR (via `arch`).
6. `backtest/`: ✅ Kupiec, Christoffersen, Basel traffic light (done in week 1); DQ; tick and FZ0 losses;
   McNeil–Frey ES test; Diebold–Mariano; Model Confidence Set.
7. First full econometric results on P1, plus a write-up.

## Week 3 — Machine learning, stress and interpretability
8. Linear quantile regression (control), QRF, gradient boosting (quantile loss),
   ML-volatility + FHS hybrids; yearly nested tuning with an embargo.
9. Stress analysis: ex-ante episodes, VIX > 30 regime, loss-differential regressions.
10. SHAP and permutation importance; importance stability across refits.

## Week 4 — Robustness, dashboard, write-up
11. Robustness: rolling vs expanding window, P2 and P3, α = 2.5% *(stretch: Acerbi–Szekely Z2, h = 10)*.
12. Open the 2023–2026 lockbox **once**; report it as is.
13. Streamlit dashboard (reads saved results; no model fitting in the app).
14. Research report (proposal format for supervisor / Mitacs) and a clean-clone reproducibility run.

## Later / partner-dependent
- CRSP or partner P&L via the provider interface; multivariate models (DCC-GARCH, copulas);
  CAViaR; intraday realized volatility.
- Dependency locking (`uv lock`) and GitHub Actions CI.

## Archived (equity stock-selection plan, superseded 2026-11-06)
Not needed for the risk study: index/ETF portfolios replace stock-level universes. Code and
docs remain available at tag `v0.1-equity-research`.
- M2 parts 2–3: point-in-time stock universe, liquidity filters, S&P 500 constituents.
- M3 factor framework (momentum, reversal, low-vol, size); M4 factor evaluation (IC, quintiles);
  M5 long–short backtester; M6 factor tear sheets.
