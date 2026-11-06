# Roadmap

Each milestone ends with passing tests and a short write-up of what you learned.

## Week 1
- **M1 — Data pipeline & returns (done).** Schema, providers, snapshots,
  validation, returns, truncation test, CLI.
- **M2 — Universe & calendar.** ~~Exchange calendar (`exchange_calendars`)~~ (done), point-in-time
  universe object (membership by date), liquidity filters, a broader universe
  (e.g. S&P 500 constituents with an explicit survivorship warning).

## Week 2
- **M3 — Factor framework.** `Factor` interface (inputs at t → score at t), cross-sectional
  winsorising and z-scoring *per date*, momentum (12-1), short-term reversal,
  low volatility, size. Every factor gets an automatic `assert_point_in_time` test.
- **M4 — Factor evaluation.** Rank IC / IC decay, quintile portfolios, long-short spreads,
  Newey–West t-stats, turnover. Ties directly to Empirical Finance coursework.

## Week 3
- **M5 — Backtester.** Rebalance schedule, weights, transaction-cost model, performance
  and risk metrics (Sharpe, drawdown, beta, factor exposures vs Fama–French).
- **M6 — Reporting.** Reproducible research notebook template + HTML tear sheet.

## Later
- **CRSP via WRDS:** implement `CRSPProvider` (mapping already documented), cross-check
  our returns against `RET`, add delisting returns, PERMNO identity.
- Compustat fundamentals with reporting lags; Fama–French data loader.
- Dependency locking (`uv lock`), GitHub Actions CI running tests + ruff on every push.
