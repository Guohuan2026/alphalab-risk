# AlphaLab-Risk v1.0.0

*Released 2026-11-12. Package version 1.0.0; MIT License.*

AlphaLab-Risk compares eleven econometric and machine-learning models for one-day portfolio
Value-at-Risk and Expected Shortfall on three portfolios (SPY, a 60/40 ETF portfolio and
XIU.TO), with a pre-registered design, a 2023–2026 held-out lockbox opened once, and a frozen
simulation study of how reliable each backtest is on data like these.

## Highlights

- **Research report v2** (`docs/REPORT.md`; PDF `docs/AlphaLab-Risk_Report.pdf`): every claim labelled
  [Inference], [Descriptive], [Exploratory] or [Unsupported test].
- **Main results:**
  - The ML models' VaR forecasts fail Christoffersen's conditional-coverage test in 11–12 of
    12 development cells; FHS-GJR's conditional-coverage null is not rejected in 11 of 12
    [Inference].
  - Every ML model loses ground relative to FHS-GJR in crisis episodes; tree models saturate
    outside their training range (October 2008) [Descriptive].
  - Held-out lockbox: NOT REPLICATED under the pre-stated rule (gradient boosting had lower
    mean FZ0 in the calm 2023–2026 period) [Descriptive].
  - Diebold–Mariano, the Model Confidence Set and the stress regression over-reject badly
    under realistic persistence, so comparative findings are descriptive, not significant.
- **Negative results preserved:** failed calibration tests and the retired ES regression
  backtest, with designs, code and decisions.
- **Reproducible:** locked dependencies, hashed data snapshots, 341 offline tests, CI, and a
  fresh-clone rebuild of the P1 forecasts and backtests from the original snapshots.

## Assets

- `docs/AlphaLab-Risk_Report.pdf` (in the repository; can also be attached to the GitHub
  release): the research report.
- Dashboard screenshots: `docs/images/`.
- Source code (automatic GitHub archive), MIT License. No market data are included
  (`docs/DATA.md`); the licence does not cover third-party data.

## Known limitations

See report §8 and `docs/LIMITATIONS.md`. In particular: comparison tests are mis-sized, so
comparative findings are descriptive; calibration "support" holds within the simulated
designs; the lockbox has low power at 99%; univariate portfolio models; public data.

## Upgrading / reproducing

`uv pip install -e ".[dev]" -c requirements-lock.txt`, then `make ci`. Real data must be
downloaded by the user (`docs/DATA.md`); new downloads may differ slightly from the original
snapshots because Yahoo Finance revises history.
