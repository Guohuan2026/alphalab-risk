# Data: sources, terms of use and how to obtain them

**No market data are distributed with this repository.** `data/raw/`, `data/processed/` and
`data/results/` are excluded by `.gitignore` and have never been committed (checked over the
full git history, 2026-11-12). Every download is stored locally as an immutable snapshot with
a SHA-256 manifest, so a given run's inputs can be identified exactly.

| Data | Source | Used for | Redistribution |
|---|---|---|---|
| SPY, EFA, IEF, LQD, XIU.TO daily prices and dividends; ^VIX; ^SP500TR (cross-check only) | Yahoo Finance, via the open-source `yfinance` library | Portfolio returns, range-based variance, VIX features | **Not redistributed.** Yahoo's terms restrict redistribution, and `yfinance` is an unofficial interface intended for personal and research use. Users download their own copy. |
| BAA10Y (Moody's Baa yield minus 10-year Treasury), T10Y3M (10-year minus 3-month Treasury) | FRED, Federal Reserve Bank of St. Louis | Credit- and term-spread features | **Not redistributed.** FRED series may carry third-party copyright (Moody's for the Baa yield); FRED's terms apply. |
| Synthetic prices | Generated in code (`provider: synthetic`) | Tests and the offline demo | Freely reproducible from the code |
| Simulation study outputs | Generated in code (`alphalab.backtest.calibration`) | Report §6.1 | Reproducible from the code and seed; not committed |

## How to obtain the data legitimately

1. Install the project (README, "Installation and usage").
2. Run `alphalab ingest --config configs/<config>.yaml` for each portfolio
   (`risk_p1_spy.yaml`, `risk_p2_balanced.yaml`, `risk_p3_xiu.yaml`). This downloads prices
   from Yahoo Finance and the two spreads from FRED for your own use, and saves a hashed
   snapshot under `data/raw/`.
3. Run `alphalab build`, then `forecast` and `backtest` with the same config.

**Reproducibility caveat.** Yahoo Finance revises history (dividends, splits, occasional
price corrections), so a new download may differ slightly from the snapshots behind the
logged results (P1 prices `20261106T222941Z_e325d363`, features `20261106T222947Z_d0d2e72e`).
With those original snapshots and `requirements-lock.txt`, the P1 forecasts and backtests
rebuild bit-for-bit (verified in fresh clones on 2026-11-07 and 2026-11-12;
`docs/RESULTS_LOG.md`). The snapshots
are not published; sharing them privately for verification (for example with a supervisor) is
possible only where the data providers' terms allow it.

## Licensed data (not used)

The provider interface has a documented CRSP stub. CRSP and similar licensed data must never
be committed; a partner project would keep such data on its own systems.
