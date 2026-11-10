# Known limitations

Research-design limitations of the risk study (statistical power, univariate portfolio
approach, tuning budget, ...) are listed in [RESEARCH_DESIGN.md §9](RESEARCH_DESIGN.md).
This file lists limitations of the **code and data as built**, ordered roughly by how much
they can distort conclusions.

1. **Survivorship bias.** Severe for the archived stock-selection universe
   (`configs/yahoo_us_core.yaml`, a hand-picked list of today's large caps). Much smaller
   for the risk study, which uses indices and ETFs that did not delist. It remains in the
   sense that we study markets that *survived and grew* (US, Canada). A full fix for
   stock-level work is CRSP with point-in-time membership and delisting returns (`DLRET`).
2. **Unofficial data source.** `yfinance` scrapes an undocumented Yahoo endpoint. It can
   rate-limit, change format, or return errors without notice. Treat it as a teaching
   and prototyping source, never as ground truth.
3. **Vendor restatements.** Yahoo re-adjusts history after every new split, and data can
   be silently corrected. Immutable snapshots make *your* runs reproducible, but two
   snapshots taken months apart may legitimately differ.
4. **Ticker ≠ identity.** For Yahoo, `security_id` is the ticker. Ticker changes, reuse
   and mergers are not handled. CRSP `PERMNO` solves this.
5. **Close-to-close, no costs.** Returns assume you trade at the close with no spread,
   commission, slippage or market impact. Fine for factor research; insufficient for
   strategy P&L.
6. **Currency.** No FX handling. Mixing USD and CAD listings (e.g. `RY.TO`) would mix
   currencies. Keep a single-currency universe until FX conversion is added.
7. **One calendar per run.** Gap checks, month completeness and synthetic dates use a
   single exchange calendar (`calendar:` in the config, default NYSE `XNYS`). A universe
   mixing exchanges with different holidays (e.g. NYSE + Toronto) will see false gaps or
   `non_session_date` warnings for the other exchange. The calendar also includes
   unscheduled closures (Hurricane Sandy 2012-10-29, 2025-01-09) that were only known
   in hindsight; this affects bookkeeping only, never signal values. It covers
   1925-01-01 to about one year ahead; dates outside that range are not supported.
   *(M1's Mon–Fri calendar, which wrongly dropped months ending on Good Friday, was
   replaced in M2.)*
8. **Dividends only as cash on ex-date.** Special dividends, return of capital, spin-offs
   and rights issues are not modelled separately. Spin-offs in particular can appear as
   large negative price-only days.
9. **No intraday or point-in-time fundamentals** yet; earnings/book data come later.
10. **Live path depends on the network.** The Yahoo download path is covered by an
    offline normalisation test; the live test (`pytest -m network`) passed on the
    owner's Mac on 2026-11-06 but can break whenever Yahoo changes its API (see #2).
11. **SPY vs S&P 500 TR index, before 2009 (open).** Our SPY total return tracks `^SP500TR`
    to within 0.13%/yr over 1993–2026 (about the expense ratio), and daily correlation is
    about 0.999 from 2009 on. But in 1993–2008 daily correlation is only 0.94–0.98, with up
    to 59 days a year differing by more than 0.5%. Cause not yet established (closing-time
    differences or stale index prints are candidates). Because daily risk forecasts are
    sensitive to this, P1 models will also be run on `^SP500TR` as a robustness check.
12. **XIU.TO year-end reinvested distributions (resolved 2026-11-06, Clarification 6: 3 dates
    corrected at build).** Some December
    "dividends" (e.g. $0.46 on 2000-12-27, 3.5% of price) come with no price drop. That is
    typical of reinvested (non-cash) distributions with unit consolidation. Treating them
    as cash creates fake gains (+3.8% that day). Must be resolved, for example by
    cross-checking against Yahoo's adjusted close or iShares distribution records, before
    P3 results are used. Related: Yahoo shows XIU trading on 2001-09-11 (a partial TSX
    session before the halt), which `exchange_calendars` marks closed; the row is kept and
    flagged as `non_session_date`.
13. **Feature timing and proxies.** FRED data are the *latest vintage*. That is harmless for
    market yields, which are not revised, but not point-in-time for macro series (those
    would need ALFRED vintages). VIX settles at 4:15 pm ET, after the equity close: fine
    for an end-of-day forecast, not for one produced at 4:00 pm. Range-based variance
    ignores the overnight gap, so it understates total variance. P3 (Canada) uses US
    stress indicators (VIX, US spreads) as global proxies.
14. **Quantile forest implementation detail and library dependence (found 2026-11-07 by the
    clean-clone check).** All QRF results used `quantile-forest` 1.4.2 with its defaults:
    `max_samples_leaf=1` (each leaf keeps ONE training value, so a forecast distribution
    has ~300 values, one per tree) and linear quantile interpolation. That is coarser than
    Meinshausen's original QRF, which keeps every leaf value, and may contribute to the QRF's
    poor deep-tail results; the model was not changed after the fact (results are frozen).
    Version 1.5.0 gives materially different quantiles even with these settings pinned
    (about 20–40% at the 5% level in a test case), so QRF results reproduce only with the
    locked versions in `requirements-lock.txt`, which the setup script installs.
    Even with locked versions, QRF output is not bit-identical across machines: a small test
    forest gave a different FZ0 on a GitHub Linux runner than on the Mac (2026-11-08), while
    HS on the same data matched exactly. Tree splits are sensitive to last-bit floating-point
    differences (likely CPU-specific numerical kernels; not verified). The bit-identical
    reproduction claim therefore holds on the original Mac setup, not across platforms.

15. **Late supplementary tests (DQ and Acerbi–Szekely Z2).** Both were pre-registered but
    implemented after the lockbox was opened (2026-11-07 and 2026-11-08), so they are
    reported for the development samples only. Z2's p-value assumes a Student-t tail shape
    matched to each model's median ES/VaR ratio, a zero location, and a shape that does not
    change over time; it is an approximation to simulating from each model's own predictive
    distribution, which is not stored.

16. **Ten-day results are descriptive and use one scaling rule (Clarification 8).** The
    10-day table scales 1-day forecasts by √10 rather than modelling 10-day losses directly
    (no multi-step GARCH simulation, no ML trained on 10-day targets), so it measures
    whether √10 scaling holds for each model, not the best achievable 10-day forecast.
    Only 4–7 breaches are expected at 1% (414–653 non-overlapping windows), so no tests are
    run. Development samples only.

17. **ESR backtest: added late, asymptotic p-values with poor finite-sample size
    (Addition 9).** The ESR test was not pre-registered; it was added on 2026-11-10, after
    the lockbox was opened, so it is exploratory and run on development samples only. Its
    p-values are asymptotic. With correct forecasts in two simulated designs, the strict
    (two-sided) version rejected 21–22% of the time at the 5% level for T = 1,000 and 9–10.5%
    for T = 5,000, so strict-ESR p-values are shown but not read as 5% tests. The one-sided
    intercept test rejected only 0.5–2.3%: it is conservative, so not rejecting is weak
    evidence that ES is adequate. These sizes come from two data-generating processes and
    may differ for real returns. The estimator was not compared with the authors' R package
    (esreg; R is not installed); joint optimality is checked numerically instead.
