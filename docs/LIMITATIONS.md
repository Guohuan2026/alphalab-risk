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
