# Known limitations (Milestones 1–2)

Ordered roughly by how much they can distort research conclusions.

1. **Survivorship bias (severe).** The universe is a hand-picked list of companies that
   exist and are large *today*. Yahoo has little data on delisted firms. Any backtest on
   it overstates returns and understates risk. Fix: CRSP with point-in-time universe
   membership and delisting returns (`DLRET`).
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
