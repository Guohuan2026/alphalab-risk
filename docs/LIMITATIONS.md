# Known limitations (Milestone 1)

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
7. **Calendar.** The synthetic provider uses Mon–Fri business days (no holidays).
   `drop_incomplete_last` uses business days too, so if a month's last business day is
   an exchange holiday (e.g. Good Friday 2018-03-30) and the data end the day before,
   that month is wrongly dropped. An exchange calendar is planned.
8. **Dividends only as cash on ex-date.** Special dividends, return of capital, spin-offs
   and rights issues are not modelled separately. Spin-offs in particular can appear as
   large negative price-only days.
9. **No intraday or point-in-time fundamentals** yet; earnings/book data come later.
10. **Live path verified offline only by the author's sandbox.** The Yahoo download path
    is covered by an offline normalisation test; the live network test must be run on
    your machine (`pytest -m network`).
