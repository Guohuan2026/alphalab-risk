"""Explicit, auditable data corrections applied at build time (data/raw is never edited).

Why explicit lists instead of automatic rules: every correction is a research decision. A
written list in the config can be reviewed, cited and diffed; an automatic rule can change
silently when a vendor restates history.

Supported:
- ``non_cash_dividends``: {security_id: [dates]}. Distributions that were reinvested
  (units consolidated, no cash paid, no price drop), so they must not be added to total
  return. Example: XIU.TO year-end reinvested distributions (docs/LIMITATIONS.md #12),
  identified as the dividend dates on which Yahoo's own adjusted close implies no payout.
"""

from __future__ import annotations

import pandas as pd

KNOWN = {"non_cash_dividends"}


def apply_corrections(
    prices: pd.DataFrame, corrections: dict | None
) -> tuple[pd.DataFrame, list[dict]]:
    """Return corrected prices and a log of every change (for the build manifest)."""
    if not corrections:
        return prices, []
    unknown = set(corrections) - KNOWN
    if unknown:
        raise ValueError(f"unknown correction types: {sorted(unknown)}")
    out = prices.copy()
    log = []
    for sid, dates in corrections.get("non_cash_dividends", {}).items():
        for d in dates:
            m = (out["security_id"] == sid) & (out["date"] == pd.Timestamp(d))
            if m.sum() != 1 or not (out.loc[m, "dividend"] > 0).all():
                raise ValueError(
                    f"non_cash_dividends: no dividend row for {sid} on {d}; check the config"
                )
            before = float(out.loc[m, "dividend"].iloc[0])
            out.loc[m, "dividend"] = 0.0
            log.append(
                {
                    "type": "non_cash_dividend",
                    "security_id": sid,
                    "date": str(pd.Timestamp(d).date()),
                    "dividend_removed": before,
                }
            )
    return out, log
