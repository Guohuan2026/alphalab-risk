"""CRSP provider: interface placeholder for the WRDS migration (not implemented yet).

Planned mapping from CRSP daily stock file (crsp.dsf + crsp.dsenames/msenames)
to the canonical schema:

    canonical      CRSP source                 notes
    -------------  --------------------------  ------------------------------------------
    security_id    PERMNO                      stable across ticker changes
    ticker         TICKER (from names, as-of)  point-in-time join on namedt..nameendt
    date           DATE
    close          abs(PRC)                    negative PRC = bid/ask midpoint, no trade
    open/high/low  OPENPRC / ASKHI / BIDLO     coverage is partial in early years
    volume         VOL
    dividend       from crsp.dsedist (DIVAMT)  or bypass and use RET directly
    split_ratio    from FACPR: 1 + FACPR       on split ex-dates
    price_basis    RAW

CRSP already supplies RET (total return) and RETX (price return). When we migrate,
the return engine should cross-check its own computation against RET, and we must
incorporate delisting returns (DLRET) to avoid survivorship bias.
Universe membership must be point-in-time (SHRCD in (10, 11), EXCHCD in (1, 2, 3)).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from alphalab.data.providers.base import DataProvider
from alphalab.data.schema import PriceBasis


class CRSPProvider(DataProvider):
    name = "crsp"
    price_basis = PriceBasis.RAW

    def __init__(self, wrds_username: str | None = None):
        self.wrds_username = wrds_username

    def fetch(self, tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError(
            "CRSP/WRDS support is planned for a later milestone. "
            "See the module docstring for the field mapping."
        )
