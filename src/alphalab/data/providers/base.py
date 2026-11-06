"""Abstract provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd

from alphalab.data.schema import PriceBasis


class DataProvider(ABC):
    """A source of daily prices + corporate actions in the canonical schema.

    Subclasses must set ``name`` and ``price_basis`` and implement ``fetch``.
    ``fetch`` must treat ``start`` and ``end`` as INCLUSIVE calendar dates.
    """

    name: str
    price_basis: PriceBasis

    @abstractmethod
    def fetch(self, tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
        """Return canonical long-format prices for ``tickers`` in [start, end]."""

    def metadata(self) -> dict:
        """Provenance stored in the snapshot manifest. Override to add library versions."""
        return {"provider": self.name, "price_basis": str(self.price_basis)}
