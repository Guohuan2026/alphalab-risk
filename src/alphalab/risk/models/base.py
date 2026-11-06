"""The contract every risk model follows. This is what makes the horse race fair.

Two calls, mirroring how a bank runs a production model:

    fit(train)          estimate parameters on the training window. Called only on
                        refit dates (e.g. every 21 sessions).
    predict(history)    one-day-ahead VaR / ES / sigma for the day AFTER the last
                        element of ``history``, using the parameters from the last fit.
                        Called every day, so filters (EWMA/GARCH variance) stay current.

``train`` is what the estimation window allows (expanding or rolling); ``history`` is
every return known at the origin. Both end at the origin t. Neither ever contains t+1.

Units: returns are daily simple returns. VaR and ES are POSITIVE loss numbers
(VaR 0.023 = "lose more than 2.3% with probability alpha").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Forecast:
    """One origin's forecast for the next day."""

    var: dict[float, float]  # alpha -> VaR
    es: dict[float, float]  # alpha -> ES
    sigma: float = np.nan  # model volatility forecast if the model has one

    def as_row(self) -> dict[str, float]:
        row = {f"var_{a:g}": v for a, v in self.var.items()}
        row |= {f"es_{a:g}": v for a, v in self.es.items()}
        row["sigma"] = self.sigma
        return row


@dataclass
class RiskModel(ABC):
    alphas: Sequence[float] = field(default=(0.01, 0.025, 0.05))
    name: str = ""

    def __post_init__(self):
        self.alphas = tuple(float(a) for a in self.alphas)
        if not all(0 < a < 0.5 for a in self.alphas):
            raise ValueError(f"alphas must be tail probabilities in (0, 0.5): {self.alphas}")
        self.name = self.name or type(self).__name__.lower()

    @property
    def min_history(self) -> int:
        """Observations needed before the first forecast."""
        return 2

    def fit(self, train: np.ndarray) -> RiskModel:  # noqa: B027  (default: nothing to estimate)
        """Estimate parameters. Default: nothing to estimate."""
        return self

    def fit_info(self) -> dict:
        """Estimated parameters after the last fit (saved per refit date). Default: none."""
        return {}

    @abstractmethod
    def predict(self, history: np.ndarray) -> Forecast:
        """Forecast for the day after ``history[-1]``."""
