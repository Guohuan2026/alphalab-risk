"""Historical simulation (HS): tomorrow's loss distribution = the last N realised losses.

    VaR_a = empirical (1 − a) quantile of {L_{t−N+1}, …, L_t},  L = −r
    ES_a  = mean of those losses that are >= VaR_a

No model and no parameters: the industry baseline. Two known weaknesses, which are the
reason the stress analysis exists:
- Slow reaction: one crash day moves VaR only when it enters the window, and the effect
  lasts exactly N days, then disappears abruptly ("ghost effect").
- Few tail points: at a = 1% and N = 250 the quantile rests on about 2.5 observations,
  so VaR and especially ES are noisy.
Quantile convention: numpy's default linear interpolation (type 7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alphalab.risk.models.base import Forecast, RiskModel


@dataclass
class HistoricalSimulation(RiskModel):
    window: int = 250

    def __post_init__(self):
        super().__post_init__()
        if self.window < 20:
            raise ValueError("HS window must be at least 20 observations")
        self.name = self.name if self.name != "historicalsimulation" else f"hs{self.window}"

    @property
    def min_history(self) -> int:
        return self.window

    def predict(self, history: np.ndarray) -> Forecast:
        if len(history) < self.window:
            raise ValueError(f"HS needs {self.window} observations, got {len(history)}")
        loss = -np.asarray(history[-self.window :], dtype=float)
        var, es = {}, {}
        for a in self.alphas:
            v = float(np.quantile(loss, 1 - a))
            var[a] = v
            es[a] = float(loss[loss >= v].mean())
        return Forecast(var, es)
