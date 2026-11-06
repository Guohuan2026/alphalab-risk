"""Forecast targets (LABELS). These deliberately look forward and must never be model inputs.

Loss convention: L = −r (a 3% fall is a loss of +0.03). VaR and ES are reported as positive
loss numbers. Stamped at origin t, the label is the loss realised on t+1, the day the risk
forecast made at the close of t is about.
"""

from __future__ import annotations

import pandas as pd


def next_day_loss(ret: pd.Series) -> pd.Series:
    """LABEL: L_{t+1} = −r_{t+1}, stamped at t. NaN on the last date (no t+1 yet)."""
    return (-ret.sort_index().shift(-1)).rename("loss_next")
