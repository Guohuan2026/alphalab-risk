"""Loss functions (scoring rules) for ranking VaR and (VaR, ES) forecasts.

Why loss functions? Coverage tests ask "is this model acceptable?". To ask "is model A
better than model B?" we need a score where lower is better ON AVERAGE for the true
forecast. Such scores are called *strictly consistent*: the expected score is minimised
only by the true quantity.

- VaR is a quantile, and quantiles are elicitable: the tick (quantile) loss is strictly
  consistent for them.
- ES alone is NOT elicitable: no score is minimised by the true ES by itself (Gneiting
  2011). The PAIR (VaR, ES) is jointly elicitable (Fissler & Ziegel 2016). FZ0 (Patton,
  Ziegel & Chen 2019) is the member of that family whose score differences do not depend
  on the units of returns, which makes it the standard choice for comparing ES forecasts.

Sign conventions. Inside this module the formulas are written in RETURN space, as in the
papers: y = return, v = the α-quantile of returns (negative), e = ES in returns (negative,
e ≤ v). Our forecasts store positive LOSS numbers, so v = −VaR and e = −ES. The public
functions take our positive VaR/ES and convert.

    tick:  QL(y, v)    = (α − 1{y ≤ v}) (y − v)                                  ≥ 0
    FZ0:   FZ0(y,v,e)  = −1{y ≤ v}(v − y)/(α e) + v/e + ln(−e) − 1               any sign
"""

from __future__ import annotations

import numpy as np


def tick_loss(ret, var, alpha: float) -> np.ndarray:
    """Quantile loss of VaR forecasts (positive VaR) against realised returns."""
    y = np.asarray(ret, dtype=float)
    v = -np.asarray(var, dtype=float)
    return (alpha - (y <= v)) * (y - v)


def fz0_loss(ret, var, es, alpha: float) -> np.ndarray:
    """FZ0 joint loss of (VaR, ES) forecasts (positive numbers) against realised returns.

    Requires ES > 0 (i.e. e < 0 in return space); a non-positive ES forecast is not a
    meaningful downside-risk forecast and raises.
    """
    y = np.asarray(ret, dtype=float)
    v = -np.asarray(var, dtype=float)
    e = -np.asarray(es, dtype=float)
    if np.any(e >= 0):
        raise ValueError("FZ0 needs ES > 0 (a loss) for every forecast")
    hit = (y <= v).astype(float)  # the indicator 1{y ≤ v} as 0.0 / 1.0
    return -hit * (v - y) / (alpha * e) + v / e + np.log(-e) - 1.0
