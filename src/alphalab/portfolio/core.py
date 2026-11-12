"""Daily portfolio returns from asset total returns and fixed target weights.

The risk study models each portfolio as ONE return series r_p,t (see RESEARCH_DESIGN.md §9),
so this module is the bridge from asset data to every risk model.

Math (simple returns, NOT log returns: only simple returns aggregate across assets):

    r_p,t   = Σ_i w_i,t · R_i,t                      w_i,t = weights at the START of day t
    w_i,t+1 = w_i,t · (1 + R_i,t) / (1 + r_p,t)      weights drift with relative performance

Rebalancing resets the start-of-day weights to the targets w*:
- ``"daily"``  : every day (a "constant-mix" portfolio; r_p,t = Σ w*_i R_i,t).
- ``"monthly"``: on the first session of each month, i.e. trades at the close of the
  previous month's last session. Whether day t starts a new month is decided by comparing
  t with the PREVIOUS row only. Asking "is t the last day of its month?" would need the
  next row, which is look-ahead (tests/test_portfolio.py checks this with the truncation test).

Ignored: transaction costs, cash drag, intraday rebalancing.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

REBALANCE_RULES = ("daily", "monthly")


def validate_weights(weights: Mapping[str, float], tol: float = 1e-9) -> dict[str, float]:
    """Weights must be finite and sum to 1 (shorts allowed, leverage not)."""
    if not weights:
        raise ValueError("portfolio weights are empty")
    w = {str(k): float(v) for k, v in weights.items()}
    if not all(np.isfinite(v) for v in w.values()):
        raise ValueError(f"portfolio weights must be finite: {w}")
    total = sum(w.values())
    if abs(total - 1.0) > tol:
        raise ValueError(f"portfolio weights must sum to 1, got {total:.6g}")
    return w


def portfolio_returns(
    ret_wide: pd.DataFrame,
    weights: Mapping[str, float],
    rebalance: str = "monthly",
) -> pd.DataFrame:
    """Daily portfolio total return plus the start-of-day weights that produced it.

    Parameters
    ----------
    ret_wide : date x asset matrix of daily simple total returns (``to_wide(..., "ret")``).
    weights  : target weights by asset id; must sum to 1.
    rebalance: ``"daily"`` or ``"monthly"``.

    Returns
    -------
    DataFrame indexed by date with column ``ret`` (portfolio return) and ``w_<asset>``
    (start-of-day weights). It starts on the first date on which EVERY asset has a return.
    A missing asset return after that raises: silently treating it as 0% would understate
    risk exactly when data are bad.
    """
    if rebalance not in REBALANCE_RULES:
        raise ValueError(f"rebalance must be one of {REBALANCE_RULES}, got {rebalance!r}")
    w_target = validate_weights(weights)
    assets = list(w_target)
    missing = [a for a in assets if a not in ret_wide.columns]
    if missing:
        raise ValueError(f"no returns for portfolio assets: {missing}")

    R = ret_wide[assets].sort_index()
    complete = R.notna().all(axis=1)
    if not complete.any():
        raise ValueError("assets never all have returns on the same date")
    R = R.loc[complete.idxmax() :]  # first date with a full cross-section
    holes = R.index[R.isna().any(axis=1)]
    if len(holes):
        raise ValueError(
            f"{len(holes)} date(s) with missing asset returns after the start, first "
            f"{holes[0].date()}; investigate the validation report (date_gap) before building"
        )

    target = np.array([w_target[a] for a in assets])
    r = R.to_numpy()
    months = R.index.year * 12 + R.index.month
    port = np.empty(len(R))
    w_start = np.empty_like(r)
    w = target.copy()
    for t in range(len(R)):
        new_month = t == 0 or months[t] != months[t - 1]
        if rebalance == "daily" or new_month:
            w = target.copy()
        w_start[t] = w
        port[t] = w @ r[t]
        w = w * (1.0 + r[t]) / (1.0 + port[t])  # drift to end of day t

    out = pd.DataFrame(w_start, index=R.index, columns=[f"w_{a}" for a in assets])
    out.insert(0, "ret", port)
    out.index.name = "date"
    return out
