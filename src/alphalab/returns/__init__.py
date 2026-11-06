"""Return construction from canonical prices."""

from alphalab.returns.core import (
    compound_returns,
    compute_daily_returns,
    forward_returns,
    log_returns,
    to_wide,
    total_return_index,
    trailing_returns,
)

__all__ = [
    "compound_returns",
    "compute_daily_returns",
    "forward_returns",
    "log_returns",
    "to_wide",
    "total_return_index",
    "trailing_returns",
]
