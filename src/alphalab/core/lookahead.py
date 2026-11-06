"""Truncation test for look-ahead bias.

Idea: if a transformation f is point-in-time, the value it produces for date d
must be identical whether you give it data up to d or the full history:

    f(data[:d]).loc[d] == f(data).loc[d]

If adding FUTURE rows changes the value at d, f used the future. This catches
centred windows, full-sample normalisation, backward-adjusted levels, negative
shifts, and similar bugs. It cannot catch leakage that is already baked into the
input data (e.g. a survivorship-biased universe) - that is a data problem.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LookaheadViolation:
    date: pd.Timestamp
    column: str
    full_value: float
    truncated_value: float


def check_point_in_time(
    func: Callable[[pd.DataFrame], pd.DataFrame],
    data: pd.DataFrame,
    dates: Sequence[pd.Timestamp] | None = None,
    n_checks: int = 10,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> list[LookaheadViolation]:
    """Return violations of the truncation test (empty list == passed)."""
    if dates is None:
        idx = data.index
        # Skip the very last date: truncation there is the full sample, a trivial pass.
        positions = np.unique(np.linspace(0, len(idx) - 2, n_checks).astype(int))
        dates = [idx[p] for p in positions]

    full = func(data)
    violations: list[LookaheadViolation] = []
    for d in dates:
        trunc = func(data.loc[:d])
        full_row = full.loc[d] if d in full.index else pd.Series(np.nan, index=full.columns)
        trunc_row = trunc.loc[d] if d in trunc.index else pd.Series(np.nan, index=full.columns)
        for c in full.columns:
            a = float(full_row.get(c, np.nan))
            b = float(trunc_row.get(c, np.nan))
            same = (np.isnan(a) and np.isnan(b)) or np.isclose(a, b, rtol=rtol, atol=atol)
            if not same:
                violations.append(LookaheadViolation(pd.Timestamp(d), str(c), a, b))
    return violations


def assert_point_in_time(
    func: Callable[[pd.DataFrame], pd.DataFrame], data: pd.DataFrame, **kwargs
) -> None:
    v = check_point_in_time(func, data, **kwargs)
    if v:
        first = v[0]
        raise AssertionError(
            f"look-ahead detected in {getattr(func, '__name__', func)}: {len(v)} violation(s); "
            f"first at {first.date.date()} [{first.column}] full={first.full_value:.6g} "
            f"truncated={first.truncated_value:.6g}"
        )
