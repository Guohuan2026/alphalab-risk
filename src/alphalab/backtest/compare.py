"""Comparing models: Diebold–Mariano tests and the Model Confidence Set.

Diebold & Mariano (1995). For two models' losses, d_t = L^A_t − L^B_t. H0: E[d] = 0 (equal
predictive ability). Daily risk losses are autocorrelated (volatility clusters), so the
variance of the mean uses a Newey–West HAC estimate with a Bartlett kernel:

    LRV = γ_0 + 2 Σ_{k=1..q} (1 − k/(q+1)) γ_k,     q = ⌊4 (n/100)^{2/9}⌋ by default
    DM  = d̄ / √(LRV / n)  ~ N(0, 1)

DM < 0 means model A has the LOWER (better) average loss.

Model Confidence Set (Hansen, Lunde & Nason 2011), via ``arch.bootstrap.MCS``. With many
models, pairwise DM tests pile up false discoveries. The MCS instead keeps eliminating the
worst model until the remaining set cannot be told apart from the best at the chosen level
(here 10%), using a block bootstrap that respects autocorrelation. Reading: "these models
are, with 90% confidence, the best; the data cannot separate them".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DMResult:
    mean_diff: float
    stat: float
    p_value: float
    lag: int
    n: int

    def to_dict(self) -> dict:
        return asdict(self)


def newey_west_lrv(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    lrv = x @ x / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (x[k:] @ x[:-k]) / n
    return float(lrv)


def default_lag(n: int) -> int:
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def diebold_mariano(loss_a, loss_b, lag: int | None = None) -> DMResult:
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("loss series must be aligned and of equal length")
    d = a - b
    n = len(d)
    q = default_lag(n) if lag is None else lag
    lrv = newey_west_lrv(d, q)
    if lrv <= 0:
        return DMResult(float(d.mean()), 0.0, 1.0, q, n)
    stat = d.mean() / np.sqrt(lrv / n)
    return DMResult(float(d.mean()), float(stat), float(2 * stats.norm.sf(abs(stat))), q, n)


def model_confidence_set(
    losses: pd.DataFrame,
    size: float = 0.10,
    reps: int = 2000,
    block_size: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """MCS p-value and membership for each model (columns of ``losses``)."""
    from arch.bootstrap import MCS

    if losses.isna().any().any():
        raise ValueError("losses contain NaN; align models on common dates first")
    block = block_size or max(5, default_lag(len(losses)))
    mcs = MCS(losses, size=size, reps=reps, block_size=block, method="R", seed=seed)
    mcs.compute()
    out = mcs.pvalues.rename(columns={"Pvalue": "mcs_pvalue"})
    out["in_mcs"] = out.index.isin(mcs.included)
    out.index.name = "model"
    return out.reset_index()
