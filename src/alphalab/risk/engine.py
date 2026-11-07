"""Out-of-sample forecast engine: walk forward through time, one forecast per day.

At each origin t (a trading day):
    1. if t is a refit date: model.fit(train), where train = returns up to and including t
       (all of them for "expanding", the last ``window_size`` for "rolling");
    2. forecast = model.predict(history), where history = all returns up to and including t;
    3. store the forecast at index t. It is the forecast for the loss on t+1.

Nothing after t is ever passed to the model. That is checked by the engine-level truncation
test: running the engine on returns[:d] must give exactly the same forecast at origin d
as the full run.

Refit dates are every ``refit_every`` origins, counted from the FIRST eligible origin. That
anchor does not depend on how much data comes later, which is what keeps the truncation test
valid.

Lockbox: origins whose target day falls on or after ``lockbox_start`` are not produced
unless ``include_lockbox=True``. The held-out period cannot be peeked at by accident while
models are still being developed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphalab.features.targets import next_day_loss
from alphalab.risk.models.base import Forecast, RiskModel

log = logging.getLogger(__name__)
WINDOWS = ("expanding", "rolling")


@dataclass(frozen=True)
class EngineSettings:
    min_train: int = 1000  # sessions of history required before the first forecast
    refit_every: int = 21  # sessions between parameter re-estimations
    window: str = "expanding"
    window_size: int | None = None  # for "rolling"
    oos_start: str | None = None  # first allowed origin (if enough history by then)
    lockbox_start: str | None = None  # first TARGET date held out
    var_floor: float = 1e-4  # safety floor: VaR >= 1 bp and ES >= VaR (RESEARCH_DESIGN Dev. 2)

    def __post_init__(self):
        if self.window not in WINDOWS:
            raise ValueError(f"window must be one of {WINDOWS}")
        if self.window == "rolling" and (self.window_size is None or self.window_size < 2):
            raise ValueError("rolling window needs window_size >= 2")
        if self.var_floor <= 0:
            raise ValueError("var_floor must be positive (FZ0 needs ES > 0)")
        if self.min_train < 2 or self.refit_every < 1:
            raise ValueError("min_train must be >= 2 and refit_every >= 1")


def forecast_origins(
    index: pd.DatetimeIndex, s: EngineSettings, include_lockbox: bool = False
) -> np.ndarray:
    """Integer positions of the origins at which forecasts are made."""
    n = len(index)
    pos = np.arange(s.min_train - 1, n)  # history length pos + 1 >= min_train
    if s.oos_start is not None:
        pos = pos[index[pos] >= pd.Timestamp(s.oos_start)]
    if s.lockbox_start is not None and not include_lockbox:
        before = index[index < pd.Timestamp(s.lockbox_start)]
        if len(before) == 0:
            return pos[:0]
        # The last pre-lockbox day is itself excluded: its target is inside the lockbox.
        pos = pos[index[pos] < before[-1]]
    return pos


def apply_floor(fc: Forecast, floor: float) -> dict[str, float]:
    """Safety floor applied identically to every model: VaR_a >= floor and ES_a >= VaR_a.

    A downside-risk forecast of zero or less ("tomorrow's 5% worst case is a gain") is not
    a valid forecast, and FZ0 cannot score it. Flooring instead of dropping keeps every model
    on the same dates, and the floored forecast is then scored (FZ0 penalises a near-zero
    risk forecast heavily). The ``floored`` column records every intervention.
    """
    row = fc.as_row()
    floored = False
    for a in fc.var:
        v, e = fc.var[a], fc.es[a]
        v2 = max(v, floor)
        e2 = max(e, v2)
        floored |= (v2 != v) or (e2 != e)
        row[f"var_{a:g}"], row[f"es_{a:g}"] = v2, e2
    row["floored"] = floored
    return row


def run_forecasts(
    returns: pd.Series,
    make_model: Callable[[], RiskModel],
    settings: EngineSettings,
    include_lockbox: bool = False,
    fits: list[dict] | None = None,
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Walk-forward forecasts. Index: origin date t; columns var_*, es_*, sigma.

    If ``fits`` is a list, one dict per refit is appended: the origin date plus the
    model's ``fit_info()`` (e.g. GARCH parameters), giving parameter paths over time.
    """
    r = returns.sort_index()
    if r.isna().any():
        raise ValueError("returns contain NaN; build the portfolio series first")
    values = r.to_numpy(dtype=float)
    origins = forecast_origins(r.index, settings, include_lockbox)
    model = make_model()
    X = None
    if model.uses_features:
        if features is None:
            raise ValueError(f"{model.name} needs features; build them first")
        X = features.reindex(r.index)  # exact date alignment; missing dates become NaN rows
    every = model.refit_every or settings.refit_every  # per-model override (e.g. yearly)
    rows = []
    for k, t in enumerate(origins):
        if k % every == 0:
            lo = 0 if settings.window == "expanding" else max(0, t + 1 - settings.window_size)
            if X is None:
                model.fit(values[lo : t + 1])
            else:
                model.fit(values[lo : t + 1], X.iloc[lo : t + 1])
            info = model.fit_info()
            if fits is not None and info:
                fits.append({"origin": r.index[t], **info})
        if X is None:
            fc = model.predict(values[: t + 1])
        else:
            fc = model.predict(values[: t + 1], X.iloc[: t + 1])
        rows.append(apply_floor(fc, settings.var_floor))
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(r.index[origins], name="origin"))
    log.info(
        "%s: %d forecasts %s..%s",
        model.name,
        len(out),
        out.index[0].date() if len(out) else "-",
        out.index[-1].date() if len(out) else "-",
    )
    return out


def attach_outcomes(forecasts: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    """Add the realised next-day loss (the LABEL) and its date, for evaluation only.

    Origins with no next day in ``returns`` (the very last date) are dropped: their
    outcome is not known yet.
    """
    r = returns.sort_index()
    nxt = pd.Series(r.index[1:], index=r.index[:-1])
    out = forecasts.join(next_day_loss(r).rename("loss_next")).assign(target_date=nxt)
    return out.dropna(subset=["loss_next"])
