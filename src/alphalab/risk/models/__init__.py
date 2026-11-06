"""Risk models and a small registry so configs can name them.

Config example:  models: [{name: hs250, type: hs, window: 250}, {type: ewma, lam: 0.94}]
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial

from alphalab.risk.models.base import Forecast, RiskModel
from alphalab.risk.models.ewma import EWMANormal
from alphalab.risk.models.garch import GarchT
from alphalab.risk.models.hs import HistoricalSimulation

MODEL_TYPES: dict[str, Callable[..., RiskModel]] = {
    "hs": HistoricalSimulation,
    "ewma": EWMANormal,
    "garch": partial(GarchT, asymmetric=False, tails="t"),  # GARCH(1,1)-t
    "gjr": partial(GarchT, asymmetric=True, tails="t"),  # GJR-GARCH(1,1)-t
    "fhs": partial(GarchT, asymmetric=True, tails="fhs"),  # filtered HS on GJR volatility
}


def build_model(spec: dict, alphas: Sequence[float]) -> RiskModel:
    """Instantiate a model from a config dict: ``type`` picks the class, the rest are fields."""
    spec = dict(spec)
    kind = spec.pop("type", None)
    if kind not in MODEL_TYPES:
        raise ValueError(f"unknown model type {kind!r}; expected one of {sorted(MODEL_TYPES)}")
    return MODEL_TYPES[kind](alphas=tuple(alphas), **spec)


__all__ = [
    "MODEL_TYPES",
    "EWMANormal",
    "GarchT",
    "Forecast",
    "HistoricalSimulation",
    "RiskModel",
    "build_model",
]
