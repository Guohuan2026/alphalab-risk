"""Risk models and a small registry so configs can name them.

Config example:  models: [{name: hs250, type: hs, window: 250}, {type: ewma, lam: 0.94}]
"""

from __future__ import annotations

from collections.abc import Sequence

from alphalab.risk.models.base import Forecast, RiskModel
from alphalab.risk.models.ewma import EWMANormal
from alphalab.risk.models.hs import HistoricalSimulation

MODEL_TYPES: dict[str, type[RiskModel]] = {"hs": HistoricalSimulation, "ewma": EWMANormal}


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
    "Forecast",
    "HistoricalSimulation",
    "RiskModel",
    "build_model",
]
