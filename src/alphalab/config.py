"""Typed configuration loaded from YAML.

Reproducibility rule: ``end`` must be an explicit past date. "Up to today" makes a
run depend on when you pressed Enter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from alphalab.core.calendar import get_calendar
from alphalab.portfolio.core import REBALANCE_RULES, validate_weights


@dataclass
class ValidationConfig:
    extreme_return: float = 0.4
    max_gap_sessions: int = 5  # trading sessions on the exchange calendar
    vendor_adj_tolerance: float = 0.002


@dataclass
class PortfolioConfig:
    """Fixed target weights; the risk study forecasts this portfolio's daily return."""

    name: str
    weights: dict[str, float]
    rebalance: str = "monthly"  # "daily" (constant mix) or "monthly"

    def __post_init__(self):
        self.weights = validate_weights({k.strip().upper(): v for k, v in self.weights.items()})
        if self.rebalance not in REBALANCE_RULES:
            raise ValueError(f"portfolio.rebalance must be one of {REBALANCE_RULES}")


@dataclass
class FeaturesConfig:
    """Inputs for risk features (see docs/METHODOLOGY.md §7)."""

    yahoo_levels: list[str] = field(default_factory=lambda: ["^VIX"])  # same-day, log changes
    fred: list[str] = field(default_factory=lambda: ["BAA10Y", "T10Y3M"])  # lagged, differences
    fred_lag_sessions: int = 1  # FRED publishes the next morning
    range_asset: str | None = None  # asset whose OHLC gives range-based variance

    def __post_init__(self):
        self.yahoo_levels = [s.strip().upper() for s in self.yahoo_levels]
        self.fred = [s.strip().upper() for s in self.fred]
        if self.range_asset is not None:
            self.range_asset = self.range_asset.strip().upper()
        if self.fred and self.fred_lag_sessions < 1:
            raise ValueError("features.fred_lag_sessions must be >= 1 (FRED publishes next day)")


@dataclass
class Config:
    provider: str
    universe: list[str]
    start: str
    end: str
    data_dir: Path = Path("data")
    monthly_min_obs: int = 15
    calendar: str = "XNYS"  # exchange_calendars code, e.g. XNYS, XTSE, XLON
    provider_options: dict = field(default_factory=dict)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    portfolio: PortfolioConfig | None = None
    features: FeaturesConfig | None = None

    def __post_init__(self):
        if not self.universe:
            raise ValueError("universe must contain at least one ticker")
        s, e = pd.Timestamp(self.start), pd.Timestamp(self.end)
        if s >= e:
            raise ValueError(f"start ({self.start}) must be before end ({self.end})")
        if e.normalize() >= pd.Timestamp.today().normalize():
            raise ValueError("end must be a fixed date in the past for reproducibility")
        self.data_dir = Path(self.data_dir)
        self.universe = [t.strip().upper() for t in self.universe]
        if len(set(self.universe)) != len(self.universe):
            raise ValueError("universe contains duplicate tickers")
        if self.portfolio is not None:
            outside = sorted(set(self.portfolio.weights) - set(self.universe))
            if outside:
                raise ValueError(f"portfolio assets not in universe: {outside}")
        if self.features is not None:
            if self.portfolio is None:
                raise ValueError("features need a portfolio (they describe its return series)")
            ra = self.features.range_asset
            if ra is not None and ra not in self.universe:
                raise ValueError(f"features.range_asset {ra!r} not in universe")
        try:
            get_calendar(self.calendar)
        except Exception as exc:  # exchange_calendars raises its own InvalidCalendarName
            raise ValueError(f"unknown exchange calendar {self.calendar!r}") from exc


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    val_raw = raw.pop("validation", None) or {}
    if "max_gap_bdays" in val_raw:  # renamed in M2 when gaps moved to the exchange calendar
        raise ValueError("validation.max_gap_bdays was renamed to max_gap_sessions")
    val = ValidationConfig(**val_raw)
    port_raw = raw.pop("portfolio", None)
    port = PortfolioConfig(**port_raw) if port_raw else None
    feat_raw = raw.pop("features", None)
    feat = FeaturesConfig(**feat_raw) if feat_raw is not None else None
    return Config(**raw, validation=val, portfolio=port, features=feat)
