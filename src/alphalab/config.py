"""Typed configuration loaded from YAML.

Reproducibility rule: ``end`` must be an explicit past date. "Up to today" makes a
run depend on when you pressed Enter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml


@dataclass
class ValidationConfig:
    extreme_return: float = 0.4
    max_gap_bdays: int = 5
    vendor_adj_tolerance: float = 0.002


@dataclass
class Config:
    provider: str
    universe: list[str]
    start: str
    end: str
    data_dir: Path = Path("data")
    monthly_min_obs: int = 15
    provider_options: dict = field(default_factory=dict)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

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


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    val = ValidationConfig(**(raw.pop("validation", None) or {}))
    return Config(**raw, validation=val)
