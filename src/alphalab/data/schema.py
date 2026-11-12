"""Canonical daily price schema shared by every data provider.

Every provider (Yahoo today, CRSP later) must return a *long* DataFrame with the
columns below, one row per (security_id, date). Downstream code never sees
vendor-specific column names, which is what lets us swap Yahoo for CRSP later
without touching return or factor code.

Conventions
-----------
- ``date``: tz-naive trading date (exchange-local calendar date).
- ``security_id``: the stable identifier. For Yahoo this is the ticker (NOT stable
  through ticker changes/delistings); for CRSP it will be PERMNO.
- ``close``: closing price on the basis declared by the provider (see PriceBasis).
- ``dividend``: cash dividend per share with ex-date == ``date``, expressed per
  share *as of that date* (i.e. post-split if a split happens the same day).
- ``split_ratio``: new shares per old share on the split ex-date (2.0 for a 2-for-1),
  1.0 on all other days.
Extra columns are allowed (e.g. ``vendor_adj_close``) and are carried through.
"""

from __future__ import annotations

from enum import StrEnum

import pandas as pd


class PriceBasis(StrEnum):
    """How a provider's ``close`` column relates to corporate actions."""

    RAW = "raw"
    """As-traded prices. Returns must apply ``split_ratio`` (CRSP PRC behaves like this)."""

    SPLIT_ADJUSTED = "split_adjusted"
    """History already rescaled for splits, not dividends (Yahoo 'Close' with auto_adjust=False)."""


KEY = ["security_id", "date"]
STRING_COLS = ["security_id", "ticker"]
FLOAT_COLS = ["open", "high", "low", "close", "volume", "dividend", "split_ratio"]
REQUIRED_COLUMNS = ["date", *STRING_COLS, *FLOAT_COLS]


class SchemaError(ValueError):
    """Raised when a frame does not satisfy the canonical schema."""


def empty_frame() -> pd.DataFrame:
    """Return an empty frame with canonical columns and dtypes."""
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in REQUIRED_COLUMNS})
    df["date"] = pd.Series(dtype="datetime64[ns]")
    for c in STRING_COLS:
        df[c] = pd.Series(dtype="string")
    return df


def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Cast dtypes, normalise dates, sort by key. Does not drop or invent rows."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")
    out = df.copy()
    dates = pd.to_datetime(out["date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    out["date"] = dates.dt.normalize().astype("datetime64[ns]")
    for c in STRING_COLS:
        out[c] = out[c].astype("string")
    for c in FLOAT_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out["dividend"] = out["dividend"].fillna(0.0)
    out["split_ratio"] = out["split_ratio"].fillna(1.0)
    return out.sort_values(KEY, kind="mergesort").reset_index(drop=True)


def check_schema(df: pd.DataFrame) -> None:
    """Raise SchemaError if ``df`` violates structural rules of the schema."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise SchemaError("'date' must be a datetime column")
    if getattr(df["date"].dt, "tz", None) is not None:
        raise SchemaError("'date' must be tz-naive")
    for c in FLOAT_COLS:
        if not pd.api.types.is_float_dtype(df[c]):
            raise SchemaError(f"'{c}' must be float, got {df[c].dtype}")
    if df.duplicated(KEY).any():
        n = int(df.duplicated(KEY).sum())
        raise SchemaError(f"{n} duplicate (security_id, date) rows")


# ---------- series schema (non-price data: index levels, macro/market series) ----------
SERIES_COLUMNS = ["date", "security_id", "value"]
"""Long format for feature inputs such as VIX or FRED spreads. ``security_id`` holds the
series name (e.g. ``"^VIX"``, ``"BAA10Y"``). Values may be negative (spreads), so the price
rules (positive close, dividends, splits) do not apply."""


def check_series_schema(df: pd.DataFrame) -> None:
    """Raise SchemaError if ``df`` is not a valid long series frame."""
    missing = [c for c in SERIES_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required series columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]) or df["date"].dt.tz is not None:
        raise SchemaError("'date' must be a tz-naive datetime column")
    if not pd.api.types.is_float_dtype(df["value"]):
        raise SchemaError(f"'value' must be float, got {df['value'].dtype}")
    if df.duplicated(KEY).any():
        raise SchemaError(f"{int(df.duplicated(KEY).sum())} duplicate (security_id, date) rows")
