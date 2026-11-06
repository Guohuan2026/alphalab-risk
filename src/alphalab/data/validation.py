"""Data-quality checks. Errors block the build; warnings are reported for review.

Nothing here silently "fixes" data. Suspicious rows are flagged, so a human decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphalab.data.schema import KEY, PriceBasis, SchemaError, check_schema
from alphalab.returns.core import compute_daily_returns


@dataclass
class Issue:
    severity: str  # "error" | "warning"
    check: str
    security_id: str | None
    date: pd.Timestamp | None
    detail: str


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    def add(self, severity, check, detail, security_id=None, date=None):
        self.issues.append(Issue(severity, check, security_id, date, detail))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_frame(self) -> pd.DataFrame:
        cols = ["severity", "check", "security_id", "date", "detail"]
        return pd.DataFrame([i.__dict__ for i in self.issues], columns=cols)

    def summary(self) -> str:
        if not self.issues:
            return "validation: no issues"
        counts = self.to_frame().groupby(["severity", "check"]).size()
        return "validation:\n" + "\n".join(f"  {s:8s} {c:28s} {n}" for (s, c), n in counts.items())


def _rows(report, mask, df, severity, check, detail_fn, limit=50):
    for _, r in df[mask].head(limit).iterrows():
        report.add(severity, check, detail_fn(r), r["security_id"], r["date"])
    extra = int(mask.sum()) - limit
    if extra > 0:
        report.add(severity, check, f"... and {extra} more rows")


def validate_prices(
    df: pd.DataFrame,
    price_basis: PriceBasis | str,
    extreme_return: float = 0.4,
    max_gap_bdays: int = 5,
    vendor_adj_tolerance: float = 0.002,
) -> ValidationReport:
    rep = ValidationReport()
    try:
        check_schema(df)
    except SchemaError as e:
        rep.add("error", "schema", str(e))
        if "missing required columns" in str(e):
            return rep

    dup = df.duplicated(KEY, keep=False)
    _rows(rep, dup, df, "error", "duplicate_key", lambda r: "duplicate (security_id, date)")

    bad_close = df["close"].isna() | (df["close"] <= 0)
    _rows(
        rep, bad_close, df, "error", "nonpositive_or_missing_close", lambda r: f"close={r['close']}"
    )
    _rows(
        rep,
        df["dividend"] < 0,
        df,
        "error",
        "negative_dividend",
        lambda r: f"dividend={r['dividend']}",
    )
    _rows(
        rep,
        df["split_ratio"] <= 0,
        df,
        "error",
        "invalid_split_ratio",
        lambda r: f"split_ratio={r['split_ratio']}",
    )

    # OHLC consistency (tolerance for vendor rounding). Warning, because OHLC is not
    # used for returns yet and vendors are often slightly inconsistent.
    tol = 1e-6
    hl = df["high"].notna() & df["low"].notna()
    bad_ohlc = hl & (
        (df["high"] < df["low"] * (1 - tol))
        | (df["close"] > df["high"] * (1 + tol))
        | (df["close"] < df["low"] * (1 - tol))
    )
    _rows(
        rep,
        bad_ohlc,
        df,
        "warning",
        "ohlc_inconsistent",
        lambda r: f"o={r['open']:.4g} h={r['high']:.4g} l={r['low']:.4g} c={r['close']:.4g}",
    )

    if rep.errors:  # returns are meaningless on structurally broken data
        return rep

    rets = compute_daily_returns(df, price_basis)
    extreme = rets["ret"].abs() > extreme_return
    _rows(
        rep,
        extreme,
        rets,
        "warning",
        "extreme_daily_return",
        lambda r: f"ret={r['ret']:.2%} (check split/dividend handling or bad tick)",
    )

    # Calendar gaps: count business days between consecutive observations.
    for sid, g in df.groupby("security_id", sort=False):
        d = g["date"].to_numpy(dtype="datetime64[D]")
        if len(d) < 2:
            continue
        gaps = np.busday_count(d[:-1], d[1:])
        for i in np.flatnonzero(gaps > max_gap_bdays):
            rep.add(
                "warning",
                "date_gap",
                f"{gaps[i]} business days since {d[i]}",
                sid,
                pd.Timestamp(d[i + 1]),
            )

    # Cross-check against the vendor's adjusted close when available (Yahoo).
    if "vendor_adj_close" in df.columns:
        s = df.sort_values(KEY)
        vend = s.groupby("security_id", sort=False)["vendor_adj_close"].pct_change()
        diff = rets["ret"].to_numpy() - vend.to_numpy()
        mask = pd.Series(np.abs(diff) > vendor_adj_tolerance, index=rets.index)
        tmp = rets.assign(vendor_ret=vend.to_numpy(), diff=diff)
        _rows(
            rep,
            mask,
            tmp,
            "warning",
            "vendor_adj_mismatch",
            lambda r: f"ours={r['ret']:.5f} vendor={r['vendor_ret']:.5f}",
        )
    return rep
