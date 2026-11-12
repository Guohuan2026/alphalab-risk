"""Immutable raw-data snapshots with a provenance manifest.

Why: Yahoo restates history (e.g. after every new split), so "re-download and rerun"
does not reproduce last month's results. Each download is saved once, never
overwritten, and identified by a timestamp + content hash. Research runs then
reference a snapshot ID, not "whatever Yahoo returns today".

Layout:
    data/raw/<provider>/<snapshot_id>/prices.parquet
    data/raw/<provider>/<snapshot_id>/manifest.json
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import alphalab
from alphalab.data.schema import KEY, check_schema

DATA_FILE = "prices.parquet"  # kept for every snapshot kind so old snapshots stay readable


def content_hash(df: pd.DataFrame) -> str:
    """SHA-256 of a deterministic CSV rendering (sorted rows, sorted columns)."""
    canon = df.sort_values(KEY, kind="mergesort")[sorted(df.columns)]
    payload = canon.to_csv(index=False, date_format="%Y-%m-%d", float_format="%.12g")
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class SnapshotInfo:
    provider: str
    snapshot_id: str
    path: Path
    manifest: dict


class RawSnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root) / "raw"

    def save(
        self,
        df: pd.DataFrame,
        provider_meta: dict,
        request: dict,
        now: datetime | None = None,
        validate: Callable[[pd.DataFrame], None] = check_schema,
    ) -> SnapshotInfo:
        """Save once, never overwrite. ``validate`` is the schema check (prices by default;
        ``check_series_schema`` for feature series)."""
        validate(df)
        now = now or datetime.now(UTC)
        digest = content_hash(df)
        provider = provider_meta["provider"]
        snapshot_id = f"{now:%Y%m%dT%H%M%SZ}_{digest[:8]}"
        path = self.root / provider / snapshot_id
        if path.exists():
            raise FileExistsError(f"snapshot already exists (immutable): {path}")
        path.mkdir(parents=True)

        returned = sorted(df["security_id"].unique().tolist())
        manifest = {
            "snapshot_id": snapshot_id,
            "created_utc": now.isoformat(),
            "request": request,
            "provider": provider_meta,
            "securities_returned": returned,
            "securities_missing": sorted(set(request.get("tickers", [])) - set(returned)),
            "n_rows": len(df),
            "date_min": str(df["date"].min().date()) if len(df) else None,
            "date_max": str(df["date"].max().date()) if len(df) else None,
            "content_sha256": digest,
            "environment": {
                "alphalab": alphalab.__version__,
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
        }
        df.to_parquet(path / DATA_FILE, index=False)
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        return SnapshotInfo(provider, snapshot_id, path, manifest)

    def list(self, provider: str) -> list[str]:
        d = self.root / provider
        return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []

    def latest(self, provider: str, request: dict | None = None) -> str:
        """Newest snapshot for ``provider``; with ``request``, the newest whose saved request
        (tickers, start, end) equals it. Without the filter, several configs sharing one
        provider (e.g. three Yahoo portfolios) would silently pick each other's data."""
        snaps = self.list(provider)
        if request is not None:
            snaps = [s for s in snaps if self._request(provider, s) == request]
        if not snaps:
            what = f" matching request {request}" if request is not None else ""
            raise FileNotFoundError(
                f"no snapshots for provider '{provider}'{what} under {self.root}; run ingest first"
            )
        return snaps[-1]  # IDs start with a UTC timestamp, so lexical order == time order

    def _request(self, provider: str, snapshot_id: str) -> dict | None:
        manifest = json.loads((self.root / provider / snapshot_id / "manifest.json").read_text())
        return manifest.get("request")

    def load(
        self, provider: str, snapshot_id: str, verify: bool = True
    ) -> tuple[pd.DataFrame, dict]:
        path = self.root / provider / snapshot_id
        manifest = json.loads((path / "manifest.json").read_text())
        df = pd.read_parquet(path / DATA_FILE)
        if verify and content_hash(df) != manifest["content_sha256"]:
            raise ValueError(f"content hash mismatch for {path}: file was modified")
        return df, manifest
