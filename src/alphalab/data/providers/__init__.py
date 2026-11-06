"""Data providers. Each one converts a vendor's format into the canonical schema."""

from __future__ import annotations

from alphalab.data.providers.base import DataProvider


def get_provider(name: str, **kwargs) -> DataProvider:
    """Factory: build a provider by name. Imports lazily so optional deps stay optional."""
    name = name.lower()
    if name == "yahoo":
        from alphalab.data.providers.yahoo import YahooProvider

        return YahooProvider(**kwargs)
    if name == "synthetic":
        from alphalab.data.providers.synthetic import SyntheticProvider

        return SyntheticProvider(**kwargs)
    if name == "crsp":
        from alphalab.data.providers.crsp import CRSPProvider

        return CRSPProvider(**kwargs)
    raise ValueError(f"unknown provider '{name}' (expected yahoo | synthetic | crsp)")


__all__ = ["DataProvider", "get_provider"]
