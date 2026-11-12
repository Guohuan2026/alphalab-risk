"""Read-only data layer for the Streamlit dashboard.

The dashboard never fits a model: everything shown is read from the saved results under
``data/results/`` (written by ``alphalab forecast / backtest / stress / explain``), so what
is on screen is exactly what is in the results log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alphalab.explain import FAMILIES
from alphalab.risk.engine import attach_outcomes

# Fixed model -> colour slot (colour follows the model, never its rank). Validated categorical
# palette (dataviz skill reference instance), light and dark steps. The 8 key models get the
# 8 slots in the validated order; the 3 remaining baselines share a neutral grey.
MODEL_ORDER = ["fhs_gjr", "mlvol_fhs", "linqr_l1", "gbq", "qrf", "gjr_t", "har_fhs", "ewma"]
PALETTE = {
    "light": [
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ],
    "dark": [
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    ],
}
OTHER_COLOR = {"light": "#8a8984", "dark": "#8f8e88"}
LABELS = {
    "fhs_gjr": "FHS-GJR",
    "mlvol_fhs": "RF-vol + FHS",
    "linqr_l1": "Lasso QR (control)",
    "gbq": "Gradient boosting",
    "qrf": "Quantile forest",
    "gjr_t": "GJR-GARCH-t",
    "har_fhs": "HAR-FHS",
    "ewma": "EWMA",
    "garch_t": "GARCH-t",
    "hs250": "HS-250",
    "hs500": "HS-500",
    "linqr": "Linear QR",
}
PORTFOLIO_LABELS = {
    "p1_spy": "P1 · SPY (headline)",
    "p2_balanced_60_40": "P2 · 60/40 balanced",
    "p3_xiu": "P3 · XIU.TO (Canada)",
}


def model_color(model: str, mode: str = "light") -> str:
    mode = "dark" if mode == "dark" else "light"
    if model in MODEL_ORDER:
        return PALETTE[mode][MODEL_ORDER.index(model)]
    return OTHER_COLOR[mode]


def label(model: str) -> str:
    return LABELS.get(model, model)


@dataclass(frozen=True)
class Run:
    portfolio: str
    snapshot: str
    tag: str | None
    path: Path

    @property
    def name(self) -> str:
        base = PORTFOLIO_LABELS.get(self.portfolio, self.portfolio)
        return f"{base} · {self.tag}" if self.tag else f"{base} · expanding"

    @property
    def manifest(self) -> dict:
        f = self.path / "forecast_manifest.json"
        return json.loads(f.read_text()) if f.exists() else {}

    @property
    def has_lockbox(self) -> bool:
        return (self.path / "forecasts_lockbox.parquet").exists()

    @property
    def lockbox_start(self) -> str | None:
        return (self.manifest.get("risk") or {}).get("lockbox_start")

    @property
    def processed(self) -> Path | None:
        p = self.manifest.get("processed")
        return Path(p) if p else None


def discover_runs(data_dir: str | Path = "data") -> list[Run]:
    """Every results folder that holds forecasts, main runs first, then tagged runs."""
    root = Path(data_dir) / "results"
    runs = []
    for f in sorted(root.glob("*/*/forecasts.parquet")) + sorted(
        root.glob("*/*/*/forecasts.parquet")
    ):
        rel = f.parent.relative_to(root).parts
        tag = rel[2] if len(rel) == 3 else None
        runs.append(Run(rel[0], rel[1], tag, f.parent))
    order = {p: i for i, p in enumerate(PORTFOLIO_LABELS)}
    return sorted(runs, key=lambda r: (r.tag is not None, order.get(r.portfolio, 99), r.portfolio))


def _sfx(lockbox: bool) -> str:
    return "_lockbox" if lockbox else ""


def table(run: Run, name: str, lockbox: bool = False) -> pd.DataFrame | None:
    """A saved CSV table (e.g. 'backtest_losses', 'stress_regimes'), or None if missing."""
    f = run.path / f"{name}{_sfx(lockbox)}.csv"
    return pd.read_csv(f) if f.exists() else None


def esr_with_quarantine(esr: pd.DataFrame, screen: pd.DataFrame | None) -> pd.DataFrame:
    """ESR table with quarantined cells' statistics blanked (Addition 9, amendment 8).

    Without a screen file every lasso-QR row is blanked as well (fail safe: the model is
    quarantined until a corrected estimator is validated)."""
    out = esr.copy()
    stat_cols = [
        c
        for c in out.columns
        if c not in ("model", "alpha", "n", "n_tail") and not c.endswith("_status")
    ]
    if screen is not None:
        bad = screen.loc[screen.quarantined.astype(bool), ["model", "alpha"]]
        mask = out.set_index(["model", "alpha"]).index.isin(bad.set_index(["model", "alpha"]).index)
    else:
        mask = out.model.eq("linqr_l1").to_numpy()
    out.loc[mask, stat_cols] = float("nan")
    out["quarantined"] = mask
    return out


def returns(run: Run) -> pd.Series | None:
    if run.processed is None or not (run.processed / "portfolio_daily.parquet").exists():
        return None
    return pd.read_parquet(run.processed / "portfolio_daily.parquet")["ret"]


def features(run: Run) -> pd.DataFrame | None:
    if run.processed is None or not (run.processed / "features_daily.parquet").exists():
        return None
    return pd.read_parquet(run.processed / "features_daily.parquet")


def forecasts(run: Run, lockbox: bool = False) -> pd.DataFrame | None:
    """Long frame: origin index, model, var_*, es_*, sigma, floored, loss_next, target_date.

    Lockbox view keeps only targets inside the held-out period."""
    f = run.path / f"forecasts{_sfx(lockbox)}.parquet"
    r = returns(run)
    if not f.exists() or r is None:
        return None
    fc = pd.read_parquet(f)
    parts = [
        attach_outcomes(g.drop(columns="model"), r).assign(model=m)
        for m, g in fc.groupby("model", sort=False)
    ]
    out = pd.concat(parts)
    if lockbox and run.lockbox_start:
        out = out[out["target_date"] >= pd.Timestamp(run.lockbox_start)]
    return out


def headline(run: Run, lockbox: bool = False) -> pd.DataFrame | None:
    """One row per model and alpha: FZ0 rank/MCS, hit rate, CC p, ES severity."""
    losses, cov, es = (
        table(run, n, lockbox) for n in ("backtest_losses", "backtest_coverage", "backtest_es")
    )
    if losses is None or cov is None:
        return None
    out = losses.merge(
        cov[["model", "alpha", "n", "hits", "hit_rate", "p_cc"]], on=["model", "alpha"]
    )
    if es is not None:
        out = out.merge(
            es[["model", "alpha", "severity_ratio", "p_raw"]], on=["model", "alpha"], how="left"
        )
    out["label"] = out["model"].map(label)
    return out.sort_values(["alpha", "mean_fz0"]).reset_index(drop=True)


def shap_by_family(run: Run, model: str = "mlvol_fhs") -> pd.DataFrame | None:
    """SHAP contributions to VaR summed by feature family, plus base and VaR."""
    f = run.path / f"explain_shap_{model}.parquet"
    if not f.exists():
        return None
    s = pd.read_parquet(f)
    feats = [c for c in s.columns if c not in ("base", "var_shap")]
    fam = s[feats].T.groupby(lambda c: FAMILIES.get(c, "other")).sum().T
    return fam.assign(base=s["base"], var=s["var_shap"])


def market_conditions(run: Run, start: str, end: str) -> dict | None:
    r, f = returns(run), features(run)
    if r is None:
        return None
    x = r.loc[start:end]
    out = {
        "days": int(len(x)),
        "ann_vol": float(x.std() * 252**0.5),
        "worst_day": float(x.min()),
        "worst_date": str(x.idxmin().date()) if len(x) else None,
        "big_moves": int((x.abs() > 0.04).sum()),
    }
    if f is not None and "vix" in f:
        v = f["vix"].loc[start:end]
        out |= {"mean_vix": float(v.mean()), "share_vix_gt30": float((v > 30).mean())}
    return out
