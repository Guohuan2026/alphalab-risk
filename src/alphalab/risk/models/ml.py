"""Machine-learning risk models (RESEARCH_DESIGN §5, specified in Deviation 3).

All three share ONE tuning skeleton (``_TunedFeatureModel``), so the protocol is identical by
construction, and identical to the linear control's:

    training pairs (x_s, r_{s+1}) → last ``val_frac`` held out, ``embargo`` pairs dropped
    between → each grid config fitted on the earlier part → validation score = mean tick
    loss over the VaR levels → best config (ties: first) refitted on the whole window.

Models
------
QRF ("qrf")       Quantile Regression Forest (Meinshausen 2006). A random forest whose
                  leaves keep the training targets, so one fitted forest gives the whole
                  conditional distribution: any quantile, no crossing.
GBQ ("gbq")       Gradient boosting with the quantile (pinball) loss, one model per level
                  (scikit-learn HistGradientBoosting; XGBoost unavailable, Deviation 3).
                  Quantiles are sorted across levels to remove crossing.
ML-vol ("mlvol")  Random forest predicting |r_{s+1}| (a volatility proxy), then FHS: VaR/ES =
                  σ̃_t × empirical VaR/ES of residuals z = r/σ̃. The residuals use OUT-OF-BAG
                  predictions (each tree predicts only days it did not train on); in-sample
                  forest fits nearly memorise |r| and would make z, and so VaR, too small.
                  ML learns the scale from all days; the tail shape comes from the data, the
                  answer to "too few tail points" for direct quantile models at 1%.

ES for QRF/GBQ: mean of quantiles at α(k−½)/K, K = 5 (as the control). Trees need no
scaling or clipping: they are invariant to monotone transforms and cannot extrapolate
beyond the training targets.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

from alphalab.backtest.losses import tick_loss
from alphalab.risk.models.base import Forecast, RiskModel
from alphalab.risk.models.linear import es_levels


@dataclass
class _TunedFeatureModel(RiskModel):
    features: Sequence[str] | None = None
    grid: Sequence[dict] = field(default_factory=list)
    es_k: int = 5
    val_frac: float = 0.2
    embargo: int = 1
    min_obs: int = 500
    seed: int = 0
    uses_features: ClassVar[bool] = True
    stateless: ClassVar[bool] = True

    def __post_init__(self):
        super().__post_init__()
        if self.refit_every is None:
            self.refit_every = 252  # Deviation 3: ML models refit and re-tune yearly
        self.grid = [dict(g) for g in (self.grid or self.default_grid())]
        self.levels = np.unique(
            np.concatenate([[a, *es_levels(a, self.es_k)] for a in self.alphas])
        )
        self.cols: list[str] = []
        self.best: dict = {}
        self.val_losses: list[float] = []
        self._obj = None

    @staticmethod
    @abstractmethod
    def default_grid() -> list[dict]: ...

    @abstractmethod
    def _fit_one(self, cfg: dict, X: np.ndarray, y: np.ndarray, levels) -> object: ...

    @abstractmethod
    def _var(self, obj, X: np.ndarray) -> dict[float, np.ndarray]:
        """VaR (positive loss) per alpha for each row; used for tuning."""

    @abstractmethod
    def _forecasts(self, obj, X: np.ndarray) -> list[Forecast]: ...

    @property
    def min_history(self) -> int:
        return self.min_obs

    def _pairs(self, train: np.ndarray, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(train, dtype=float)[1:]
        Xs = X[self.cols].iloc[:-1].to_numpy(dtype=float)
        ok = np.isfinite(Xs).all(axis=1)
        if ok.sum() < 100:
            raise ValueError(f"{self.name}: only {ok.sum()} complete training rows")
        return Xs[ok], y[ok]

    def _split(self, n: int) -> tuple[int, int]:
        """(n_fit, n_val): tuning part, then embargo, then validation part."""
        n_val = int(n * self.val_frac)
        return n - n_val - self.embargo, n_val

    def fit(self, train: np.ndarray, X: pd.DataFrame):
        self.cols = list(self.features) if self.features is not None else list(X.columns)
        Xs, y = self._pairs(train, X)
        n_fit, n_val = self._split(len(y))
        self.val_losses = []
        best, best_loss = self.grid[0], np.inf
        for cfg in self.grid:
            obj = self._fit_one(cfg, Xs[:n_fit], y[:n_fit], self.alphas)
            var = self._var(obj, Xs[-n_val:])
            loss = float(np.mean([tick_loss(y[-n_val:], var[a], a).mean() for a in self.alphas]))
            self.val_losses.append(loss)
            if loss < best_loss:
                best, best_loss = cfg, loss
        self.best = best
        self._obj = self._fit_one(best, Xs, y, self.levels)
        return self

    # ---------- interpretability (H4, Clarification 5) ----------
    def permutation_importance(
        self, train: np.ndarray, X: pd.DataFrame, repeats: int = 5, seed: int = 0
    ) -> pd.DataFrame:
        """Rise in validation tick loss when one feature is shuffled.

        Uses exactly the tuning split of ``fit``: the chosen configuration is fitted on the
        tuning part and scored on the validation part, so the importance is out-of-sample
        for the model being scored. Call after ``fit`` (needs the chosen configuration)."""
        if not self.best:
            raise RuntimeError("call fit() before permutation_importance()")
        Xs, y = self._pairs(train, X)
        n_fit, n_val = self._split(len(y))
        obj = self._fit_one(self.best, Xs[:n_fit], y[:n_fit], self.alphas)
        Xv, yv = Xs[-n_val:], y[-n_val:]

        def score(Xm):
            var = self._var(obj, Xm)
            return float(np.mean([tick_loss(yv, var[a], a).mean() for a in self.alphas]))

        base = score(Xv)
        rng = np.random.default_rng(seed)
        rows = []
        for j, col in enumerate(self.cols):
            rises = []
            for _ in range(repeats):
                Xp = Xv.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                rises.append(score(Xp) - base)
            rows.append(
                {"feature": col, "importance": float(np.mean(rises)), "std": float(np.std(rises))}
            )
        return pd.DataFrame(rows)

    def shap_var(self, X: pd.DataFrame, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """(base value, per-feature contributions) to VaR_alpha for each row: TreeSHAP,
        exact additivity, so base + contributions.sum(1) equals the model's VaR."""
        raise NotImplementedError(f"{self.name}: TreeSHAP does not explain this model")

    def fit_info(self) -> dict:
        info = {f"cfg_{k}": v for k, v in self.best.items()}
        info["val_loss"] = min(self.val_losses) if self.val_losses else np.nan
        return info

    def predict_block(self, X: pd.DataFrame) -> list[Forecast]:
        x = X[self.cols].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            bad = X.index[~np.isfinite(x).all(axis=1)][0]
            raise ValueError(f"{self.name}: missing feature values at {bad}")
        return self._forecasts(self._obj, x)

    def predict(self, history: np.ndarray, X: pd.DataFrame) -> Forecast:
        return self.predict_block(X.iloc[[-1]])[0]

    # shared by the two direct-quantile models
    def _quantile_forecasts(self, Q: np.ndarray) -> list[Forecast]:
        """Q: (n_rows, n_levels) return quantiles at self.levels -> Forecasts."""
        Q = np.sort(Q, axis=1)  # no crossing
        pos = {lv: i for i, lv in enumerate(self.levels)}
        out = []
        for row in Q:
            var = {a: -float(row[pos[a]]) for a in self.alphas}
            es = {
                a: -float(np.mean([row[pos[t]] for t in es_levels(a, self.es_k)]))
                for a in self.alphas
            }
            out.append(Forecast(var, es))
        return out


@dataclass
class QuantileRandomForest(_TunedFeatureModel):
    n_estimators: int = 300
    max_features: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if self.name == "quantilerandomforest":
            self.name = "qrf"

    @staticmethod
    def default_grid() -> list[dict]:
        return [{"min_samples_leaf": m} for m in (10, 50, 200)]

    def _fit_one(self, cfg, X, y, levels):
        from quantile_forest import RandomForestQuantileRegressor

        return RandomForestQuantileRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            # Pinned explicitly: these were the library defaults when every logged result was
            # produced (quantile-forest 1.4.2). max_samples_leaf=1 keeps ONE training value per
            # leaf, so a forecast distribution has ~n_estimators values: coarser than
            # Meinshausen's original QRF (LIMITATIONS #14). Kept, not changed: results are frozen.
            max_samples_leaf=1,
            n_jobs=-1,
            random_state=self.seed,
            **cfg,
        ).fit(X, y)

    def _q(self, obj, X, levels) -> np.ndarray:
        # interpolation pinned: quantile-forest 1.5.0 changed its default from "linear" to
        # "averaged_inverted_cdf", which moved forecasts by ~20% (found by the clean-clone check).
        q = np.asarray(obj.predict(X, quantiles=[float(t) for t in levels], interpolation="linear"))
        return q.reshape(len(X), len(levels))

    def _var(self, obj, X):
        q = self._q(obj, X, self.alphas)
        return {a: -q[:, i] for i, a in enumerate(self.alphas)}

    def _forecasts(self, obj, X):
        return self._quantile_forecasts(self._q(obj, X, self.levels))


@dataclass
class GradientBoostingQuantile(_TunedFeatureModel):
    learning_rate: float = 0.05
    max_iter: int = 200

    def __post_init__(self):
        super().__post_init__()
        if self.name == "gradientboostingquantile":
            self.name = "gbq"

    @staticmethod
    def default_grid() -> list[dict]:
        return [
            {"max_leaf_nodes": leaves, "min_samples_leaf": m}
            for leaves in (7, 31)
            for m in (50, 200)
        ]

    def _fit_one(self, cfg, X, y, levels):
        from sklearn.ensemble import HistGradientBoostingRegressor

        return {
            float(t): HistGradientBoostingRegressor(
                loss="quantile",
                quantile=float(t),
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                early_stopping=False,  # deterministic, no internal validation split
                random_state=self.seed,
                **cfg,
            ).fit(X, y)
            for t in levels
        }

    def _var(self, obj, X):
        return {a: -obj[float(a)].predict(X) for a in self.alphas}

    def _forecasts(self, obj, X):
        Q = np.column_stack([obj[float(t)].predict(X) for t in self.levels])
        return self._quantile_forecasts(Q)

    def shap_var(self, X, alpha):
        import shap

        model = self._obj[float(alpha)]
        e = shap.TreeExplainer(model)
        sv = e.shap_values(X[self.cols].to_numpy(dtype=float))
        return np.full(len(X), -float(np.ravel(e.expected_value)[0])), -sv  # VaR = −quantile


@dataclass
class MLVolFHS(_TunedFeatureModel):
    n_estimators: int = 300
    max_features: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if self.name == "mlvolfhs":
            self.name = "mlvol_fhs"

    @staticmethod
    def default_grid() -> list[dict]:
        return [{"min_samples_leaf": m} for m in (50, 200, 500)]

    def _fit_one(self, cfg, X, y, levels):
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            oob_score=True,
            n_jobs=-1,
            random_state=self.seed,
            **cfg,
        ).fit(X, np.abs(y))
        # Parallel FITTING is deterministic (same trees for the same seed), but parallel
        # PREDICTION sums the tree outputs in thread order, and floating-point addition is not
        # associative, so repeated runs differ in the last bits (found by a test). Predict
        # single-threaded for bit-identical reruns.
        rf.set_params(n_jobs=1)
        scale = np.maximum(rf.oob_prediction_, 1e-6)  # honest (out-of-bag) fitted scale
        loss_z = -(y / scale)
        tail = {}
        for a in self.alphas:
            q = float(np.quantile(loss_z, 1 - a))
            tail[a] = (q, float(loss_z[loss_z >= q].mean()))
        return {"rf": rf, "tail": tail}

    def _sigma(self, obj, X) -> np.ndarray:
        return np.maximum(obj["rf"].predict(X), 1e-6)

    def _var(self, obj, X):
        s = self._sigma(obj, X)
        return {a: s * obj["tail"][a][0] for a in self.alphas}

    def shap_var(self, X, alpha):
        import shap

        q = self._obj["tail"][alpha][0]
        e = shap.TreeExplainer(self._obj["rf"])
        sv = e.shap_values(X[self.cols].to_numpy(dtype=float))
        return np.full(len(X), q * float(np.ravel(e.expected_value)[0])), q * sv  # VaR = q·σ̃

    def _forecasts(self, obj, X):
        s = self._sigma(obj, X)
        out = []
        for si in s:
            var = {a: float(si * q) for a, (q, _) in obj["tail"].items()}
            es = {a: float(si * e) for a, (_, e) in obj["tail"].items()}
            out.append(Forecast(var, es, float(si)))
        return out
