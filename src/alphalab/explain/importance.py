"""H4: do the ML models respond to economically meaningful drivers?

Rules (RESEARCH_DESIGN Clarification 5, committed before any value was computed):

- Permutation importance at every yearly refit, on that refit's validation window.
- TreeSHAP contributions to VaR (lowest α) for every out-of-sample forecast, for the models
  that support it (mlvol_fhs, gbq). Exactly additive: base + contributions = VaR.
- Criteria: C1 top-3 features by mean permutation importance are volatility/VIX features;
  C2 positive correlation between each volatility/VIX feature's value and its SHAP
  contribution; C3 median Spearman correlation of importance vectors between consecutive
  refits ≥ 0.6.

The explain pass refits the models on the forecast engine's schedule. Because refits are
bit-identical (determinism tests), these are the same models that produced the forecasts;
``explain_model`` returns the SHAP-implied VaR so callers can verify that against the
saved forecasts.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from alphalab.risk.engine import EngineSettings, forecast_origins

log = logging.getLogger(__name__)

FAMILIES: dict[str, str] = {
    **{
        f: "volatility"
        for f in ("rv_1", "rv_5", "rv_22", "rv_66", "abs_ret", "pk_1", "gk_1", "gk_5", "gk_22")
    },
    **{f: "vix" for f in ("vix", "vix_chg_5")},
    **{f: "credit" for f in ("baa10y", "baa10y_chg_22")},
    **{f: "other" for f in ("ret", "dd_252", "t10y3m", "t10y3m_chg_22")},
}
STABILITY_MIN = 0.6


def explain_model(
    returns: pd.Series,
    features: pd.DataFrame,
    make_model,
    settings: EngineSettings,
    repeats: int = 5,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Permutation importance per refit and SHAP contributions per forecast for one model."""
    r = returns.sort_index()
    values = r.to_numpy(dtype=float)
    X = features.reindex(r.index)
    origins = forecast_origins(r.index, settings)
    model = make_model()
    every = model.refit_every or settings.refit_every
    alpha = min(model.alphas)
    perm, shap_rows = [], []
    for k in range(0, len(origins), every):
        t = origins[k]
        lo = 0 if settings.window == "expanding" else max(0, t + 1 - settings.window_size)
        model.fit(values[lo : t + 1], X.iloc[lo : t + 1])
        imp = model.permutation_importance(values[lo : t + 1], X.iloc[lo : t + 1], repeats, seed)
        perm.append(imp.assign(refit_origin=r.index[t]))
        block = origins[k : k + every]
        Xb = X.iloc[block]
        try:
            base, contrib = model.shap_var(Xb, alpha)
        except NotImplementedError:
            continue
        df = pd.DataFrame(contrib, index=Xb.index, columns=model.cols)
        df.insert(0, "base", base)
        df["var_shap"] = base + contrib.sum(axis=1)
        shap_rows.append(df)
    out = {"permutation": pd.concat(perm, ignore_index=True)}
    if shap_rows:
        s = pd.concat(shap_rows)
        s.index.name = "origin"
        out["shap"] = s
    log.info("%s: explained %d refits", model.name, len(perm))
    return out


def h4_criteria(
    perm: pd.DataFrame, shap: pd.DataFrame | None, features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(criteria table, sign table) for one model, per Clarification 5."""
    rows = []
    mean_imp = perm.groupby("feature")["importance"].mean().sort_values(ascending=False)
    top3 = list(mean_imp.index[:3])
    fams = [FAMILIES.get(f, "other") for f in top3]
    rows.append(
        {
            "criterion": "C1_concentration",
            "value": ", ".join(f"{f} ({fa})" for f, fa in zip(top3, fams, strict=True)),
            "passed": all(fa in ("volatility", "vix") for fa in fams),
        }
    )
    signs = pd.DataFrame(columns=["feature", "family", "corr_value_vs_contribution"])
    if shap is not None:
        vals = features.reindex(shap.index)
        sign_rows = []
        for f in [c for c in shap.columns if FAMILIES.get(c) in ("volatility", "vix")]:
            c = float(np.corrcoef(vals[f], shap[f])[0, 1])
            sign_rows.append({"feature": f, "family": FAMILIES[f], "corr_value_vs_contribution": c})
        signs = pd.DataFrame(sign_rows)
        positive = signs.corr_value_vs_contribution > 0
        rows.append(
            {
                "criterion": "C2_economic_sign",
                "value": f"{int(positive.sum())}/{len(signs)} positive",
                "passed": bool(positive.all()),
            }
        )
    wide = perm.pivot(index="refit_origin", columns="feature", values="importance").sort_index()
    rho = [stats.spearmanr(wide.iloc[i], wide.iloc[i + 1]).statistic for i in range(len(wide) - 1)]
    med = float(np.median(rho)) if rho else np.nan
    rows.append(
        {
            "criterion": "C3_stability",
            "value": f"median Spearman {med:.3f} over {len(rho)} consecutive refits",
            "passed": bool(med >= STABILITY_MIN),
        }
    )
    return pd.DataFrame(rows), signs
