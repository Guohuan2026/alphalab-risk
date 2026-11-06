"""One function that turns evaluated forecasts into every backtest table.

Input: {model name: frame with var_*, es_*, sigma, loss_next, target_date}, all on the same
origins. Output tables:

- coverage : Kupiec / Christoffersen / CC per model and α (+ Basel traffic light at 1%)
- es       : McNeil–Frey per model and α
- losses   : mean tick and FZ0 loss per model and α, with Model Confidence Set membership
- dm       : pairwise Diebold–Mariano tests on FZ0 per α (A vs B; stat < 0 = A better)
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from alphalab.backtest.compare import diebold_mariano, model_confidence_set
from alphalab.backtest.coverage import coverage_tests, traffic_light
from alphalab.backtest.es import mcneil_frey
from alphalab.backtest.losses import fz0_loss, tick_loss


def align(evs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict every model to the origins they all share (fair comparison)."""
    common = None
    for f in evs.values():
        common = f.index if common is None else common.intersection(f.index)
    return {k: f.loc[common] for k, f in evs.items()}


def evaluate(
    evs: dict[str, pd.DataFrame], alphas, mcs_reps: int = 2000, seed: int = 0
) -> dict[str, pd.DataFrame]:
    evs = align(evs)
    cov, es_rows, loss_rows, dm_rows = [], [], [], []
    for a in alphas:
        fz, tk = {}, {}
        for name, ev in evs.items():
            loss, var, es = ev["loss_next"], ev[f"var_{a:g}"], ev[f"es_{a:g}"]
            hits = (loss > var).to_numpy()
            row = {"model": name, **coverage_tests(hits, a).to_dict()}
            row["first_target"] = str(ev["target_date"].iloc[0].date())
            row["last_target"] = str(ev["target_date"].iloc[-1].date())
            if np.isclose(a, 0.01):
                row |= {f"tl_{k}": v for k, v in traffic_light(hits).items()}
            cov.append(row)
            sigma = ev["sigma"] if ev["sigma"].notna().all() else None
            es_rows.append(
                {"model": name, **mcneil_frey(loss, var, es, a, sigma, seed=seed).to_dict()}
            )
            fz[name] = fz0_loss(-loss, var, es, a)
            tk[name] = tick_loss(-loss, var, a)
        fz_df, tk_df = pd.DataFrame(fz), pd.DataFrame(tk)
        m_fz = model_confidence_set(fz_df, reps=mcs_reps, seed=seed).set_index("model")
        m_tk = model_confidence_set(tk_df, reps=mcs_reps, seed=seed).set_index("model")
        for name in evs:
            loss_rows.append(
                {
                    "model": name,
                    "alpha": a,
                    "mean_fz0": fz_df[name].mean(),
                    "mcs_p_fz0": m_fz.loc[name, "mcs_pvalue"],
                    "in_mcs_fz0": bool(m_fz.loc[name, "in_mcs"]),
                    "mean_tick": tk_df[name].mean(),
                    "mcs_p_tick": m_tk.loc[name, "mcs_pvalue"],
                    "in_mcs_tick": bool(m_tk.loc[name, "in_mcs"]),
                }
            )
        for x, y in combinations(evs, 2):
            dm_rows.append(
                {
                    "alpha": a,
                    "model_a": x,
                    "model_b": y,
                    **diebold_mariano(fz_df[x], fz_df[y]).to_dict(),
                }
            )
    losses = pd.DataFrame(loss_rows)
    losses["rank_fz0"] = losses.groupby("alpha")["mean_fz0"].rank().astype(int)
    return {
        "coverage": pd.DataFrame(cov),
        "es": pd.DataFrame(es_rows),
        "losses": losses,
        "dm": pd.DataFrame(dm_rows),
    }
