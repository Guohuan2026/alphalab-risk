"""AlphaLab-Risk dashboard. Run from the repo root:

    streamlit run dashboard/app.py

Read-only: shows saved results from data/results (run the pipeline first). Set
ALPHALAB_DATA_DIR to point at another data folder.
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alphalab.dashboard import data as D

st.set_page_config(page_title="AlphaLab-Risk", page_icon="📉", layout="wide")

DATA_DIR = os.environ.get("ALPHALAB_DATA_DIR", "data")
MODE = "dark" if getattr(getattr(st.context, "theme", None), "type", "light") == "dark" else "light"
INK = {"light": "#52514e", "dark": "#c3c2b7"}[MODE]
GRID = {"light": "rgba(0,0,0,0.08)", "dark": "rgba(255,255,255,0.10)"}[MODE]
MUTED = {"light": "#8a8984", "dark": "#8f8e88"}[MODE]
FAMILY_COLORS = dict(
    zip(["volatility", "vix", "credit", "other"], D.PALETTE[MODE][:3] + [MUTED], strict=True)
)
DEFAULT_MODELS = ["fhs_gjr", "mlvol_fhs", "linqr_l1", "gbq"]
MAX_LINES = 4  # categorical series cap for line charts


def style(
    fig: go.Figure, height: int = 380, y_title: str | None = None, pct: bool = False
) -> go.Figure:
    """Recessive grid, one y-axis, legend on top, transparent surface (follows the theme)."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
        legend=dict(orientation="h", y=1.08, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(
        gridcolor=GRID, zeroline=False, title=y_title, tickformat=".1%" if pct else None
    )
    return fig


def fmt_p(p: float) -> str:
    return "–" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


# ---------------------------------------------------------------- sidebar
runs = D.discover_runs(DATA_DIR)
st.sidebar.title("AlphaLab-Risk")
st.sidebar.caption("ML vs econometric VaR / ES forecasting")
if not runs:
    st.title("AlphaLab-Risk")
    st.info(
        f"No results found under `{DATA_DIR}/results`. Run the pipeline first, e.g.\n\n"
        "`alphalab ingest/build/forecast/backtest --config configs/risk_p1_spy.yaml`"
    )
    st.stop()

run = runs[
    [r.name for r in runs].index(st.sidebar.selectbox("Portfolio / run", [r.name for r in runs]))
]
samples = ["Development (1997–2022)"] + (["Lockbox (2023–2026)"] if run.has_lockbox else [])
lockbox = st.sidebar.radio("Sample", samples).startswith("Lockbox")
alpha = st.sidebar.select_slider(
    "Tail probability α",
    options=[0.01, 0.025, 0.05],
    value=0.025,
    format_func=lambda a: f"{a:.1%} (VaR {1 - a:.1%})",
)
st.sidebar.divider()
st.sidebar.caption(
    "Read-only view of saved results. Rules for every analysis were committed before the "
    "results existed (docs/RESEARCH_DESIGN.md)."
)

head = D.headline(run, lockbox)
tabs = st.tabs(["Overview", "Forecasts", "Backtests", "Stress", "Explain", "Lockbox", "About"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    st.header(run.name)
    st.markdown(
        "**Can machine-learning models improve portfolio downside-risk forecasts versus "
        "econometric methods, particularly during market stress?** "
        f"Sample: **{'lockbox, opened once' if lockbox else 'development'}**."
    )
    if head is None:
        st.warning("No backtest tables for this run yet (`alphalab backtest`).")
    else:
        h = head[head.alpha == alpha].copy()
        best = h.iloc[0]
        fhs = h[h.model == "fhs_gjr"]
        c = st.columns(4)
        c[0].metric("Forecasts per model", f"{int(best.n):,}")
        c[1].metric("Lowest FZ0 loss", best.label)
        c[2].metric("Models in the 90% MCS", int(h.in_mcs_fz0.sum()))
        if len(fhs):
            c[3].metric(f"FHS-GJR hit rate (target {alpha:.1%})", f"{fhs.hit_rate.iloc[0]:.2%}")
        ref = fhs.mean_fz0.iloc[0] if len(fhs) else h.mean_fz0.min()
        h["diff"] = h.mean_fz0 - ref
        fig = go.Figure(
            go.Bar(
                x=h["diff"],
                y=h["label"],
                orientation="h",
                marker=dict(color=[D.model_color(m, MODE) for m in h.model], cornerradius=4),
                customdata=h[["mean_fz0", "rank_fz0"]],
                hovertemplate="%{y}<br>FZ0 − FHS-GJR: %{x:+.4f}<br>mean FZ0 %{customdata[0]:.4f} · rank %{customdata[1]}<extra></extra>",
            )
        )
        style(fig, height=40 + 30 * len(h)).update_layout(hovermode="closest", showlegend=False)
        fig.update_yaxes(autorange="reversed", title=None, gridcolor="rgba(0,0,0,0)")
        fig.update_xaxes(
            title="Mean FZ0 loss minus FHS-GJR (negative = better)",
            showgrid=True,
            gridcolor=GRID,
            zeroline=True,
            zerolinecolor=INK,
        )
        st.subheader(f"Joint VaR–ES loss (FZ0) at α = {alpha:.1%}")
        st.plotly_chart(fig, width="stretch", theme="streamlit")
        show = h[
            ["label", "mean_fz0", "rank_fz0", "in_mcs_fz0", "mcs_p_fz0", "hit_rate", "p_cc"]
        ].rename(
            columns={
                "label": "Model",
                "mean_fz0": "Mean FZ0",
                "rank_fz0": "Rank",
                "in_mcs_fz0": "In MCS",
                "mcs_p_fz0": "MCS p",
                "hit_rate": "Hit rate",
                "p_cc": "Coverage p (CC)",
            }
        )
        st.dataframe(
            show.style.format(
                {
                    "Mean FZ0": "{:.4f}",
                    "MCS p": "{:.3f}",
                    "Hit rate": "{:.2%}",
                    "Coverage p (CC)": fmt_p,
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Coverage p (CC) is supported inference. MCS membership and MCS p are descriptive "
            "only: the MCS excluded equally good models far too often in a size check under "
            "realistic persistence " + "(Milestone 16 calibration; docs/EVIDENCE_STATUS.md)."
        )

# ---------------------------------------------------------------- forecasts
with tabs[1]:
    st.header("VaR forecasts against realised losses")
    fc = D.forecasts(run, lockbox)
    if fc is None:
        st.warning("No forecasts for this run.")
    else:
        models = [m for m in DEFAULT_MODELS if m in set(fc.model)]
        pick = st.multiselect(
            f"Models (up to {MAX_LINES})",
            sorted(fc.model.unique(), key=D.label),
            default=models,
            format_func=D.label,
            max_selections=MAX_LINES,
        )
        lo, hi = fc.target_date.min().date(), fc.target_date.max().date()
        default = (pd.Timestamp("2020-01-02").date(), pd.Timestamp("2020-06-30").date())
        if not (lo <= default[0] <= hi):
            default = (lo, min(hi, (pd.Timestamp(lo) + pd.Timedelta(days=180)).date()))
        rng = st.slider(
            "Target dates", min_value=lo, max_value=hi, value=default, format="YYYY-MM-DD"
        )
        w = fc[(fc.target_date.dt.date >= rng[0]) & (fc.target_date.dt.date <= rng[1])]
        col = f"var_{alpha:g}"
        loss = w.drop_duplicates("target_date").set_index("target_date")["loss_next"]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=loss.index,
                y=loss,
                mode="markers",
                name="Realised loss",
                marker=dict(size=5, color=MUTED),
                hovertemplate="loss %{y:.2%}<extra></extra>",
            )
        )
        for m in pick:
            g = w[w.model == m].set_index("target_date")
            color = D.model_color(m, MODE)
            fig.add_trace(
                go.Scatter(
                    x=g.index,
                    y=g[col],
                    mode="lines",
                    name=f"{D.label(m)} VaR",
                    line=dict(width=2, color=color),
                    hovertemplate="%{y:.2%}",
                )
            )
            hits = g[g.loss_next > g[col]]
            fig.add_trace(
                go.Scatter(
                    x=hits.index,
                    y=hits.loss_next,
                    mode="markers",
                    showlegend=False,
                    name=f"{D.label(m)} exceedance",
                    marker=dict(symbol="x-thin", size=10, line=dict(width=2, color=color)),
                    hovertemplate=f"{D.label(m)} breached<extra></extra>",
                )
            )
        st.plotly_chart(
            style(fig, 440, f"Loss / VaR at α = {alpha:.1%}", pct=True), width="stretch"
        )
        st.caption(
            "Dots: realised next-day loss. Lines: VaR forecast made the day before. ✕: a breach (loss > VaR) "
            "in that model's colour."
        )
        counts = [
            {
                "Model": D.label(m),
                "Breaches": int((w[w.model == m].loss_next > w[w.model == m][col]).sum()),
                "Expected": round(alpha * w[w.model == m].shape[0], 1),
            }
            for m in pick
        ]
        st.dataframe(pd.DataFrame(counts), hide_index=True)

# ---------------------------------------------------------------- backtests
with tabs[2]:
    st.header("Backtests")
    cov, es, dm = (
        D.table(run, n, lockbox) for n in ("backtest_coverage", "backtest_es", "backtest_dm")
    )
    if cov is None:
        st.warning("No backtest tables for this run.")
    else:
        st.subheader("Coverage: are breaches as frequent as α and unclustered?")
        cv = cov[cov.alpha == alpha].sort_values("p_cc", ascending=False)
        cols = ["Model", "n", "hits", "hit_rate", "p_pof", "p_ind", "p_cc"]
        dq = D.table(run, "backtest_dq", lockbox)  # development samples only (RESULTS_LOG)
        if dq is not None:
            cv = cv.merge(
                dq[dq.alpha == alpha][["model", "p_value"]].rename(columns={"p_value": "p_dq"}),
                on="model",
                how="left",
            )
            cols.append("p_dq")
        cv = cv.assign(Model=cv.model.map(D.label))[cols]
        st.dataframe(
            cv.rename(
                columns={
                    "hits": "Breaches",
                    "hit_rate": "Hit rate",
                    "p_pof": "Kupiec p",
                    "p_ind": "Independence p",
                    "p_cc": "CC p",
                    "p_dq": "DQ p",
                }
            ).style.format(
                {
                    "Hit rate": "{:.2%}",
                    "Kupiec p": fmt_p,
                    "Independence p": fmt_p,
                    "CC p": fmt_p,
                    "DQ p": fmt_p,
                }
            ),
            hide_index=True,
            width="stretch",
        )
        if dq is None and lockbox:
            st.caption("DQ test: development samples only (added after the lockbox was opened).")
        if es is not None:
            st.subheader(
                "Expected Shortfall (McNeil–Frey): are tail losses as large as ES predicted?"
            )
            e = es[es.alpha == alpha].sort_values("severity_ratio")
            ecols = ["Model", "n_hits", "severity_ratio", "p_raw", "p_std"]
            z2 = D.table(run, "backtest_z2", lockbox)  # development samples only (RESULTS_LOG)
            if z2 is not None:
                e = e.merge(
                    z2[z2.alpha == alpha][["model", "z2", "p_value"]].rename(
                        columns={"p_value": "p_z2"}
                    ),
                    on="model",
                    how="left",
                )
                ecols += ["z2", "p_z2"]
            esr = D.table(run, "backtest_esr", lockbox)  # development samples only (Addition 9)
            if esr is not None:
                esr = D.esr_with_quarantine(esr, D.table(run, "backtest_esr_screen", lockbox))
                e = e.merge(
                    # Addition 9, amendment 11: ESR p-values are retired from inference;
                    # only the descriptive statistics are shown (the tables keep the p columns)
                    esr[esr.alpha == alpha][["model", "strict_g1", "strict_wald", "int_t"]],
                    on="model",
                    how="left",
                )
                ecols += ["strict_g1", "strict_wald", "int_t"]
            e = e.assign(Model=e.model.map(D.label))[ecols]
            st.dataframe(
                e.rename(
                    columns={
                        "n_hits": "Breaches",
                        "severity_ratio": "Realised / forecast ES",
                        "p_raw": "p (raw)",
                        "p_std": "p (standardised)",
                        "z2": "Z2",
                        "p_z2": "Z2 p",
                        "strict_g1": "ESR slope",
                        "strict_wald": "ESR Wald",
                        "int_t": "ESR intercept t",
                    }
                ).style.format(
                    {
                        "Realised / forecast ES": "{:.3f}",
                        "p (raw)": fmt_p,
                        "p (standardised)": fmt_p,
                        "Z2": "{:.3f}",
                        "Z2 p": fmt_p,
                        "ESR slope": "{:.2f}",
                        "ESR Wald": "{:.1f}",
                        "ESR intercept t": "{:.2f}",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Ratio above 1 = ES understates real tail losses (the dangerous direction). "
                "Acerbi–Szekely Z2 below 0 = breaches too frequent and/or too severe. "
                "McNeil–Frey and Z2 p-values are descriptive in most cells: both over-reject "
                "when volatility dynamics are misspecified (Milestone 16 calibration; "
                "docs/EVIDENCE_STATUS.md)"
                + (
                    "."
                    if z2 is not None
                    else "; Z2 is development-only (added after the lockbox was opened)."
                )
            )
            if esr is not None:
                st.caption(
                    "ESR (descriptive diagnostics only, development samples): slope 1 and "
                    "intercept t near 0 mean the forecast ES matches the realised tail; a "
                    "negative intercept t points to ES understated. No p-values are shown: ESR "
                    "inference was retired after its bootstrap failed the pre-specified size "
                    "check under serial dependence (Addition 9, amendment 11; "
                    "docs/EVIDENCE_STATUS.md)."
                )
                q = esr.loc[esr.quarantined, "model"].map(D.label).unique()
                if len(q):
                    st.caption(
                        "ESR statistics unavailable (degenerate covariance fit; ESR quarantine "
                        f"only, other results unaffected): {', '.join(q)}."
                    )
        if dm is not None:
            st.subheader("Diebold–Mariano on FZ0 (row vs column; negative = row better)")
            st.caption(
                "Descriptive only: the DM test over-rejected badly (up to 50% at nominal 5%) "
                "in a size check with persistent loss differentials, so these p-values do not "
                "establish significance " + "(Milestone 16 calibration; docs/EVIDENCE_STATUS.md)."
            )
            d = dm[dm.alpha == alpha]
            names = sorted(
                set(d.model_a) | set(d.model_b), key=lambda m: (m not in D.MODEL_ORDER, D.label(m))
            )
            mat = pd.DataFrame(index=names, columns=names, dtype=float)
            pmat = mat.copy()
            for _, row in d.iterrows():
                mat.loc[row.model_a, row.model_b], mat.loc[row.model_b, row.model_a] = (
                    row.stat,
                    -row.stat,
                )
                pmat.loc[row.model_a, row.model_b] = pmat.loc[row.model_b, row.model_a] = (
                    row.p_value
                )
            lim = float(min(4, max(2, mat.abs().max().max())))
            mid = {"light": "#f0efec", "dark": "#383835"}[MODE]
            blue, red = D.PALETTE[MODE][0], D.PALETTE[MODE][7]
            fig = go.Figure(
                go.Heatmap(
                    z=mat.values,
                    x=[D.label(m) for m in names],
                    y=[D.label(m) for m in names],
                    zmin=-lim,
                    zmax=lim,
                    colorscale=[[0, blue], [0.5, mid], [1, red]],
                    customdata=pmat.values,
                    xgap=2,
                    ygap=2,
                    hovertemplate="%{y} vs %{x}<br>DM stat %{z:.2f}<br>p %{customdata:.3f}<extra></extra>",
                    colorbar=dict(title="DM stat"),
                )
            )
            style(fig, 460).update_layout(hovermode="closest")
            fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- stress
with tabs[3]:
    st.header("Stress periods (H3)")
    if lockbox:
        st.info(
            "Stress analysis covers the development sample only: no lockbox episodes were pre-specified."
        )
    reg, ols, ev = (
        D.table(run, n) for n in ("stress_regimes", "stress_regression", "stress_event_paths")
    )
    if reg is None:
        st.warning("No stress tables for this run (`alphalab stress`).")
    else:
        pick = st.multiselect(
            "Models",
            sorted(reg.model.unique(), key=D.label),
            default=[m for m in DEFAULT_MODELS if m in set(reg.model)],
            format_func=D.label,
            max_selections=MAX_LINES,
            key="stress_models",
        )
        regimes = [r for r in ("calm", "episodes", "vix_le30", "vix_gt30") if r in set(reg.regime)]
        r = reg[(reg.alpha == alpha) & reg.regime.isin(regimes) & reg.model.isin(pick)]
        names = {
            "calm": "Calm",
            "episodes": "Crisis episodes",
            "vix_le30": "VIX ≤ 30",
            "vix_gt30": "VIX > 30",
        }
        fig = go.Figure()
        for m in pick:
            g = r[r.model == m].set_index("regime").reindex(regimes)
            fig.add_trace(
                go.Bar(
                    x=[names[x] for x in regimes],
                    y=g.hit_rate,
                    name=D.label(m),
                    marker=dict(color=D.model_color(m, MODE), cornerradius=4),
                    customdata=g[["hits", "expected_hits"]],
                    hovertemplate="%{y:.2%} (%{customdata[0]} breaches, %{customdata[1]:.1f} expected)",
                )
            )
        fig.add_hline(
            y=alpha,
            line=dict(color=INK, width=1, dash="dot"),
            annotation_text=f"target {alpha:.1%}",
        )
        style(fig, 380, "Breach rate", pct=True).update_layout(
            barmode="group", bargap=0.25, bargroupgap=0.08
        )
        st.subheader(f"Breach rate by regime at α = {alpha:.1%}")
        st.plotly_chart(fig, width="stretch")
        if ols is not None:
            st.subheader("Loss differential vs FHS-GJR: calm (a) and extra in stress (b)")
            o = ols[(ols.alpha == alpha)].assign(Model=lambda x: x.model.map(D.label))
            st.dataframe(
                o[["dummy", "Model", "calm_diff", "p_calm", "stress_shift", "p_shift", "n_stress"]]
                .rename(
                    columns={
                        "dummy": "Stress definition",
                        "calm_diff": "a: calm",
                        "p_calm": "p(a)",
                        "stress_shift": "b: stress shift",
                        "p_shift": "p(b)",
                        "n_stress": "Stress days",
                    }
                )
                .style.format(
                    {
                        "a: calm": "{:+.4f}",
                        "b: stress shift": "{:+.4f}",
                        "p(a)": fmt_p,
                        "p(b)": fmt_p,
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Positive = worse than FHS-GJR. A non-significant b is not proof of no stress effect: FZ0 is "
                "noisy on volatile days (see METHODOLOGY §14)."
            )
        if ev is not None:
            st.subheader(
                "How fast VaR reacts after the largest losses (1% VaR, relative to 5 days before)"
            )
            ev = ev.set_index("k")
            fig = go.Figure()
            for m in pick:
                if m in ev:
                    fig.add_trace(
                        go.Scatter(
                            x=ev.index,
                            y=ev[m],
                            name=D.label(m),
                            mode="lines",
                            line=dict(width=2, color=D.model_color(m, MODE)),
                            hovertemplate="%{y:.2f}×",
                        )
                    )
            fig.add_vline(
                x=0, line=dict(color=INK, width=1, dash="dot"), annotation_text="crash day"
            )
            style(fig, 360, "VaR relative to k = −5")
            fig.update_xaxes(title="Trading days relative to the large loss")
            st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- explain
with tabs[4]:
    st.header("What drives the ML forecasts (H4)")
    perm, crit = D.table(run, "explain_permutation"), D.table(run, "explain_criteria")
    if perm is None:
        st.warning("No interpretability tables for this run (`alphalab explain`).")
    else:
        ml_models = sorted(perm.model.unique(), key=D.label)
        start = ml_models.index("mlvol_fhs") if "mlvol_fhs" in ml_models else 0
        m = st.selectbox("Model", ml_models, index=start, format_func=D.label)
        imp = perm[perm.model == m].groupby("feature").importance.mean().sort_values().tail(10)
        fam = [D.FAMILIES.get(f, "other") for f in imp.index]
        fig = go.Figure(
            go.Bar(
                x=imp.values,
                y=imp.index,
                orientation="h",
                marker=dict(color=[FAMILY_COLORS[f] for f in fam], cornerradius=4),
                customdata=fam,
                hovertemplate="%{y} (%{customdata})<br>%{x:.2e}<extra></extra>",
            )
        )
        style(fig, 380).update_layout(hovermode="closest", showlegend=False)
        fig.update_xaxes(
            title="Rise in validation tick loss when the feature is shuffled",
            showgrid=True,
            gridcolor=GRID,
        )
        fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
        st.subheader("Top-10 permutation importance (mean over yearly refits)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Colour = feature family: " + " · ".join(f"{k}" for k in FAMILY_COLORS))
        if crit is not None:
            st.dataframe(
                crit[crit.model == m][["criterion", "value", "passed"]],
                hide_index=True,
                width="stretch",
            )
        sh = D.shap_by_family(run, m) if m in ("mlvol_fhs", "gbq") else None
        if sh is not None:
            st.subheader("SHAP contribution to the 1% VaR over time, by feature family")
            fig = go.Figure()
            # Lines, not a stacked area: contributions can be negative, and stacking mixed signs
            # draws shapes that correspond to no quantity.
            for f in ["volatility", "vix", "credit", "other"]:
                if f in sh:
                    fig.add_trace(
                        go.Scatter(
                            x=sh.index,
                            y=sh[f],
                            name=f,
                            mode="lines",
                            line=dict(width=1.5, color=FAMILY_COLORS[f]),
                            hovertemplate="%{y:+.2%}",
                        )
                    )
            fig.add_hline(y=0, line=dict(color=INK, width=1))
            style(fig, 380, "Push on VaR vs the model's average", pct=True)
            st.plotly_chart(fig, width="stretch")
            if m == "mlvol_fhs":
                st.caption(
                    "2008: the model refit in Jan 2008 had seen VIX ≤ 45.7, so its VaR stayed capped near 3.8% while "
                    "VIX reached 70 (trees cannot extrapolate). RESULTS_LOG, H4."
                )
            else:
                st.caption(
                    "Gradient boosting: SHAP explains the raw 1% quantile model; on 66% of days the reported VaR was "
                    "re-sorted by the quantile-crossing fix (RESULTS_LOG, H4)."
                )

# ---------------------------------------------------------------- lockbox
with tabs[5]:
    st.header("Lockbox 2023–2026: opened once")
    dev_h, lb_h = D.headline(run, False), D.headline(run, True) if run.has_lockbox else None
    if lb_h is None:
        st.info("No lockbox results for this run.")
    else:
        st.error(
            "**Not replicated** by the pre-stated rule (protocol pushed before the run): in the calm "
            "2023–26 period, feature-based models had lower mean FZ0 than FHS-GJR. On SPY the differences "
            "are not significant; FHS-GJR stayed well calibrated.",
            icon="⚠️",
        )
        comp = dev_h[dev_h.alpha == alpha][["label", "rank_fz0", "hit_rate"]].merge(
            lb_h[lb_h.alpha == alpha][["label", "rank_fz0", "hit_rate", "in_mcs_fz0"]],
            on="label",
            suffixes=(" dev", " lockbox"),
        )
        st.dataframe(
            comp.sort_values("rank_fz0 lockbox")
            .rename(
                columns={
                    "label": "Model",
                    "rank_fz0 dev": "FZ0 rank (development)",
                    "hit_rate dev": "Hit rate (development)",
                    "rank_fz0 lockbox": "FZ0 rank (lockbox)",
                    "hit_rate lockbox": "Hit rate (lockbox)",
                    "in_mcs_fz0": "In MCS (lockbox)",
                }
            )
            .style.format({"Hit rate (development)": "{:.2%}", "Hit rate (lockbox)": "{:.2%}"}),
            hide_index=True,
            width="stretch",
        )
        dev_m = D.market_conditions(run, "1997-01-14", "2022-12-29")
        lb_m = D.market_conditions(run, "2023-01-03", "2026-10-30")
        if dev_m and lb_m:
            st.subheader("Market conditions (descriptive)")
            c = st.columns(3)
            if "share_vix_gt30" in lb_m:
                c[0].metric(
                    "Days with VIX > 30",
                    f"{lb_m['share_vix_gt30']:.1%}",
                    f"{lb_m['share_vix_gt30'] - dev_m['share_vix_gt30']:+.1%} vs development",
                    delta_color="off",
                )
            c[1].metric(
                "Annualised volatility",
                f"{lb_m['ann_vol']:.1%}",
                f"{lb_m['ann_vol'] - dev_m['ann_vol']:+.1%}",
                delta_color="off",
            )
            c[2].metric(
                "Daily moves > 4%",
                lb_m["big_moves"],
                f"{lb_m['big_moves'] - dev_m['big_moves']:+d}",
                delta_color="off",
            )
            st.caption(
                "The calm-regime explanation was formed after seeing the lockbox result: post hoc, untested."
            )

# ---------------------------------------------------------------- about
with tabs[6]:
    st.header("About this study")
    st.markdown(
        """
- **Question.** Can ML improve VaR and Expected Shortfall forecasts versus econometric models, especially in stress?
- **Design.** 11 models (HS, EWMA, GARCH-t, GJR-t, FHS-GJR, HAR-FHS, lasso QR control, quantile forest,
  gradient boosting, RF-volatility + FHS), 1-day forecasts, walk-forward with no look-ahead, frozen at git tag
  `prereg-v1`; every later rule committed before its results (Deviations and Clarifications 1–6).
- **Evaluation.** Kupiec / Christoffersen coverage, DQ, McNeil–Frey and Acerbi–Szekely Z2 ES tests, FZ0 joint loss, Diebold–Mariano,
  Model Confidence Set, stress regimes, SHAP and permutation importance, lockbox opened once.
- **Findings.** In development, no ML model beat FHS-GJR, and ML lost ground in crises. In the calm 2023–26 lockbox,
  feature-based models did better on average: the conclusion is regime-dependent, with FHS-GJR the robust choice.
- **Read more.** `docs/RESEARCH_DESIGN.md`, `docs/METHODOLOGY.md`, `docs/RESULTS_LOG.md`, `docs/LIMITATIONS.md`.
"""
    )
