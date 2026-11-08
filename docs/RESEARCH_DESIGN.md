# AlphaLab-Risk — Research Design

**Machine Learning for Portfolio Risk Forecasting and Stress Resilience**

Status: **frozen at git tag `prereg-v1` (2026-11-06)**, branch `alphalab-risk`. The pre-pivot
code is preserved at tag `v0.1-equity-research`. This design was first written (commit
`d935f1d`) before any risk result existed. Before the freeze, only the development-sample
results of the HS and EWMA baselines on P1 had been seen; they are recorded in
[RESULTS_LOG.md](RESULTS_LOG.md). No GARCH-family, ML or lockbox result had been produced.

**Deviation rule.** After `prereg-v1`, this file is not edited silently. Any change to the
hypotheses, data, forecast contract, model list, tuning budget or evaluation is added to the
"Deviations" section at the end, with the date, the reason, and whether it was made before or
after seeing the affected results. `git diff prereg-v1 -- docs/RESEARCH_DESIGN.md` shows every
change. This is the honest substitute for formal pre-registration.

---

## 1. Research question and why it matters

> Can machine-learning models improve the accuracy and robustness of portfolio downside-risk
> forecasts compared with traditional econometric methods, particularly during periods of
> financial market stress?

Banks report 99% VaR and 97.5% Expected Shortfall under Basel FRTB. Pension funds and asset
managers use the same quantities for risk budgets. Model validation teams have to decide
whether a new model (often ML) is *actually* better, or only better in-sample. The academic
evidence is mixed: GARCH-family models with fat-tailed innovations are hard to beat at short
horizons, and ML gains are often reported without proper tests or out-of-sample discipline.
The contribution here is a **careful, reproducible horse race**, not a claim that ML wins.

**Possible contributions**

1. A like-for-like comparison: same data, horizon, confidence levels, estimation windows
   and information set for every model.
2. Separating *where* ML gains come from. A linear quantile regression on the same features
   isolates "better inputs" from "non-linearity" (see H2b).
3. A stress-period analysis with episodes defined **ex ante**, with an explicit discussion
   of statistical power.
4. An open, tested pipeline that a partner can rerun on its own portfolio (the industry
   adaptation path, §10).

## 2. Hypotheses

Notation: `r_{t+1}` is the portfolio's simple total return, the loss is `L_{t+1} = −r_{t+1}`,
and α is the tail probability (1%, 2.5%, 5%). VaR and ES are reported as positive loss numbers:

```
VaR_α,t : P_t( L_{t+1} > VaR_α,t ) = α          (forecast made at close t)
ES_α,t  = E_t[ L_{t+1} | L_{t+1} ≥ VaR_α,t ]     (mean loss beyond VaR)
```

| # | Hypothesis (H₀ in brackets) | Test |
|---|---|---|
| H1 | ML VaR forecasts are correctly calibrated out-of-sample [exceedance rate = α, exceedances independent] | Kupiec POF, Christoffersen IND and CC, Engle–Manganelli DQ |
| H2a | ML models have lower expected joint VaR–ES loss than the best econometric model [equal predictive ability] | FZ0 loss; Diebold–Mariano with HAC errors; Model Confidence Set |
| H2b | Any ML gain comes from non-linearity, not only from extra features [ML ≡ linear quantile regression with the same features] | Same as H2a, ML vs linear QR |
| H3 | Relative performance differs in stress periods [loss differential the same in stress and calm] | DM on stress subsample; differential regressed on a stress dummy (HAC) |
| H4 | ML risk forecasts respond to economically meaningful drivers (volatility, VIX, credit spread) | SHAP and permutation importance; stability across refits |

**The null result is a valid result.** If GJR-GARCH-t is inside the Model Confidence Set
and ML is not, that is a publishable and defensible finding: it says that complexity is not
rewarded. The write-up will treat it as such.

## 3. Data

All sources are public and free, so anyone can reproduce the results. Each download is an
immutable, hashed snapshot (code reused from M1).

| Role | Series | Source | History |
|---|---|---|---|
| P1 — US equity | SPY total return (our own dividend-inclusive returns) | Yahoo | 1993– |
| P1 cross-check | S&P 500 Total Return index `^SP500TR` | Yahoo | 1988– |
| P2 — balanced 60/40 | SPY 40%, EFA 20%, IEF 30%, LQD 10%, monthly rebalance | Yahoo | 2002-07– |
| P3 — Canadian equity | XIU.TO (iShares S&P/TSX 60), CAD, TSX calendar | Yahoo | 1999-10– |
| Feature | VIX close | Yahoo `^VIX` | 1990– |
| Feature | Credit spread Moody's Baa − 10y Treasury | FRED `BAA10Y` | 1986– |
| Feature | Term spread 10y − 3m | FRED `T10Y3M` | 1982– |
| Feature | Daily range (high/low) for range-based volatility | Yahoo OHLC | per asset |

Verified on 2026-11-06: every series above downloads, with the first dates shown.

**Why ETF/index portfolios instead of single stocks?** It removes most of the survivorship
bias flagged in M1 (an index or ETF does not "delist"). It matches how risk teams actually
forecast (portfolio-level P&L). It also keeps the study within what Yahoo can supply reliably.

**Data limitations**
- Yahoo is unofficial and restates history. This was observed in this project:
  `vendor_adj_close` moved within an hour. Hence the immutable snapshots, and returns are
  built from raw closes plus dividends.
- Early `^GSPC` OHLC values may simply be copies of the close (no real range). This must be
  checked before using range-based volatility for any series before the 1960s. The main
  sample starts in 1993, so it does not bind there.
- FRED values for day *d* are usually published on *d+1*. All FRED features are **lagged by
  one session**.
- P3 is in CAD and is never mixed with USD assets (limitation #6).

## 4. Forecast design (the comparability contract)

Every model obeys the same contract. This is what makes the comparison fair.

| Element | Main specification | Robustness |
|---|---|---|
| Horizon | **h = 1 trading day** | h = 10 days, non-overlapping, descriptive only (too few exceedances to test) |
| VaR levels | α = 1% (Basel), 5% (more exceedances, more power) | 2.5% |
| ES level | α = 2.5% (FRTB 97.5%) | 1%, 5% |
| Information set | data published by the end of day t (after the close, before the t+1 open); FRED lagged one session | — |
| Estimation window | **expanding**, minimum 1,000 sessions (~4 years) | rolling 1,000 sessions |
| Refit frequency | parameters re-estimated every 21 sessions; filters (EWMA/GARCH variance, features) updated daily | refit every 5 sessions |
| OOS period | P1: 1997-01 → 2022-12 | — |
| Lockbox | **2023-01-01 → 2026-10-30**, evaluated **once**, after all models are frozen | — |

**Out-of-sample sizes and power.** P1 from 1997 to 2022 is about 6,500 forecasts. At α = 1%
that means about 65 expected exceedances, which is enough for Kupiec and Christoffersen. A
single stress episode such as COVID (about 50 sessions) expects 0.5 exceedances at 1%, so no
test has power there. Stress results therefore (a) pool the episodes, (b) emphasise α = 5%,
and (c) compare *losses* (every day counts) rather than only exceedance counts.

## 5. Models

### Econometric benchmarks
| Model | VaR / ES | Notes |
|---|---|---|
| Historical simulation (HS) | empirical quantile and tail mean of the last 250 / 500 returns | No model, slow to react; the industry baseline |
| EWMA (RiskMetrics) | σ²_{t+1} = λσ²_t + (1−λ)r²_t, λ = 0.94; normal quantiles | Fixed λ, so there is no estimation risk |
| GARCH(1,1)-t | `arch` package, Student-t innovations, maximum likelihood | Volatility clustering and fat tails |
| GJR-GARCH(1,1)-t | adds a leverage term γ·r²·1[r<0] | Losses raise volatility more than gains |
| Filtered HS (FHS) | GARCH σ × empirical quantile of standardised residuals | Strong hybrid benchmark (Barone-Adesi et al.) |
| HAR on range volatility | linear regression of next-day variance on 1/5/22-day realized variance (Corsi 2009) | Linear volatility model, bridge to the ML-volatility models |

### Machine-learning models
| Model | How VaR / ES are produced |
|---|---|
| Linear quantile regression (QR) | one regression per α on the feature set. **The control for H2b** |
| Quantile Regression Forest (QRF) | the conditional distribution comes from the forest's leaves (Meinshausen 2006); VaR and ES read directly from it |
| Gradient boosting, quantile loss (XGBoost `reg:quantileerror`; fallback scikit-learn `HistGradientBoosting`) | one model per quantile level |
| ML-volatility + FHS (RF / XGBoost) | ML forecasts log next-day variance; VaR/ES = σ̂ × empirical quantile/tail mean of standardised residuals |

**ES from quantile models.** ES is the average of VaR over deeper tail levels:
`ES_α = (1/α) ∫₀^α VaR_u du`. It is approximated by averaging predicted quantiles at
u = α(k − ½)/K, k = 1…K (K = 5), with monotonicity enforced by sorting. This matters
because ES on its own is not *elicitable* (no loss function is minimised by the true ES
alone), while the pair (VaR, ES) is (Fissler & Ziegel 2016). That is why models are compared
with a joint loss (§7).

**A known weakness to discuss, not hide.** At α = 1% with 1,000 training days, a direct
quantile model sees about 10 tail observations. The ML-volatility + FHS design exists for
exactly this reason: ML learns the volatility from *all* observations, and the tail shape
comes from residuals. Deep learning (LSTM etc.) is deliberately out of scope: there is too
little daily data for its parameter count, and it is hard to interpret for validation.

### Features (all stamped at close t, predicting t+1)
- Lagged return, |r|, r²; realized variance over 1/5/22/66 days (HAR structure).
- Parkinson and Garman–Klass range-based variance (uses the OHLC columns the schema already holds).
- Drawdown from the trailing 252-day high of the **forward-built** TRI. Never a vendor's
  adjusted level (M1 rule).
- VIX level and 5-day change; credit spread and term spread (FRED, lagged one session).
- For P2: rolling 63-day stock–bond correlation.

## 6. Leakage safeguards (what a validator will check)

| Risk | Safeguard |
|---|---|
| Feature uses future data | every feature function passes `assert_point_in_time` (truncation test, M1) |
| Forecast engine peeks ahead | engine-level truncation test: the forecast for t+1 made on `data[:t]` must equal the one produced in the full run |
| Training label overlaps forecast date | at origin t, training pairs are (x_s, r_{s+1}) with s+1 ≤ t; tuning validation set separated by an h-day embargo |
| Scaling / imputation fitted on future data | scalers and imputers fitted inside each training window only |
| Publication lag | FRED series shifted one session; VIX settles at 4:15 pm ET, after the 4:00 pm equity close but before the t+1 open, so it is in the end-of-day-t information set |
| Hyperparameters tuned on the test set | tuning on the last 20% of each training window, re-done yearly, small pre-declared grids |
| Researcher degrees of freedom | design frozen at tag `prereg-v1`; lockbox 2023–2026 opened once; all runs logged with snapshot hash and config |
| Regime labels leak | stress dummies are used **only** to split the evaluation, never as model inputs |
| Vendor backward-adjusted levels | forbidden in features (M1 rule); TRI built forward |

## 7. Evaluation

**Absolute tests (is each model acceptable?)**
- Kupiec (1995) proportion-of-failures likelihood ratio, χ²(1): exceedance rate = α.
- Christoffersen (1998) independence χ²(1) and conditional coverage χ²(2): exceedances do
  not cluster.
- Engle–Manganelli (2004) Dynamic Quantile test: the hit sequence is unpredictable from
  lagged hits and VaR.
- Basel traffic light over rolling 250-day windows at 99% (green/yellow/red), for industry
  relevance.
- ES: McNeil–Frey (2000) exceedance-residual test (bootstrap; standardised where the model
  provides σ, unstandardised otherwise). Acerbi–Szekely (2014) Z2 for models with a full
  predictive distribution (EWMA, GARCH family, FHS, QRF).

**Comparative tests (which model is better?)**
- Quantile (tick) loss for VaR: `(α − 1[L > VaR])·(VaR − L)`, sign-adjusted for losses.
- FZ0 joint loss for (VaR, ES) (Patton, Ziegel & Chen 2019), a strictly consistent scoring rule.
- Diebold–Mariano tests on loss differentials with Newey–West standard errors.
- Model Confidence Set (Hansen, Lunde & Nason 2011) at 90%, via `arch.bootstrap.MCS`. This
  handles the many-models problem better than a pile of pairwise tests.

**Stress robustness**
- Ex-ante episodes: LTCM (Aug–Oct 1998), dot-com (2000-03 → 2002-10), GFC (2007-07 → 2009-03),
  euro crisis (Jul–Oct 2011), Aug 2015, Q4 2018, COVID (2020-02-20 → 2020-04-30),
  2022 rate shock. Lockbox episodes (e.g. Apr 2025 tariff shock) are evaluated once only.
- Data-driven regime: VIX at the forecast origin above 30. Known at t, so a legitimate
  conditioning variable for evaluation.
- Reported: losses and exceedances by regime; DM on the stress subsample; the loss
  differential regressed on a stress dummy (HAC); how fast each model's VaR reacts after a shock.

**Interpretability**: TreeSHAP for tree models (per-date attributions, e.g. what drove VaR
up in March 2020); permutation importance on the validation window; stability of importances
across refits; GARCH parameter paths for comparison.

**Multiple testing**: 3 portfolios × 2–3 levels × about 10 models × subsamples. The headline
claims are limited to P1, α ∈ {1%, 5%}, ES 2.5%, using the MCS. Everything else is labelled
robustness.

## 8. Architecture

```
src/alphalab/
  data/          KEEP   schema, snapshots, validation, providers (yahoo, synthetic, crsp stub)
                 ADD    providers/fred.py
  core/          KEEP   calendar.py, lookahead.py (truncation test)
  returns/       KEEP   total returns, forward-built TRI, compounding
  portfolio/     ADD    weights → portfolio return series (daily / monthly rebalance)
  features/      ADD    HAR & range volatility, VIX, spreads; each PIT-tested
  risk/
    models/      ADD    RiskModel interface: fit(train) → forecast(t) = {VaR_α, ES_α, σ}
                        hs, ewma, garch (GARCH, GJR, FHS), har, ml_quantile (QR, QRF, XGB), ml_vol
    engine.py    ADD    rolling/expanding out-of-sample loop, refit schedule, seeds, caching
  backtest/      ADD    coverage.py (Kupiec, Christoffersen, DQ, traffic light), es.py,
                        losses.py (tick, FZ0), compare.py (DM, MCS)
  stress/        ADD    episode definitions, regime splits, subsample reports
  explain/       ADD    SHAP, permutation importance
dashboard/       ADD    Streamlit app: reads saved results only, never fits models
configs/         ADD    risk_p1_spy.yaml, risk_p2_balanced.yaml, risk_p3_xiu.yaml
```

Pipeline: `ingest → build (returns, portfolio, features) → forecast → backtest → report`.
Each stage writes parquet files plus a manifest that names its inputs (snapshot hash, config,
code version), so any number in the report can be traced back to raw data.

**Synthetic ground truth for model tests.** The synthetic provider will gain a GARCH-t
generator. Its *true* conditional VaR and ES are then known exactly, so unit tests can check
that (a) GARCH recovers the truth, (b) the coverage tests accept correct forecasts and reject
HS during simulated volatility bursts, and (c) the losses rank the true model first.

## 9. Limitations to state up front
- **Power:** stress subsamples are short; conclusions about stress are suggestive.
- **Univariate portfolio approach:** each portfolio is modelled as one return series with
  fixed target weights. Multivariate models (DCC-GARCH, copulas) are out of scope; this is
  common in VaR backtesting research but ignores weight dynamics.
- **One market history:** 1997–2022 holds a handful of crises; results may not generalise
  to crises of a different kind.
- **Public data quality** (Yahoo); no intraday realized volatility (range-based proxies instead).
- **Tuning budget:** small grids are chosen for honesty and compute; a richer search might help ML.
- **h = 10 days** cannot be tested formally with this sample size.

## 10. Industry adaptation (Mitacs / partner path)

The provider interface was designed so that a partner's own P&L or holdings can be plugged
in without touching model or backtest code. Typical partner questions this design answers:
- *Bank market risk / model validation:* challenger-model study for a 99% VaR / 97.5% ES
  model, with Basel traffic light and FRTB-style ES backtests (relevant to OSFI's model risk
  guideline E-23 and to SR 11-7-style validation).
- *Pension fund / asset manager:* risk forecasting for a multi-asset policy portfolio (P2 is
  the template), with stress diagnostics.

## 11. Key references
Kupiec (1995); Christoffersen (1998); McNeil & Frey (2000); Engle & Manganelli (2004);
Glosten, Jagannathan & Runkle (1993); Bollerslev (1986); Barone-Adesi, Giannopoulos &
Vosper (1999); Meinshausen (2006); Corsi (2009); Hansen, Lunde & Nason (2011);
Acerbi & Szekely (2014); Fissler & Ziegel (2016); Nolde & Ziegel (2017);
Patton, Ziegel & Chen (2019); Basel Committee, *Minimum capital requirements for market risk* (FRTB).

## Deviations from `prereg-v1`

Format: date · section · change · reason · made before/after seeing affected results.

1. **2026-11-06 · §5 · Clarification (not a change): failed GARCH refits.** The design did
   not say what happens when a refit fails. Rule implemented: a refit counts as failed if the
   optimiser does not converge, or the estimates are non-stationary (persistence ≥ 1), have
   ν ≤ 2 or ω ≤ 0. The previous parameters are then kept and the failure is counted; the
   first fit must succeed. Reason: an integrated or explosive variance process is not a valid
   forecasting model, and dropping forecasts would make samples differ across models. The rule
   was written with the model code, before the first real run. That run then showed it
   binding for plain GARCH-t (13/312 refits); the rule was not altered after seeing that.

2. **2026-11-06 · §5, §4 · Change: the H2b control becomes L1-penalised (lasso) quantile
   regression, refit yearly; a 1 bp safety floor applies to every model.** Made **after
   seeing** the plain control's development-sample forecasts on P1, before any tree/boosting
   ML model existed.
   - *What happened.* Plain linear QR on the 17 features produced 27 forecasts (of 6,535)
     with VaR ≤ 0, i.e. a "5% worst case" that is a gain. The cause is collinear features
     (Parkinson vs Garman–Klass variance; rv_5 vs |r|) with large offsetting coefficients,
     and extrapolation on stress days 4–9 standard deviations from the training mean. FZ0
     cannot score ES ≤ 0.
   - *Steps taken, in order.* (a) Clip each feature to the training window's [0.5%, 99.5%]
     quantiles: 13 invalid days remained. (b) The owner chose lasso QR on the same features,
     with λ tuned at each refit by the design's tuning protocol (last 20% of the training
     window, 1-pair embargo, grid {0, 1e-5, 1e-4, 1e-3} including "no penalty", criterion =
     mean tick loss over the VaR levels): 4 invalid days remained, all in 1998–99 when the
     tuning sample is smallest. (c) The owner chose a safety floor for every model: VaR ≥
     0.0001 and ES ≥ VaR, with every floored forecast flagged (`floored`, `n_floored`).
   - *Refit frequency.* The exact L1 solver is about 70× slower than unpenalised IRLS, so the
     control refits every 252 sessions instead of 21 (a per-model `refit_every`). Whether the
     ML models also refit yearly is decided, and recorded, before they are first run.
   - *Direction of the bias.* Regularising and clipping make the control stronger, so H2b
     (ML beats the linear control) becomes harder to support: conservative. The floor
     changes no other model (verified: zero floored forecasts for HS, EWMA, GARCH family,
     HAR). Plain QR stays in the code and its failure is reported as a stress-robustness
     finding (RESULTS_LOG).

3. **2026-11-06 · §4, §5 · Specification of the tree/boosting ML models.** Written **before
   any of these models was fitted on real data** (only timing probes on P1 features, which
   produce no forecasts or scores).
   - *XGBoost → scikit-learn.* XGBoost cannot load on the project Mac (missing `libomp`), so
     the design's named fallback is used: `HistGradientBoostingRegressor(loss="quantile")`.
   - *Refit frequency.* All ML models refit and re-tune every 252 sessions (yearly), like the
     control `linqr_l1`; forecasts still use the day's features x_t. The econometric models
     keep monthly refits, which is a handicap for ML and so conservative for H2a. The
     ML-vs-control comparison (H2b) is on equal terms.
   - *Tuning (identical for every ML model and the control).* Last 20% of the training
     pairs, 1-pair embargo, criterion = mean tick loss over the VaR levels, ties → first
     grid entry. `random_state = 0` at every fit. All 17 features, untransformed (trees are
     invariant to monotone transforms).
   - *Grids (fixed now):*
     - QRF (`quantile-forest`): 300 trees, `max_features = 0.5`;
       `min_samples_leaf` ∈ {10, 50, 200}.
     - Gradient boosting, one model per quantile level: learning rate 0.05, 200 iterations;
       (`max_leaf_nodes`, `min_samples_leaf`) ∈ {7, 31} × {50, 200}.
     - ML-volatility + FHS: random forest (300 trees, `max_features = 0.5`) predicting
       |r_{s+1}|; `min_samples_leaf` ∈ {50, 200, 500}. FHS tails from **out-of-bag**
       residuals z = r_{s+1}/σ̃_s (in-sample forest fits are overfitted and would make the
       residuals too small). The design said "RF / XGBoost": RF is used, one hybrid only, to
       limit the number of compared models.
   - *ES:* QRF and boosting average quantiles at α(k−½)/K, K = 5 (as the control);
     quantiles sorted across levels. The 1 bp floor applies.

4. **2026-11-06 · §7 · Clarification (not a change): exact stress-analysis rules.** Written
   **before any stress-period statistic was computed**. The P1 full-sample results
   (RESULTS_LOG) had been seen; per-regime numbers had not.
   - *Episodes* (inclusive; a forecast belongs to an episode if its **target date**, the day
     whose loss it forecasts, falls inside): LTCM 1998-08-01 → 1998-10-31; dot-com
     2000-03-01 → 2002-10-31; GFC 2007-07-01 → 2009-03-31; euro crisis 2011-07-01 →
     2011-10-31; Aug 2015 2015-08-01 → 2015-08-31; Q4 2018 2018-10-01 → 2018-12-31;
     COVID 2020-02-20 → 2020-04-30; 2022 rate shock 2022-01-01 → 2022-10-31 (the S&P 500
     peak-to-trough window, which the design left unspecified).
   - *VIX regime:* VIX at the forecast **origin** > 30 (known at t). Portfolios without a
     VIX feature skip it.
   - *Benchmark for regime DM tests and loss-differential regressions:* FHS-GJR, the
     best full-sample econometric model. Regression: d_t = L_model,t − L_FHS,t =
     a + b·D_t + e_t with Newey–West standard errors (default lag); H3 is tested by b
     (b > 0: the model loses ground in stress). Losses: FZ0 at each α.
   - *Reported per regime* (calm, pooled episodes, each episode, VIX > 30, VIX ≤ 30): n,
     expected and observed hits, Kupiec p, mean FZ0, ES severity, and an MCS on the pooled
     episodes and on VIX > 30. Expected hits are shown so that under-powered cells (e.g.
     COVID at 1%: 0.5 expected) are visible, not over-read.
   - *Reaction speed:* event study around the 1% largest realised losses in the sample
     (target days). Each model's VaR_1% path from 5 sessions before to 20 after, divided by
     its value 5 sessions before. Summarised by days until VaR first reaches 1.5× that level.

5. **2026-11-06 · §2 (H4), §7 · Clarification (not a change): testable H4 rules.** Written
   **before any importance or SHAP value was computed on real data** (only a speed and
   additivity probe on P1 features, with no numbers inspected beyond timing and error).
   - *Models:* the three ML models. SHAP (TreeSHAP) for `mlvol_fhs` (the forest's scale
     prediction; VaR = σ̃·q is linear in it, so VaR attributions = q × scale attributions)
     and `gbq` at α = 1% (the 1% quantile model; VaR attributions = −SHAP). Not for `qrf`:
     TreeSHAP explains a forest's mean, not its quantiles. `qrf` gets permutation importance
     only.
   - *Permutation importance:* at every yearly refit, the chosen configuration is fitted on
     the tuning part and each feature is permuted on the validation part (5 repeats,
     seed 0). Importance = rise in mean tick loss over the VaR levels.
   - *Feature families:* volatility = {rv_1, rv_5, rv_22, rv_66, abs_ret, pk_1, gk_1, gk_5,
     gk_22}; VIX = {vix, vix_chg_5}; credit = {baa10y, baa10y_chg_22}; other = {ret,
     dd_252, t10y3m, t10y3m_chg_22}.
   - *H4 criteria* (all three needed for "supported"; each reported separately):
     (C1) concentration: the top-3 features by mean permutation importance (averaged over
     refits) are all volatility or VIX features;
     (C2) economic sign: for every volatility and VIX feature, the correlation between the
     feature's value and its SHAP contribution to VaR is positive (higher volatility or VIX
     → higher VaR) for `mlvol_fhs` and `gbq`;
     (C3) stability: the median Spearman rank correlation of permutation-importance vectors
     between consecutive refits is ≥ 0.6.
   - *Descriptive (not tests):* SHAP decomposition of VaR around 2008-09/10 and
     2020-02/03; the linear control's coefficient paths and the GARCH parameter paths for
     comparison.

6. **2026-11-06 · §3, §4 · Clarification (not a change): robustness runs.** Written before
   any P2, P3 or rolling-window forecast was produced.
   - *P3 data correction* (resolving LIMITATIONS #12 by the cross-check named there in
     advance): 3 of XIU.TO's 107 distributions (2000-12-27, 2003-12-29, 2004-12-29) are
     treated as non-cash. These are exactly the dividend dates on which Yahoo's own adjusted
     close implies no payout and the validation flags `vendor_adj_mismatch`. Applied at build
     from an explicit list in the config; `data/raw` untouched; logged in the build manifest.
   - *Rolling window:* P1 re-run with a rolling 1,000-session estimation window
     (`configs/risk_p1_spy_rolling.yaml`), everything else identical, results in a separate
     folder (`run_tag`).
   - *Reading rule:* robustness results are reported next to the main ones; the headline
     conclusions stay those of P1 with the expanding window (§7, multiple testing).

7. **2026-11-08 · §7 · Clarification (not a change): Acerbi–Szekely Z2 test.** Written and
   committed before any Z2 value was computed. §7 lists Z2 but does not say how its critical
   values are obtained, and the models' full predictive distributions are not stored.
   - *Statistic* (loss convention, equivalent to Acerbi & Szekely's Z2 with X = −L):
     Z2 = 1 − (1/(T·α)) Σ_t L_t·1[L_t > VaR_t] / ES_t. E[Z2] = 0 if VaR and ES are right;
     Z2 < 0 means risk is understated. One-sided test.
   - *Null distribution:* Monte Carlo (10,000 draws, seed 0) under a shape-matched
     Student-t: ν is chosen so that the t's ES/VaR ratio at α equals the median of the
     model's own ES_t/VaR_t (normal if that ratio is at or below the normal one; ν ≥ 2.05).
     Location is taken as zero (daily mean loss ≈ 0). p = share of null draws ≤ observed Z2.
   - *Scope:* computed for every model (the statistic needs only VaR, ES and L); the
     pre-registered set (EWMA, GARCH family, FHS, QRF) is flagged. Development samples only,
     for the same reason as DQ: the lockbox has already been opened.
   - *Reading rule:* supplementary absolute ES test next to McNeil–Frey; it changes no
     headline ranking (FZ0 remains the comparative criterion).
