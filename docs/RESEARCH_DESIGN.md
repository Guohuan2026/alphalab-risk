# AlphaLab-Risk — Research Design

**Machine Learning for Portfolio Risk Forecasting and Stress Resilience**

Status: design draft, 2026-11-06, branch `alphalab-risk`. The pre-pivot code is preserved at
tag `v0.1-equity-research`. This document is written *before* any out-of-sample risk result
has been looked at. When it is final, it is frozen with a git tag (`prereg-v1`), so that every
later deviation is visible in the history. That is the honest substitute for formal
pre-registration.

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
