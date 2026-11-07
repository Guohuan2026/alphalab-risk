# AlphaLab-Risk: Machine Learning for Portfolio Risk Forecasting and Stress Resilience

**Research report, draft v1 (2026-11-07)**
Guohuan Yu, MSc Finance (supervised-project pathway), HEC Montréal
Code, data provenance and full history: `github.com/Guohuan2026/alphalab-risk` (private),
branch `alphalab-risk`. Every number below is taken from `docs/RESULTS_LOG.md`.

---

## Abstract

Can machine-learning (ML) models forecast portfolio downside risk better than the
econometric methods banks and asset managers use, particularly during market stress? I
compare eleven one-day Value-at-Risk (VaR) and Expected Shortfall (ES) models on US equity
(SPY, the headline portfolio), a 60/40 multi-asset portfolio and Canadian equity (XIU.TO).
The models are historical simulation, EWMA, GARCH-t, GJR-GARCH-t, filtered historical
simulation (FHS), HAR, a lasso quantile-regression control, a quantile regression forest,
gradient boosting and a random-forest volatility model with FHS tails. Forecasts are
strictly out-of-sample and walk-forward, without look-ahead. The design was frozen before any
model result existed, every later rule was committed before the result it governs, and a
2023–2026 lockbox was opened exactly once. In the 1997–2022 development sample no ML model
beat FHS-GJR on the joint VaR–ES (FZ0) loss; direct-quantile ML models were significantly
worse even than the linear control in the deep tail; and every feature-based model lost
ground in crises. Tree models cannot forecast risk beyond the range of their training data:
in October 2008 the forest's VaR stayed capped while VIX reached 70. These patterns held
across portfolios and estimation windows. **They did not replicate in the lockbox**: in the
calm 2023–2026 period, feature-based models had lower average loss than FHS-GJR (not
significant on SPY), while FHS-GJR stayed well calibrated. The evidence supports a
regime-dependent conclusion. FHS-GJR is the robust choice, best in stress and calibrated
everywhere, while feature-based ML can reduce average loss in calm markets.

---

## 1. Research question and contribution

> Can machine-learning models improve the accuracy and robustness of portfolio downside-risk
> forecasts compared with traditional econometric methods, particularly during periods of
> financial market stress?

**Why it matters.** Banks report 99% VaR and 97.5% ES under the Basel FRTB framework, and
pension funds and asset managers set risk budgets from the same quantities. ML methods are
increasingly proposed as replacements, but published comparisons often lack strict
out-of-sample discipline, proper scoring rules, or any test of behaviour in crises, which is
the regime where risk models matter most.

**Hypotheses** (pre-registered at git tag `prereg-v1`):

| | Hypothesis |
|---|---|
| H1 | ML VaR forecasts are correctly calibrated out-of-sample. |
| H2a | ML has lower joint VaR–ES loss than the best econometric model. |
| H2b | Any ML gain comes from non-linearity, not merely from extra inputs (vs a linear model on the same features). |
| H3 | Relative performance differs in stress periods. |
| H4 | ML forecasts respond to economically meaningful drivers, stably. |

**Contributions.**
1. A like-for-like comparison: same data, horizon, confidence levels, information set and
   tuning protocol for every model.
2. A control model separating "better inputs" from "non-linearity" (H2b).
3. A stress analysis with ex-ante episodes, a VIX regime and reaction-speed event studies,
   with explicit attention to statistical power.
4. A mechanism-level explanation of why tree-based ML fails in stress (saturation).
5. A research-integrity process (design freeze, pre-committed clarifications, results log,
   once-only lockbox, bit-identical reproduction from a clean clone) that a model-validation
   team can audit.

## 2. Background

**VaR and ES.** With loss L = −r and tail probability α, VaR_α is the loss exceeded with
probability α, and ES_α = E[L | L ≥ VaR_α] is the mean loss beyond it. ES captures tail
severity, which VaR ignores; that is why FRTB moved capital to ES at 97.5%.

**Elicitability.** To *rank* forecasts we need a scoring rule whose expected value is
minimised by the true forecast. Quantiles (VaR) have one: the tick loss. ES alone does not
(Gneiting 2011), but the pair (VaR, ES) does (Fissler & Ziegel 2016). I use the FZ0 loss of
Patton, Ziegel & Chen (2019).

**Models.** GARCH (Bollerslev 1986) and GJR-GARCH (Glosten, Jagannathan & Runkle 1993)
capture volatility clustering and the leverage effect; FHS (Barone-Adesi et al. 1999)
combines GARCH volatility with empirical residual tails; HAR (Corsi 2009) models volatility
over daily, weekly and monthly horizons; quantile regression forests (Meinshausen 2006) and
gradient boosting with the pinball loss are the main ML quantile learners.

## 3. Data

| Portfolio | Composition | Calendar | Out-of-sample development | Lockbox |
|---|---|---|---|---|
| **P1 (headline)** | SPY total return (dividends included) | NYSE | 1997-01-14 → 2022-12-29 (6,535 days) | 2023-01-03 → 2026-10-30 |
| P2 | 40% SPY, 20% EFA, 30% IEF, 10% LQD, monthly rebalance | NYSE | 2006-07 → 2022-12 (4,142) | same |
| P3 | XIU.TO (S&P/TSX 60), CAD | TSX | 2003-09 → 2022-12 (4,838) | same |

**Sources.** Prices, dividends and VIX from Yahoo Finance; credit (Moody's Baa − 10y) and term
(10y − 3m) spreads from FRED. Every download is an immutable snapshot with a SHA-256 hash;
each build records the snapshot it used.

**Features** (17, all stamped with information published by the end of day t): realized
variance over 1/5/22/66 days (HAR structure), Parkinson and Garman–Klass range variance,
drawdown from the forward-built wealth index, VIX and its 5-day change, and the two spreads
with their 22-day changes. FRED series are lagged one session (they are published the next
morning).

**Data quality.** Total returns are built from raw closes plus dividends, never from Yahoo's
backward-adjusted levels, which embed future information. Three XIU.TO year-end
distributions that were reinvested rather than paid are treated as non-cash (an explicit,
logged correction; raw data untouched). SPY tracks the S&P 500 total-return index to within
0.13%/yr, but daily agreement is weaker before 2009 (a documented open limitation).

## 4. Methodology

### 4.1 The forecast contract

All models forecast the next day's loss at α ∈ {1%, 2.5%, 5%}. Main specification: expanding
estimation window with at least 1,000 sessions, parameters re-estimated every 21 sessions,
filters updated daily. Feature-based models refit and re-tune yearly (Deviation 3), which
handicaps ML relative to the monthly econometric refits. A walk-forward engine produces one
forecast per day using only data up to that day. An engine-level truncation test checks that a
forecast computed on data cut off at date d equals the one computed in the full run.

### 4.2 Models

| Family | Model | VaR / ES from |
|---|---|---|
| Baselines | HS-250, HS-500 | empirical quantile and tail mean of past losses |
| | EWMA (RiskMetrics, λ = 0.94) | normal quantiles × EWMA volatility |
| Econometric | GARCH(1,1)-t, GJR-GARCH(1,1)-t | Student-t quantiles × GARCH volatility (`arch`) |
| | **FHS-GJR** | GJR volatility × empirical quantiles of standardised residuals |
| Linear | HAR-FHS | OLS volatility on 1/5/22-day realized variance + FHS tails |
| | **Lasso QR (H2b control)** | L1-penalised linear quantile regression on the 17 features |
| ML | Quantile regression forest | forest leaf distributions (`quantile-forest`) |
| | Gradient boosting | one pinball-loss model per quantile level (scikit-learn) |
| | RF-volatility + FHS | random forest for \|r\| (a volatility proxy) × out-of-bag residual tails |

ES for the quantile models is the average of predicted quantiles at α(k − ½)/5, k = 1…5. A
safety floor (VaR ≥ 1 bp, ES ≥ VaR) applies to every model (Deviation 2).

**Tuning.** One shared protocol for all ML models and the control. Hold out the last 20% of
each training window with a one-day embargo, evaluate small grids fixed in advance with the
mean tick loss, refit the best configuration on the whole window. Seeds are fixed.

### 4.3 Leakage safeguards

Every feature passes a truncation test, and the label (next-day loss) is checked to *fail*
it. Scalers and clipping bounds are fitted on training data only. Stress regimes are used only
to split the evaluation, never as inputs. The lockbox was enforced in code: forecasts into it
required an explicit flag.

### 4.4 Evaluation

- **Calibration:** Kupiec (frequency), Christoffersen (independence, conditional coverage),
  Basel traffic light.
- **ES:** McNeil–Frey exceedance-residual test (bootstrap).
- **Ranking:** mean FZ0 and tick loss; Diebold–Mariano tests with Newey–West errors; the Model
  Confidence Set (MCS, Hansen, Lunde & Nason 2011) at 90%, the headline comparison because it
  controls for comparing many models.
- **Stress:** eight ex-ante episodes (by target date) and a VIX-above-30 regime (at the
  forecast origin); loss differentials regressed on a stress dummy with HAC errors; an event
  study of VaR reaction after the largest losses.
- **Interpretability:** permutation importance on each yearly validation window; TreeSHAP
  attributions to VaR; three pre-stated criteria (concentration, economic sign, stability).

Every test was validated on simulated GJR-GARCH-t data with known true VaR and ES before
being applied to market data. For example, FZ0 ranks the true model first, and the tests have
roughly correct size.

### 4.5 Research integrity

The design was frozen at tag `prereg-v1` before any GARCH, ML or lockbox result existed (only
baseline HS/EWMA results had been seen, and they are logged). Six later decisions were
committed before the results they govern (Deviations and Clarifications 1–6 in
`RESEARCH_DESIGN.md`). Every result is logged in the order it was seen. The lockbox protocol,
including the rule for "replicated", was pushed to GitHub before the lockbox was opened.

## 5. Results

### 5.1 Calibration (H1): only FHS-GJR passes everywhere (P1, development)

| Model | 1% hits | 2.5% hits | 5% hits | Conditional-coverage p (1 / 2.5 / 5%) |
|---|---|---|---|---|
| **FHS-GJR** | **1.13%** | **2.86%** | **5.37%** | **0.32 / 0.15 / 0.36** |
| GJR-GARCH-t | 1.39% | 3.52% | 6.12% | 0.004 / <0.001 / <0.001 |
| RF-vol + FHS | 1.45% | 3.05% | 5.91% | <0.001 / 0.008 / <0.001 |
| Lasso QR (control) | 1.65% | 3.55% | 6.63% | <0.001 at every level |
| Quantile forest | 1.76% | 3.40% | 6.06% | <0.001 at every level |
| Gradient boosting | 2.92% | 5.31% | 8.28% | <0.001 at every level |
| EWMA | 2.08% | 3.64% | 5.95% | <0.001 / <0.001 / 0.002 |
| HS-250 | 1.56% | 3.43% | 5.71% | <0.001 at every level |

HS fails through clustered breaches (slow reaction). EWMA tracks volatility, but its normal
tails are too thin: realized tail losses exceed its ES by 14% at 2.5%. The t-tailed GARCH
models over-hit because SPY's losses are negatively skewed. FHS, with GJR volatility and
empirical tails, is the only model not rejected at any level. **H1 is not supported for the
ML models.**

### 5.2 Accuracy (H2): no ML model beats FHS-GJR

FZ0 rank among 11 models (P1, development; ✓ = in the 90% MCS):

| Model | 1% | 2.5% | 5% |
|---|---|---|---|
| FHS-GJR | **1** ✓ | **1** ✓ | 2 ✓ |
| RF-vol + FHS | 2 ✓ | 2 ✓ | **1** ✓ |
| GJR-GARCH-t | 3 ✓ | 3 | 3 |
| Lasso QR (control) | 4 ✓ | 6 ✓ | 10 ✓ |
| Quantile forest | 9 | 8 | 7 |
| Gradient boosting | 10 | 9 | 8 |

Diebold–Mariano on FZ0: FHS-GJR beats the quantile forest (p = 0.009, 0.003, 0.0007) and
gradient boosting (p = 0.003, 0.0005, 0.0002). It is statistically tied with RF-vol + FHS
(p = 0.82, 0.79, 0.95). The linear control beats both direct-quantile ML models at 1%
(p = 0.004, 0.0006) and 2.5% (p = 0.036, 0.005).

- **H2a is not supported.** The best ML model ties with FHS-GJR but is worse calibrated.
- **H2b is not supported, and is partly reversed:** non-linear quantile learners are worse
  than the linear control in the deep tail. With only about 10–84 training observations below
  the 1% quantile, trees cannot learn the extreme tail. Gradient boosting's 1% VaR is breached
  2.9× as often as it should be. ML helped only in the hybrid design, which uses it to model
  volatility and lets the residuals supply the tail.

### 5.3 Stress (H3): supported, and it explains the "tie"

The eight pooled crisis episodes cover 1,604 days; the VIX-above-30 regime covers 688 days.

| Model | 1% hits, calm | 1% hits, episodes | 1% hits, VIX > 30 |
|---|---|---|---|
| FHS-GJR | 1.05% | **1.37%** | **1.02%** |
| RF-vol + FHS | 1.05% | 2.68% | 3.05% |
| Quantile forest | 1.03% | 3.99% | 5.38% |
| Gradient boosting | 2.21% | 5.11% | 5.52% |
| HS-250 | 0.75% | 4.05% | 4.07% |

Regressing the FZ0 differential against FHS-GJR on an episode dummy shows the hybrid is
significantly **better in calm periods** (a = −0.105, −0.067, −0.050; p ≤ 0.0015) and
significantly **worse in crises** (b = +0.48, +0.31, +0.20; p ≤ 0.015). The two effects cancel
over the full sample. Every feature-based model loses ground in stress (b > 0, p ≤ 0.01 at
2.5%). At 2.5% and 5%, FHS-GJR is the only member of the crisis-period MCS.

**Mechanism: reaction speed.** The day after one of the 1% largest losses, GJR/FHS VaR rises
67%; HAR and EWMA rise 37–41%; the tree models only 23–29%, and they never reach 1.5× their
pre-shock level.

**H3 is supported.**

### 5.4 Interpretability (H4): sensible drivers, unstable explanations

| Model | C1: top-3 drivers are volatility/VIX | C2: economic signs | C3: stability ≥ 0.6 |
|---|---|---|---|
| RF-vol + FHS | ✓ (VIX, gk_5, gk_1) | ✗ (9/11 positive) | ✗ (0.54) |
| Quantile forest | ✓ | n/a (no TreeSHAP for quantiles) | ✗ (0.35) |
| Gradient boosting | ✗ (credit-spread change 3rd) | ✗ (8/11) | ✓ (0.69) |

**H4 is not supported** under the pre-stated all-three rule. The direction is sensible: VIX
and range-based volatility dominate, and VIX's contribution rises with its level (correlation
0.79–0.87). But signs of correlated volatility features are unreliable, and importance
rankings shift from year to year.

**The saturation result.** The hybrid in use during 2008 had been trained on data whose
maximum VIX was 45.7. Its 1% VaR was 3.81% on 12 September 2008 (VIX 25.7) and 3.82% on
10 October 2008 (VIX 69.9), with identical SHAP contributions: both days fall in the same
leaves. FHS-GJR moved from 4.12% to 11.73%. A tree cannot forecast a risk level outside its
training range; GARCH extrapolates through its recursion. This is the mechanism behind H3.

### 5.5 Robustness

| Setting | FHS-GJR FZ0 rank (1/2.5/5%) | RF-vol + FHS rank | Quantile forest / boosting ranks |
|---|---|---|---|
| P1, expanding (main) | 1 / 1 / 2 | 2 / 2 / 1 | 9, 8, 7 / 10, 9, 8 |
| P2, 60/40 | 1 / 1 / 1 | 6 / 7 / 7 | 10, 9, 8 / 11, 11, 10 |
| P3, Canada | 1 / 1 / 1 | 5 / 3 / 3 | 8, 9, 9 / 9, 8, 8 |
| P1, rolling 1,000 days | 1 / 1 / 1 | 5 / 5 / 5 | 10, 10, 8 / 11, 11, 11 |

FHS-GJR ranks first in 11 of 12 cells and is always in the MCS. Every ML model loses ground in
stress in every setting (p ≤ 0.03). **Not robust:** the hybrid's calm-period advantage is
significant only for P1 with an expanding window (P2 p = 0.046; P3 p = 0.68; rolling
p = 0.31).

### 5.6 Lockbox 2023–2026: not replicated

The lockbox was opened once, about 961 days per portfolio. Integrity check: re-forecasting the
development period inside the lockbox run reproduced the saved forecasts bit-for-bit for all
three portfolios.

By the rule pushed before the run (FHS-GJR in the MCS at 2.5% and 5%, and neither the
quantile forest nor gradient boosting with lower mean FZ0), **the headline did not replicate
in P1, P2 or P3.**

On **P1**, the lasso control, gradient boosting and the hybrid rank 1–3 and FHS-GJR ranks 5–6.
The differences are mostly not significant: DM p = 0.07–0.65, except the control beating
FHS-GJR at 2.5% (p = 0.037). FHS-GJR remained well calibrated (hit rates 1.04%, 1.98%, 4.79%;
CC p = 0.89, 0.37, 0.63) and in the MCS. On **P2**, some ML advantages are significant (the
hybrid at 2.5%, p = 0.011).

The lockbox was unusually calm. VIX was above 30 on 1.6% of days against 10.5% in development;
annualized SPY volatility was 14.9% against 19.9%; there were 4 daily moves larger than 4%
against 86. Combined with H3 (feature-based models are better in calm markets, worse in
stress), the reversal is consistent with a regime effect. **That explanation was formed after
seeing the result. It is untested and labelled post hoc.** The lockbox contains essentially
no stress, so it cannot test stress robustness.

## 6. Discussion

**The answer to the research question is regime-dependent.**
- In crises, the econometric FHS-GJR is clearly superior. It stays calibrated, reacts fastest,
  and is the only model in the crisis-period MCS.
- In calm markets, feature-based models (including a regularized *linear* one) can reduce
  average loss.
- Over a full cycle that includes crises, nothing beat FHS-GJR in development. In a calm
  holdout, ML did better on average.

**What ML is and is not good for in VaR/ES.** ML adds value as a *volatility* model fed with
rich information (VIX, range-based variance). It fails as a direct *tail* model: there are too
few tail observations, and trees cannot extrapolate beyond their training range. The hybrid
design is the right architecture, and its stress weakness has a clear mechanical cause.

**Practical implications.**
- *Bank market risk / model validation:* FHS-GJR is the defensible production model; an ML
  challenger should not replace it without a stress override.
- *Pension funds and asset managers* who weight average accuracy more heavily could use ML in
  calm regimes, with a guard for stress.
- A regime switch (for example on VIX) or a max(ML, GARCH) rule is the natural next
  hypothesis. It was **not** tested here, because adding it after seeing the results would be
  post hoc.

## 7. Limitations

1. **Statistical power.** About 65 expected 1% breaches in development but only about 10 in
   the lockbox; crisis cells at 1% are thin.
2. **Univariate portfolio approach.** Each portfolio is modelled as one return series with
   fixed weights; no multivariate (DCC, copula) models.
3. **One market history.** A handful of crises; future crises may differ.
4. **Public data.** Yahoo is unofficial and restates history (immutable snapshots mitigate
   this); there is no intraday realized volatility; SPY's daily agreement with the index is
   weaker before 2009.
5. **Implementation details.** The quantile forest used the library default of one training
   value per leaf (about 300 values per forecast distribution), coarser than Meinshausen's
   original, and its results depend on the library version (locked). HAR coefficients are
   attenuated because daily squared returns are noisy proxies. Gradient boosting's per-level
   quantiles crossed on 66% of days.
6. **Control-model changes.** The lasso control and the safety floor were chosen after seeing
   the plain linear control fail (Deviation 2). The change strengthens the control, which is
   conservative for H2b, and is fully documented.
7. **Tuning budget.** Small fixed grids, chosen for honesty and compute; richer searches might
   help ML.
8. **The lockbox reversal is unexplained** beyond a post-hoc regime hypothesis.

## 8. Industry adaptation (Mitacs partnership)

The pipeline is built to take a partner's own data. The provider interface accepts any price
or P&L series; portfolios are configured by weights; every model, test and report is
config-driven. A partner project could:

- **Bank (model validation / market risk):** run a challenger study of the partner's 99% VaR
  and 97.5% ES model, with Basel traffic-light and FRTB-style ES backtests, under OSFI's model
  risk management guideline (E-23).
- **Pension fund / asset manager:** forecast risk for the actual policy portfolio (P2 is the
  template), with crisis diagnostics and a pre-registered regime-switching rule tested on new
  data.

The integrity process (frozen design, committed clarifications, results log, lockbox) maps
directly onto model-validation documentation standards.

## 9. Reproducibility

- **Data:** immutable, hashed snapshots.
- **Code:** a complete git history on GitHub, including the design freeze and lockbox protocol.
- **Environment:** exact package versions in `requirements-lock.txt`. A clean-clone check
  showed that without it the project was not reproducible one day later: `quantile-forest`
  1.5.0 changes forest quantiles.
- **Verification:** a fresh clone with the documented one-command setup passes 254 tests and
  rebuilds all 11 models' P1 forecasts and backtest tables bit-for-bit.
- **Exploration:** a read-only Streamlit dashboard (`make dashboard`) shows the saved results.

## 10. Conclusion

Under strict out-of-sample discipline, proper scoring rules and pre-committed analysis rules,
machine learning did not improve one-day portfolio VaR/ES forecasts over filtered historical
simulation with GJR-GARCH volatility across a full market cycle. It was clearly worse in
crises, because tree models cannot extrapolate risk beyond their training range. A held-out
calm period reversed the average-loss ranking without overturning FHS-GJR's calibration.
The defensible conclusion is that the value of ML in risk forecasting is regime-dependent and
lies in volatility modelling, not tail modelling. A risk system that uses it needs a stress
safeguard.

---

## References

Barone-Adesi, G., Giannopoulos, K. & Vosper, L. (1999). VaR without correlations for
portfolios of derivative securities. *Journal of Futures Markets*.
Basel Committee on Banking Supervision. *Minimum capital requirements for market risk* (FRTB).
Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of
Econometrics*.
Christoffersen, P. (1998). Evaluating interval forecasts. *International Economic Review*.
Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of
Financial Econometrics*.
Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business &
Economic Statistics*.
Fissler, T. & Ziegel, J. F. (2016). Higher order elicitability and Osband's principle. *Annals
of Statistics*.
Glosten, L., Jagannathan, R. & Runkle, D. (1993). On the relation between the expected value
and the volatility of the nominal excess return on stocks. *Journal of Finance*.
Gneiting, T. (2011). Making and evaluating point forecasts. *JASA*.
Hansen, P. R., Lunde, A. & Nason, J. M. (2011). The model confidence set. *Econometrica*.
Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal
of Derivatives*.
Lundberg, S. M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions.
*NeurIPS*.
McNeil, A. J. & Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic
financial time series. *Journal of Empirical Finance*.
Meinshausen, N. (2006). Quantile regression forests. *JMLR*.
Patton, A. J., Ziegel, J. F. & Chen, R. (2019). Dynamic semiparametric models for expected
shortfall (and Value-at-Risk). *Journal of Econometrics*.

## Appendix: decision timeline (git)

| Commit / tag | What it fixed in advance |
|---|---|
| `d935f1d` | Research design written before any risk result |
| `prereg-v1` (`46c21ea`) | Design frozen (only HS/EWMA development results seen) |
| Deviation 1 (in `100f0a1`) | Rule for failed GARCH refits |
| Deviation 2 (in `f1b3efc`) | Lasso control and safety floor, after the plain control failed |
| `2773c4f` | ML specification (Deviation 3), before any ML fit |
| `a65c036` | Stress rules (Clarification 4), before any stress statistic |
| `2bb1d58` | H4 criteria (Clarification 5), before any importance value |
| `b7ab38d` | Robustness specification (Clarification 6), before P2/P3/rolling results |
| `dc8af86` | Lockbox protocol, pushed to GitHub before the lockbox was opened |
| `9925df7` | Lockbox results, as observed |
| `69e8a1c` | Dependency lock after the clean-clone check |
