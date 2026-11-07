# Results log

A dated record of every result **looked at**, in order. It documents what was known when
each design decision was made. That matters for credibility: results seen before the design
freeze (`prereg-v1`) could have influenced it. Lockbox results (targets from 2023-01-01) are
logged only once, at the end.

---

## 2026-11-06 — Baselines, P1 (SPY), development sample, before `prereg-v1`

Setup: expanding window, min 1,000 sessions, refit every 21; origins 1997-01-14 →
2022-12-29 (6,535 one-day forecasts per model); code at commit after `8220717`.

| Model | α | Hit rate | Kupiec p | Christoffersen IND p | CC p |
|---|---|---|---|---|---|
| HS-250 | 1% | 1.56% | 2.5e-05 | 2.9e-05 | 2.2e-08 |
| HS-250 | 5% | 5.71% | 0.010 | 3.3e-09 | 9.3e-10 |
| HS-500 | 1% | 1.65% | 1.3e-06 | 1.7e-07 | 9.1e-12 |
| HS-500 | 5% | 5.78% | 0.0045 | 1.8e-07 | 2.1e-08 |
| EWMA-normal | 1% | 2.08% | 1.7e-14 | 0.032 | 1.7e-14 |
| EWMA-normal | 2.5% | 3.64% | 3.0e-08 | 0.27 | 1.2e-07 |
| EWMA-normal | 5% | 5.95% | 0.00059 | 0.54 | 0.0023 |

Basel traffic light (1%, share of 250-day windows): HS-250 64% green / 7% red; HS-500 63% /
12%; EWMA 41% / 7%.

Reading: every baseline is rejected. HS is too slow (COVID, 2020-02-19 → 03-20: HS-500's
VaR rose only 2.65% → 4.50%, with 9 hits in 23 days). EWMA tracks volatility (independence
mostly not rejected) but its normal tails are too thin (2.1% hits at 1%). This matches the
behaviour on simulated GJR-GARCH-t data. These results were seen before the design freeze.
They do not change the pre-specified model list (GARCH-t, GJR-GARCH-t, FHS, ML), which was
written before any risk result existed (RESEARCH_DESIGN.md, commit `d935f1d`).

---

## 2026-11-06 — GARCH family, P1 (SPY), development sample, after `prereg-v1`

Same setup as above (6,535 forecasts per model, origins 1997-01-14 → 2022-12-29).
Pre-specified models; first results after the freeze.

| Model | α | Hit rate | Kupiec p | IND p | CC p |
|---|---|---|---|---|---|
| GARCH(1,1)-t | 1% | 1.48% | 0.00024 | 0.25 | 0.00061 |
| GARCH(1,1)-t | 5% | 6.53% | 5.2e-08 | 0.67 | 3.3e-07 |
| GJR-GARCH-t | 1% | 1.39% | 0.0026 | 0.18 | 0.0044 |
| GJR-GARCH-t | 2.5% | 3.52% | 6.4e-07 | 0.97 | 4.2e-06 |
| GJR-GARCH-t | 5% | 6.12% | 5.8e-05 | 0.92 | 0.00030 |
| **FHS (GJR)** | 1% | 1.13% | 0.29 | 0.27 | **0.32** |
| **FHS (GJR)** | 2.5% | 2.86% | 0.067 | 0.53 | **0.15** |
| **FHS (GJR)** | 5% | 5.37% | 0.17 | 0.65 | **0.36** |

Reading:
- The GARCH family removes the clustering that sinks HS (independence no longer rejected).
- With symmetric Student-t tails, GARCH-t and GJR-t still over-hit at every level: SPY's
  losses are negatively skewed, which a symmetric t cannot place.
- FHS, with the same GJR volatility and empirical residual tails, is the **first model not
  rejected at any level**. It is now the benchmark the ML models must beat on *loss*, not
  only on coverage.
- GJR parameter paths (312 monthly refits): α ≈ 0 (median 0.000), so positive returns
  barely move volatility; γ 0.06–0.21, rising from about 0.10 (1997) to 0.20 (2020–22);
  ν 4.8–8.5; persistence about 0.99.
- Plain GARCH-t hit the stationarity boundary (α + β = 1.00000, with the optimiser
  converged) in 13 of 312 refits (2001–2003, Oct–Nov 2008). By rule (see Deviations,
  clarification 1) those refits kept the previous parameters. GJR had 0 such cases: the
  leverage term absorbs what plain GARCH can only fit with infinite persistence.

Caveat: coverage tests have low power against wrong-but-adequate volatility. On simulated
data GARCH-t passed despite 4× the volatility error of GJR. "Not rejected" does not mean
best; the loss-based comparison (next step) decides that.

---

## 2026-11-06 — Loss-based comparison and ES backtests, P1 (SPY), development sample

Same 6,535 forecasts per model as above. FZ0 = joint (VaR, ES) loss; MCS at 90%
(2,000 block-bootstrap reps); McNeil–Frey with 10,000 bootstrap reps (so p = 9.999e-05
means p < 1/10,001).

**Mean FZ0 loss (lower = better) and Model Confidence Set**

| Model | α = 1% | α = 2.5% | α = 5% |
|---|---|---|---|
| HS-250 | −3.098 | −3.372 | −3.593 |
| HS-500 | −3.009 | −3.305 | −3.532 |
| EWMA-normal | −3.097 | −3.461 | −3.699 |
| GARCH-t | −3.294 (in MCS) | −3.527 | −3.721 |
| GJR-GARCH-t | −3.319 (in MCS) | −3.551 | −3.745 |
| **FHS-GJR** | **−3.338** (in MCS) | **−3.570** (MCS = {FHS}) | **−3.755** (MCS = {FHS}) |

**Diebold–Mariano, FZ0, other model vs FHS-GJR** (positive stat = FHS better)

| vs FHS | α = 1% | α = 2.5% | α = 5% |
|---|---|---|---|
| GJR-t | 1.69 (p 0.091) | 2.63 (p 0.009) | 2.29 (p 0.022) |
| GARCH-t | 1.67 (p 0.094) | 2.59 (p 0.010) | 3.11 (p 0.002) |
| EWMA | 3.84 (p 0.0001) | 3.74 (p 0.0002) | 3.53 (p 0.0004) |
| HS-250 | 2.77 (p 0.006) | 3.74 (p 0.0002) | 4.38 (p < 0.0001) |

**McNeil–Frey ES test** (severity = realised mean tail loss / forecast ES; 1 = right):
EWMA 1.12–1.18 (p < 0.0001 at every α); HS-250 1.06–1.10 (p 0.004–0.023); GARCH-t,
GJR-t and FHS 0.97–1.02 (raw p ≥ 0.17). Standardised version: GJR-t rejected at 5%
(p = 0.010), GARCH-t borderline (p = 0.035), FHS not rejected (p ≥ 0.24).

Reading:
- FHS-GJR is the econometric benchmark on every criterion: coverage, ES, FZ0, MCS.
- At 1% the data cannot separate FHS from the t-tail GARCH models (DM p ≈ 0.09; all three
  in the MCS). Few tail events means low power, as anticipated in RESEARCH_DESIGN §4.
- Normal tails (EWMA) understate ES most: realised tail losses are 14% above forecast at
  the FRTB level (2.5%).
- Bar for the ML models (H2a): beat FHS-GJR on FZ0 and enter or displace it in the MCS.

---

## 2026-11-06 — Linear benchmarks (HAR-FHS, linear QR), P1 (SPY), development sample

**Plain linear QR failed (finding; led to Deviation 2).** On the 17 features it gave 27 of
6,535 forecasts with VaR ≤ 0 (worst 5% VaR −25.9%, the day after the 1997-10-27 crash, when
features were 4–9 sd from their training mean). Collinear features (Parkinson vs
Garman–Klass: standardised 1% coefficients −0.0147 and +0.0134) offset each other until
an unusual day breaks the offset. Clipping features to their training range left 13 invalid
days (incl. Lehman, 2008-09-15). Tuned lasso QR (`linqr_l1`) left 4, all in 1998–99
(λ = 0.001 chosen in most years, zeroing 6–12 of 17 features; weak λ in the first years when
the tuning sample is smallest). A 1 bp floor handles those 4.

Reproducibility check: re-running all models gave **bit-identical** forecasts for HS-250,
HS-500, EWMA, GARCH-t, GJR-t, FHS-GJR and HAR-FHS; the floor touched only the 4
`linqr_l1` days.

| Model | α | Hit rate | CC p | ES severity (p) | Mean FZ0 | FZ0 rank (of 8) | In MCS |
|---|---|---|---|---|---|---|---|
| FHS-GJR | 1% | 1.13% | 0.32 | 0.99 (0.61) | −3.338 | 1 | yes |
| HAR-FHS | 1% | 1.29% | <0.001 | 1.00 (0.54) | −3.309 | 4 | yes |
| linqr_l1 | 1% | 1.65% | <0.001 | 0.99 (0.70) | −3.317 | 3 | yes |
| FHS-GJR | 2.5% | 2.86% | 0.15 | 0.97 (0.89) | −3.570 | 1 | yes |
| HAR-FHS | 2.5% | 2.98% | 0.014 | 1.00 (0.54) | −3.523 | 4 | no |
| linqr_l1 | 2.5% | 3.55% | <0.001 | 1.03 (0.062) | −3.517 | 5 | yes (p 0.14) |
| FHS-GJR | 5% | 5.37% | 0.36 | 0.98 (0.88) | −3.755 | 1 | yes (alone) |
| HAR-FHS | 5% | 5.81% | 0.012 | 0.99 (0.66) | −3.702 | 4 | no |
| linqr_l1 | 5% | 6.63% | <0.001 | 1.05 (0.004) | −3.574 | 7 | no (p 0.09) |

DM (FZ0), FHS-GJR vs: HAR-FHS p = 0.40 / 0.016 / 0.0001 (1% / 2.5% / 5%); linqr_l1
p = 0.66 / 0.15 / 0.22.

**Sensitivity to the floor.** `linqr_l1`'s 5% mean FZ0 is dominated by ONE floored day,
1999-01-06 (FZ0 = 951.7: a 1 bp VaR forecast followed by a real loss). Excluding the 4
floored days, its mean FZ0 is −3.334 / −3.547 / −3.730, still behind FHS-GJR at every
level. That one day is also why DM cannot separate linqr_l1 from FHS at 5% despite the large
mean gap: one extreme loss inflates the variance. The floor rule was fixed before these
numbers were seen; the headline uses it as computed.

Reading: neither linear benchmark beats FHS-GJR. The linear control over-hits at every
level (1.65% at 1%). FHS-GJR remains the sole MCS member at 5%. These are the bars for the
ML models in the next step: beat FHS-GJR (H2a) and beat `linqr_l1` (H2b).

---

## 2026-11-06 — Machine-learning models, P1 (SPY), development sample (main H2 evidence)

Specification fixed and committed before any ML fit (Deviation 3, commit `2773c4f`). All
11 models on the same 6,535 origins (1997-01-14 → 2022-12-29). The 8 earlier models
reproduced **bit-identically**. Floored forecasts: only `linqr_l1` (4). ML refit and
re-tune yearly (26 refits).

**FZ0 rank (of 11) and MCS membership**

| Model | α = 1% | α = 2.5% | α = 5% |
|---|---|---|---|
| FHS-GJR | **1** (MCS) | **1** (MCS) | 2 (MCS) |
| ML-vol + FHS | 2 (MCS) | 2 (MCS) | **1** (MCS) |
| GJR-t | 3 (MCS) | 3 | 3 |
| linqr_l1 (control) | 4 (MCS) | 6 (MCS) | 10 (MCS, p 0.27) |
| HAR-FHS | 5 (MCS) | 5 | 5 |
| QRF | 9 | 8 | 7 |
| Gradient boosting | 10 | 9 | 8 |

**Calibration of the ML models** (hit rate; McNeil–Frey severity)

| Model | 1% | 2.5% | 5% | ES severity at 2.5% |
|---|---|---|---|---|
| ML-vol + FHS | 1.45% (CC p 0.0001) | 3.05% (p 0.008) | 5.91% (p < 0.001) | 1.05 (p 0.024) |
| QRF | 1.76% | 3.40% | 6.06% | 1.10 (p 0.0006) |
| Gradient boosting | **2.92%** | **5.31%** | **8.28%** | 1.09 (p 0.0004) |
| FHS-GJR (reference) | 1.13% (p 0.32) | 2.86% (p 0.15) | 5.37% (p 0.36) | 0.97 (p 0.89) |

**Diebold–Mariano (FZ0)**
- FHS-GJR vs QRF: FHS better at every level (p = 0.009, 0.003, 0.0007).
- FHS-GJR vs GBQ: FHS better (p = 0.003, 0.0005, 0.0002).
- FHS-GJR vs ML-vol: no difference (p = 0.82, 0.79, 0.95).
- `linqr_l1` vs QRF and GBQ: the **linear control is better** at 1% (p = 0.004, 0.0006)
  and 2.5% (p = 0.036, 0.005); no difference at 5%.
- `linqr_l1` vs ML-vol: no difference (p = 0.85, 0.19, 0.22).

**Tuning choices** (26 yearly refits): QRF leaf 50 in 17 years; GBQ the smallest trees
(7 leaves) in 25; ML-vol leaf 50 in 15; lasso λ = 0.001 in 20. The data favoured simple
configurations.

**Reading, against the pre-registered hypotheses (development sample only):**
- **H2a (ML beats the best econometric model): not supported.** No ML model has
  significantly lower FZ0 than FHS-GJR. The best ML model, the hybrid, is statistically
  tied with it (DM p ≥ 0.79; both in the MCS at every level) but is worse calibrated
  (rejected by CC at every level; ES severity 1.05).
- **H2b (gains come from non-linearity): not supported, and partly reversed.** The two
  direct-quantile ML models are significantly **worse** than the linear control at 1% and
  2.5%. Where ML helps at all (the hybrid), it is in modelling *volatility*, not the tail
  directly.
- **Why direct-quantile ML fails in the tail:** with only ~10–84 training observations below
  the 1% quantile (training windows of 1,000–8,400 days), trees cannot learn the extreme tail. Gradient boosting's 1% VaR is
  breached 2.9× as often as it should be, and both understate ES by about 10%. This is
  the "few tail points" weakness anticipated in RESEARCH_DESIGN §5; the hybrid design
  (ML for scale, residuals for the tail) is the one that works.
- Stress-period analysis (H3) and interpretability (H4) come next. The lockbox remains closed.

---

## 2026-11-06 — Stress analysis (H3), P1 (SPY), development sample

Rules fixed and committed before any stress statistic (Clarification 4, commit `a65c036`).
Regimes by target date: 8 episodes pooled = 1,604 days (calm 4,931); VIX at origin > 30 =
688 days. Power: at 1%, 16.0 expected hits in the episodes and 6.9 when VIX > 30 (thin);
at 5%, 80.2 and 34.4.

**Hit rates by regime**

| Model | 1% calm | 1% episodes | 1% VIX>30 | 5% calm | 5% episodes | 5% VIX>30 |
|---|---|---|---|---|---|---|
| FHS-GJR | 1.05% | **1.37%** | **1.02%** | 4.77% | 7.23% | **4.07%** |
| GJR-t | 1.30% | 1.68% | 1.02% | 5.54% | 7.92% | 5.23% |
| ML-vol + FHS | 1.05% | 2.68% | 3.05% | 5.01% | 8.67% | 7.99% |
| linqr_l1 | 1.01% | 3.62% | 3.20% | 5.37% | 10.47% | 8.14% |
| QRF | 1.03% | 3.99% | 5.38% | 4.70% | 10.22% | 10.76% |
| Gradient boosting | 2.21% | 5.11% | 5.52% | 7.00% | 12.22% | 13.52% |
| HS-250 | 0.75% | 4.05% | 4.07% | 4.06% | 10.79% | 10.61% |
| EWMA | 1.89% | 2.68% | 1.74% | 5.21% | 8.23% | 5.67% |

**Loss-differential regressions vs FHS-GJR** (FZ0; d = a + b·episode + e, HAC).
a = difference in calm periods, b = additional difference in stress; positive = worse than FHS.

| Model | α | a (p) | b (p) |
|---|---|---|---|
| ML-vol + FHS | 1% | **−0.105 (0.0015)** | **+0.477 (0.015)** |
| ML-vol + FHS | 2.5% | **−0.067 (0.0008)** | **+0.306 (0.008)** |
| ML-vol + FHS | 5% | **−0.050 (0.0002)** | **+0.197 (0.013)** |
| QRF | 2.5% | +0.041 (0.23) | +0.434 (0.010) |
| Gradient boosting | 2.5% | +0.046 (0.18) | +0.554 (0.002) |
| linqr_l1 | 2.5% | −0.013 (0.76) | +0.268 (0.002) |
| GJR-t | 2.5% | +0.011 (0.15) | +0.030 (0.094) |

**MCS within stress:** pooled episodes at 2.5% and 5% = {FHS-GJR} alone; at 1% {FHS-GJR,
GJR-t, ML-vol, HAR}. VIX > 30 at 1% and 2.5%: large sets (too few hits to separate models).

**Reaction speed** (66 events = the 1% largest daily losses; VaR_1% relative to 5 sessions
before; k = 1 is the first forecast after the loss):

| Model | k = 1 | k = 5 | Days to 1.5× |
|---|---|---|---|
| GJR-t / FHS-GJR | 1.67 | 1.63 | 1 |
| GARCH-t | 1.42 | 1.53 | 4 |
| HAR-FHS | 1.41 | 1.60 | 3 |
| EWMA | 1.37 | 1.50 | 5 |
| ML-vol + FHS | 1.38 | 1.37 | never |
| linqr_l1 | 1.28 | 1.37 | never |
| Gradient boosting / QRF | 1.29 / 1.23 | 1.32 / 1.25 | never |
| HS-250 / HS-500 | 1.16 / 1.08 | 1.23 / 1.11 | never |

**Reading (H3: supported; relative performance does change in stress):**
- FHS-GJR is the only model whose calibration survives stress (1.37% at 1% in the
  episodes; 1.02% when VIX > 30), and the sole MCS member in the episodes at 2.5% and 5%.
- The ML hybrid's full-sample tie with FHS-GJR is **two opposite, significant effects**:
  better in calm markets (p ≤ 0.0015 at every α) and worse in crises (p ≤ 0.015). It is a
  calm-weather model.
- Every feature-based model, linear or ML, loses ground in stress (b > 0, p ≤ 0.01 at
  2.5%). The common cause is reaction speed. GARCH-family VaR jumps 67% the day after a
  large loss; the tree models move 23–29% and never reach 1.5×, because a tree cannot
  forecast beyond the range of training outcomes in its leaves. GJR-t's shift is small
  (p = 0.09).
- Practical message for a risk team: an ML scale model may be worth using in calm regimes,
  but an econometric model must govern in stress, e.g. a regime switch on VIX or a
  max(ML, GARCH) rule. That rule is a hypothesis for future work, not tested here (it
  would be a post-hoc model).

---

## 2026-11-06 — Interpretability (H4), P1 (SPY), development sample

Criteria fixed and committed before any importance value (Clarification 5, commit
`2bb1d58`). 26 yearly refits per model; permutation importance on each validation window
(5 repeats); TreeSHAP for every out-of-sample forecast of `mlvol_fhs` and `gbq` (1%).

**Same models as the forecasts?** `mlvol_fhs`: the SHAP-implied VaR equals the saved
forecasts on 100% of 6,535 days (max difference 2e-16). `gbq`: identical on only **34%** of
days. On the other 66% the separately fitted per-level quantile models crossed and the
sorting fix changed the reported 1% VaR (by up to 1.0 percentage point). SHAP explains
boosting's raw 1% model, not its reported VaR on those days.

**Pre-registered criteria**

| Model | C1 top-3 vol/VIX | C2 economic sign | C3 stability ≥ 0.6 | H4 |
|---|---|---|---|---|
| ML-vol + FHS | ✅ vix, gk_5, gk_1 | ❌ 9/11 positive (abs_ret −0.56, rv_1 −0.28) | ❌ 0.544 | not supported |
| QRF | ✅ gk_5, vix, pk_1 | n/a (no SHAP) | ❌ 0.348 | not supported |
| Gradient boosting | ❌ gk_5, vix, baa10y_chg_22 (credit) | ❌ 8/11 (abs_ret −0.56, rv_22 −0.17; rv_1 unused → undefined) | ✅ 0.691 | not supported |

(For boosting, `rv_1` is never used, so its SHAP values are all zero and the correlation is
undefined, counted as not positive per the rule. Excluding it would not change the result:
two negative signs remain.)

**What the models rely on.** Share of positive permutation importance: volatility 46–59%,
VIX 18–44%, credit 2–14%, other 9–14%. The hybrid relies most on VIX (vix + vix_chg_5 = 44%)
and range-based volatility (gk_5, gk_1). The return-based HAR features (rv_*) matter much
less than the range-based ones.

**Why ML fails in stress, shown directly (descriptive).** The hybrid's model used through
2008 (refit 2008-01-22) had seen VIX at most 45.7 in training. Its 1% VaR was 3.81% on
2008-09-12 (VIX 25.7) and 3.82% on 2008-10-10 (VIX 69.9), with identical SHAP
contributions: both days fall in the same leaves, so the forest is **saturated**. FHS-GJR
moved from 4.12% to 11.73%; QRF from 3.27% to 3.18%. In March 2020, after training on
2008 (max VIX 80.9), the hybrid reached 11.19% (FHS-GJR 19.81%). A tree model cannot
forecast a risk level outside the range of its training outcomes; GARCH extrapolates through
its recursion. This is the mechanism behind the H3 result.

**Reading:** H4 is **not supported** under the pre-stated all-three rule for any model.
The *direction* is economically sensible (VIX and range volatility dominate; VIX's
contribution rises with its level, correlation 0.79–0.87). But (i) SHAP signs for correlated
volatility features are not reliable (|r| gets a negative sign once range volatility is
included), and (ii) importance rankings shift a lot from year to year (median Spearman
0.35–0.69). For model validation, the explanations are suggestive, not stable enough to
govern a model.

---

## 2026-11-07 — Robustness: P2 (60/40), P3 (Canada), P1 rolling window

Specification committed before any of these forecasts (Clarification 6, commit `b7ab38d`).
P3 built with the 3 XIU.TO non-cash corrections (logged in the build manifest; remaining
warning: the 2001-09-11 partial TSX session). Same 11 models, same rules, dev sample only.

| Setting | Forecasts | FHS-GJR FZ0 rank (1/2.5/5%) | FHS-GJR CC p (1/2.5/5%) | RF-vol + FHS rank | QRF / GBQ ranks |
|---|---|---|---|---|---|
| **P1 SPY, expanding (main)** | 6,535 (1997–2022) | 1 / 1 / 2 | 0.32 / 0.15 / 0.36 | 2 / 2 / 1 | 9,8,7 / 10,9,8 |
| P2 60/40 balanced | 4,142 (2006–2022) | 1 / 1 / 1 | **0.027** / 0.36 / 0.27 | 6 / 7 / 7 | 10,9,8 / 11,11,10 |
| P3 XIU.TO (CAD) | 4,838 (2003–2022) | 1 / 1 / 1 | 0.15 / 0.46 / 0.86 | 5 / 3 / 3 | 8,9,9 / 9,8,8 |
| P1 rolling 1,000 | 6,535 (1997–2022) | 1 / 1 / 1 | 0.31 / 0.48 / 0.40 | 5 / 5 / 5 | 10,10,8 / 11,11,11 |

MCS (FZ0): FHS-GJR is in the MCS in every setting at every level; it is the **sole** member
at 2.5% and 5% in the rolling run, and at 2.5% in P2.

**Stress shift vs FHS-GJR** (episode dummy, FZ0, α = 2.5%; b > 0 = loses ground in stress):

| Model | P1 | P2 | P3 | P1 rolling |
|---|---|---|---|---|
| RF-vol + FHS | +0.31 (p 0.008) | +0.98 (0.006) | +0.27 (0.023) | +0.50 (0.020) |
| QRF | +0.43 (0.010) | +1.34 (0.003) | +0.60 (0.028) | +0.75 (0.009) |
| Gradient boosting | +0.55 (0.002) | +2.06 (0.0002) | +0.53 (0.003) | +1.16 (0.0006) |
| linqr_l1 | +0.27 (0.002) | +0.80 (0.002) | −0.06 (0.40) | +0.60 (0.002) |
| GJR-t | +0.03 (0.094) | +0.09 (0.002) | +0.06 (0.015) | +0.04 (0.087) |

**Calm-period difference, RF-vol vs FHS-GJR** (a < 0 = hybrid better in calm; α = 2.5%):
P1 −0.067 (p 0.0008); P2 −0.043 (p 0.046); P3 −0.006 (p 0.68); rolling −0.021 (p 0.31).

**Reading:**
- **Robust across all four settings:** FHS-GJR ranks first on FZ0 at 11 of 12
  (setting × α) cells and is always in the MCS. No ML model beats it (H2a not supported
  anywhere). Direct-quantile ML (QRF, boosting) is always in the bottom four. Every ML model
  loses ground in stress, significantly, in every setting (H3 supported everywhere).
- **Not robust (state as P1-specific):** the hybrid's significant calm-market advantage
  (P1 strong, P2 borderline, P3 and rolling absent). The "calm-weather model" reading holds
  for SPY with an expanding window only.
- **The rolling window hurts ML most** (1,000 training days, ~10 below the 1% quantile):
  QRF/boosting fall to ranks 10–11 and the lasso control needs 5 floored forecasts. This is
  consistent with the "few tail points" mechanism.
- FHS-GJR's only coverage rejection is P2 at 1% (1.40% hits, p = 0.027): one rejection in
  12 cells, about what a 5% test produces by chance, but reported.

---

## 2026-11-07 — LOCKBOX PROTOCOL (written and pushed before any lockbox number exists)

- **What is frozen:** code, configs and model specifications at commit `2cba50d` (pushed to
  GitHub 2026-11-07 14:55 UTC). No model, grid, feature or rule may change because of the
  lockbox results.
- **Period:** targets 2023-01-01 → 2026-10-30 (~960 trading days). Raw snapshots were taken
  on 2026-11-06 and are unchanged (hash-verified at load).
- **Runs, each executed once:** P1 (headline), then P2 and P3 (robustness), with
  `alphalab forecast --include-lockbox` then `alphalab backtest --include-lockbox`. Models
  are refitted on their usual schedule through the lockbox, using only data available at
  each origin (as in a live deployment).
- **Integrity check:** the lockbox run also re-forecasts the development period. Those
  forecasts must be bit-identical to the saved development forecasts, proving nothing
  changed.
- **Reported, whatever the outcome:** coverage, ES test, FZ0 ranks, MCS and DM, as in the
  development sample. **Power warning, stated in advance:** about 9.6 expected hits at 1%
  and 48 at 5%, so the lockbox can confirm or contradict the *direction* of the main
  findings but cannot by itself establish small differences.
- **Pre-stated reading:** the main findings (FHS-GJR ≥ every ML model on FZ0; direct-quantile
  ML worst) are *consistent* with the lockbox if FHS-GJR is in the lockbox MCS at 2.5% and
  5% and neither QRF nor GBQ has lower mean FZ0 than FHS-GJR. Otherwise they are reported as
  *not replicated*.
- **Not done:** no stress-episode analysis inside the lockbox. Clarification 4 defined no
  lockbox episodes (e.g. the April 2025 tariff shock), and defining one now would be post hoc.

---

## 2026-11-07 — LOCKBOX RESULTS (opened once; protocol `dc8af86`, pushed 15:28 UTC)

Runs: P1, P2, P3 with `--include-lockbox`, frozen code `2cba50d` + protocol commit.
**Integrity: for all three portfolios, the development-period forecasts in the lockbox run
are bit-identical to the saved development forecasts.** ~961 lockbox forecasts per model
(targets 2023-01-03 → 2026-10-30); no floored forecasts.

**Pre-stated rule** (FHS-GJR in MCS at 2.5% and 5%, AND neither QRF nor GBQ has lower mean
FZ0): **NOT REPLICATED in P1, P2 or P3.** GBQ has lower mean FZ0 than FHS-GJR in all three;
QRF does in P1 and P2. FHS-GJR is in the MCS at 2.5% and 5% in P1 and P3, but not at 2.5% in P2.

**P1 (headline), FZ0 rank of 11 and MCS:**

| Model | 1% rank | 2.5% rank | 5% rank | Hit rates (1 / 2.5 / 5%) |
|---|---|---|---|---|
| linqr_l1 (control) | 1 | 1 | 1 | 0.73% / 1.56% / 4.37% |
| Gradient boosting | 3 | 2 | 2 | 1.35% / 2.50% / 5.20% |
| ML-vol + FHS | 2 | 3 | 3 | 1.14% / 2.50% / 4.16% |
| QRF | 7 | 4 | 4 | 0.83% / 1.56% / 3.95% |
| FHS-GJR | 5 (MCS) | 6 (MCS) | 5 (MCS) | **1.04% / 1.98% / 4.79%** (CC p 0.89 / 0.37 / 0.63) |
| GJR-t | 6 | 7 | 7 | 1.56% / 2.71% / 6.14% |
| EWMA, HS | 9–11 | 9–11 | 9–11 | |

DM (FZ0), FHS-GJR vs: linqr_l1 p = 0.12 / **0.037** / 0.081 (the control is better at 2.5%);
GBQ p = 0.65 / 0.11 / 0.15; ML-vol p = 0.25 / 0.13 / 0.065; QRF p = 0.50 / 0.57 / 0.16.
On P1 the ML models are better on average but **not significantly**, with ~10–48 expected
hits.

**P2:** GBQ / ML-vol / linqr_l1 rank 1–3; FHS-GJR 5–6 (outside the MCS at 2.5%). ML better
than FHS-GJR significantly in places (ML-vol p = 0.030 / 0.011 / 0.052; GBQ p = 0.015 at 1%).
**P3:** almost every model in the MCS; GBQ rank 1–2; FHS-GJR 3–5; the only significant DM is
FHS-GJR beating QRF at 5% (p = 0.022).

**Market conditions (descriptive):** lockbox mean VIX 17.4 vs 20.6 in development; VIX > 30
on 1.6% of days vs 10.5%; annualised SPY volatility 14.9% vs 19.9%; 4 days with |r| > 4% vs
86; worst day −5.85% (2025-04-04).

**Reading:**
1. **The development-sample headline does not carry over to 2023–2026.** In the lockbox,
   feature-based models (the lasso control, boosting, the RF-volatility hybrid) have lower
   average FZ0 than FHS-GJR. On the headline portfolio the differences are not significant;
   on P2 some are.
2. **What did hold:** FHS-GJR stayed well calibrated in all three portfolios (P1 CC p ≥ 0.37)
   and remained in the MCS in 7 of 9 (portfolio × α) cells. HS and EWMA stayed near the
   bottom. Nothing in the pipeline changed (bit-identical development forecasts).
3. **A possible explanation, stated as post hoc and untested:** the lockbox was unusually
   calm (VIX > 30 on 1.6% of days), and the development sample found feature-based models
   *better in calm periods and worse in stress* (H3). The lockbox reversal is consistent with
   that pattern, but this reading was formed after seeing the result. Confirming it would need
   a new pre-specified test on new data. The lockbox itself contains almost no stress (one
   episode, April 2025), so it cannot test stress robustness.
4. **Implication for the overall conclusion:** "econometric models dominate" is
   regime-dependent, not a general law. The defensible statement is: FHS-GJR is the most
   *robust* choice (calibrated in every period and portfolio, and best in stress), while
   feature-based ML can have lower average loss in calm periods. Which is better depends on
   how much weight the user puts on stress performance.

---

## 2026-11-07 — Clean-clone reproducibility check

1. **First clone** (commit `f3ae5cc`, fresh environment): 1 test failed. The fresh install
   got `quantile-forest` 1.5.0 instead of 1.4.2, and QRF quantiles moved materially even with
   identical settings (LIMITATIONS #14). Fix: `requirements-lock.txt` with the exact versions
   behind every logged result, installed by `scripts/setup_mac.sh`; QRF settings made
   explicit (commit `69e8a1c`). No result changed.
2. **Second clone** (commit `69e8a1c`, fresh isolated environment, documented setup only):
   254 tests passed, lint clean, live Yahoo tests passed. P1 rebuilt end-to-end from the
   original raw snapshots (prices `20261106T222941Z_e325d363`, features
   `20261106T222947Z_d0d2e72e`): **all 11 models' 6,535 forecasts and all four backtest
   tables are bit-identical to the logged results.**

Lesson for the write-up: without the lock file the project was *not* reproducible one day
later. Reproducibility needs locked data (snapshots), locked code (git) **and** a locked
environment.
