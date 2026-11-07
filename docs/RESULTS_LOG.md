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
