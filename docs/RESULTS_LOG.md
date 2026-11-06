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
