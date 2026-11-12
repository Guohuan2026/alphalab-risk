# Evidence status: what each result can and cannot support

Status as of 2026-11-11 (Addition 9, amendment 11, owner-approved). This page is the
authoritative guide to how strongly each kind of result in `RESULTS_LOG.md` and `REPORT.md`
may be read. It changes no number: it classifies them.

**Why this page exists.** The Bayer–Dimitriadis ESR backtest failed its pre-specified
validation: under serially dependent tail hits its bootstrap over-rejected (block 20,
nominal 5%: strict 11.7% and 25.4%, intercept 8.5% and 16.6% in the two dependence designs;
amendment 10(k)). That failure is specific to ESR, but it shows that a test can be correct in
textbook conditions and still mis-sized under the dependence found in daily risk data. So
every other test was re-examined here, and none is assumed valid merely because ESR was
retired.

## The four categories

| Category | Meaning | May be stated as |
|---|---|---|
| **A. Supported inference** | The test is valid for the null it states (by construction and by size checks in simulation); remaining caveats are finite-sample ones, listed. | "rejected / not rejected at 5%", with the caveat |
| **B. Descriptive** | Estimates, rankings and diagnostics. Correct as measurements; no sampling-error claim attached. | "higher / lower", "ranked first", never "significantly" |
| **C. Inference pending calibration** | Standard, asymptotically justified tests whose finite-sample size under the persistence and heavy tails of these data has **not** been checked. | Reported with p-values, labelled *provisional*; conclusions must not rest on them alone |
| **D. Retired** | The method failed validation. | Statistics only, as diagnostics; no p-values or GO/NO-GO |

## Test by test

| Test (code) | Used for | Null hypothesis | Category | Basis, and what is still needed |
|---|---|---|---|---|
| Kupiec POF + Christoffersen IND/CC (`coverage.py`) | H1 | Hits are i.i.d. Bernoulli(α) (correct conditional coverage) | **A** (CC and IND) | Dependence is the *alternative* under this null, not a nuisance, so clustering correctly counts against the model. Checked: hand-computed values; size ≈ 5% and power against clustered hits in simulation. Caveats: χ² approximation with few hits (lockbox ≈ 10 hits at 1%; sparse n11 cell in IND); exact or Monte Carlo p-values for the 1% lockbox cells would remove it. **POF alone is not dependence-robust** if read as a test of the hit *rate* only (its variance assumes independence). |
| Engle–Manganelli DQ (`coverage.py`) | H1 (supplementary, development only) | Demeaned hits are a martingale difference, unpredictable from lags and VaR | **A**, with a finite-sample caveat | Same logic as CC. Size ≈ 5% in the project's simulation. The DQ χ² is known to be over-sized at low α and moderate T (Dumitrescu, Hurlin & Pham 2012); a Monte Carlo p-value at 1% would close this. |
| Basel traffic light (`coverage.py`) | Description | — | **B** | A regulatory counting rule, not a test. |
| McNeil–Frey (`es.py`) | ES adequacy | Mean exceedance residual is zero given a hit | **C** | Under a fully correct model the hit-day residuals form a martingale difference, so the i.i.d. bootstrap of their mean is asymptotically valid. Not checked: size with few hits (≈ 65 at 1% on P1; ≈ 10 in the lockbox) and heavy-tailed residuals; only power was simulated. The **severity ratio** itself is **B**. |
| Acerbi–Szekely Z2 (`es.py`) | ES adequacy (development only) | Losses i.i.d. with correct VaR/ES, tail shape = shape-matched Student-t | **C** | The p-value assumes the data's tail shape equals the Student-t matched to the median ES/VaR ratio. Size ≈ 5% was verified only when that assumption holds; robustness to a wrong tail shape is unchecked. The **Z2 statistic** itself is **B**. |
| Diebold–Mariano on FZ0 / tick loss (`compare.py`) | H2a, H2b, H3 | Equal expected loss | **C** | Newey–West HAC, Bartlett kernel, lag ⌊4(n/100)^{2/9}⌋ (10 on P1 development, 9 on P2/P3, 6 in the lockbox), no small-sample correction. Asymptotically dependence-robust, but FZ0 differentials are heavy-tailed and persistent, where such DM tests can over-reject. Needed: a size simulation with persistent, heavy-tailed differentials; Harvey–Leybourne–Newbold and fixed-b sensitivity. Lockbox p-values near 5% (0.037, 0.011, 0.015, 0.022) are the most exposed. |
| Model Confidence Set (`compare.py`, `arch`) | H2a headline | The set contains the best model | **C** | Block bootstrap with block = max(5, DM lag): 9–10 days on the development samples, 6 in the lockbox. The ESR work showed blocks up to 40 too short for persistent tail dependence, so MCS membership could be too narrow. Needed: block-length sensitivity (e.g. 10/20/40) and a size check. |
| HAC regression on a stress dummy (`compare.py`) | H3 | Loss differential equal in stress and calm | **C** | Stress periods are long, persistent episodes; a short HAC lag can understate the standard error of the dummy, so p-values like "p ≤ 0.0015" may be too small. Needed: lag sensitivity and a size check with episode-length dummies. |
| SHAP, permutation importance (`interpret`) | H4 | — | **B** | Descriptive by design; stability across refits is reported as robustness, not as inference. |
| Ten-day horizon table (`horizon.py`) | Description | — | **B** | Declared descriptive (Clarification 8). |
| Bayer–Dimitriadis ESR (`es.py`, `esr_*`) | Exploratory | ES regression intercept 0 (and slope 1) | **D (retired)** | Asymptotic p-values: over-sized under dependence (amendments 5–6). Pre-registered scl_sp bootstrap: unusable (boundary fits). Amended ind bootstrap: NO-GO (amendment 10). Estimates γ̂, Wald and t are kept as diagnostics. P1 lasso QR's ESR statistics stay blank (degenerate covariance); that quarantine concerns ESR only. |

## What the headline findings rest on

| Finding (REPORT.md) | Rests on | Status |
|---|---|---|
| FHS-GJR is the best-calibrated model: the only one passing Kupiec and Christoffersen in every development cell, and still calibrated in the lockbox (P1 CC p = 0.89 / 0.37 / 0.63) | CC/IND (A); hit rates (B) | **Supported**, with the few-hits caveat at 1% in the lockbox |
| DQ rejects every model, including FHS-GJR on P1 | DQ (A, finite-sample caveat) | **Supported**, development only; the 1% p-values should get a Monte Carlo check |
| Ranking by mean FZ0; "ML does not beat FHS-GJR" (development); lockbox reversal (feature-based models lower mean FZ0) | Mean losses and ranks (B) | **Descriptive facts.** "Lower average loss" is safe; "significantly lower" is provisional (C) |
| Pre-stated lockbox rule "NOT REPLICATED" | Mean-FZ0 ordering (B) and MCS membership (C) | The rule fails on its descriptive half (GBQ has lower mean FZ0 in all three portfolios), so the verdict does **not** depend on MCS calibration |
| DM significance (e.g. FHS-GJR beats QRF and GBQ, development) and MCS membership | DM, MCS (C) | **Provisional** until calibrated |
| H3: ML better in calm, worse in crises (HAC dummy p ≤ 0.015) | HAC regression (C); breach rates by regime (B) | Direction and breach rates are descriptive; significance is **provisional** |
| H4: drivers unstable across refits | SHAP/permutation (B) | Descriptive |
| ESR results | ESR (D) | Diagnostics only |

## Labels that always apply

- **Held-out vs exploratory.** Only the pre-registered tests run once on the lockbox
  (coverage, McNeil–Frey, FZ0, DM, MCS; 2026-11-07) are held-out evidence. DQ, Z2, ESR, the horizon table
  and everything in Addition 9 were added after the lockbox was opened: development samples
  only, exploratory.
- **Point estimates are unaffected** by any of this: hit rates, losses, ranks, γ̂ and Z2
  values are measurements; the categories concern only how much sampling-error inference
  they can carry.

## What would move a test from C to A

One validation milestone covers all of category C at low cost (simulation only, no new data,
no lockbox): size of DM (with and without HLN correction), MCS (blocks 10/20/40) and the H3
HAC regression under persistent, heavy-tailed loss differentials calibrated to the real
FZ0 series; size of McNeil–Frey and Z2 at the real numbers of hits and under a wrong tail
shape. Acceptance rules must be written down before running, as for amendment 6.
