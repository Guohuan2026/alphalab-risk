# Evidence status: what each result can and cannot support

Status as of **2026-11-12** (Milestone 16 calibration, RESEARCH_DESIGN Addition 10;
previous version 2026-11-11, Addition 9 amendment 11). This page is the authoritative guide
to how strongly each kind of result in `RESULTS_LOG.md` and `REPORT.md` may be read. It
changes no number: it classifies them.

**Why this page exists.** The Bayer–Dimitriadis ESR backtest failed its pre-specified
validation under serially dependent tail hits (amendment 10(k)). That showed a test can be
correct in textbook conditions and still mis-sized under the dependence found in daily risk
data, so every other test was then put through a finite-sample size check with designs,
nulls, thresholds, seeds and failure handling frozen before running (Addition 10, reviewed by
Codex before and after; pre-run fixes in 10(j)). A test that failed was classified as
unsupported; no method was modified after the results were seen.

## The four categories

| Category | Meaning | May be stated as |
|---|---|---|
| **A. Supported inference** | Size within the frozen tolerance in every gated design for that sample length and α (simulated dependence envelope; Addition 10). | "rejected / not rejected at 5%", with any caveat listed |
| **B. Descriptive** | Estimates, rankings and diagnostics; correct as measurements, no sampling-error claim. Includes the statistics of tests classified unsupported. | "higher / lower", "ranked first", never "significantly" |
| **C. Pending** | Not yet checked. | *(no test is now in this category)* |
| **D. Unsupported / retired** | The test failed its size check (Milestone 16) or its validation (ESR). | Statistic as a diagnostic only; no p-value, significance or GO/NO-GO |

## Calibration results (Milestone 16; R = 2,000, seed 20261112; nominal 5%, limit 6.4%; MCS 90%, limit 11.9%)

Sample lengths: T = 6,535 (P1 development) and T = 961 (lockbox length). Rates are the worst
gated design. Full tables: `data/results/calibration/{diff,es}_reps2000_seed20261112/`
(`cells.csv`, `decisions.csv`, `hit_strata_n961_a1.csv`).

| Test | T = 6,535 | T = 961 | Decision |
|---|---|---|---|
| Diebold–Mariano (Newey–West, default lag) | 46.3% (Dhigh) | 50.1% (Dhigh) | **D — unsupported** |
| DM, fixed-b fallback (b = 0.1) | 7.6% (Dhigh) | 20.7% (Dhigh) | fallback fails → unsupported |
| H3 stress-dummy HAC regression | 30.2% (Dhigh) | 30.2% (Dhigh) | **D — unsupported** (fixed-b diagnostic 7.1–17.3%, never adoptable) |
| MCS 90% (block max(5, lag)) | 98.5% (Dhigh) | 99.2% (Dhigh) | **D — unsupported** |
| MCS, block-40 fallback | 86.3% (Dhigh) | 96.5% (Dhigh) | fallback fails → unsupported |
| Christoffersen CC | ≤ 5.8% all α | ≤ 5.6% all α | **A** at every α and length |
| Christoffersen IND | ≤ 6.2% all α | 1%, 2.5%: ≤ 1.9%; **5%: 9.0%** | **A**, except lockbox-length 5% (D) |
| Kupiec POF | ≤ 6.4% all α | 2.5%, 5% pass; **1%: 7.0%** | **A**, except lockbox-length 1% (D; discreteness) |
| Engle–Manganelli DQ | 2.5%, 5% pass; **1%: 7.8%** | 5% pass; **1%: 11.1%, 2.5%: 7.5%** | **A** at 5% (and 2.5% at T = 6,535); **D** otherwise |
| McNeil–Frey (raw / standardised) | 1% pass; **2.5%, 5% fail** (B2: 8.0–14.9%) | **1%, 2.5% unavailable** (too few hits: 17–37% no result); 5%: raw passes, std fails | **D** except raw at T = 961 α 5% and raw/std at T = 6,535 α 1% |
| Acerbi–Székely Z2 (t-shape MC p-value) | **13.6–30.7%** (B1, B2) | **13.6–26.5%** | **D — unsupported** |
| Bayer–Dimitriadis ESR | — | — | **D — retired** (amendment 11) |

Reading the table:
- **Sanity designs pass**: with i.i.d. normal or t(3) differentials DM is 4.1–5.1% and H3
  5.2–6.0%, and a post-hoc check (not frozen; informational) gave the MCS code 11.7% (i.i.d.)
  and 8.3% (stochastic volatility and stress, no persistent mean) at its 10% level. The
  failures are therefore caused by dependence, not by the code.
- **Why DM, H3 and MCS fail**: the designs add a slowly mean-reverting component to the loss
  differential, matched to the real FZ0 series' autocorrelation and lag-10 / lag-100
  long-run-variance ratios. A Newey–West lag of 6–10 cannot capture it. Even **Dmid**,
  matched to the *median* real targets, gives DM 13.9–16.7%, H3 9.3–11.4% and MCS
  39–53% false exclusions. The stronger designs imply a true long-run-variance ratio of
  about 15–17, which the data cannot confirm or rule out beyond lag 100: the exact size of
  the over-rejection is uncertain, its direction is not.
- **Coverage and DQ**: rates are for correct forecasts (designs A and C). In the
  missed-volatility designs B1/B2 these tests reject 17–100%, which is correct power, not
  size distortion: clustered hits are their alternative.
- **Sparse hits (T = 961, α = 1%, ≈ 10 expected hits)**: POF rejects every sample with 0–4
  hits by construction; CC is conservative (3.6%); McNeil–Frey gives no result in up to a
  third of samples; DQ over-rejects (11%).

## Test by test

| Test (code) | Used for | Category | What may be said |
|---|---|---|---|
| Christoffersen CC (`coverage.py`) | H1 | **A** | Reject / not reject at 5%, development and lockbox, every α. |
| Christoffersen IND | H1 | **A** (D at T = 961, α 5%) | As CC; lockbox 5% IND p-values descriptive. |
| Kupiec POF | H1 | **A** (D at T = 961, α 1%) | Lockbox 1% POF is over-sized (discreteness). A *non-rejection* there remains informative; a rejection would not be. |
| DQ (`coverage.py`; development only) | H1 supplementary | **A** at 5% and at 2.5% on T = 6,535; **D** at 1% | 1% DQ p-values descriptive. A Monte Carlo DQ (rebuilding lagged-hit regressors each draw) would be needed to restore them. |
| Basel traffic light | Description | **B** | A counting rule, not a test. |
| McNeil–Frey (`es.py`) | ES adequacy | **D** in most cells | p-values descriptive except raw at T = 6,535 α 1% and T = 961 α 5%; the **severity ratio** stays **B**. |
| Acerbi–Székely Z2 (`es.py`; development only) | ES adequacy | **D** | Over-rejects whenever volatility dynamics are misspecified, even with correct conditional tails assumed; the **Z2 statistic** stays **B**. |
| Diebold–Mariano (`compare.py`) | H2a, H2b | **D** | Mean loss differences are **B**; "significantly better/worse" is not supported, in development or lockbox. |
| Model Confidence Set (`compare.py`) | H2a headline | **D** | Membership is descriptive; the procedure excludes equally good models far too often under realistic dependence, so exclusion is not evidence of inferiority. |
| HAC stress-dummy regression (`compare.py`) | H3 | **D** | Coefficient signs and sizes and regime breach rates are **B**; p-values are not supported. |
| SHAP, permutation importance | H4 | **B** | Descriptive by design. |
| Ten-day horizon table | Description | **B** | Declared descriptive (Clarification 8). |
| Bayer–Dimitriadis ESR | Exploratory | **D (retired)** | γ̂, Wald and t as diagnostics only. |

## What the headline findings rest on

| Finding (REPORT.md) | Rests on | Status after calibration |
|---|---|---|
| FHS-GJR is the best-calibrated model: passes Kupiec and Christoffersen in every development cell; in the lockbox its conditional coverage is not rejected (P1 CC p = 0.89 / 0.37 / 0.63) | CC (A), POF/IND mostly A; hit rates (B) | **Stands as "not rejected"**, never as "calibration confirmed": a non-rejection with ≈ 10 expected hits at 1% has limited power. The 1% POF and 5% IND tests at lockbox length are over-sized (size, not power), so their non-rejections are not weakened by that distortion. |
| ML models fail coverage (H1 not supported for ML) | CC/POF (A) | **Stands** (development cells; T = 6,535 is calibrated at every α). |
| DQ rejects every model, including FHS-GJR on P1 (p = 0.019, 0.002, 0.011) | DQ | **Stands at 2.5% and 5%** (p = 0.002, 0.011); the 1% rejections are descriptive. |
| Ranking by mean FZ0; "ML does not beat FHS-GJR" (development); lockbox reversal | Mean losses, ranks (B) | **Descriptive facts.** "Lower average loss" is safe; every "significantly" from DM is withdrawn. |
| DM significance (FHS-GJR beats QRF and GBQ; control vs FHS-GJR at 2.5% in the lockbox, p = 0.037; P2 hybrid p = 0.011) and MCS membership | DM, MCS (D) | **Withdrawn as inference**; reported as descriptive. |
| Pre-stated lockbox rule "NOT REPLICATED" | Mean-FZ0 ordering (B) and MCS membership (D) | **Unchanged**: the rule fails on its descriptive half (GBQ has lower mean FZ0 in all three portfolios), whatever the MCS does. |
| H3: ML better in calm, worse in crises (HAC p ≤ 0.015) | HAC regression (D); breach rates by regime (B) | **Direction descriptive only**; significance withdrawn. The consistency of sign across 12 robustness cells is descriptive support, not a test. |
| H4: drivers unstable across refits | SHAP/permutation (B) | Descriptive. |
| ESR results | ESR (D) | Diagnostics only. |

## Labels that always apply

- **Held-out vs exploratory.** Only the pre-registered tests run once on the lockbox
  (coverage, McNeil–Frey, FZ0, DM, MCS; 2026-11-07) are held-out evidence; of those, only
  CC, IND (1%, 2.5%) and POF (2.5%, 5%) now carry supported inference at lockbox length.
  DQ, Z2, ESR, the horizon table, Addition 9 and the Milestone 16 calibration were added
  after the lockbox was opened: exploratory. The calibration used simulated data and the
  development stress calendar only; the lockbox was not accessed.
- **Scope of "supported".** A pass holds within the simulated dependence envelope
  (Addition 10(b), (c), (j)), not beyond it. MCS conclusions also assume independent model
  deviations around a common volatility.
- **Point estimates are unaffected**: hit rates, losses, ranks, γ̂, Z2 and regression
  coefficients are measurements; the categories concern only sampling-error inference.

## What could restore the unsupported tests (not started; each needs a new frozen design)

- **DM / H3 / MCS:** inference robust to long memory in the differential (e.g. larger
  fixed-b bandwidths, b ≈ 0.3–0.5, or a subsampling or block-of-blocks approach), validated
  again on the same frozen designs. None is known in advance to pass Dhigh.
- **DQ at 1%, IND/POF at lockbox length:** Monte Carlo (Dufour 2006) p-values.
- **McNeil–Frey, Z2:** no simple remedy; dependence in the residuals and a wrong tail shape
  both bias them.
