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
| **D. Unsupported / retired** | The test failed its size check (Milestone 16) or its validation (ESR). | Statistic as a diagnostic only; no significance or GO/NO-GO. A p-value may be quoted only to document an earlier claim, never as evidence |

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
- **What "supported" means**: size within the frozen tolerance in every gated design for that
  test, length and α, i.e. size control within the simulated envelope. It says nothing about
  power, and each test is gated separately, so CC can pass where IND fails (T = 961, α 5%).
- **Sanity designs pass**: with i.i.d. normal or t(3) differentials DM is 4.1–5.1% and H3
  5.2–6.0% (frozen designs). Two post-hoc MCS runs (not frozen, outside every decision;
  R = 300) gave 11.7% (11 i.i.d. models) and 8.3% (stochastic volatility and stress, no
  persistent mean) at the 10% level. This is consistent with the persistent component
  driving the distortion in these designs; it is a diagnostic check, not a full
  verification of the implementation.
- **Why DM, H3 and MCS fail**: the designs add a slowly mean-reverting component to the loss
  differential, matched to the real FZ0 series' autocorrelation and lag-10 / lag-100
  long-run-variance ratios. A Newey–West lag of 6–10 cannot capture it. Even **Dmid**,
  matched to the *median* real targets, gives DM 13.9–16.7%, H3 9.3–11.4% and MCS
  39–53% false exclusions. The matching used finite-lag *estimates* (lag 10 and lag 100).
  The untruncated dependence the designs imply is larger: for the AR components alone, with
  calm-period innovation variance normalised to 1 and ignoring the stress multiplier and
  the volatility normalisation, the long-run-to-variance ratio is about 2.9 (Dmid), 16.8
  (Dhigh) and 14.7 (Dhigh2). The sample cannot identify persistence beyond lag 100. So the
  results show that these tests are vulnerable under DGPs that match the observed
  statistics, including a median-matched one; they do not establish the same distortion
  in the actual series.
- **Coverage and DQ**: rates are for correct forecasts (designs A and C). In the
  missed-volatility designs B1/B2, IND, CC and DQ reject 17–100%, which is power: clustered
  hits are their alternative. **POF is different**: its null (correct overall breach rate)
  can still hold in B1/B2, so its 27–67% rejection there is size distortion under clustered
  breaches. POF is supported only where breaches are close to independent; CC is the primary
  coverage test.
- **Sparse hits (T = 961, α = 1%, ≈ 10 expected hits)**: POF rejects every sample with 0–4
  hits by construction; CC (3.6%) and IND (1.2%) are conservative; McNeil–Frey gives no result in up to a
  third of samples; DQ over-rejects (11%).

## Test by test

| Test (code) | Used for | Category | What may be said |
|---|---|---|---|
| Christoffersen CC (`coverage.py`) | H1 | **A** | Reject / not reject at 5%, development and lockbox, every α. P2/P3 lengths lie between the simulated ones: provisionally supported by endpoint checks. |
| Christoffersen IND | H1 | **A** (D at T = 961, α 5%) | As CC; lockbox 5% IND p-values descriptive. |
| Kupiec POF | H1 | **A** (D at T = 961, α 1%) | Lockbox 1% POF is over-sized (discreteness). A *non-rejection* there remains informative; a rejection would not be. |
| DQ (`coverage.py`; development only) | H1 supplementary | **A** at 5% and at 2.5% on T = 6,535; **D** at 1% | 1% DQ p-values descriptive. A Monte Carlo DQ (rebuilding lagged-hit regressors each draw) would be needed to restore them. |
| Basel traffic light | Description | **B** | A counting rule, not a test. |
| McNeil–Frey (`es.py`) | ES adequacy | **D** in most cells | p-values descriptive except three supported cells: raw and standardised at T = 6,535 α 1%, raw at T = 961 α 5%; the **severity ratio** stays **B**. |
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
| FHS-GJR is the best-calibrated model (draft v1). Checked against saved results 2026-11-12: its CC null is not rejected in 11 of 12 development cells (rejected on P2 at 1%, p = 0.027); on P1 it is the only model never rejected by Kupiec or Christoffersen; in the lockbox its CC null is not rejected in 9 of 9 cells, as for four other models (P1 CC p = 0.89 / 0.37 / 0.63) | CC (A), POF/IND mostly A; hit rates (B) | **Restated:** FHS-GJR's VaR conditional-coverage null is not rejected; CC meets the simulated size criterion, with sparse-hit power limitations (≈ 10 expected hits at 1% in the lockbox). This does not confirm correct calibration, ES adequacy, or superiority over other models; "best-calibrated" is a descriptive summary (fewest rejections: 1 of 12 development cells; on P1 the only model never rejected), not a test result. |
| ML models fail coverage (H1 not supported for ML) | CC/POF (A) | **Stands** (development cells; T = 6,535 is calibrated at every α). |
| DQ rejects every model, including FHS-GJR on P1 (p = 0.019, 0.002, 0.011) | DQ | **Restated per portfolio:** at 5% DQ rejects all 11 models on P1 and P2 and 10 of 11 on P3 (FHS-GJR not rejected, p = 0.43), checked against the saved `backtest_dq.csv`. P1 (T = 6,535) is calibrated at 2.5% and 5%, so FHS-GJR's P1 rejections there (p = 0.002, 0.011) stand. P2/P3 (T = 4,142 / 4,838) lie between the simulated lengths: 5% passed at both, 2.5% only at T = 6,535. All 1% rejections are descriptive. |
| Ranking by mean FZ0; "ML does not beat FHS-GJR" (development); lockbox reversal | Mean losses, ranks (B) | **Descriptive facts.** "Lower average loss" is safe; every "significantly" from DM is withdrawn. |
| DM significance (FHS-GJR beats QRF and GBQ; control vs FHS-GJR at 2.5% in the lockbox, p = 0.037; P2 hybrid p = 0.011) and MCS membership | DM, MCS (D) | **Withdrawn as inference**; reported as descriptive. |
| Pre-stated lockbox rule "NOT REPLICATED" | Mean-FZ0 ordering (B) and MCS membership (D) | **Unchanged**: the rule fails on its descriptive half (GBQ has lower mean FZ0 in all three portfolios), whatever the MCS does. |
| H3: ML better in calm, worse in crises (HAC p ≤ 0.015) | HAC regression (D); breach rates by regime (B) | **Restated (draft v2):** every ML model's loss relative to FHS-GJR worsens in crisis episodes in all 36 model × setting × level cells (correlated; descriptive consistency, not a test); a calm-period advantage holds only for the RF-vol hybrid (P1, P2, rolling; ≈ 0 on P3), not for the quantile forest or gradient boosting. Significance withdrawn. |
| H4: drivers unstable across refits | SHAP/permutation (B) | Descriptive. |
| ESR results | ESR (D) | Diagnostics only. |

## Labels that always apply

- **Held-out vs exploratory.** Only the pre-registered tests run once on the lockbox
  (coverage, McNeil–Frey, FZ0, DM, MCS; 2026-11-07) are held-out evidence; of those, only
  CC, IND (1%, 2.5%), POF (2.5%, 5%; independent breaches) and raw McNeil–Frey at 5% now carry supported inference at lockbox length.
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
