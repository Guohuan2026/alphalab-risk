# Autonomous run summary — 2026-11-11

Run window: start 18:16:02 EST, hard stop 20:16 EST (CLAUDE.md "Autonomous run rules").
Branch `esr-robust-start` (local commits only; see Open issues). Everything here is
**EXPLORATORY: development sample, added after the lockbox was opened; none of it is
held-out evidence.**

## Bottom line

**Amendment 10 (ESR bootstrap studentised by esreg's independence covariance): NO-GO.**
The amended bootstrap has no numerical failures, but it over-rejects under serially
dependent tail hits, so its p-values are not valid tests. Per the frozen rule (amendment
10(g)) work on the estimator stopped; the real-data bootstrap was not launched. ESR p-values
remain descriptive only.

## What was done

1. **Amendment frozen before any result** (`docs/RESEARCH_DESIGN.md`, Addition 9,
   amendments 9 and 10; commit `595ed7a`). Amendment 9 records the robust-start attempt
   (gate failed, not adopted). Amendment 10: primary V2 = esreg `vcovA(sigma_est="ind",
   misspec=FALSE)`; secondary V1 = `ind, misspec=TRUE`. Finding from esreg's own source:
   `sigma_est="ind"` alone still runs the location-scale fit through `cdf_at_quantile`;
   misspec=FALSE is exactly cdf = α, so only V2 removes the fit (Codex's choice).
2. **Implementation** (`es.cov_variant`, `esr-sim --cov`, separate output folders). The
   pre-registered default path is unchanged: it reproduces a saved legacy chunk bit for bit.
   Tests: 83 ESR tests + new variant and summary tests pass; ruff clean.
3. **Reference check vs R esreg 0.6.3** (coefficients fixed): V2's ES covariance matches to
   ≤ 1e-13 in 25 of 30 comparisons. The other 5 (≤ 2%) are entirely the documented tie
   policy: esreg's tail variance uses `u <= 0`, so the quantile regression's interpolated
   points (residual ≈ ±1e-16) enter by rounding; the port counts them by its tie band. With
   the exact rule the port reproduces esreg to 10 digits. V1 agrees to ≤ 1.2% and fails on
   boundary fits as specified.
4. **Codex review of the implementation:** algebra confirmed; one MAJOR (a survivors-only
   pass was possible) fixed *before* results were read as amendment 10(j): usable now
   requires a complete run, ≤ 5% failed full-sample fits and ≤ 5% failed draws; Wilson
   intervals and failure counts added. The 6.4% threshold was never changed.
5. **Simulation** (designs A, B1, B2; T = 2,500; R = 1,000; seed 20261111; 4 low-priority
   cores; V2 18:08–18:57, V1 18:57–19:40).

## Results (5% level; block 20 is the gate; Wilson 95% in brackets)

| Design | Strict, V2 block 20 | Intercept, V2 block 20 | Gate (≤ 6.4%) |
|---|---|---|---|
| A (correct forecasts) | 5.6% [4.3, 7.2] | 3.5% [2.5, 4.8] | pass |
| B1 (missed vol, φ 0.95) | 11.7% [9.9, 13.8] | 8.5% [6.9, 10.4] | **fail** |
| B2 (missed vol, φ 0.99) | 25.4% [22.8, 28.2] | 16.6% [14.4, 19.0] | **fail** |

- Failures (V2): 0 of 1,000 full samples, 0 failed draws in every design and block. **The
  absence of failures is not evidence of valid inference**: the size check fails.
- Block-length sensitivity (strict / intercept, blocks iid · 10 · 20 · 40):
  B1 24.5 · 15.3 · 11.7 · 10.1% / 12.3 · 8.6 · 8.5 · 5.5%;
  B2 44.7 · 36.8 · 25.4 · 19.8% / 20.0 · 18.4 · 16.6 · 14.5%;
  A 4.5–5.6% / 3.5–4.3% at every block.
- Serial dependence: longer blocks reduce over-rejection monotonically, so the dependence
  is the cause, but no block tested is long enough in B2 (and only block 40 for the
  intercept in B1). The bootstrap accounts for the dependence only partly.
- Asymptotic p-values: strict A 11.0%, B1 38.3%, B2 58.6% → NO-GO as tests.
- V1 (secondary): unusable; full-sample fit failures A 98, B1 409, B2 437 of 1,000.
- Monte Carlo sds (replication resampling incl. the pooled critical value): 0.9–3.0 points.
- Files: `data/results/esr_sim/n2500_reps1000_seed20261111_cov-ind_nomis/rejection_rates.csv`,
  `..._cov-ind_mis/rejection_rates.csv`.

## GO / NO-GO

| Method | Strict | Intercept |
|---|---|---|
| Block bootstrap, V2 (amendment 10) | NO-GO | NO-GO |
| Block bootstrap, V1 | unusable | unusable |
| Asymptotic | NO-GO | NO-GO |
| Pre-registered scl_sp bootstrap | NO-GO (unusable: boundary fits) | NO-GO |

All real-data cells: ESR p-values descriptive only. Lasso QR stays quarantined.

## Decisions made by Codex during this run (all in `docs/DECISIONS.md`)

- Freeze V2 (ind, misspec=FALSE) as primary; V1 as reported secondary; gate on V2 only.
- Implementation review: adopt the survivors-only fix (10(j)), Wilson CIs, failure counts.
- NO-GO confirmed for both tests (Agent HUD m-b5125423b0e5, 19:42).
- Recommended next milestone (for the owner to decide; not started): **retire ESR inference to
  descriptive status by a new amendment**, keep the descriptive results, and check the other
  backtests (Acerbi–Székely Z2, McNeil–Frey) for dependence-robust calibration separately.
- Codex's caveat: alternatives (HAC studentisation, longer blocks, score bootstraps) would each
  be a new research programme with fresh validation; none is known to work, and HAC plus a
  block bootstrap is not automatically pivotal.

## Real-data cost (for reference only — not applicable after NO-GO)

Measured under V2: main run 120 non-lasso cells × 999 draws ≈ 88 CPU-h ≈ 22 h on 4 workers
(P1 7.2 h, P2 3.1 h, P3 4.0 h, P1-rolling 7.6 h), plus ≈ 7 h block sensitivity. Not
requested, because the method failed validation.

## Open issues (for the owner)

1. **Methodological problem that remains:** under persistent clustering of tail hits the
   circular block bootstrap of the iid-studentised ESR statistic does not reproduce its
   sampling distribution at T = 2,500 with blocks ≤ 40. Any fix is a new design amendment
   and needs your approval (not started, per 10(g)).
2. **Push:** commits are local on `esr-robust-start`. The only sanctioned push path
   (`.claude/autonomy.py publish`) works from `alphalab-risk` only, so pushing needs either a
   merge into `alphalab-risk` or a change to the pre-push hook/settings; both need your
   approval. `--no-verify` was not used.
3. **Merge:** `esr-robust-start` → `alphalab-risk` awaits your review (amendments 9–10,
   end-solution audit, variant code).
4. **`.claude/settings.json`:** not created (it would bypass your `Edit(/.claude/**)` deny
   and the autonomy rules); run it yourself with `!` if wanted. Note `pip install` and
   `git commit` conflicts described in the chat.
5. Legacy P1 run: 996/1,320 chunks, stopped by budget; its 2,479 lasso draws are degenerate
   (quarantined, never counted). Superseded in any case by the NO-GO.
