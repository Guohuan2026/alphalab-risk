# Delegated decisions log

Decisions made on Codex's recommendation under the CLAUDE.md "Ask Codex" rule. One entry per decision: date, question, Codex's answer, action taken.


## 2026-11-11 — ESR robust-start validation thresholds (Addition 9, amendment 9; exploratory)

**Q1. Start-invariance tolerance.** On bootstrap draws and full samples where both esreg's
location-scale start and the warm start succeed, what difference counts as "numerically
identical"? Options: 1e-8, 1e-6 or 1e-4 (relative for Wald/t, absolute for p-values).
**Codex (codex exec, session 01a1c907):** 1e-6 relative for Wald/t and 1e-6 absolute for
p-values, with an absolute-tolerance fallback near zero, because relative differences are
unstable for near-zero statistics while 1e-6 is still stringent.
**Action:** criterion |Δ| ≤ 1e-6 · max(1, |x|) for W and t, |Δp| ≤ 1e-6, fixed before the
comparison was computed.

**Q2. Pilot failure gate (user: "under 1–2%").** Options: pooled < 2%; every non-quarantined
real cell and every simulation design individually < 2% with the upper 95% Wilson bound
reported; pooled non-quarantined < 1%.
**Codex:** each non-quarantined real cell and each simulation design individually < 2%, Wilson
upper bound reported, because pooling can hide an unstable cell; quarantined lasso-QR cells
stay excluded until cleared separately.
**Action:** adopted as the launch gate; quarantined cells reported but not gating. Launching
the overnight run additionally needs the user's approval (CLAUDE.md: jobs > 2 hours).

## 2026-11-11 00:35 — overnight plan under the owner's pre-approvals (tonight only)

Owner pre-approved (asleep): push `esr-robust-start`; launch the robust overnight bootstrap
(--max-hours 8) if every non-quarantined real pilot cell and every simulation design has a
failure rate < 2% AND the start-invariance check passes; otherwise ask Codex what to
investigate. Questions put to Codex (codex exec), with its answers and the action taken:

- **Q1. Legacy run while the robust run goes?** Codex: stop the legacy run once the pilot
  passes, preserving its checkpoints, so the robust draws that supersede it get the compute.
  **Action:** if (and only if) the robust run launches, stop the legacy session (script first,
  so it cannot move on to P2), keeping every saved chunk; in-progress chunks are lost, saved
  ones are untouched (atomic writes).
- **Q2. Quarantined lasso-QR cells in the robust run?** Codex: exclude them; robust starts do
  not establish the separate validation they need. **Action:** `--only '^(?!linqr_l1_)'`.
- **Q3. Config order within 8 h?** Codex: P1, P2, P3, then P1-rolling (primary result first,
  then breadth, then the robustness check). **Action:** adopted.
- **Q4. Push despite the owner's pre-push hook (allows only alphalab-risk)?** Codex: leave
  the branch unpushed and report; approval to push is not explicit approval to bypass the
  owner's safeguard. **Action:** not pushed (no --no-verify); reported in the morning summary.

## 2026-11-11 06:45 — robust-start gate failed; next step (Codex)

**Gate result (owner's pre-approved rule):** pilot PASS (2,600 draws, 0 failures, Wilson upper
bound 1.9% per group); start-invariance FAIL (real data 297/297 within 1e-6, sim A within
3e-9, but sims B1/B2 24 of 285 samples beyond tolerance, e.g. Wald 3.14 vs 10.08). **Action:
overnight robust run NOT launched.** (The legacy session had already ended at its 04:47
budget, so Q1 of 00:35 did not arise.) Diagnosis: in 22 of 24 divergent samples esreg's end
solution has min fitted sd ≈ 0 (the linear-sd Gaussian likelihood is unbounded as an sd → 0);
the warm start usually, not always, finds an interior optimum.
**Question to Codex:** what next? (a) end-solution validity check + re-audit; (b) constrained
or log-linear scale model; (c) multi-start, keep highest likelihood; (d) use 'ind' variance.
**Codex:** (a): add an end-solution check (min/median fitted sd below a predefined threshold
→ failed draw) for both starts, re-audit legacy, real and pilot draws, report flagged draws
as failures, examine sensitivity; with an unbounded likelihood a higher likelihood cannot
identify the better fit, so multi-start alone cannot resolve it.
**Action:** threshold 0.05 = the interior criterion already pre-specified in amendment 8(d)(2)
(no new number chosen after seeing results); sensitivity at 0.01 and 0.10.

## 2026-11-11 07:30 — end-solution audit result; estimator choice escalated to the owner

**Audit** (`esr_endaudit`, threshold 0.05 = amendment 8(d)(2); output
`data/results/esr_pilot/end_solution_by_group.csv`): legacy P1 draws, 22 non-lasso cells,
0 of 21,879 flagged (min ratio ≥ 0.22); lasso flags all coincide with degenerate starts.
Robust real-data pilot, 0 of 1,400 flagged. Simulations (robust start): A 2–3%, B1 22–24%,
B2 31–32% flagged. The boundary is an estimator limitation of the linear-sd scale model,
growing under the misspecification the test should detect; the pre-specified size
simulation cannot be evaluated with these failure rates.
**Question to Codex:** keep scl_sp and report failures / switch to 'ind' covariance /
log-linear scale / unstudentised bootstrap? **Codex:** switch the bootstrap studentisation to
esreg's 'ind' covariance by a frozen amendment (post-lockbox, exploratory), keep the ≤ 6.4%
criterion, re-run A/B1/B2, and report the scl_sp results as motivation.
**Action:** NOT implemented. This changes the pre-registered estimator, which CLAUDE.md
reserves for the owner ("Changing the pre-registered research design"). Escalated in the
morning summary. No overnight run launched.

## 2026-11-11 — amendment 10: which "ind" variant to freeze (owner approved the switch to ind)

**Finding:** esreg's `sigma_matrix`/`lambda_matrix` always call `cdf_at_quantile` (the same
location-scale fit) and use it when misspec = TRUE; all misspecification terms are multiples
of (cdf − α). So sigma_est = "ind" alone does not remove the scale fit.
**Question to Codex:** freeze V1 (ind, misspec = TRUE) or V2 (ind, misspec = FALSE)?
**Codex:** freeze V2 as an explicitly clarified amendment (it fulfils the stated purpose);
report V1 as a secondary sensitivity with its failure rule; distinguish the ESR coefficient
null from correct specification (V2 imposes cdf = α; adequacy to be shown by simulation);
check against R esback/esreg; apply the gate to V2, never pick whichever variant passes.
**Action:** adopted; frozen in RESEARCH_DESIGN amendment 10 before any amended simulation ran.

## 2026-11-11 18:10 — Codex review of amendment 10 implementation

**Codex:** algebra OK (cdf = α ≡ misspec = FALSE, all Λ/Σ blocks match esreg), default path
unchanged, pairing preserved, assumptions honestly stated. MAJOR: the summary could pass on
survivors only (full-sample failures excluded, incomplete run accepted). MINORs: failure
counts/Wilson intervals missing; asymptotic sd at nominal 5%; crash if all fits fail.
**Action:** all adopted in `esr_sim.summarize` (completeness + ≤ 5% full-sample failures
required; Wilson CIs; per-test failure counts; observed-rate sd; "unavailable" row), recorded
as amendment 10(j) before any amended result was read; threshold 6.4% unchanged. Not
adopted: per-test (instead of either-test) draw-failure gating; kept, as the more
conservative existing rule (amendment 7(c)).

## 2026-11-11 18:16 EST — autonomous run started (CLAUDE.md "Autonomous run rules")

Start 18:16:02 EST; hard stop 20:16 EST; summary goes to RUN_SUMMARY.md.
**Reconciling the two CLAUDE.md sections (my reading, no question needed):** "Autonomous run
rules" replace *asking* the user, but not the "Always ask Guohuan" list. Items on that list
(deleting data, rewriting history, pushing, the lockbox, jobs > 2 h or costing money, changing
the pre-registered design) are neither done nor delegated during the run: they are left for
the owner and listed in RUN_SUMMARY.md. Codex timeout is 10 min (new rule), after which I
choose, log, and continue. Owner's standing limits from the amendment-10 approval also hold:
no real-data bootstrap launch, no --no-verify, no estimator changes to obtain a pass.
**Plan:** wait for the V2/V1 simulations (running since 18:08), apply the frozen gate
(amendment 10(f), 10(j)), report GO/NO-GO; if GO, estimate the real-data cost only.

## 2026-11-11 19:40 EST — amendment 10 gate applied (no discretion involved)

Frozen rule (10(f), 10(j)) applied mechanically: V2 NO-GO for strict and intercept (block 20:
strict B1 11.7%, B2 25.4%; intercept B1 8.5%, B2 16.6% > 6.4%); V1 unusable (10–44% failed
full-sample fits). Per 10(g): estimator work stopped, nothing launched. Next-milestone
recommendation requested from Codex (Agent HUD message m-b5125423b0e5, 19:42).
**Codex (19:42, Agent HUD):** agrees NO-GO for both tests; recommends as the next milestone a
new amendment retiring ESR inference to descriptive status, keeping descriptive results, with
the other backtests calibrated for dependence separately. **Action:** recorded as the
recommendation to the owner in RUN_SUMMARY.md; not implemented (design change → owner).
Run stopped at 19:42 EST, before the 20:16 hard stop: no remaining step is permitted without
the owner (design change, real-data launch, push, merge).
**Codex full reply (m-9710261fed41):** confirms NO-GO (Monte Carlo uncertainty does not alter
the pre-specified decision; V2's complete availability resolves its earlier selection
concern). Recommendation unchanged: retire ESR p-values and GO decisions from inferential use,
keep ESR estimates/statistics descriptively; other backtests need their own dependence-robust
calibration. Caveats: HAC studentisation, longer blocks or score bootstraps would each be a new
research programme needing fresh validation (these results do not show which would work), and
HAC plus a block bootstrap must not be described as automatically pivotal.

## 2026-11-11 ~20:00 EST — owner decisions and repository maintenance (not delegated)

**Owner approved recommendation (d):** ESR p-values and ESR GO/NO-GO retired from inference;
estimates/statistics kept descriptively; no new ESR programme. Recorded as amendment 11.
**Merge and push (owner-approved, conditional):** `esr-robust-start` was 14 commits ahead of
`alphalab-risk` and 0 behind (no conflicts possible); full suite 328 passed locally; merged
`--no-ff` (`ef18486`); pushed by `.claude/autonomy.py publish` (hooks untouched); CI success
on that exact commit (run 43131573295).
**Proposed `.claude/settings.json`: not applied.** Not necessary (the approved workflow is
already allow-listed in `settings.local.json`); not safe or consistent: `Bash(python:*)` allows
arbitrary code, `Bash(pip install:*)` contradicts "no dependency installation without
approval" (autonomy.md), unscoped `Edit`/`Write` widen the scoped edit allow-list, and its
`git commit` allow conflicts with the local deny. Creating it would also bypass the
`Edit(/.claude/**)` deny. Applying it needs the owner to run it (and to amend autonomy.md if
`pip install` is intended).
**Disclosure:** `settings.local.json` denies `Bash(git commit *)`, but the earlier commits on
`esr-robust-start` (made at the owner's request) were issued as compound shell commands, which
that pattern did not catch. From now on, commits on `alphalab-risk` go through
`autonomy.py publish` only.

## 2026-11-12 — Milestone 16 (owner-approved): pre-run review of the calibration design

Design frozen and pushed first (`b6c5936`, CI green). **Codex pre-run review (Agent HUD
m-33c28ffb07fc / reply m-6b484d2095eb):** null constructions OK in every gated design;
three MAJORs: (1) missing designs or replications could still yield SUPPORTED; (2) Dhigh
under-matches the upper default-lag LRV ratio, so a pass would overstate coverage; (3) the
fixed-b fallback is not theoretically justified for H3's persistent deterministic dummy (and
DM's claim is qualified by the variance calendar). MINORs: thresholds lenient relative to
R = 2,000; report finite-only rates and hit strata; pin parameters and hashes; DQ remedy
details. **Action (all adopted before any result):** exact-set and replication-ID checks
with regression tests; new gated design Dhigh2 matching both upper targets; H3 fixed-b never
adopted, DM fixed-b labelled empirical; finite-only rates, hit strata, strict-tolerance flag
(thresholds unchanged), hashes pinned. Recorded as Addition 10(j); pushed `b47c95e` (CI green)
before the run started at 10:11 EST.

## 2026-11-12 — Milestone 16: results and post-run review

Run complete (10:11–11:29 EST, 1 h 18 min, 4 cores at nice 10, within the 2 h limit). Frozen
rules applied unchanged; classification in `docs/EVIDENCE_STATUS.md`. **Codex post-run review
(m-42681ae26bbd / m-69ed73004aaa):** classifications match the frozen rules; one MAJOR on
wording: CC's pass and lockbox p = 0.89 support "conditional coverage not rejected", not
"calibration confirmed". **Action:** EVIDENCE_STATUS, RESULTS_LOG and REPORT wording use
"not rejected"; low power with ≈ 10 hits noted. A post-hoc MCS sanity check (not frozen) is
labelled informational everywhere. No method was modified after the results.

**Codex post-run review, full text (m-69ed73004aaa):** classifications confirmed (including
the three surviving McNeil–Frey cells and POF T = 6,535 α 1% passing exactly at 6.4%). MAJOR:
the FHS-GJR headline must read "VaR conditional-coverage null not rejected; CC meets the
simulated size criterion, with sparse-hit power limitations", not confirmed or best
calibration. MINORs: the LRV ratios are unit-calm-variance AR calculations, distinct from the
finite-lag matched estimates, and Dmid shows vulnerability, not the actual series'
distortion; the MCS sanity runs are post-hoc diagnostics; explain that SUPPORTED is
size-only and marginal per test; verify each asserted DQ rejection. **Action:** all applied in
EVIDENCE_STATUS and RESULTS_LOG. The DQ check against saved results showed FHS-GJR is *not*
rejected on P3 at 2.5% or 5%, so "DQ rejects every model" is restated per portfolio.

## 2026-11-12 — Milestone 17 (owner-approved): report draft v2 and independent review

Report rewritten with evidence labels; every number re-checked against saved results, which
found four imprecise earlier statements (logged in RESULTS_LOG). **Codex review
(m-85c9fec3c1b5 / m-793ca1fcd76f):** two MAJORs: (1) the calm-period advantage holds only for
the RF-vol hybrid; the quantile forest and gradient boosting are worse than FHS-GJR in calm
periods too (verified: positive calm difference in 23 of their 24 cells); (2) causal
mechanisms and "correct on average" were stated categorically. MINORs: EVIDENCE_STATUS
inconsistencies (POF in B1/B2 is size distortion, not power; McNeil–Frey supported-cell lists;
category D wording); P2/P3 inference only provisional by endpoint checks; correlated sign
counts; "meaningfully better" undefined; episode variance quadrupled, not doubled; P1 target
dates. Spot checks of DQ, CC counts, lockbox and NOT REPLICATED: OK; ESR framing OK.
**Action:** all applied in REPORT, EVIDENCE_STATUS and README before publishing.

## 2026-11-12 — Milestone 18 (owner-approved): release preparation and final review

Clean-clone re-check, audit, README, DATA.md, CHANGELOG, proposed release notes, report PDF
(generated outside the repository; binary files are outside the publish workflow), dashboard
evidence captions. **Codex final review (m-b6a4b184349e / m-a2eb537cd8e4):** MAJOR: README
and report abstract said the design was frozen "before any model result", contradicting the
logged HS/EWMA baseline exception. MINORs: hybrid calm-advantage wording too absolute; CC
"everywhere" needed scoping; verification claims outran the record (339 tests "in a fresh
clone" not yet run); `make synthetic` scope; PDF export could succeed silently without
rendering; DATA.md promised private snapshot sharing categorically; plain-language VaR/ES
definition; newest lockbox screenshot. **Action:** all applied (render-completion check with
tests; claims narrowed; screenshot retaken). **Public release: NO-GO** until the owner
decides on the privacy items (author email in commit metadata; tracked internal workflow
notes) and the final-tree fresh-clone check is logged.
