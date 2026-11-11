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
