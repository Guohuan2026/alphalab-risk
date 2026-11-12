# Changelog

Dates are those of the commits. Research results are logged in full, in the order they were
seen, in `RESULTS_LOG.md`; this file summarises versions.

## v1.0.0 (proposed; not yet tagged)

Portfolio release candidate (Milestone 18).

- **Report v2** (`REPORT.md`, 2026-11-12): every claim labelled by evidence strength;
  significance claims based on Diebold–Mariano, MCS and the stress regression withdrawn;
  figures; PDF export (`python -m alphalab.report_pdf`).
- **Backtest calibration study** (Milestone 16, RESEARCH_DESIGN Addition 10): frozen size
  simulations; DM, MCS, stress regression, Z2 and most McNeil–Frey cells unsupported;
  Christoffersen CC supported everywhere. Classification in `EVIDENCE_STATUS.md`.
- **ES regression backtest retired** (Addition 9, amendments 1–11): esreg port validated, but
  the bootstrap failed its size check under dependence; estimates kept as diagnostics.
- **Dashboard**: evidence-status notes next to every p-value and on the lockbox verdict.
- **Reproducibility**: fresh-clone check repeated; a test-isolation defect fixed (the
  dashboard test left `__main__` replaced, which made later process-pool tests fail in a clone
  without saved results); `DATA.md` documents data sources, terms and how to obtain them.
- **README** rewritten for external readers.

## v0.9 (2026-11-07, untagged): lockbox opened

- Lockbox protocol pushed (`dc8af86`), lockbox run once for P1, P2 and P3 (`9925df7`):
  NOT REPLICATED under the pre-stated rule.
- Dependency lock (`requirements-lock.txt`) after a clean-clone check found `quantile-forest`
  1.5.0 changed results.
- Post-lockbox, development-only additions: DQ test, Acerbi–Székely Z2 (Clarification 7),
  ten-day horizon table (Clarification 8), Streamlit dashboard.

## v0.5 (2026-11-06, untagged): development study

- Design frozen at `prereg-v1` (`46c21ea`); GARCH family and FHS; ML models (Deviation 3);
  stress analysis (Clarification 4); interpretability criteria (Clarification 5); robustness
  on P2, P3 and a rolling window (Clarification 6).

## v0.1-equity-research (tag)

- The original AlphaLab equity-research platform (data pipeline, returns, calendar,
  look-ahead checks), reused unchanged by AlphaLab-Risk.
