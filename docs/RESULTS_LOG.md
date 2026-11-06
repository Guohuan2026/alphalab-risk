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
