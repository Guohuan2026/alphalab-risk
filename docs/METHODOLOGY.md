# Methodology

## 1. Intuition first

A return should answer: *if I held one share from yesterday's close to today's close,
how much did my wealth grow?* Two corporate actions break the naive `P_t / P_{t-1} - 1`:

- **Split** (2-for-1): price halves, but you now own 2 shares. Wealth unchanged.
- **Dividend**: price drops by roughly the dividend on the ex-date, but you received cash.

Ignore splits → fake −50% days. Ignore dividends → understate returns by the yield
(≈ 2–3%/yr for dividend payers, much more for some sectors), which biases value and
low-volatility factors.

## 2. Formulas

Let `P_t` be the raw close, `D_t` the dividend per share going ex on `t` (per post-split
share), and `s_t` the split ratio (new shares per old, 1 on normal days). Holding one
share from `t-1`, you own `s_t` shares at `t`, each worth `P_t` plus `D_t` cash:

```
Total return     R_t  = s_t (P_t + D_t) / P_{t-1} − 1
Price return     RX_t = s_t P_t / P_{t-1} − 1
Income           Y_t  = s_t D_t / P_{t-1}              (R_t = RX_t + Y_t exactly)
Log return       r_t  = ln(1 + R_t)                    (additive over time)
Period return    R_[a,b] = exp(Σ r_t) − 1 = Π(1 + R_t) − 1
Total-return idx TRI_t = Π_{u ≤ t} (1 + R_u),  TRI = 1 on first price date
```

Worked example (tested in `test_returns.py`): close 102 → 51.5 on a 2-for-1 split day:
`R = 2 × 51.5 / 102 − 1 = +0.98%`, not −49.5%.

**Price basis.** CRSP `PRC` is raw (`s_t` must be applied). Yahoo's `Close` with
`auto_adjust=False` is already split-adjusted, so AlphaLab treats `s_t = 1` for Yahoo.
Each provider declares its basis; applying `s_t` twice is as wrong as not applying it,
and `test_unhandled_split_is_flagged` shows validation catching that.

## 3. Why we don't use Yahoo's "Adj Close" for levels

Yahoo builds `Adj Close` *backward*: today's value equals the actual price, and each
past value is multiplied down by every later dividend/split factor. So the adjusted
level on 2018-06-01 changes when a dividend is paid in 2026 — it encodes the future.
Any signal built on adjusted *levels* (e.g. "price relative to 52-week high", price
thresholds) leaks. Returns computed from it are fine-ish; levels are not.

AlphaLab instead builds the TRI forward from 1.0 using only past returns, and keeps
Yahoo's value as `vendor_adj_close` purely for cross-checking. Small mismatches on
ex-dates are expected: Yahoo's factor is `(1 − D/P_{t−1})`, giving
`P_t / (P_{t−1} − D)` versus our `(P_t + D) / P_{t−1}`. The gap is second-order
(≈ (D/P)·R) and stays within the default tolerance of 0.2% for typical dividends.

## 4. Point-in-time rules (look-ahead bias)

A value stamped at date `t` may only use information available at the close of `t`.

| Rule | Implementation |
|---|---|
| Signals use the past only | `trailing_returns(window, skip)` — rolling window ending at `t − skip` |
| Targets look forward explicitly | `forward_returns(h)` = return over `t+1 … t+h`, stamped at `t`; a *label*, never an input |
| No backward-adjusted levels | TRI built forward; vendor adj close only for checks |
| No incomplete periods | `compound_returns(drop_incomplete_last=True)` drops a month still in progress |
| No moving data | immutable, hashed snapshots; builds record the snapshot they used |
| Verified, not assumed | `check_point_in_time`: `f(data[:d])[d]` must equal `f(data)[d]` |

The truncation test is itself tested against known leaks (centred windows, full-sample
z-scores, backward-adjusted levels, `shift(-1)`). During development it caught a real
bug in AlphaLab's own TRI code: the base value was set using whether the *next* row had
a return. The fix anchors the base on price availability. This is why the test exists.

**Momentum example** (preview of Milestone 3): with monthly data, signal at month-end
`t` is `trailing_returns(window=11, skip=1)` = cumulative return over months `t−11 … t−1`
(skipping month `t` to avoid short-term reversal); its target is `forward_returns(1)` =
month `t+1`. Signal and target never overlap.

**What the truncation test cannot see:** leakage baked into the data itself —
survivorship (today's tickers), restated fundamentals, or using a filing before its
public release date. Those need point-in-time *data*, which is what CRSP/Compustat
point-in-time datasets provide.

## 5. Monthly aggregation details
- Month label = calendar month-end (`"ME"`).
- `min_obs` (default 15) blanks months with too few trading days (e.g. listing month).
- Missing days inside a month are skipped, i.e. treated as 0% — validation warns about
  gaps larger than `max_gap_sessions` (default 5) trading sessions so you can investigate.
- A month is "complete" if the data reach its last **trading session**, not its last
  weekday. March 2018 ended on Good Friday (closed), so data ending Thu 29 March is a
  complete month.

## 6. Trading calendar

"Business day" (Mon–Fri) and "trading day" differ on roughly 9 NYSE holidays a year,
plus rare unscheduled closures. AlphaLab uses `exchange_calendars` (config key
`calendar`, default `XNYS`) wherever it needs to count or enumerate days:

| Where | What changes |
|---|---|
| `validate_prices` — `date_gap` | gap = sessions between observations; Thanksgiving week is not "missing data" |
| `validate_prices` — `non_session_date` | warns on a price row dated on a closed day (vendor glitch, timezone shift) |
| `compound_returns(drop_incomplete_last=True)` | compares data end with the period's last session |
| `SyntheticProvider` | generates prices only on sessions, so offline data looks like real data |

Why this is not look-ahead: a calendar records when the market was *open*, not prices.
Scheduled holidays are published years ahead. Unscheduled closures are only known
afterwards, but they can only change bookkeeping (a gap count, whether a month is
finished), never a signal value.

## 7. Risk features (AlphaLab-Risk)

**Information set.** A feature stamped at session *t* may use anything *published by the end
of day t*: after the close, before the next open. The risk forecast it feeds is for the loss
on *t+1*. The label `next_day_loss` (L_{t+1} = −r_{t+1}) lives in `features/targets.py`. It
deliberately fails the truncation test, and a test checks that it does.

| Family | Columns | Formula / rule | Why |
|---|---|---|---|
| Returns | `ret`, `abs_ret`, `rv_1` | r_t, \|r_t\|, r_t² | Most recent shock |
| HAR realized variance | `rv_5`, `rv_22`, `rv_66` | mean of r² over 5/22/66 sessions (full window required) | Volatility is persistent at week/month/quarter horizons (Corsi 2009) |
| Range variance | `pk_1`, `gk_1`, `gk_5`, `gk_22` | Parkinson (ln H/L)²/(4 ln 2); Garman–Klass ½(ln H/L)² − (2 ln 2 − 1)(ln C/O)² | The intraday range is a more efficient variance proxy than r²; split-adjusted OHLC ratios are unaffected by splits |
| Drawdown | `dd_252` | wealth / trailing-252 max − 1, with wealth = Π(1 + r) built forward | Stress state; never uses vendor backward-adjusted levels |
| VIX | `vix`, `vix_chg_5` | same-day close; 5-session log change | Option-implied (forward-looking) volatility |
| Credit / term spreads | `baa10y`, `t10y3m`, `*_chg_22` | FRED, **lagged 1 session**; 22-session difference (spreads can be negative, so no logs) | Funding and recession stress |

**Alignment (`align_asof`).** Series dated on their own calendars (FRED observes on
bond-market days, VIX on CBOE days) are put onto the portfolio's trading sessions by an
*as-of* join. The value at session *t* is the latest observation dated on or before the
session `lag` trading days earlier. Example: a FRED value observed Thursday is published
Friday morning and so is first used on Friday; with Friday a holiday (Good Friday), it is
first used the following Monday. A value more than 5 sessions old becomes NaN, so a
discontinued series cannot pose as a calm market.

**Snapshots.** Feature series are stored like prices (immutable, hashed), in a long
`date, security_id, value` format under `data/raw/features/`. Every build records which
feature snapshot it used.

## 8. Simulated ground truth (GJR-GARCH-t)

`risk/simulate.py` generates returns from a GJR-GARCH(1,1) with unit-variance Student-t
innovations. Because σ_t is known at t−1, the *true* VaR and ES of each day are known in
closed form:

```
VaR_a,t = −μ + σ_t · q_a,   q_a = −c·t_a
ES_a,t  = −μ + σ_t · s_a,   s_a =  c · f_ν(t_a)/a · (ν + t_a²)/(ν − 1),   c = √((ν−2)/ν)
```

The formulas are checked against 4 million Monte Carlo draws, and the simulated true VaR
has the right exceedance rate. Later, every risk model is first tested on this data. A
correct GARCH should recover σ_t; coverage tests should accept the true VaR and reject a
slow model during simulated volatility bursts. A useful fact for interviews: under a normal
distribution ES/VaR at 1% is 1.15, but with ν = 6 it exceeds 1.2. So a model that is
calibrated for VaR under normality still understates ES when tails are fat.

## 9. Forecast engine and VaR coverage tests

**Model contract** (`risk/models/base.py`). `fit(train)` estimates parameters on refit dates
(every 21 sessions); `predict(history)` returns next-day VaR, ES and σ every day from all
returns known at the origin. The same split is used in production: parameters are reviewed
monthly, filters update daily. VaR/ES are positive loss numbers; α is the *tail
probability* (0.01 = 99% VaR).

**Walk-forward engine** (`risk/engine.py`). For each origin *t*: refit if due, then
forecast the loss on *t+1*. Refit dates count from the first eligible origin, so they do not
depend on later data. **Engine truncation test:** running on `returns[:d]` must give the
identical forecast at origin *d* (tested for every model). **Lockbox:** origins whose target
falls on or after `lockbox_start` (2023-01-01) are refused unless `--include-lockbox` is
passed. Lockbox results are written and tested separately, never blended with the
development sample.

**Baselines.**
- *Historical simulation* (HS-250, HS-500): empirical quantile and tail mean of the last N
  losses. It reacts slowly, and a crash day drops out abruptly after N days.
- *EWMA-normal* (RiskMetrics, λ = 0.94): σ²_{t+1} = λσ²_t + (1−λ)r_t², VaR = σz_α,
  ES = σφ(z_α)/α. It reacts fast, but its tails are normal (thin).

**Coverage tests** (`backtest/coverage.py`). Hit = L_{t+1} > VaR_t. Under a correct model
hits are i.i.d. Bernoulli(α).
- *Kupiec POF*: is the hit frequency α? LR ~ χ²(1).
- *Christoffersen IND*: P(hit | hit yesterday) = P(hit | no hit yesterday)? LR ~ χ²(1).
- *Conditional coverage*: CC = POF + IND ~ χ²(2).
- *Basel traffic light* (99% VaR, 250-day windows): green 0–4, yellow 5–9, red 10+.

How the tests themselves were validated:
1. **Hand-computed values:** 5 hits in 250 days at 1% gives LR = 1.956, p = 0.162.
2. **Size:** with a correct model, CC rejects about 5% of the time at the 5% level. So a
   true model *will* occasionally be rejected. In the simulations, the true VaR failed once
   in six checks (p = 0.011); that is expected, not a bug.
3. **Power:** clustered hits at the correct frequency pass Kupiec but fail IND.
4. **Simulated truth:** on GJR-GARCH-t data, the true VaR passes; EWMA-normal under-covers
   at 1% (fat tails) but not at 5%; HS's hits cluster.

What these tests cannot do: they say whether a model is *acceptable*, not which model is
*better*. That needs loss functions (tick loss, FZ0) with Diebold–Mariano tests and a
Model Confidence Set (week 2). They also ignore ES; ES backtests follow in week 2.

## 10. GARCH-family models

`risk/models/garch.py`; config types `garch`, `gjr`, `fhs`.

```
r_t = μ + σ_t z_t,  z_t ~ unit-variance t(ν)
σ²_t = ω + (α + γ·1[ε_{t−1}<0]) ε²_{t−1} + β σ²_{t−1}        (γ = 0: plain GARCH)
```

- **Estimation:** maximum likelihood with `arch` on percent returns, converted to decimals.
  Monthly refits, warm-started from the previous estimates.
- **Daily filter:** the same function that generates the simulated truth
  (`gjr_variance_filter`), so the timing is identical. It is updated one step per day
  between refits, and a test checks this equals a full recompute.
- **Tails:** GARCH-t and GJR-t use Student-t quantiles. **FHS** multiplies σ_{t+1} by the
  empirical VaR/ES of the standardised residuals ẑ = (r − μ)/σ over the training window.
  The first 50 residuals are dropped because the filter's start value still matters there.
- **Failed refits:** non-convergence, persistence ≥ 1, ν ≤ 2 or ω ≤ 0 means the previous
  parameters are kept and the failure counted (RESEARCH_DESIGN Deviations, clarification 1).
  Parameters at every refit are saved to `fit_params.parquet`.

**Validation on simulated GJR-GARCH-t data:**
- Parameters are recovered: γ within 0.04, persistence within 0.015, ν within 1.5.
- GJR's σ forecast is within about 2% (median) of the true σ. Plain GARCH, misspecified
  here, has about 4× the error and still passes coverage, which is a reminder of how little
  power coverage tests have.
- Coverage is compared with the true VaR *on the same simulated path*, never against a fixed
  p-value. On one path the true 5% VaR itself failed CC (p = 0.026): a Type I error that a
  correct model cannot be expected to beat.

## 11. Comparing models: losses, ES backtests, DM and MCS

Coverage tests ask whether a model is *acceptable*; these tools ask which is *better*.

- **Scoring rules** (`backtest/losses.py`). A loss is *strictly consistent* if its expected
  value is minimised only by the true forecast. VaR (a quantile) has the **tick loss**
  `(α − 1{y ≤ v})(y − v)`. ES alone has no such loss: it is not elicitable (Gneiting 2011).
  The *pair* (VaR, ES) is (Fissler & Ziegel 2016). **FZ0** (Patton, Ziegel & Chen 2019):
  `−1{y ≤ v}(v − y)/(α e) + v/e + ln(−e) − 1`, written in return space (v, e < 0). Its
  differences do not depend on the units of returns. Tests confirm that misstating VaR
  and/or ES by ±10% raises the mean FZ0.
- **McNeil–Frey** (`backtest/es.py`). On VaR-hit days, ξ = L − ES should average zero;
  H1: ES too small. Bootstrap p-value from centred residuals; a standardised version
  divides by σ. Fewer than 5 hits gives NaN, not a p-value. `severity_ratio` = realised
  mean tail loss / forecast ES.
- **Diebold–Mariano** (`backtest/compare.py`). d = loss_A − loss_B; Newey–West HAC
  variance (Bartlett, lag ⌊4(n/100)^{2/9}⌋). *Measured caveat:* with AR(1) differentials
  (φ = 0.5) the naive t-test falsely rejects ~30%, HAC ~10%. So HAC removes most of the
  distortion and DM p-values are mildly liberal under strong autocorrelation.
- **Model Confidence Set** (Hansen, Lunde & Nason 2011, `arch.bootstrap.MCS`, range
  statistic, block bootstrap, 90%): the set of models that cannot be distinguished from the
  best. This is the **headline** comparison, because it controls for comparing many models
  at once.

Validation on simulated GJR-GARCH-t data (true VaR/ES known): FZ0 ranks the truth first,
then GJR, FHS, GARCH, EWMA, HS. The MCS keeps {truth, GJR, FHS} and drops HS and EWMA. DM
finds GJR better than EWMA (p = 0.015). McNeil–Frey rejects EWMA's normal ES
(severity 1.10) and accepts the truth and GJR.

## 12. Feature-based models and the linear benchmarks

**Feature contract.** Feature models receive the feature rows for the same dates as the
returns. Training pairs are (x_s, r_{s+1}), the last being (x_{t−1}, r_t); the forecast
for t+1 uses x_t. Tests check this with a model that records what it is given, and the
engine truncation test is repeated with features.

**HAR-FHS** (`har`). OLS of r²_{s+1} on [1, RV1_s, RV5_s, RV22_s] (Corsi 2009); variance
floored at 5% of the training variance; FHS tails from residuals z = r/σ̂. *Known bias:* the
regressors are noisy proxies of σ², so OLS coefficients are attenuated (errors in
variables). On simulated GJR data they sum to about 0.71 against a true persistence of 0.98.
Intraday realized variance would reduce this.

**Linear quantile regression.** `linqr`: q_τ(r_{s+1}|x_s) = x_s'β_τ for τ ∈ {α} ∪
{α(k−½)/K}, K = 5; ES = −mean of the sub-α quantiles; quantiles sorted across levels to
prevent crossing; features clipped to training [0.5%, 99.5%] quantiles and standardised
with training moments. `linqr_l1` (the H2b control): the same model with an L1 penalty,
λ tuned at each yearly refit on the last 20% of the training pairs (1-pair embargo;
statistics for tuning from the earlier part only), exact LP solver (HiGHS).

**Why plain linear QR failed on SPY** (a finding, RESULTS_LOG): collinear features get
large offsetting coefficients, and on stress days (e.g. after 1997-10-27, Lehman
2008-09-15) the offset breaks and the 5% quantile turns positive. Clipping alone was not
enough; L1 shrinkage plus the floor were needed (RESEARCH_DESIGN Deviation 2).

**Safety floor** (engine, every model): VaR ≥ 1 bp and ES ≥ VaR. Invalid forecasts are
flagged and scored, never dropped, so all models stay on the same dates.

## 13. Machine-learning models (`risk/models/ml.py`, specified in Deviation 3)

| Model | What it learns | VaR / ES |
|---|---|---|
| `qrf`: quantile regression forest (Meinshausen 2006) | a forest whose leaves keep the training targets, i.e. a full conditional distribution | quantiles at α and α(k−½)/K; ES = mean of the sub-α quantiles |
| `gbq`: gradient boosting, pinball loss (scikit-learn `HistGradientBoostingRegressor`) | one boosted model per quantile level | same as QRF; quantiles sorted to stop crossing |
| `mlvol_fhs`: RF volatility + FHS | a random forest for \|r_{s+1}\|, a volatility proxy | σ̃_t × empirical VaR/ES of **out-of-bag** residuals r/σ̃ |

- **Identical tuning** (one shared class, also used by the lasso control): last 20% of the
  training pairs, 1-pair embargo, mean tick loss over the VaR levels, ties → first config,
  refit + re-tune every 252 sessions, `random_state = 0`.
- **Why out-of-bag residuals:** a forest's in-sample fit nearly memorises |r|, so in-sample
  residuals are too small and VaR would be too low. A test shows the out-of-bag tail is
  larger.
- **Block prediction:** between refits these models depend only on x_t, so the engine
  predicts a refit block in one call. A test checks the result is identical to day-by-day
  prediction.
- **Determinism (found by a test):** scikit-learn random forests predicting with
  `n_jobs = -1` sum tree outputs in thread order, and floating-point addition is not
  associative, so repeated runs differed in the last bits and flipped a tuning tie. Fitting
  stays parallel; prediction is single-threaded. Every ML model now has a bit-identical
  refit test.
- **Validation on simulated GJR-GARCH-t data:** all three land between the truth and HS on
  FZ0 but do not beat the true parametric model, a check that the evaluation does not
  automatically reward flexibility. The quantile forest had 2 floored forecasts even
  there: a leaf whose training returns are mostly gains can give a positive 5% quantile.

## 14. Stress analysis (H3) (`stress/analysis.py`; rules: Clarification 4)

- **Regimes:** 8 ex-ante episodes, assigned by *target date*; calm = no episode. VIX regime
  by VIX at the *origin* (> 30), known when the forecast is made. Conditioning on the
  outcome would bias the comparison.
- **Per regime and model:** n, expected and observed hits, Kupiec p, mean FZ0, ES severity.
  The expected-hits column is the power warning (e.g. 6.9 expected at 1% when VIX > 30).
- **Tests vs the benchmark (FHS-GJR):** DM within regimes; HAC regression d = a + b·D + e,
  where *a* is the calm-period difference and *b* the extra difference in stress (H3: b ≠ 0).
  The HAC regression matches statsmodels exactly (test). Regimes with < 30 days (or < 30
  outside) get no DM, regression or MCS: a short sample can contain no episode day, and the
  dummy would be degenerate (a bug found by a test).
- **Power lesson from simulated data:** HS broke down in the high-volatility regime (hit
  rate 2.6× the target), yet its FZ0 stress shift was not significant (p ≈ 0.2), because
  FZ0 is very noisy on volatile days. A non-significant b is therefore *not* evidence that
  stress makes no difference; read it together with the hit rates.
- **Event study:** the 1% largest losses; VaR path k = −5..20 relative to k = −5; the
  committed summary is "days to 1.5×". Because large losses usually strike when VaR is
  already elevated, the k = 1 and k = 5 ratios are reported too, as extra description.

## 15. Interpretability (H4) (`explain/importance.py`; rules: Clarification 5)

- **Permutation importance:** at each yearly refit, the chosen configuration is refitted on
  the tuning part and each feature is shuffled on the validation part (5 repeats). The
  rise in mean tick loss is that feature's importance, measured out of sample for the model
  being scored. It answers: "how much worse do forecasts get without this information?"
- **TreeSHAP** (Lundberg & Lee): an exact decomposition of a tree model's prediction into
  a base value plus one contribution per feature, with base + contributions = prediction
  (tested to 1e-9). For the RF-volatility hybrid, VaR = q·σ̃ is linear in the forest's
  prediction, so VaR contributions = q × SHAP. For boosting, SHAP explains the *raw* 1%
  quantile model; the reported VaR differs on days where the per-level models cross and
  are sorted. The explain run reports that share. TreeSHAP cannot explain the quantile
  forest's quantiles (it explains a forest's mean), so QRF gets permutation importance only.
- **Same models as the forecasts:** the explain pass refits on the engine's schedule;
  refits are bit-identical, and `explain_check.csv` compares the SHAP-implied VaR with the
  saved forecasts.
- **Criteria (fixed before results):**
  - C1: the top-3 features by mean permutation importance are volatility or VIX features.
  - C2: every volatility/VIX feature's SHAP contribution rises with its value.
  - C3: median Spearman rank correlation of importances between consecutive refits ≥ 0.6.
- **Caveat:** importance is not causation. Correlated features (rv_5, rv_22, gk_22) share
  credit: permuting one while its twins remain understates the family's importance. That is
  why C1 is stated at the family level.

## 16. Reproducibility

- **Data:** immutable, hash-verified raw snapshots; every build and forecast run records the
  snapshot it used.
- **Code:** git history on GitHub (`Guohuan2026/alphalab-risk`), with design, clarifications
  and lockbox protocol committed before the results they govern.
- **Environment:** `requirements-lock.txt` pins the exact package versions behind every
  logged result; `scripts/setup_mac.sh` installs them as constraints. *Why it is needed:* a
  clean clone on 2026-11-07 installed `quantile-forest` 1.5.0 instead of 1.4.2, and its
  quantile forecasts differed materially from the logged ones (LIMITATIONS #14). Unpinned
  "latest" installs are not reproducible research.
- **Determinism:** fixed seeds; single-threaded forest prediction; bit-identical refit
  tests; dev-period forecasts re-checked bit-for-bit in the lockbox runs.

## 17. Engle–Manganelli Dynamic Quantile test (`backtest/coverage.py: dq_test`)

Regress the demeaned hit Hit_t = 1[L_t > VaR_t] − α on a constant, its own 4 lags and VaR_t.
Under a correct VaR all coefficients are zero; DQ = β̂'X'Xβ̂ / (α(1 − α)) ~ χ²(6). It nests
Kupiec (constant) and Christoffersen (lag 1) and adds the VaR regressor, which catches
breaches that depend on the forecast's level. Example from the tests: breaches at 8% when
VaR is low and 2% when it is high average 5%, show no clustering, and pass both classical
tests, but DQ rejects them with p < 1e-6. Output: `backtest_dq.csv`. Development samples only
(added after the lockbox was opened; RESULTS_LOG).

## 18. Acerbi–Szekely Z2 ES test (`backtest/es.py: acerbi_szekely_z2`; Clarification 7)

Z2 = 1 − (1/(T·α)) Σ_t L_t·1[L_t > VaR_t] / ES_t. If VaR and ES are right, E[Z2] = 0; a
negative value means breaches are too frequent, too severe, or both. Unlike McNeil–Frey,
which looks only at the model's own breach days, Z2 scores frequency and severity together.

Its null distribution depends on the shape of the tail, and the models' full predictive
distributions are not stored. The p-value is therefore Monte Carlo (10,000 draws) under a
*shape-matched* Student-t: ν is chosen so that the t's ES/VaR ratio equals the median of the
model's own ES_t/VaR_t (EWMA's normal tails give ν ≈ ∞). Only breaches matter, so a draw is
a Binomial(T, α) number of breaches with t tail quantiles; scale cancels in L/ES.

Validated in the tests: the ν inversion recovers 3, 6 and 20 exactly; under H0 with a
time-varying scale the test rejects about 5% of the time; with VaR right and ES 20% too
small (5,000 days) its power is about 0.83, against 1.00 for McNeil–Frey, which is sharper
for pure ES errors. Output: `backtest_z2.csv`, with a `preregistered` flag for the models
named in RESEARCH_DESIGN §7. Development samples only (LIMITATIONS #15).
