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
  gaps larger than 5 business days so you can investigate.
