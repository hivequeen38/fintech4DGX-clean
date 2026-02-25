# Feature Backlog

Features considered but not yet added to selected_columns.
Prioritized by expected signal value for a 1–3 week horizon.

---

## Top 5 Priority Picks (highest expected SHAP, implement first)

| # | Feature(s) | Category | Status |
|---|---|---|---|
| 1 | `earnings_is_bmo`, `earnings_is_amc` | Earnings timing | **IMPLEMENTED** 2026-02-25 — computed & stored in TMP.csv; NOT in selected_columns yet (constant for all-AMC universe; activate when BMO stock added) |
| 2 | `ret_5d_rel_SPY`, `ret_10d_rel_SPY`, `ret_5d_rel_SMH`, `ret_10d_rel_SMH` | Cross-sectional alpha | **IMPLEMENTED** 2026-02-25 |
| 3+4 | `iv_30d`, `iv_skew_30d`, `iv_term_ratio` | Options IV | **IMPLEMENTED** 2026-02-25 — extracted from existing AV HISTORICAL_OPTIONS fetch in same API call; backfill auto-runs on next training pass |
| 5 | `short_interest_pct_float` (or `days_to_cover`) | Short interest | Backlog — low cadence limits daily utility |

**Note:** `is_earnings_day`, `is_post_earnings_day_1/2/3` were removed from top-5 — largely redundant
with `dte`/`dse` already in all profiles. `earnings_is_bmo/amc` is the genuinely new piece.

---

## EPS Analyst Estimate features (AV EARNINGS_ESTIMATES)

**Already implemented** (in `fetch_eps_estimates.py`, all available in TMP CSVs):
All columns listed here are already computed and forward-filled into the daily
DataFrame by `fetchBulkData.py`. Adding them to `selected_columns` is all that
is needed to activate them for training.

**Active in all profiles for all tickers** (as of 2026-02-24):
- `eps_est_avg` — raw AV consensus average EPS estimate (upcoming quarter, raw AV units)

**Commented out (BACKLOG — leakage risk):**
Forward-fill of end-of-quarter revision stats contaminates earlier months in each quarter.
- `eps_rev_30_pct` — % revision in consensus over 30 days
- `eps_rev_7_pct` — % revision in consensus over 7 days
- `eps_breadth_ratio_30` — net analyst conviction, 30-day window
- `eps_dispersion` — (high−low)/|avg|, analyst disagreement

### Tier 2 — Add after resolving leakage or validating on reference_no_shuffle

| Feature | Formula / source | Notes |
|---|---|---|
| `eps_rev_accel` | `eps_rev_7_pct − eps_rev_30_pct` | Is revision momentum accelerating? Compute in `fetch_eps_estimates._compute_tier1_features` |
| `eps_breadth_ratio_7` | `(up_7 − down_7) / (up_7 + down_7 + 1)` | Already computed, not yet in selected_columns |
| `log_analyst_count` | `log1p(analyst_count)` | Coverage proxy (liquidity / attention). Already computed. |
| `rev_rev_30_pct` | Same formula for revenue estimate avg | Needs `rev_est_avg_30d` field — AV doesn't return revenue revision history, only current avg. Skip for now. |

### Tier 3 — Cross-features (after Tier 2 validated)

| Feature | Formula | Notes |
|---|---|---|
| `eps_revision_signal` | `eps_rev_30_pct / (eps_dispersion + 0.01)` | Revision scaled by uncertainty — high signal when both large & confident |
| `eps_rev_breadth_cross` | `eps_rev_30_pct × eps_breadth_ratio_30` | Magnitude × analyst conviction — needs compute step |
| `eps_rev_x_dte` | `eps_rev_30_pct × earn_in_20` | Estimate revision gated by earnings proximity; pre/post earnings drift interaction |

---

## 1. Earnings Event Structure

DTE/DSE (days-to/since earnings) are already implemented. These additional flags are the
next highest-impact block — a large fraction of 1–3 week moves are earnings-adjacent.

### 1a. Day-of / post-earnings flags (derive from existing report dates — Low difficulty)

| Feature | Formula / source | Notes |
|---|---|---|
| `is_earnings_day` | 1 if date == report_date, else 0 | Binary flag; derive from `next_report_date` / `last_report_date` already in TMP |
| `is_post_earnings_day_1` | 1 if dse == 1 | Immediate reaction day |
| `is_post_earnings_day_2` | 1 if dse == 2 | — |
| `is_post_earnings_day_3` | 1 if dse == 3 | 3-day post-earnings drift window |

### 1b. Earnings time-of-day (BMO / AMC) — PLANNED

**Data sources researched 2026-02-25:**

| Source | Field | Values | Historical? | Notes |
|---|---|---|---|---|
| **AV EARNINGS** (already fetched) | `reportTime` in `quarterlyEarnings` | `"pre-market"` / `"post-market"` | **Yes — full history** | Best source. Already returned by the EARNINGS call in `fetchBulkData.py`. |
| AV EARNINGS_CALENDAR (already fetched) | `timeOfTheDay` column (CSV) | `"pre-market"` / `"post-market"` / blank | **Upcoming only** | Use for the next upcoming report; more reliable than EARNINGS placeholder for not-yet-reported entries. |
| yfinance `earnings_dates` | index timezone-aware timestamp `.hour` | hour < 12 = BMO; hour >= 16 = AMC; hour == 9 = unconfirmed | Yes (≤100 rows) | No API key. Encodes timing in the DatetimeTZDtype index hour. |
| Nasdaq.com API (unofficial) | `time` field in JSON | `"time-pre-market"` / `"time-after-hours"` | **No** — past dates return `"time-not-supplied"` | Upcoming only, undocumented, can break. |

**Critical caveat:** AV EARNINGS `reportTime` for the most recent not-yet-reported quarter (the "upcoming" placeholder) can be wrong. Verified: NVDA upcoming 2026-02-25 returned `"pre-market"` but always reports `"post-market"`. For upcoming reports, use `AV EARNINGS_CALENDAR timeOfTheDay` instead.

**Implementation plan:**
1. In `fetchBulkData.py`, extract `reportTime` for each quarter from the already-fetched `income_df` / EARNINGS response
2. Build a `{report_date → bmo_flag}` mapping (alongside existing `fiscal_to_report_map`)
3. For the next upcoming report: cross-check with EARNINGS_CALENDAR `timeOfTheDay` (already fetched for the report date)
4. Forward-fill `earnings_is_bmo` and `earnings_is_amc` as binary features across daily rows (using same window convention as DTE/DSE: active from prior report date to next)
5. Add to all profiles in all param files

**Why this matters for leakage analysis:**
- BMO report day → open price already reflects the reaction; `report_date` row is post-reaction
- AMC report day → report hasn't happened during trading hours; `report_date` row is pre-reaction
- Combined with `dte=0`, the model can learn different behavior for BMO vs AMC earnings-day rows

### 1c. Earnings surprise history (rolling window stats)

| Feature | Formula / source | Notes |
|---|---|---|
| `avg_surprise_4q` | mean of last 4 quarters' `surprisePercentage` | Trend of beat/miss; AV EARNINGS already fetched |
| `std_surprise_4q` | stdev of last 4 quarters' `surprisePercentage` | Consistency of surprise magnitude |
| `beat_rate_8q` | fraction of last 8 quarters with positive surprise | Analyst calibration signal |

---

## 2. Options-Implied Features — Implementation Plan

Often dominate SHAP at 1–3 week horizons because they are forward-looking and update
continuously. Currently we only have coarse put/call volume ratios.

### Data Sourcing — The Core Problem

Individual stock implied volatility requires a paid historical options data source.
Free sources (yfinance, AV) provide only the **current** options chain — no historical snapshots.
Without historical IV you cannot backfill the training data.

**Source evaluation (as of 2026-02-25):**

| Source | Historical IV? | Cost | Notes |
|---|---|---|---|
| yfinance `.option_chain()` | No — current chain only | Free | Can start collecting going forward; no backfill |
| CBOE DataShop | Yes | $500+/month | Authoritative, expensive |
| Polygon.io Options Plan | Yes (since ~2020) | ~$199/month | Good coverage, reliable |
| ORATS | Yes | $150/month | Options-focused, best quality/price for retail |
| Intrinio options | Yes | ~$500/month | Full chain history |
| Barchart OnDemand | Partial | $100+/month | Less reliable history |
| **VIXCLS (FRED)** | **Yes — already fetched** | **Free** | SPY 30D IV; not individual stock IV, but correlated |

### Phase 1 — Proxies from existing data (no new data source needed)

Use VIXCLS (already in `robust_features`) as a market-IV proxy and compute stock-level vol
from price history. These are approximations but cost-free to implement now.

| Feature | Formula / source | Notes |
|---|---|---|
| `rv_10d` | rolling 10-day std of daily returns × √252 | Realized vol; already calculable from price history |
| `rv_20d` | rolling 20-day version | — |
| `vix_rv_ratio` | `VIXCLS / rv_20d` | Market IV premium over stock realized vol; rough signal |
| `rv_term_ratio` | `rv_5d / rv_20d` | Short- vs medium-term realized vol acceleration |

These are poor substitutes for actual stock IV but are zero-cost and can be activated immediately.

### Phase 2 — Forward-only IV collection pipeline (yfinance, ~6 month build-up)

Build a daily options snapshot fetcher that runs alongside the main pipeline:
1. After market close each trading day, fetch the options chain for each symbol
2. Extract ATM IV for the two nearest expiries (≤7d and ≤35d out)
3. Append to a per-symbol options history CSV (e.g., `NVDA_iv_history.csv`)
4. After 6+ months of daily collection, activate as training features

Implementation notes:
- yfinance `Ticker.option_chain(expiry)` gives `impliedVolatility` per contract
- ATM IV = IV of the call/put closest to current stock price at each expiry
- Run at 4:15 PM ET (after market close, after AV daily update)

### Phase 3 — Paid source (if Phase 2 data shows signal)

If Phase 2's IV features show meaningful SHAP uplift after 6 months, evaluate ORATS (~$150/month)
for historical backfill. This would immediately give full training coverage.

**Priority order for options features (once data exists):**
1. `iv_earn_move` — straddle price / stock price at front expiry; pairs with DTE; directly predicts earnings reaction size
2. `iv_term_7_30` — `IV_7D − IV_30D`; backwardation signals near-term event (usually earnings)
3. `iv_30d` — standalone ATM IV level; less directional than term structure but good baseline
4. `rv_iv_ratio` — realized/implied spread; vol risk premium signal

### 2a. Implied volatility levels (Medium difficulty — needs options data source)

| Feature | Formula / source | Notes |
|---|---|---|
| `iv_30d` | ATM 30-day implied volatility | Primary IV signal; CBOE / yfinance options chain |
| `iv_7d` | ATM 7-day (nearest expiry) IV | Shorter-horizon fear gauge |
| `iv_90d` | ATM 90-day IV | Baseline long-run expectation |

### 2b. IV term structure & skew

| Feature | Formula | Notes |
|---|---|---|
| `iv_term_7_30` | `IV_7D − IV_30D` | Contango/backwardation; positive = near-term fear |
| `iv_term_30_90` | `IV_30D − IV_90D` | Longer-term structure slope |
| `iv_skew` | 25-delta put IV − 25-delta call IV | Tail risk expectation; put vs call demand |
| `iv_earn_move` | front-expiry straddle price / stock price | Implied move into earnings; high signal pre-earnings |

### 2c. Realized vs implied

| Feature | Formula | Notes |
|---|---|---|
| `rv_iv_ratio` | `RV_10D / IV_30D` | Vol risk premium; <1 means IV overpriced vs realized |
| `iv_rv_diff` | `IV_30D − RV_10D` | Alternative: level difference |

### 2d. Open interest features

| Feature | Formula / source | Notes |
|---|---|---|
| `oi_put_call_ratio` | total put OI / total call OI | OI ratios often more informative than volume ratios |
| `oi_gamma_proxy` | sum of (call OI − put OI) × delta × 100 at each strike | Dealer gamma exposure estimate; charm/vanna proxies require full chain |

---

## 3. Cross-Sectional / Peer & Sector Returns

Models learn "stock moving with/against its group" better with explicit relative returns
than raw index levels. Sector ETFs for semiconductors: SMH, SOXX.

### 3a. Relative returns vs sector/market — **IMPLEMENTED 2026-02-25**

Computed in `trendAnalysisFromTodayNew.py` §3 (Idiosyncratic return block).
Active in selected_columns of all profiles for all 5 tickers.

| Feature | Formula | Stocks active |
|---|---|---|
| `ret_5d_rel_SPY` | `price_change_5 − SPY_close.pct_change(5)` | ALL (NVDA, CRDO, PLTR, APP, INOD) |
| `ret_10d_rel_SPY` | `price_change_10 − SPY_close.pct_change(10)` | ALL |
| `ret_5d_rel_SMH` | `price_change_5 − SMH_close.pct_change(5)` | NVDA, CRDO only (SMH not fetched for others) |
| `ret_10d_rel_SMH` | `price_change_10 − SMH_close.pct_change(10)` | NVDA, CRDO only |
| `price_change_10` | `adjusted_close.pct_change(10)` | ALL (added to fill the 1/5/15 gap) |

Also added `price_change_10` and the relative returns to `robust_features` for NVDA and CRDO.

Backlog: `ret_3d_rel_SPY`, `ret_3d_rel_SMH` (3-day window not yet added; add if 5d/10d show signal).

### 3b. Rolling beta and idiosyncratic return (Medium difficulty)

| Feature | Formula | Notes |
|---|---|---|
| `rolling_beta_spy_60d` | OLS beta of daily stock returns on SPY over 60 days | Systematic risk exposure |
| `residual_ret_10d` | `ret_10d_stock − rolling_beta_spy_60d × ret_10d_SPY` | Idiosyncratic component of recent move |

### 3c. Factor proxies (simple ETF-based)

| Feature | Source | Notes |
|---|---|---|
| `momentum_factor_ret_10d` | MTUM ETF return | Momentum factor exposure |
| `growth_vs_value_10d` | IWF vs IWD return spread | Regime signal |

---

## 4. Short Interest / Borrow

Short crowding + catalysts often drive sharp 1–3 week moves.

| Feature | Source | Notes |
|---|---|---|
| `short_interest_pct_float` | FINRA / yfinance `info['shortPercentOfFloat']` | Bi-monthly FINRA release; yfinance has ~2-week delay |
| `days_to_cover` | short_interest / avg_daily_volume | Also from yfinance; same cadence |
| `short_interest_chg` | Δ short_interest vs prior FINRA report | Direction of short buildup/cover |
| `borrow_fee_rate` | Stock Borrow desk / IB API | Hard to source reliably; useful for crowded shorts |

---

## 5. Volume & Flow

Current volume features are good; these are the incremental improvements.

| Feature | Formula / source | Notes |
|---|---|---|
| `dollar_volume_zscore` | z-score of `close × volume` over 20d | Normalizes across price regimes better than share volume |
| `rel_volume_vs_weekday` | today's volume / avg volume for same weekday (30d) | Intraday seasonality proxy; flags unusual activity on specific days |
| `smh_fund_flow_proxy` | SMH volume z-score or return | ETF flow proxy for sector demand |
| `institutional_ownership_pct` | 13F filings / WhaleWisdom / AV | Slow (quarterly); useful as regime feature |

---

## 6. Price-Action Features

Simple but often higher SHAP than MACD/RSI variants because they are more direct.

### 6a. Multi-horizon momentum (explicit clean return set)

| Feature | Formula | Notes |
|---|---|---|
| `ret_3d` | `(close_t / close_{t-3}) − 1` | We have change_1/5/15 but not a clean ret_N set |
| `ret_5d` | `(close_t / close_{t-5}) − 1` | — |
| `ret_10d` | `(close_t / close_{t-10}) − 1` | — |
| `ret_20d` | `(close_t / close_{t-20}) − 1` | — |

### 6b. Gap features (require open price)

| Feature | Formula | Notes |
|---|---|---|
| `gap_open_to_close` | `(close − open) / open` | Intraday direction; already have OHLC? |
| `gap_close_to_open` | `(open_t − close_{t-1}) / close_{t-1}` | Overnight gap; news/futures reaction proxy |

### 6c. Trend strength

| Feature | Formula | Notes |
|---|---|---|
| `log_price_slope_10d` | OLS slope of `log(close)` over 10 days | More robust trend signal than MACD crossover |
| `log_price_slope_20d` | OLS slope of `log(close)` over 20 days | Medium-term trend direction |

### 6d. Realized distribution shape

| Feature | Formula | Notes |
|---|---|---|
| `realized_skew_20d` | skewness of daily returns over 20 days | Tail direction; negative = left-tail risk |
| `realized_kurtosis_20d` | excess kurtosis of daily returns over 20 days | Fat-tail proxy |

---

## 7. Corporate Actions / Supply Events

Can drive predictable short-term patterns around known dates. Generally low-frequency
and hard to source reliably at scale.

| Feature | Source | Notes |
|---|---|---|
| `days_to_exdiv` | dividend calendar (yfinance / AV) | Ex-dividend date proximity; predictable buying/selling pattern |
| `is_exdiv_week` | 1 if days_to_exdiv ≤ 5 | Binary flag version |
| `secondary_offering_flag` | SEC S-3 / 424B filings | Share dilution events; complex to parse reliably |
| `lockup_expiration_flag` | IPO lockup calendars | Relevant for recently-IPO'd names (CRDO, INOD) |
| `buyback_active_flag` | 10b5-1 plan filings / press releases | Hard to automate; strong signal when active |

---

## Other existing ideas (pre-categorization)

Previously noted — now merged into categories above or retained here:

| Feature | Source | Notes |
|---|---|---|
| `insider_buy_sell_ratio` | SEC Form 4 / OpenInsider | Strong signal but complex to fetch reliably; see Corporate Actions §7 |
| `news_sentiment_score` | AV NEWS_SENTIMENT (already fetched) | Currently not in selected_columns for most profiles; low-hanging fruit |

---

## Notes on activation

To activate any backlog feature:
1. Verify it exists in `{SYMBOL}_TMP.csv` (it may already be computed)
2. Add the column name to `selected_columns` in the relevant `{SYMBOL}_param.py`
3. Run the integration test or smoke test to confirm no KeyError
4. Re-run training and compare F1 against baseline (use `reference_no_shuffle` for honest eval)

**Baseline** (NVDA `reference_no_shuffle`, target_size=1, end_date=2026-02-23, with `eps_est_avg`):
- Test F1 c0 (neutral): 0.178 | c1 (UP): 0.555 | c2 (DN): 0.051
- Test AUC c0: 0.629 | c1: 0.588 | c2: 0.512

**Baseline** (NVDA `reference` shuffle, target_size=1, end_date=2026-02-23, post-bug-fix, pre-EPS):
- F1 class 0 (neutral): 0.716 | F1 class 1 (UP): 0.214 | F1 class 2 (DN): 0.212

---

## Technical Debt / Code Review

| Task | Priority | Notes |
|---|---|---|
| Code review CP Ratio code | Medium | Review `get_historical_cp_ratios_with_sentiments_new()` and all CP ratio processing in `fetchBulkData.py` for correctness, edge cases, and IV backfill behavior after IV feature addition |
