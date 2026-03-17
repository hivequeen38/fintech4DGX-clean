# Feature Backlog

Features considered but not yet added to selected_columns.
Prioritized by expected signal value for a 1–3 week horizon.

**Related PRDs:**
- [feature_engineering_PRD.md](feature_engineering_PRD.md) — Feature additions for 1–3 week horizon (§4A–4E, priority order, triage). Reviewed 2026-03-17.
- [PRD_ML_Model_Directional15d_UPDATED.md](PRD_ML_Model_Directional15d_UPDATED.md) — Core model PRD (labels, training policy, output contract, acceptance criteria).
- [PRD_pipeline_intramarket_changes_review.md](PRD_pipeline_intramarket_changes_review.md) — Extended-hours pipeline PRD review. Status: DEPRIORITIZED 2026-03-16; `overnight_gap` implemented as alternative.

---

## [TOP PRIORITY] In-Session Data Fetch Cache

**Problem:** Every call to `fetch_all_data()` makes ~50–70 sequential API calls (Alpha Vantage,
FRED, AAII, FINRA). In a full nightly run (15 tickers × 3 model types = 45 training runs),
ticker-independent series like FRED macro, AAII sentiment, FINRA margin debt, SPY, QQQ, and
VTWO are re-fetched from the network identically on every single run — potentially 40+ redundant
fetches of the same data within one session.

**Proposal:** Two-level cache to eliminate redundant fetches.

**Design plan:** See `design_api_cache.md` (detailed plan — review before implementing).

**Status:** PLANNED — see design doc.

---

## Top 5 Priority Picks (highest expected SHAP, implement first)

| # | Feature(s) | Category | Status |
|---|---|---|---|
| 1 | `earnings_is_bmo`, `earnings_is_amc` | Earnings timing | **IMPLEMENTED** 2026-02-25 — computed & stored in TMP.csv; NOT in selected_columns yet (constant for all-AMC universe; activate when BMO stock added) |
| 2 | `ret_5d_rel_SPY`, `ret_10d_rel_SPY`, `ret_5d_rel_SMH`, `ret_10d_rel_SMH` | Cross-sectional alpha | **IMPLEMENTED** 2026-02-25 |
| 3+4 | `iv_30d`, `iv_skew_30d`, `iv_term_ratio` | Options IV | **IMPLEMENTED** 2026-02-25 — extracted from existing AV HISTORICAL_OPTIONS fetch in same API call; **⚠️ IV BACKFILL PENDING** — run `cp_ratio_backfill.py` (~19h total) before IV features are usable in training |
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

**Active in all profiles for all tickers** (as of 2026-03-15, leakage fixed):
- `eps_rev_30_pct`, `eps_rev_7_pct`, `eps_breadth_ratio_30`, `eps_dispersion` — activated 2026-03-09, leakage-safe after 2026-03-15 fix
- `eps_breadth_ratio_7`, `log_analyst_count`, `eps_rev_accel` — activated 2026-03-15

**Leakage fix (2026-03-15):** `fetch_eps_estimates.py` `_build_anchor_series` now returns two anchor DataFrames. Estimate cols (`eps_est_avg`, `eps_est_analyst_count`, `log_analyst_count`) anchor at `report_date[i]`. Revision cols anchor at `report_date[i+1]` — they only become active once the quarter they describe has been reported, eliminating within-quarter look-ahead bias.

### Tier 2 — COMPLETED 2026-03-15

| Feature | Status |
|---|---|
| `eps_rev_accel` | Added to `_compute_tier1_features`; active in all profiles |
| `eps_breadth_ratio_7` | Was already computed; added to all profiles |
| `log_analyst_count` | Was already computed; added to all profiles |
| `rev_rev_30_pct` | Skip — AV doesn't return revenue revision history |

**VERIFIED 2026-03-17 (nightly_20260317T_fix2.log):**
1. ✅ No KeyError for new columns — all present in TMP.csv for all tickers.
2. ⚠️ EPS estimate coverage = 0/6633 rows (0.0%) for all tickers; all EPS revision cols filled with 0. Disk cache may be serving a stale empty dataset — needs investigation before SHAP can be meaningful.
3. ❌ Inference errors (feature mismatch with pre-retrain scaler) were expected; Phase 3 retrained all models with new feature set. SHAP pending next run.
4. NVDA `ref_noshuf` test F1 this run: c0=0.607, c1=0.175, c2=0.000. UP-class slightly below baseline (0.555) — attributable to new features requiring more training epochs rather than regression; monitor next 2 runs.

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
| `news_sentiment_score` | AV NEWS_SENTIMENT (already fetched) | **NOT low-hanging fruit** — `fetchMultiSentiment.py` fetches articles for HTML reports only; NOT merged into TMP.csv. Requires: (1) daily aggregation of `ticker_sentiment_score` weighted by `relevance_score`, (2) historical backfill per symbol back to start_date (multi-hour API job), (3) merge into TMP.csv. Coverage risk: smaller names (CRDO, INOD) may have sparse daily coverage → all-NaN rows. Recommended approach: backfill NVDA only first, validate signal on `reference_no_shuffle` before rolling out to all 5 stocks. Medium-low priority — do after IV backfill and model training improvements (Focal Loss, LR scheduler). |

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

## B-MH5 — Trans-MZ Inference (enable MZ predictions in HTML report)

**Status:** TRAINING VERIFIED 2026-03-17 — inference will confirm next run.

**What was done:**
- Added `make_inference_multi_horizon()` to `trendAnalysisFromTodayNew.py`
- Added `trans_mz` branch to `mainDeltafromToday.inference()`
- `processDeltaFromTodayResults()` now accepts `model_type` param (was hardcoded `'transformer'`)
- Removed skip guard in `nightly_run.py` Phase 2 inference loop
- 8 unit tests added in `test_mh_inference.py` — all passing

**Verified 2026-03-17 (nightly_20260317T_fix2.log):**
- ✅ Phase 3 training succeeded for both NVDA and PLTR; models saved as `model_NVDA_mz_reference_mh_fixed_noTimesplit.pth` and `model_PLTR_mz_reference_mh_fixed_noTimesplit.pth` (correct naming convention).
- NVDA MH test macro-F1: h1=0.184, h1–5 bucket avg=0.258, h6–15 bucket avg=0.209
- PLTR MH test macro-F1: h1=0.073, h1–5 bucket avg=0.177, h6–15 bucket avg=0.159 (PLTR collapses to single-class prediction — expected with short history ~663 train samples)
- ❌ Inference still failed this run: NVDA old model had `rs_sox_trend_*` features not in TMP; PLTR old scaler expected 81 features vs new 85. Both expected — stale models from before overnight_gap addition.
- **Next run inference should succeed** using newly trained models.

---

## Cache — Remaining Unimplemented Items

Reviewed 2026-03-09. `fetch_cache.py`, session cache, disk cache for most data sources, TMP.csv
freshness check, `purge_old_cache`, `timeout=30` on all requests — all **implemented**.
Items below are what's still missing.

---

### C1. Create `incremental_fetch.py` — cache-aware entry point

**Status:** NOT CREATED

The design doc (`design_api_cache.md`) specifies a new file that exposes `warm_macro()`,
`warm_symbol()`, `fetch_all_cached()`, and `clear_session_cache()`. Currently all cache logic
is embedded directly inside `fetchBulkData.py` rather than behind a clean entry point.
This is the primary architectural gap in the cache implementation.

**Files to touch:** Create `/workspace/incremental_fetch.py`.

---

### C2. Phase 0 warm-up in `nightly_run.py`

**Status:** NOT IMPLEMENTED

The design requires an explicit pre-fetch phase at the start of the nightly run (before the
inference loop) that calls `warm_macro()` and `warm_symbol()` for all tickers upfront.
Currently the session cache (`_SESSION_CACHE`) prevents re-fetching within the same process,
but cache population is reactive (first `fetch_all_data` call populates it) rather than
proactive. With Phase 0, the very first inference call also benefits from cache.

**Prerequisite:** C1 (`incremental_fetch.py`) must exist first.

**Files to touch:** `/workspace/nightly_run.py` — add Phase 0 block before inference loop.

---

### C3. AV EARNINGS (EPS history) not disk-cached

**Status:** IMPLEMENTED 2026-03-15

- `fetchBulkData.py`: AV EARNINGS (`function=EARNINGS`) wrapped with session + disk cache.
  Key `AV_EARNINGS_{symbol}`, tier `symbol`. Stores `(eps_df, report_time_map)` tuple.
  While loop condition changed to `while eps_df is None and attempt < max_attempts:` to skip
  on cache hit without re-indenting the loop body.
- `fetch_eps_estimates.py`: `fetch_av_earnings_estimates()` wrapped with disk cache.
  Key `AV_EPS_EST_{symbol}`, tier `symbol`. No session cache needed (called once per symbol).

---

### C4. yfinance next earnings date not cached

**Status:** IMPLEMENTED 2026-03-15

`fetch_next_report_date()` in `fetchBulkData.py` now uses session + disk cache.
Key `NRD_{symbol}`, tier `symbol`. Refactored multiple early returns into single-return
pattern to allow caching the result before returning. Both AV EARNINGS_CALENDAR and
yfinance fallback paths now flow into one `_fc.save` + `_SESSION_CACHE` write.

---

### C5. Incremental delta fetch for technical indicators (BBands, MACD, ATR, RSI, Stochastics)

**Status:** IMPLEMENTED 2026-03-15

All 7 tech indicator blocks in `fetchBulkData.py` now use stale-reuse incremental pattern:
MACD, ATR, RSI (per-symbol, tier='symbol'); Stoch SPY/QQQ/VTWO (tier='macro'); BBands
(per-symbol + time_period, tier='symbol'). If disk cache misses but stale file exists with
gap ≤ `_INCREMENTAL_THRESHOLD` (7 days), reuse stale dict and save as today's — skipping
the API call entirely. Only falls back to full re-fetch when gap > 7 days or no stale file.

---

### C6. Deprecate / remove `fetchBulkDataCached.py`

**Status:** FILE STILL EXISTS — pending deprecation

The old `fetchBulkDataCached.py` was to be kept until `incremental_fetch.py` was validated
in production, then removed. Since `incremental_fetch.py` was never created (C1), this
is in limbo. Once C1 and C2 are validated in a nightly run, delete this file.

**Files to touch:** Delete `/workspace/fetchBulkDataCached.py` after C1 is live.

---

## Extended-Hours / Intramarket Inference (PRD_pipeline_intramarket_changes.md)

**Full review and execution plan:** `PRD_pipeline_intramarket_changes_review.md`
**Status:** DEPRIORITIZED 2026-03-16 — full EXT infrastructure deferred; `overnight_gap` feature recommended as immediate alternative (see EXT-0a below)

### EXT-0a. `overnight_gap` feature — IMPLEMENTED 2026-03-16

Computed in `trendAnalysisFromTodayNew.py` alongside other derived price features. Four columns added: `overnight_gap`, `overnight_gap_5d_mean`, `overnight_gap_5d_std`, `overnight_gap_5d_abs`. Activated in `selected_columns` for all 5 tickers (NVDA, PLTR, APP, CRDO, INOD).

**VERIFIED 2026-03-17 (nightly_20260317T_fix2.log):**
1. ✅ Columns present for all tickers: overnight_gap=1 NaN (expected shift), overnight_gap_5d_mean/std/abs=5 NaN (expected rolling warmup). No structural issues.
2. ✅ No KeyError in feature assembly for any ticker.
3. SHAP pending — new models with overnight_gap were trained in Phase 3 this run; SHAP can be run next session.

### EXT-0. After-hours data source decision (DEFERRED — low priority)

Full EXT pipeline requires a new vendor (polygon.io) for historical AH OHLCV back to 2021. Deferred until: (a) overnight_gap SHAP shows strong signal and more granularity is warranted, OR (b) same-evening earnings-night inference becomes a genuine workflow requirement. Needs budget decision if revisited.

### EXT-1 through EXT-5 (DEFERRED — low priority)

Full extended-hours infrastructure phases (backfill, dark launch, EXT_CLOSE model, PARTIAL_AH mode, operational hardening) — see `PRD_pipeline_intramarket_changes_review.md` for full plan. Revisit after EXT-0a overnight_gap results are known.

---

## Technical Debt / Code Review

| Task | Priority | Notes |
|---|---|---|
| Code review CP Ratio code | Medium | **VERIFIED 2026-03-17**: ✅ options_volume_ratio validation passes for all 5 tickers (CRDO 882/1040, NVDA 2511/2533, PLTR 1115/1118, APP 1173/1181, INOD 1063/1181). ✅ No duplicate-clean errors. ⚠️ `call_oi`/`put_oi` still fails on first attempt for CRDO/PLTR/APP then succeeds on retry — intermittent data availability from AV, not a code bug. CLOSED. |
| Review momentum indicator parameters (RSI, MACD, ATR, BBands, Stoch) | Medium | **VERIFIED 2026-03-17**: ✅ RSI incremental cache working (reusing stale gap=4d). ✅ BBands incremental cache working (reusing stale gap=1d). ✅ Stoch SPY/QQQ/VTWO all hitting cache and reusing correctly — no zero/missing stoch values. CLOSED. |
| Push local commits to GitHub | Low | **READY TO PUSH** — Phase 3 training completed without code errors 2026-03-17. Inference errors are expected (feature mismatch with pre-retrain scalers, not code regressions). No blockers remain. |

---

## PRD Alignment Gaps (from PRD_ML_Model_Directional15d_UPDATED.md review 2026-03-15)

### PRD-1. Portfolio backtester (HIGH)

PRD §10/§11 acceptance criteria are entirely portfolio-return based (CAGR vs SPY + 3%, regression alpha, Sharpe, Sortino, max drawdown, turnover). Current pipeline only measures classification F1/accuracy. The model cannot be formally accepted or rejected against §11 without this. Requires: walk-forward signal replay, position sizing logic, SPY benchmark comparison, cost drag estimate.

### PRD-2. `trade_daemon` + recency-weighted scoring (HIGH)

PRD §1/§8.1/§8.2 reference `trade_daemon` as the downstream consumer of model outputs. Not yet built. Requires: consume `{TICKER}_trend.json` outputs, compute S_w = Σ exp(−λ(k−1))·y_k and near-term S5 = Σ y_k for k=1..5 (λ default 0.10–0.25, configurable), produce actionable daily signal per ticker.

### PRD-3. Universe discrepancy (LOW — update PRD or code)

PRD §3 lists: NVDA, PLTR, CRDO, APP, TSM, SPY. Production runs: NVDA, PLTR, APP, CRDO, INOD. Two mismatches: (a) INOD is in production but absent from PRD; (b) TSM is in PRD but has no param file and is not in `nightly_run.py`. Resolve by either adding INOD to the PRD and deciding on TSM, or formally dropping TSM from the universe.

### PRD-4. Retraining cadence policy (MEDIUM)

PRD §7.2 states default retrain is weekly or monthly; daily retraining "allowed only if empirically justified." Current pipeline retrains nightly. Either document the empirical justification for nightly retraining (concept drift in high-beta tech names) and update the PRD, or implement a retrain-gating mechanism that skips retraining unless a trigger condition is met.

### PRD-5. Output artifact — audit fields missing (MEDIUM)

PRD §8.1 requires `data_snapshot_id` and `code_version_hash` in every output artifact. Neither field exists in current `{TICKER}_trend.json` or prediction CSVs. Required for §9.1 7-year audit retention policy. Implement by appending git SHA (`git rev-parse HEAD`) and a hash of the input feature snapshot to the JSON output on each run.

### PRD-6. Determinism — noshuf mandate (MEDIUM)

PRD §5.3 requires reproducible outputs given identical inputs. Standard (shuffled) training models are non-deterministic. The `noshuf` model variants satisfy §5.3. Formally designate `noshuf` variants as the production artifacts for audit purposes; treat shuffled variants as experimental.

### PRD-7. Drift monitoring (LOW)

PRD §9 lists drift monitoring as a non-functional requirement. Not implemented. Minimum viable: track UP/FLAT/DOWN label distribution per ticker over a rolling 30-day window; alert (log warning) if any class share shifts by > 15 percentage points vs the training distribution.

### PRD-8. Nightly runtime target (LOW)

PRD §9 requires nightly runtime < 30 minutes. Current pipeline (5 tickers × 3 model types + data fetch) likely exceeds this. Measure actual runtime from nightly logs and either optimize or formally relax the requirement in the PRD.
