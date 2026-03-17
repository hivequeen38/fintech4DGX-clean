# API Response Cache -- Design Document

## Problem

Every call to `mainDeltafromToday.inference()` or `mainDeltafromToday.main()` calls
`trendAnalysisFromTodayNew.load_data_to_cache()`, which in turn calls
`fetchBulkData.fetch_all_data()`. That function makes network requests to every
external data source from scratch, regardless of whether the same data was already
fetched a few seconds ago for a different param set or symbol.

**Nightly run fetch count (as of 2026-03-04):**

| Phase | Calls to `fetch_all_data` |
|---|---|
| Phase 1 - Inference (13 transformer param sets; 2 `trans_mz` skipped) | 13 |
| Phase 2 - Training (17 param sets: 5 ref + 5 AAII + 5 no_shuffle + 2 mz) | 17 |
| **Total per night** | **30** |

Each of those 30 calls hits:

- **FRED** (~29 series: DFF, DGS10, DGS2, DGS3MO, T10Y2Y, VIXCLS, UNRATE, UMCSENT,
  BUSLOANS, RSAFS, M2SL, MORTGAGE30US, DTWEXBGS, DCOILWTICO, BSCICP03USM665S,
  FEDTARMDLR, DGORDER, USEPUINDXD, USALOLITONOSTSAM, BSCICP02USM460S, DFEDTARU,
  CORESTICKM159SFRBATL, CPIAUCSL, PCE, BOGMBBM, SAHMREALTIME, JTSJOL, PCU33443344,
  GDPC1) - **~870 redundant FRED calls per night**
- **Alpha Vantage** - SPY, QQQ, VTWO (shared indices), plus per-symbol: price,
  MACD, ATR, RSI, Stochastic, Bollinger Bands, EPS, income statement, FX (EUR, JPY, TWD)
- **FINRA** - margin-statistics.xlsx (~3 s per fetch)
- **AAII** - `sentiment.csv` read from disk (already local, but opened/parsed 30x)
- **yfinance** - next earnings dates

**Root cause of the 2026-02 stall:** `get_FRED_Data` has no timeout guard. A single
hanging FRED request stalled the run indefinitely. With 870 FRED calls per night,
even a brief outage is nearly certain to kill the pipeline.

---

## Proposed Architecture

### Cache Tiers

**Tier 1 - Macro/market-wide cache** (shared across all symbols and param sets)

Data whose content does not depend on the stock symbol being trained:

- All FRED series (~29 series)
- AAII sentiment CSV
- FINRA margin statistics XLSX
- SPY, QQQ, VTWO price series (Alpha Vantage)
- FX rates: EUR/USD, JPY/USD, TWD/USD (Alpha Vantage)
- Stochastic oscillators for SPY, QQQ, VTWO (Alpha Vantage)

**Tier 2 - Per-symbol cache** (one entry per stock ticker)

Data parameterised on the stock symbol:

- Stock price / adjusted close (Alpha Vantage TIME_SERIES_DAILY_ADJUSTED)
- MACD, ATR, RSI, Bollinger Bands (Alpha Vantage tech indicators for that symbol)
- EPS / income statement (Alpha Vantage EARNINGS + INCOME_STATEMENT)
- Sector peers (AMD, INTC, AVGO, ... for NVDA; ITA, IGV, ... for PLTR; etc.)
- Next earnings date (Alpha Vantage + yfinance)
- CP/options ratio - already handled by per-symbol CSV files; no change needed

---

### Cache Storage

**Format:** Per-date Parquet files under `/workspace/cache/`. Parquet is compact and
fast to read back with pandas. Fall back to JSON if pyarrow/fastparquet is unavailable.

**Directory layout:**

```
/workspace/cache/
  macro/
    2026-03-04_FRED_DFF.parquet
    2026-03-04_FRED_DGS10.parquet
    2026-03-04_AV_SPY_price.parquet
    2026-03-04_AV_QQQ_price.parquet
    2026-03-04_FINRA_margin.parquet
    2026-03-04_AAII_sentiment.parquet
    2026-03-04_FX_EUR_USD.parquet
    2026-03-04_FX_JPY_USD.parquet
    2026-03-04_FX_TWD_USD.parquet
    ...
  symbol/
    2026-03-04_NVDA_price.parquet
    2026-03-04_NVDA_macd.parquet
    2026-03-04_NVDA_eps.parquet
    2026-03-04_PLTR_price.parquet
    ...
  cache_manifest.json
```

**File naming convention:** `{YYYY-MM-DD}_{source_tag}.parquet`

**Freshness / TTL:** A cache file is valid if its filename date matches today's
trading date (`datetime.now(eastern).strftime('%Y-%m-%d')`). Files from prior dates
are stale but kept on disk for failure recovery. `cache_manifest.json` records
per-key write timestamps so that a restarted run skips keys already populated in
the current night.

---

### Cache Key Design

```
Tier 1 (macro):  {date}_{source}_{series_id}
  Examples:
    2026-03-04_FRED_VIXCLS
    2026-03-04_FRED_DGS10
    2026-03-04_AV_SPY_price
    2026-03-04_FINRA_margin
    2026-03-04_AAII_sentiment
    2026-03-04_FX_EUR_USD

Tier 2 (symbol): {date}_{symbol}_{data_type}
  Examples:
    2026-03-04_NVDA_price
    2026-03-04_NVDA_macd
    2026-03-04_NVDA_eps
    2026-03-04_NVDA_sector_peers
    2026-03-04_PLTR_price
```

---

### Integration Points

**New file: `/workspace/fetch_cache.py`**

A thin wrapper that checks disk before calling the real fetch, and writes results
after a successful fetch. No existing function signatures change.

```python
# fetch_cache.py  (pseudocode)
import os, pandas as pd
from datetime import datetime
import pytz

CACHE_DIR = '/workspace/cache'
MAX_STALE_DAYS = 3

def _today() -> str:
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern).strftime('%Y-%m-%d')

def _cache_path(tier: str, key: str) -> str:
    return os.path.join(CACHE_DIR, tier, f'{key}.parquet')

def get_or_fetch(key: str, tier: str, fetch_fn, *args, **kwargs) -> pd.DataFrame:
    today = _today()
    cache_key = f'{today}_{key}'
    path = _cache_path(tier, cache_key)

    if os.path.exists(path):
        return pd.read_parquet(path)   # cache hit

    try:
        df = fetch_fn(*args, **kwargs)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=True)
        return df
    except Exception as e:
        print(f'[CACHE] WARN: fetch failed for {key}: {e}. Trying stale cache.')
        return _load_stale(tier, key)

def _load_stale(tier: str, key: str) -> pd.DataFrame:
    stale_dir = os.path.join(CACHE_DIR, tier)
    # Files named YYYY-MM-DD_<key>.parquet; reverse sort = newest first
    candidates = sorted(
        [f for f in os.listdir(stale_dir) if key in f],
        reverse=True
    )
    for fname in candidates[:MAX_STALE_DAYS]:
        path = os.path.join(stale_dir, fname)
        print(f'[CACHE] STALE: serving {path}')
        return pd.read_parquet(path)
    raise RuntimeError(f'[CACHE] No cache (fresh or stale) found for {key}')

def warm_macro(config: dict, start_date: str):
    import fetchBulkData as fb
    for series_id in fb.FRED_SERIES_IDS:   # new constant list of all 29 series
        get_or_fetch(f'FRED_{series_id}', 'macro',
                     fb.fetch_fred_series_standalone, series_id, start_date)
    for sym, col in [('SPY','SPY_close'), ('QQQ','qqq_close'), ('VTWO','VTWO_close')]:
        get_or_fetch(f'AV_{sym}_price', 'macro',
                     fb.fetch_av_timeseries_standalone, sym, col, config)
    get_or_fetch('FINRA_margin',   'macro', fb.fetch_finra_standalone)
    get_or_fetch('AAII_sentiment', 'macro', fb.fetch_aaii_standalone, start_date)
    for pair in [('EUR','USD'), ('JPY','USD'), ('TWD','USD')]:
        get_or_fetch(f'FX_{pair[0]}_{pair[1]}', 'macro',
                     fb.fetch_fx_standalone, pair[0], pair[1], config)

def warm_symbol(symbol: str, config: dict, start_date: str):
    import fetchBulkData as fb
    get_or_fetch(f'{symbol}_price',        'symbol', fb.fetch_symbol_price_standalone,  symbol, config)
    get_or_fetch(f'{symbol}_macd',         'symbol', fb.fetch_macd_standalone,           symbol, config)
    get_or_fetch(f'{symbol}_atr',          'symbol', fb.fetch_atr_standalone,            symbol, config)
    get_or_fetch(f'{symbol}_rsi',          'symbol', fb.fetch_rsi_standalone,            symbol, config)
    get_or_fetch(f'{symbol}_bbands',       'symbol', fb.fetch_bbands_standalone,         symbol, config)
    get_or_fetch(f'{symbol}_eps',          'symbol', fb.fetch_eps_standalone,            symbol, config)
    get_or_fetch(f'{symbol}_sector_peers', 'symbol', fb.fetch_sector_peers_standalone,   symbol, config)

def purge_old_cache(max_age_days: int = MAX_STALE_DAYS + 1):
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(days=max_age_days)).strftime('%Y-%m-%d')
    for tier in ('macro', 'symbol'):
        tier_dir = os.path.join(CACHE_DIR, tier)
        if not os.path.isdir(tier_dir):
            continue
        for fname in os.listdir(tier_dir):
            if fname < cutoff:   # lexicographic compare on YYYY-MM-DD prefix
                os.remove(os.path.join(tier_dir, fname))
```

**Changes to `/workspace/fetchBulkData.py`**

Two classes of change, minimal surface area:

1. Extract standalone (non-merging) fetch helpers for FRED, FINRA, AAII, AV
   indices, and FX so they can be passed as `fetch_fn` to `get_or_fetch`.
2. Inside `fetch_all_data`, replace each bare call with a `get_or_fetch` call.

Example before/after for a FRED series:

```python
# Before (fetchBulkData.py ~line 2007):
df = get_FRED_Data(df, 'DFF', start_date_timestamp)

# After:
import fetch_cache
fred_dff = fetch_cache.get_or_fetch(
    key='FRED_DFF', tier='macro',
    fetch_fn=fetch_fred_series_standalone,
    series_id='DFF', start_date=start_date_timestamp
)
df = merge_feed_data_frame(df, fred_dff)
```

Also add `timeout=30` to every `requests.get` call in `fetchBulkData.py`. This is
the immediate fix for the FRED stall and can be deployed independently of the cache.

**Changes to `/workspace/nightly_run.py`**

Add Phase 0 immediately before the inference loop:

```python
# nightly_run.py -- new Phase 0
phase_banner(0, 'CACHE WARM -- prefetch all macro and per-symbol data')
import fetch_cache
fetch_cache.warm_macro(config=trendConfig_cfg, start_date=TRAIN_START_DATE)
for sym in ['CRDO', 'NVDA', 'PLTR', 'APP', 'INOD']:
    fetch_cache.warm_symbol(sym, config=trendConfig_cfg, start_date=TRAIN_START_DATE)
```

All subsequent `fetch_all_data` calls in the inference and training loops then read
from disk via the cache layer.

**How `load_cache=True/False` interacts with the new cache**

- `load_cache=True` (default): `load_data_to_cache` calls `fetch_all_data`. With
  the cache layer active, `fetch_all_data` reads Parquet files instead of hitting
  the network. The per-symbol `{SYMBOL}_TMP.csv` files continue to be written by
  `load_data_to_cache` exactly as before.
- `load_cache=False`: unchanged behaviour. Used for QA back-testing where a
  specific historical snapshot is needed; cache layer is bypassed.

---

### Failure Handling

| Scenario | Response |
|---|---|
| API returns error or times out | `get_or_fetch` catches exception, logs `[CACHE] WARN`, calls `_load_stale` |
| Stale data found within MAX_STALE_DAYS=3 | Serve stale DataFrame; print `[CACHE] STALE: serving /workspace/cache/macro/2026-03-03_FRED_DFF.parquet` |
| No stale data exists at all | Raise RuntimeError with clear message; do not silently fill zeros |
| Process restart mid-night | Parquet files already written; warm-cache step skips those keys automatically |
| FRED hangs indefinitely | `timeout=30` on `requests.get` ensures fast failure; stale cache takes over |

**Max staleness threshold:** `MAX_STALE_DAYS = 3` covers weekends plus a 1-day
buffer. Three-day-old VIX / treasury yields are acceptable for model inference, but
every stale serve must produce a visible `[CACHE] STALE` line in the log.

---

### Estimated Impact

| Source | Calls today (30 runs) | Calls after cache | Savings |
|---|---|---|---|
| FRED (~29 series x 30) | ~870 | 29 | ~841 |
| AV SPY/QQQ/VTWO (3 x 30) | 90 | 3 | 87 |
| AV FX EUR/JPY/TWD (3 x 30) | 90 | 3 | 87 |
| FINRA XLSX (30 runs) | 30 | 1 | 29 |
| AAII CSV parse (30 runs) | 30 | 1 | 29 |
| AV per-symbol (~6 calls x 3 param sets x 5 symbols) | ~90 | ~30 | ~60 |
| **Total** | **~1200** | **~67** | **~94% reduction** |

**Rough time savings:** Each FRED call is 2-5 s; AV calls 1-3 s; FINRA ~3 s.
Conservative: 1200 x 2 s average = ~40 minutes of network I/O eliminated per
nightly run, plus elimination of the timeout-failure cascade risk.

---

## Implementation Plan

### Phase 1 - Shared macro cache (biggest win)

1. Add `timeout=30` to all `requests.get` calls in `fetchBulkData.py`. Deploy
   immediately -- this alone prevents the stall that triggered this work.
2. Create `/workspace/fetch_cache.py` with `get_or_fetch`, `_load_stale`,
   `warm_macro`, `warm_symbol`, `purge_old_cache`.
3. Extract standalone (non-merging) helpers from `get_FRED_Data`, FINRA, AAII,
   AV SPY/QQQ/VTWO, and FX fetch blocks in `fetch_all_data`.
4. Wrap all FRED, FINRA, AV-shared, and FX calls in `fetch_all_data` with
   `fetch_cache.get_or_fetch`.
5. Add Phase 0 warm-cache call in `nightly_run.py`.
6. Add `test_fetch_cache.py` -- verify stale fallback triggers, cache hit/miss,
   and `purge_old_cache` behaviour. Run via:
   `python -m pytest /workspace/test_*.py -v`

### Phase 2 - Per-symbol TMP.csv freshness check

1. At the top of `load_data_to_cache` in `trendAnalysisFromTodayNew.py`, check
   whether `{SYMBOL}_TMP.csv` was written today (mtime vs. today's date string).
2. If the TMP file is fresh (same trading day), skip `fetch_all_data` entirely and
   return the CSV as the DataFrame -- zero extra storage, reuses existing mechanism.
3. If the TMP file is stale or absent, proceed with the normal `fetch_all_data` path.
4. Extend `warm_symbol` to call `fetch_all_data` for each symbol once upfront and
   write the `_TMP.csv`; all subsequent param-set calls in that night's run skip
   the fetch via the mtime check.

### Phase 3 - Cross-run persistence (survive process restarts mid-run)

1. Parquet files in `/workspace/cache/` persist across process restarts automatically
   after Phase 1. A restarted run's warm-cache step finds existing files and skips
   those fetches.
2. Add cleanup at the end of `nightly_run.py`:
   `fetch_cache.purge_old_cache(max_age_days=MAX_STALE_DAYS + 1)`
3. Optional cron backup:
   `find /workspace/cache -name '*.parquet' -mtime +4 -delete`

### Phase 4 - Incremental (delta) fetch

**Motivation:** After Phase 1–3, the cache stores each series as a full history Parquet file
keyed by `end_date`. On day X+1, every series in cache is valid up to X — only one new row
is missing per series. The current design would re-fetch the entire history from the data
source just to get that one row. Phase 4 eliminates this by fetching only the delta.

**How it works:**

```
fetch_series(series_id, end_date):
    cached_df = load latest cached Parquet for series_id
    if cached_df is None:
        → full fetch (Phase 1 path)
    cached_max_date = cached_df['date'].max()
    if cached_max_date >= end_date:
        → cache hit, return as-is (Phase 1 path)
    missing_dates = trading_days_between(cached_max_date + 1, end_date)
    if len(missing_dates) <= INCREMENTAL_THRESHOLD (e.g. 5 trading days):
        → delta fetch: call fetch_fn(from_date=cached_max_date + 1, to_date=end_date)
        → append new rows to cached_df
        → write updated Parquet back to cache
        → return combined DataFrame
    else:
        → full fetch (too many missing days — e.g. first run after a long gap)
```

**API support:** Most data sources support a date-range parameter:
- Alpha Vantage `TIME_SERIES_DAILY_ADJUSTED`: returns full history by default, but
  `outputsize=compact` returns the last 100 rows — sufficient for a 1–5 day delta.
- FRED: `observation_start` / `observation_end` parameters allow fetching a window.
- AAII / FINRA: downloaded as full files — delta fetch not applicable; use Phase 1
  freshness check (re-download at most once per day).

**Changes required:**
- Add `fetch_fn_incremental` parameter alongside `fetch_fn` in `get_or_fetch` — the
  incremental variant that accepts `from_date`/`to_date` args.
- Each standalone fetch helper in `fetchBulkData.py` gets an `incremental=False` flag
  and respects `from_date` when `incremental=True`.
- `incremental_fetch.py` orchestrates: check cache max date, decide full vs delta,
  call the right variant.

**Prerequisite:** Phases 1–3 must be stable first. The full-fetch cache (Phase 1)
is the foundation — Phase 4 just optimises the "almost-current" cache hit path.

**Note on naming:** This is what gives `incremental_fetch.py` its name. Phase 1–3
implement the "fetch once per session" behaviour; Phase 4 implements the true
day-over-day incremental update that makes subsequent nights near-instantaneous.

---

## Design Decisions (resolved 2026-03-04)

| # | Question | Decision |
|---|---|---|
| 1 | Should peer stocks (AMD, INTC, etc.) use `ticker=None`? | **Yes.** Peers are shared data — same series regardless of which ticker is being trained. Only the main ticker's price, MACD, ATR, EPS, and OPTIONS use `ticker=<symbol>`. |
| 2 | TMP.csv freshness threshold | **Accept `df['date'].max() >= end_date`.** If the cache has data beyond `end_date` (backtesting scenario), it is still valid — the training pipeline already filters by `start_date`/`end_date` after loading. |
| 3 | Phase order | **Phase 1 first** (series cache, no interface changes), then Phase 2 (TMP.csv mtime check). |
| 4 | `fetchBulkDataCached.py` deprecation | **Create `incremental_fetch.py`** (new entry point that replaces `fetch_all_data` with cache-aware logic). Migrate callers to `incremental_fetch`. Deprecate `fetchBulkDataCached.py` only after full migration is validated. The old `fetchBulkData.fetch_all_data` stays intact as the fallback. |

---

## Files to Create/Modify

| File | Action | Summary of change |
|---|---|---|
| `/workspace/incremental_fetch.py` | **Create** | New cache-aware data fetch entry point. Replaces `fetch_all_data` calls. Wraps each series fetch with `get_or_fetch`. Exposes `warm_macro()`, `warm_symbol()`, `fetch_all_cached()`, `clear_session_cache()`. Phase 4 adds delta-fetch orchestration logic here. |
| `/workspace/fetch_cache.py` | **Create** | Low-level cache layer: `get_or_fetch`, `_load_stale`, `purge_old_cache`. Disk-backed Parquet under `/workspace/cache/`. |
| `/workspace/fetchBulkData.py` | **Modify** | (1) Add `timeout=30` to all `requests.get`. (2) Extract standalone (non-merging) fetch helpers for FRED, FINRA, AAII, AV-shared, FX so they can be passed as `fetch_fn` to `get_or_fetch`. No callers changed yet. |
| `/workspace/nightly_run.py` | **Modify (Phase 1)** | Add Phase 0 `incremental_fetch.warm_macro()` + `warm_symbol()` before inference loop; add `purge_old_cache()` at end. Change `load_cache=True` calls to use `incremental_fetch.fetch_all_cached()`. |
| `/workspace/trendAnalysisFromTodayNew.py` | **Modify (Phase 2)** | In `load_data_to_cache`: check `{SYMBOL}_TMP.csv` freshness (`df['date'].max() >= end_date`). If fresh, return DataFrame directly — skip `fetch_all_data` entirely. |
| `/workspace/test_fetch_cache.py` | **Create** | Unit tests: stale fallback, cache hit/miss, `purge_old_cache`, key naming, freshness threshold (`max_date >= end_date`). No network calls — mock fetch_fn. |
| `/workspace/cache/` | **Create (dir)** | Cache root; `macro/` and `symbol/` subdirs created on first run. |
| `/workspace/fetchBulkDataCached.py` | **Deprecate (later)** | Keep until `incremental_fetch` is validated in production. Then remove. |
