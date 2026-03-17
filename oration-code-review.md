# Code Review: Call/Put Ratio (CP Ratio) Feature

**Reviewed:** 2026-03-09
**Scope:** All code paths for CP ratio fetch, computation, backfill, storage, IV feature
extraction, and integration into training/inference.

**Files reviewed:**
- `/workspace/fetchBulkData.py` — primary fetch, compute, and merge logic
- `/workspace/cp_ratio_backfill.py` — historical IV/CP backfill script
- `/workspace/unused_files/cp_ratio.py` — legacy (archived)
- `/workspace/gbdt_pipeline.py` — training pipeline (IV column defaults)

---

## Summary Table

| # | File | Line(s) | Severity | Issue |
|---|------|---------|----------|-------|
| 1 | `fetchBulkData.py` | 3296–3310, 3376–3394 | **CRITICAL** | Magic value `3.0` for sentiment ratio when bearish=0 |
| 2 | `gbdt_pipeline.py` | 159 | **CRITICAL** | Same magic value `3.0` — duplicated bug |
| 3 | `fetchBulkData.py` | 3403–3404 | **CRITICAL** | Division by zero in `options_volume_ratio` — only post-validated |
| 4 | `fetchBulkData.py` | 3284–3291, 3364–3371 | **MEDIUM** | Date alignment of CP ratio merge — potential look-ahead bias (undocumented) |
| 5 | `fetchBulkData.py` | 1580–1593 | **MEDIUM** | Silent zeros when API returns no data — indistinguishable from true zero volume |
| 6 | `fetchBulkData.py` | 1692, 1711 | **MEDIUM** | Duplicate row prevention happens after write — fragile if process crashes |
| 7 | `fetchBulkData.py`, `gbdt_pipeline.py` | 3340–3343, 239–243 | **MEDIUM** | Inconsistent IV column default fill: NaN vs 0.0 |
| 8 | `fetchBulkData.py` | 3406–3414 | **MEDIUM** | `options_volume_ratio` validation fires post-merge (too late to block bad data) |
| 9 | `fetchBulkData.py` | 1522–1546 | **MEDIUM** | No integrity check on existing CP ratio CSV — trusts it blindly |
| 10 | `fetchBulkData.py` | 1649–1651 | **LOW** | Same-day expiry exclusion (`dte > 0`) is correct but undocumented |
| 11 | `fetchBulkData.py` | 1654, 1659, 1672 | **LOW** | Expiry bucket gaps: DTE 11–19 and 46–59 are excluded from all IV buckets |
| 12 | `cp_ratio_backfill.py` | 285–288 | **LOW** | Per-date failures are silently swallowed — only final row count reported |
| 13 | `fetchBulkData.py` | 1692–1694, 1710–1714, 3288 | **LOW** | CSV write → re-read pattern (workaround for formatting bug; root cause not fixed) |

---

## Critical Findings

### C1. Magic value `3.0` for sentiment ratio — `fetchBulkData.py` lines 3296–3310, 3376–3394

When `bearish_volume == 0` and `bullish_volume > 0`, the code returns a hardcoded `3.0`:

```python
elif bullish > 0 and bearish == 0:
    return 3.0
```

**Problems:**
- Arbitrary constant with no economic justification — why 3.0 and not 2.0 or 5.0?
- Asymmetric: when `bullish=0, bearish>0` the ratio is `0.0` (not `-3.0` or a reciprocal)
- Creates artificial outliers in the feature distribution that models can overfit to
- The identical bug is duplicated in `gbdt_pipeline.py` line 159

**Recommended fix:** Use `np.nan` to signal insufficient data (no bearish volume), or replace
with a configurable cap (e.g. `min(bullish / max(bearish, 1), MAX_RATIO_CAP)`). If a magic
value must be kept, make it a named constant and document the rationale.

---

### C2. ~~Division by zero in `options_volume_ratio`~~ — **FIXED 2026-03-09** (`fetchBulkData.py` ~line 3384)

```python
df['options_volume'] = df['call_volume'] + df['put_volume']
df['options_volume_ratio'] = df['options_volume'] / df['volume']
```

If `df['volume']` (stock volume) is 0 or NaN, this produces `inf` or `nan`. The validation
alert (lines 3406–3414) fires **after** the bad value is computed and potentially written.

**Recommended fix:**
```python
df['options_volume_ratio'] = np.where(
    df['volume'] > 0,
    (df['call_volume'] + df['put_volume']) / df['volume'],
    np.nan
)
```

---

### C3. Same magic value `3.0` duplicated — `gbdt_pipeline.py` line 159

Same issue as C1, independently hardcoded in the training pipeline. Both occurrences must be
fixed together to keep inference and training consistent.

---

## Medium Findings

### M1. Date alignment of CP ratio merge — potential look-ahead bias

**Lines:** 3284–3291, 3364–3371

CP ratio is computed from same-day options data and merged on `date`. Options chains close
at 3 PM ET; stock prices close at 4 PM ET. If CP ratio from date `D` is used as a feature
predicting a label derived from the 4 PM close on the same day `D`, this is correct (no
look-ahead). However, if labels are derived from the next-day open or close, the merge
alignment introduces a 1-trading-day shift that is not documented.

**Action:** Add a comment in the merge block confirming the alignment intent. Verify against
the label construction logic in `trendAnalysisFromTodayNew.py`.

---

### M2. Silent zeros when API returns no data — `fetchBulkData.py` lines 1580–1593

When Alpha Vantage returns `{"data": []}` or an error, the row defaults to:

```python
row = {
    'date': date_str,
    'call_volume': 0.0, 'put_volume': 0.0,
    'cp_volume_ratio': 0.0, ...
}
```

A true zero-volume options day and an API failure are indistinguishable in the resulting CSV.
The post-merge alert (finding M4) catches all-zero data but does not distinguish the two cases.

**Recommended fix:** Use `np.nan` as the default for API failures and `0.0` only when the API
explicitly returns an empty chain (data key present but empty list).

---

### M3. Late duplicate prevention — `fetchBulkData.py` lines 1692, 1711

```python
results_df = results_df.drop_duplicates(subset=['date'])
results_df.to_csv(file_path, index=False)
```

Duplicates are dropped just before writing, but if the process crashes between accumulating
duplicates and dropping them, the CSV will contain duplicate rows. The `clean_cp_ratio_file()`
called at the start of the function (line 1443) partially mitigates this, but it is fragile.

**Recommended fix:** De-duplicate before appending (upsert pattern): check if a date already
exists in the file before inserting, rather than appending and deduplicating after.

---

### M4. Inconsistent IV column defaults: NaN vs 0.0

**`fetchBulkData.py` lines 3340–3343:**
```python
for _iv_col in ['iv_7d', 'iv_30d', 'iv_90d', 'iv_skew_30d', 'iv_term_ratio']:
    if _iv_col not in df_cp_ratios.columns:
        df_cp_ratios[_iv_col] = np.nan   # ← NaN
```

**`gbdt_pipeline.py` lines 239–243:**
```python
for col in IV_COLS:
    if col in feature_cols and col not in df.columns:
        df[col] = 0.0   # ← 0.0
```

NaN is the correct sentinel for "IV not available." Using 0.0 implies zero implied volatility,
which is economically meaningless and will be treated as a valid signal by tree models.

**Recommended fix:** Change `gbdt_pipeline.py` to fill with `np.nan` and confirm that the
training pipeline uses imputation (median or forward-fill) rather than assuming 0.0 is valid.

---

### M5. Validation fires post-merge — `fetchBulkData.py` lines 3406–3414

```python
non_zero_count = (df['options_volume_ratio'] > 0).sum()
if non_zero_count == 0:
    print(f"⚠️  ALERT: options_volume_ratio is ALL ZEROS for {symbol}!")
```

By this point the DataFrame has already been merged for training. The alert is a useful
signal but does not prevent bad data from flowing into model training.

**Recommended fix:** Move the validation to immediately after the CP ratio CSV is loaded
(before the merge), and raise or log prominently so the nightly run can be halted.

---

### M6. No integrity check on existing CP ratio CSV — `fetchBulkData.py` lines 1522–1546

The function reads the existing CSV and trusts it entirely:
```python
if os.path.exists(file_path):
    existing_df = pd.read_csv(file_path)
    results_df = existing_df.copy()
```

A corrupted file (truncated write, wrong column names after a schema change) would propagate
silently. The `clean_cp_ratio_file()` call handles duplicates, but not schema drift or
truncation.

**Recommended fix:** Add a column schema check after reading the CSV and raise a clear error
if expected columns are missing.

---

## Low Findings

### L1. Same-day expiry exclusion undocumented — `fetchBulkData.py` line 1651

```python
df_iv = df[(df['implied_volatility'] > 0) & (df['dte'] > 0)].copy()
```

`dte > 0` excludes same-day expiry options (0DTE), which are the most active contracts in
the current market. This is probably intentional (illiquid for historical data), but there
is no comment explaining the rationale.

---

### L2. Expiry bucket gaps — `fetchBulkData.py` lines 1654, 1659, 1672

```python
m7  = (df_iv['dte'] >= 1)  & (df_iv['dte'] <= 10)   # "7-day"
m30 = (df_iv['dte'] >= 20) & (df_iv['dte'] <= 45)   # "30-day"
m90 = (df_iv['dte'] >= 60) & (df_iv['dte'] <= 120)  # "90-day"
```

Options with DTE 11–19 and 46–59 fall in neither bucket and are silently ignored. The same
ranges are hardcoded identically in `cp_ratio_backfill.py` lines 204, 208, 219.

For most liquid names this is low impact, but the gaps should be documented. Consider
whether DTE 11–19 contracts should count toward the 30-day bucket.

---

### L3. Per-date backfill failures silently swallowed — `cp_ratio_backfill.py` lines 285–288

```python
result = fetch_cp_and_iv(symbol, dates_df, api_key)
iv_coverage = result['iv_30d'].notna().sum()
print(f"[{symbol}] Done — {len(result)} rows, {iv_coverage} with iv_30d coverage")
```

Exceptions inside the per-date loop (lines 237–239) are caught and the date is skipped.
The final summary only reports total row count, not how many dates failed.

**Recommended fix:** Accumulate a `failed_dates` list and print it at the end.

---

### L4. CSV write → re-read pattern — `fetchBulkData.py` lines 1692–1694, 1710–1714, 3288

Data is written to CSV and then immediately re-read on line 3288 with the comment "rid of
formatting error that causes NaN." This implies a known serialization bug in the write path
that is worked around rather than fixed.

**Recommended fix:** Identify what formatting issue the re-read corrects (likely date column
dtype after `to_csv` / `read_csv` round-trip) and fix it at the source, removing the
redundant I/O.

---

## Already Correct (No Action Needed)

- **OI-weighted IV fallback** (`lines 1556–1561`): Falls back to unweighted mean when all OI
  is zero; the empty-subset case is guarded by `if m7.sum() > 0:` before calling — safe ✓
- **IV term ratio division** (`lines 1677–1678`): Checks `row['iv_30d'] > 0` before dividing — safe ✓
- **Timeout guards**: `_FRED_TIMEOUT = 30` and `_AV_TIMEOUT = 30` applied to all `requests.get`
  calls — safe ✓
- **Session + disk cache for CP ratio CSV**: CP ratio data is persisted in per-symbol CSV
  files (not via `fetch_cache.py`), which is intentional per the design doc ("already handled
  by per-symbol CSV files; no change needed") ✓
- **Legacy `unused_files/cp_ratio.py`**: The `float('inf')` pattern in this file is not used
  in any active code path — no action needed ✓

---

## Priority Action List

| Priority | Item |
|---|---|
| ~~Fix now~~ **DONE** | C1 + C3: Replace magic value `3.0` — fixed 2026-03-09 |
| ~~Fix now~~ **DONE** | C2: Pre-division guard for `options_volume_ratio` — fixed 2026-03-09 |
| Fix soon | M4: Align IV default fill to `np.nan` in `gbdt_pipeline.py` |
| Fix soon | M2: Distinguish API failure (NaN) from true zero volume (0.0) |
| Document | M1: Confirm and document CP ratio date alignment vs label construction |
| Document | L1, L2: Add comments for same-day expiry exclusion and bucket gap rationale |
| Low priority | M3, M6: Upsert pattern + CSV schema validation |
| Low priority | L3: Report failed dates in backfill summary |
| Low priority | L4: Fix CSV write/re-read root cause |
