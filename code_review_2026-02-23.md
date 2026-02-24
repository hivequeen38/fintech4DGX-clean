# Code Review: ML Stock Prediction Training Pipeline
**Date:** 2026-02-23
**Files reviewed:** trendAnalysisFromTodayNew.py, mainDeltafromToday.py, fetchBulkData.py, processPrediction.py, get_historical_html.py, manual_daily_train.py, NVDA_param.py, CRDO_param.py

---

## CRITICAL ISSUES (fix before trusting any model output)

### 1. ~~Hardcoded date in production entry point~~ — NOT AN ISSUE (by design)
**File:** `manual_daily_train.py` lines 20-21
The date is intentionally set manually each day. This supports regression testing on earlier dates. Line 20 (datetime.now()) serves as a reminder of the format; line 21 is the deliberate override. No action needed.

---

### 2. Shuffled time-series data in DataLoader — ADDRESSED 2026-02-24
**File:** `trendAnalysisFromTodayNew.py` lines 709-710
`shuffle=True` in the DataLoader destroys temporal ordering. Training batches contain future data relative to some samples in the same batch — this is look-ahead bias in every epoch.
**Fix:** Set `shuffle=False` on all DataLoaders.
**Resolution:** `shuffle_splits` param added to all `*_param.py` files. `reference` keeps `shuffle_splits=True` (historical baseline). New `reference_no_shuffle` config (added to all 5 tickers) runs with `shuffle_splits=False` for honest chronological evaluation. Both run nightly — comparison reveals that shuffle=False yields honest ~0.18 macro-F1 vs artificially inflated 0.747 with shuffle=True (see test_shuffle_comparison.py results).

---

### 3. ~~Labels not moved to device in test loop~~ — FIXED 2026-02-24
**File:** `trendAnalysisFromTodayNew.py` lines 921-929
`inputs` are moved to device but `labels` are not. Works silently on CPU fallback but will crash or silently corrupt results on GPU.
**Fix:** Added `labels = labels.to(device)` immediately after `inputs = inputs.to(device)`, matching the validation loop pattern at lines 874-875.

---

### 4. Class weights created on CPU, criterion then moved to device incorrectly
**File:** `trendAnalysisFromTodayNew.py` lines 731, 788, 803-805
`class_weights_tensor` is created on CPU, passed into `CrossEntropyLoss` (still CPU), then moved to device — but the criterion already holds a reference to the CPU tensor.
**Fix:** Move `class_weights_tensor` to device *before* creating `CrossEntropyLoss`.
*(Also partially fixed in Feb 2026 session — verify the ordering is correct.)*

---

### 5. Class weight indexing assumes exactly 3 classes exist
**File:** `trendAnalysisFromTodayNew.py` lines 729-730
```python
class_weights_dict = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}
```
If only 2 classes appear in a training fold (e.g., no NEUTRAL days in a small date range), this crashes with `IndexError`.
**Fix:** Build dict dynamically: `{i: w for i, w in enumerate(class_weights)}`

---

### 6. ~~Inconsistent return tuple sizes from `fetchDateAndClosing`~~ — FIXED 2026-02-24
**File:** `mainDeltafromToday.py` lines 20-77
The function had 4 different return paths returning 2, 3, 3, and 4 values respectively. The caller at line 268 always unpacks 4 values — crashed when any 2- or 3-value path was hit.
**Fix:** Standardized to always return 4 values. Added fallback to most recent available date when requested date has no data (server-UTC vs US-Eastern timezone issue).

---

### 7. Duplicate split code (dead code, copy-paste risk)
**File:** `trendAnalysisFromTodayNew.py` lines 700-707
Lines 705-707 are an exact repeat of 702-704. If someone edits one copy and not the other, the second copy silently overwrites the change.
**Fix:** Remove lines 705-707.

---

## HIGH SEVERITY ISSUES

### 8. Debug prints left in production path
**File:** `mainDeltafromToday.py` lines 267-270
```python
print(f"fetchDateAndClosing returned: {result}")
print(f"Number of values: {len(result)}")
```
These suggest uncertainty about the return type. Leave noise in logs and indicate unstable contract.
**Fix:** Remove debug prints; add a proper assertion instead.

---

### 9. Zero-fill fallback for missing sentiment/CP ratio data
**File:** `fetchBulkData.py` lines ~732, 748, 790
When a CP ratio CSV file is missing, all sentiment columns are filled with 0. Zero is not a neutral value — it signals strong bearish sentiment in most encodings. This silently corrupts features for affected date ranges.
**Fix:** Use forward-fill (`ffill`) or mark as NaN and let the model handle it, rather than filling with 0.

---

### 10. Param files have diverged — inconsistent feature sets per stock
**File:** `NVDA_param.py` vs `CRDO_param.py` vs others
NVDA `AAII_option_vol_ratio` includes `dte/dse/earn_in_*`; CRDO `reference` includes `rs_avgo` but not all other stocks do. Models trained on different feature counts can't be compared, ensembled, or validated against each other.
**Fix:** Define a canonical feature schema; validate all param files against it at startup.

---

### 11. `dropna` silently removes training rows without logging
**File:** `trendAnalysisFromTodayNew.py` lines 302-303
The last `target_size` rows have NaN labels (they look beyond the dataset end) and are silently dropped. No log of how many rows were removed or why.
**Fix:** Log the count of dropped rows. Assert they are only the last N rows (not scattered through the data).

---

### 12. No validation that selected_columns exist in dataframe at inference
**File:** `trendAnalysisFromTodayNew.py` lines 369-371, 543-544
If a feature in `selected_columns` is missing from the fetched data (e.g., API returned incomplete data), the column selection silently produces NaNs or crashes with an obscure KeyError.
**Fix:** Assert all `selected_columns` exist before subsetting. Print a clear error naming the missing column(s).

---

### 13. Scaler file loaded without existence check
**File:** `trendAnalysisFromTodayNew.py` lines 380-381, 474-475, 547-549
If the `.joblib` scaler file is missing, the crash message is a cryptic joblib error rather than "scaler not found for NVDA/AAII_option_vol_ratio".
**Fix:** Add `os.path.exists()` check with a clear error message naming the missing file.

---

### 14. Softmax applied inconsistently across inference vs. training paths
**File:** `trendAnalysisFromTodayNew.py` lines 397, 985
Inference manually applies softmax to logits before argmax. Training validation uses argmax directly on logits. Both give the same argmax result, but probability outputs (for confidence scores) will differ. Inconsistency makes it harder to add calibration later.
**Fix:** Pick one convention and document it. For calibration use, always save raw logits.

---

### 15. No bounds/NaN check on precision/recall/F1 metrics
**File:** `trendAnalysisFromTodayNew.py` lines 889-907
If a class is absent from the test set, sklearn's `classification_report` emits NaN/zero-division warnings but execution continues. NaN metrics get written to the results file silently.
**Fix:** Check for NaN metrics after computation; warn loudly and skip saving if metrics are invalid.

---

## MEDIUM SEVERITY ISSUES

### 16. Silent date mismatch: prediction labeled with script run-date, not data date
**File:** `trendAnalysisFromTodayNew.py` lines 411, 588
Prediction timestamp uses `currentDateTime` (wall clock), but if the data ends on a market holiday or weekend, the prediction date in the results file will be wrong.
**Fix:** Extract the actual max date from the data and use that as the prediction date. Warn if it differs from wall clock by more than 3 days.

---

### 17. Batch size can exceed dataset size in small splits
**File:** `trendAnalysisFromTodayNew.py` lines 758-761
With `batch_size=128`, very small stocks or narrow date ranges could produce validation/test sets smaller than one batch. PyTorch handles this but BatchNorm layers (if added later) would break.
**Fix:** Assert each split has `>= batch_size` samples, or set `drop_last=False` and document it.

---

### 18. Reproducibility: CUDA determinism not verified
**File:** `trendAnalysisFromTodayNew.py` lines 656-670
Seeds are set but `torch.use_deterministic_algorithms(True)` is not called. On GB10 (Blackwell) or any CUDA device, matrix ops remain non-deterministic.
**Fix:** Add `torch.use_deterministic_algorithms(True)` or document that results are non-deterministic across runs.

---

### 19. Positional encoding buffer device placement (PyTorch version dependency)
**File:** `trendAnalysisFromTodayNew.py` line 131
`self.register_buffer('pe', pe)` is correct in PyTorch ≥1.9 (buffer moves with `model.to(device)`), but this dependency isn't documented. On older installs it silently stays on CPU.
**Fix:** Document minimum PyTorch version or explicitly move PE after model.to(device).

---

### 20. Hyperparameters have no validation; invalid values accepted silently
**File:** All `*_param.py` files
`dropout_rate=1.5` or `head_count=7` would be accepted and cause a downstream failure far from where the value was set.
**Fix:** Add a `validate_param()` function called at the start of training that checks ranges for all numeric hyperparameters.

---

### 21. Performance: JSON round-trip when appending results
**File:** `trendAnalysisFromTodayNew.py` lines 1054-1071
Each training run reads the full JSON results file, appends one record, and rewrites it. For 2000+ records this is slow and fragile (file corruption on kill signal).
**Fix:** Append directly to a CSV file; keep JSON only as the final rendering step.

---

### 22. Performance: `iterrows()` in HTML generation
**File:** `get_historical_html.py` line 399
`for _, row in df_sorted.iterrows()` reconstructs a Series per row. For large history tables this is slow.
**Fix:** Use `df.to_html()` with styling, or `itertuples()`.

---

## LOW SEVERITY ISSUES

### 23. Duplicate import
**File:** `trendAnalysisFromTodayNew.py` line 27
`import torch.nn as nn` appears at both line 6 and line 27.
**Fix:** Remove the duplicate.

---

### 24. Typos in function names and comments
- `make_prediciton` → `make_prediction` (`trendAnalysisFromTodayNew.py`)
- `"advise"` → `"advice"` (comment)
- `"dateshould"` → `"date should"` (comment)

---

### 25. Large blocks of commented-out code
**Files:** `trendAnalysisFromTodayNew.py`, `mainDeltafromToday.py`, `fetchBulkData.py`
Dead code scattered throughout makes it hard to read the active logic. Should be deleted (git history preserves it).

---

### 26. Inconsistent string formatting style
Mix of f-strings, `.format()`, and `+` concatenation across the codebase.
**Fix:** Standardize on f-strings.

---

### 27. Magic window sizes hardcoded inconsistently across files
`rolling(window=20)` in fetchBulkData.py vs `rolling(15)` in trendAnalysisFromTodayNew.py for similar calculations. Should live in param files.

---

### 28. Missing docstrings on most functions
Functions like `calculate_label`, `validate`, `make_prediciton_test` have no documentation of inputs, outputs, or expected shapes.

---

## SUMMARY TABLE

| # | Severity | File | Area | Description |
|---|----------|------|------|-------------|
| 1 | ~~CRITICAL~~ BY DESIGN | manual_daily_train.py:21 | Config | Manual date override — intentional for regression testing |
| 2 | ~~CRITICAL~~ ADDRESSED | trendAnalysis:709 | ML | shuffle=True destroys temporal ordering — reference_no_shuffle config added with shuffle=False; reference keeps shuffle=True for continuity |
| 3 | ~~CRITICAL~~ FIXED | trendAnalysis:921 | GPU | Labels not moved to device in test loop |
| 4 | **CRITICAL** | trendAnalysis:731-805 | GPU | Class weights on wrong device when criterion created |
| 5 | **CRITICAL** | trendAnalysis:729 | ML | Hard-coded 3-class assumption crashes on 2-class fold |
| 6 | ~~CRITICAL~~ FIXED | mainDelta:20-77 | Logic | fetchDateAndClosing returns 2/3/4-tuples; caller expects 4 — standardized to always return 4 values |
| 7 | **CRITICAL** | trendAnalysis:705-707 | Code | Duplicate split assignment overwrites valid code silently |
| 8 | HIGH | mainDelta:267-270 | Code | Debug prints left in production |
| 9 | HIGH | fetchBulkData:790 | Data | Zero-fill for missing sentiment corrupts features |
| 10 | HIGH | *_param.py | Config | Feature sets have diverged across stocks |
| 11 | HIGH | trendAnalysis:302 | Data | Silent row drop via dropna — no logging |
| 12 | HIGH | trendAnalysis:369 | Error | No check that selected_columns exist in dataframe |
| 13 | HIGH | trendAnalysis:380 | Error | No existence check before loading scaler file |
| 14 | HIGH | trendAnalysis:397,985 | ML | Softmax applied inconsistently in inference vs. train |
| 15 | HIGH | trendAnalysis:889 | ML | NaN metrics written to results file silently |
| 16 | MEDIUM | trendAnalysis:411 | Logic | Prediction timestamp from wall clock, not data max date |
| 17 | MEDIUM | trendAnalysis:758 | ML | Batch size may exceed split size in edge cases |
| 18 | MEDIUM | trendAnalysis:656 | ML | CUDA determinism not enforced |
| 19 | MEDIUM | trendAnalysis:131 | PyTorch | PE buffer device placement undocumented |
| 20 | MEDIUM | *_param.py | Config | No hyperparameter validation at startup |
| 21 | MEDIUM | trendAnalysis:1054 | Perf | JSON round-trip on every training run |
| 22 | MEDIUM | get_historical_html:399 | Perf | iterrows() on large table |
| 23 | LOW | trendAnalysis:27 | Code | Duplicate import |
| 24 | LOW | trendAnalysis | Code | Typos: make_prediciton, advise, dateshould |
| 25 | LOW | multiple | Code | Large blocks of commented-out code |
| 26 | LOW | multiple | Style | Inconsistent string formatting |
| 27 | LOW | multiple | Code | Magic window sizes inconsistent across files |
| 28 | LOW | multiple | Doc | Missing docstrings on most functions |

---

## RECOMMENDED PRIORITY ORDER FOR TOMORROW

**Fix immediately (before next training run):**
1. ~~Issue #1~~ — by design, skip
2. Issue #3 — labels.to(device) in test loop
3. Issue #4 — class_weights_tensor device order
4. Issue #5 — dynamic class weight dict

**Fix before trusting model results:**
5. Issue #2 — shuffle=False in DataLoader
6. Issue #6 — fetchDateAndClosing return tuple consistency
7. Issue #9 — zero-fill sentinel for missing sentiment

**Fix in next sprint:**
- Issues #8, #10-15 (HIGH severity)

**Defer / low priority:**
- Issues #16-28

---

## BACKLOG (future investigation)

### B1. Verify timesplit vs no time split
**Added:** 2026-02-24
**Context:** `use_time_split` param added to all `*_param.py` profiles (default `False`). `time_based_split()` uses `n_splits=3` (reduced from 5 to avoid underpowered early folds).
**To do:** Run a head-to-head comparison for NVDA (most data, 1252 rows) with `use_time_split=True` vs `False` over the same date range. Compare:
- Reported F1/accuracy on test set
- Forward prediction accuracy from the historical CSV
- Training time cost (3x slower with time split)
**Decision criteria:** If forward accuracy improves by >3% for NVDA, roll out to all tickers with sufficient data (>800 rows).
