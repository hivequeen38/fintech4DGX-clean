# PRD: Pipeline & Model Changes — Using Cached Extended-Hours Features + Intramarket Snapshots

**Owner:** Milton Soong  
**Date:** 2026-02-26  
**Status:** Draft  
**Audience:** ML/Quant engineering (feature pipeline + training + inference)  
**Scope:** Code changes after `backfill_ext_market.py` cache is available to support:
- end-of-extended-hours predictions (20:00 ET snapshot),
- **before-extended-close** predictions using `ext_last(as_of_ts)` partial after-hours snapshots,
- label alignment and leakage controls.

---

## 1. Problem Statement

After adding extended-hours awareness, we must:
- integrate cached extended-hours features into the daily feature table,
- support predictions **after earnings is out but before extended session ends** (using `ext_last(as_of_ts)`),
- keep labels and features aligned to avoid lookahead/leakage,
- ensure training/inference run modes match the same feature definition.

---

## 2. Goals

1. Load cached `ext_features` (Layer B) and merge into the model feature dataframe by (`ticker`, `date`).
2. Support two inference modes:
   - **Mode EXT_CLOSE:** as-of 20:00 ET (`ext_close` and full after-hours aggregates)
   - **Mode PARTIAL_AH:** as-of configured snapshot times (e.g., 16:30, 19:30) using cached partial features
3. Align labels to the chosen **prediction anchor**:
   - For PARTIAL_AH, anchor at `ext_last(as_of_ts)`
4. Avoid dependence on “next market open” for earnings-night predictions.

---

## 3. Non-Goals

- Re-architecting the model; focus is on feature/label consistency.
- Adding new vendors or options surface data.

---

## 4. Definitions

- **RTH close:** 16:00 ET (`close_rth`)
- **Extended close:** 20:00 ET (`ext_close`)
- **Partial after-hours anchor:** `ext_last_<asof>` (e.g., `ext_last_1630`)
- **Prediction anchor:** the price/time at which we consider the prediction to be made (start point for labels)

---

## 5. Functional Requirements

### FR1: Feature merge (daily dataset)
- Update feature assembly code to load cached `ext_features` and merge on (`ticker`, `date`).
- Keep existing `close` feature as **RTH close** (Option B), add `ext_close` + derived fields.

### FR2: Inference mode selection
Add a runtime configuration:
- `PREDICTION_MODE ∈ {EXT_CLOSE, PARTIAL_AH}`
- For `PARTIAL_AH`, specify `ASOF_SLOT ∈ {1630, 1930, ...}` (must match cache).

#### Mode EXT_CLOSE (20:00 snapshot)
Use:
- `ext_close`, `ret_after`, `after_vol_share`, `after_range`, etc.

#### Mode PARTIAL_AH (as-of slot)
Use:
- `ext_last_<slot>` as the operational “current price”
- `ret_after_sofar_<slot>`, `after_vol_share_sofar_<slot>`, etc.
- Include `minutes_since_16` or `asof_slot` as a categorical/int feature (recommended) to reduce distribution shift.

### FR3: Label alignment (critical)
Update label generation to match the selected mode.

#### If training/inference uses EXT_CLOSE
Anchor start price/time at:
- `P0 = ext_close(t)`

Label end price/time for horizon N (trading days) at:
- `P1 = ext_close(t+N)`

#### If training/inference uses PARTIAL_AH at slot S (e.g., 16:30)
Anchor start:
- `P0 = ext_last_S(t)`  (last price <= as-of time)

Label end:
- Prefer consistent endpoint:
  - `P1 = ext_close(t+N)`  (stable, available from cache)
- Classification uses `R = (P1/P0 - 1)` with your up/down/flat thresholds.

**Explicit requirement:** do not mix anchors (e.g., features include after-hours but label starts at 16:00 RTH close).

### FR4: Earnings-night prediction availability
When an AMC earnings happens on date D:
- PARTIAL_AH features for D (e.g., 16:30/19:30) must be available from cache.
- Inference can run on D evening without waiting for next market day.

Operationally:
- The production job can choose:
  - run `PARTIAL_AH` at `16:30` and/or `19:30` for tickers with earnings today,
  - run `EXT_CLOSE` at ~20:30 for all tickers.

### FR5: Missing / incomplete data behavior
- If cache flags `ext_data_incomplete=1` or missing after-hours:
  - Keep prediction running
  - Set a feature flag `ext_data_incomplete` (already in cache) to allow the model to learn downweighting
- If PARTIAL_AH slot data missing:
  - fallback to nearest earlier slot (policy) OR
  - skip ticker for that slot (policy)
- Log all fallbacks.

---

## 6. Non-Functional Requirements

- **Backtest fidelity:** training uses the same mode + slot as inference.
- **Configurability:** mode/slot selectable without code edits (config file/env var).
- **Observability:** logs include mode, slot, and counts of missing/incomplete feature rows.

---

## 7. Implementation Notes

### 7.1 Feature naming conventions (Option B)
- Existing `close` remains `close_rth`
- New:
  - `ext_close`
  - `ret_after`, `after_vol_share`, `after_range`, `after_vol`
  - `ext_last_1630`, `ret_after_sofar_1630`, etc.
- Add:
  - `asof_slot` (int: 1630/1930/2000) and/or `minutes_since_16` (e.g., 30/210/240)

### 7.2 Model input handling
- Feature count does not need to be divisible by `n_heads`; only `d_model` does.
- If your code incorrectly sets `d_model = n_features`, decouple via `Linear(n_features → d_model)`.

### 7.3 Training dataset strategy
Two viable choices:

1) **Two separate models**:
   - Model A: EXT_CLOSE (20:00)
   - Model B: PARTIAL_AH (e.g., 16:30 slot)

2) **One model with slot indicator**:
   - Mix EXT_CLOSE and PARTIAL_AH rows in training and include `asof_slot`/`minutes_since_16`.
   - This reduces maintenance but increases variance; recommended only if you have sufficient data.

Start with (1) unless you strongly prefer a single model.

---

## 8. Acceptance Criteria

1. Inference job can run **on earnings night** (date D) at PARTIAL_AH slot without waiting for date D+1:
   - features for D include `ext_last_<slot>` and `ret_after_sofar_<slot>`.
2. Labels are consistent with mode:
   - EXT_CLOSE training uses ext_close anchors
   - PARTIAL_AH training uses ext_last_<slot> anchors
3. Backtest shows no artificial lift attributable to leakage (spot-check earnings days).
4. Pipeline can still run if some after-hours data is missing/incomplete; flags are present.

---

## 9. Rollout Plan

1. Merge ext_features into the daily dataset without using it (dark launch).
2. Enable EXT_CLOSE mode at 20:30 ET (should match prior “after extended hours” concept).
3. Enable PARTIAL_AH mode for a small set of tickers with earnings and a single slot (e.g., 16:30).
4. Expand to additional slots if needed.

---

## 10. Open Questions / Decisions Needed

- Which `ASOF_SLOT` times to standardize on? Suggested: `16:30` and `19:30`.
- Should PARTIAL_AH inference run only on earnings days (calendar-driven) or always?
- Endpoint for label (P1): keep `ext_close(t+N)` or use `ext_last` at a consistent time on t+N?
