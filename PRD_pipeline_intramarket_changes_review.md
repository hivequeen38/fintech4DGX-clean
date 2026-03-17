# Review & Execution Plan: PRD_pipeline_intramarket_changes.md

**Reviewed by:** Claude / Milton Soong
**Review date:** 2026-03-16
**PRD date:** 2026-02-26
**Status:** DEPRIORITIZED 2026-03-16 — full EXT infrastructure deferred; `overnight_gap` feature recommended as immediate lower-cost alternative (see addendum below)

---

## Comments on PRD

### What's well-specified

- **Label alignment (FR3)** is the core insight and is precisely stated — do not mix anchors (no features-from-AH + label-from-RTH-close).
- **Rollout sequence (§9)** dark launch → EXT_CLOSE → PARTIAL_AH is correct.
- **Two-model recommendation (§7.3)** is right given dataset size (~20 PARTIAL_AH training rows/year across the universe).

---

### Clarifications / Concerns

**1. Missing prerequisite — `backfill_ext_market.py` doesn't exist**

The PRD scopes itself to "after the cache is available" but no extended-hours infrastructure exists in the codebase today. This needs a data source decision, a schema definition, a historical backfill, and an ongoing nightly update — this is Phase 0 and likely the most work. Alpha Vantage does not provide after-hours data in the current subscription. Need to decide: **polygon.io**, yfinance (partial AH, recent only), or another vendor.

**2. Open question §10.2 — PARTIAL_AH always vs earnings nights only**

**Recommendation: earnings nights only.** Outside earnings (and known macro catalysts), individual-name after-hours is illiquid and noisy. Running PARTIAL_AH as a permanent always-on mode would dilute the training set with low-information rows and introduce distribution shift. Trigger: `next_report_date == today` from the existing earnings calendar.

**3. Open question §10.3 — P1 endpoint**

**Use `ext_close(t+N)` as P1.** It is stable, always available from cache, and avoids needing snapshot data for every future day. Using ext_last at a consistent time on t+N would create a much deeper dependency on the after-hours cache for all days, not just earnings nights.

**4. 16:30 slot timing — consider 16:45 instead**

At 16:30 the bid-ask spread on individual names after an AMC earnings print can be wide and the price may spike/reverse within minutes. Recommend `16:45` as the minimum PARTIAL_AH slot to let prices settle after the initial reaction. `19:30` is cleaner — most conference calls are complete by then.

**5. Two-model approach confirmed correct**

With 5 tickers × ~4 earnings/year = ~20 PARTIAL_AH training rows/year, a single mixed model would be underpowered. Two separate models (EXT_CLOSE always-on, PARTIAL_AH earnings-only) is the right architecture.

**6. d_model / n_features decoupling (§7.2)**

PRD flags a potential transformer architecture bug: if `d_model` is currently set equal to `n_features`, adding ext_feature columns would silently change the model architecture. **Must be verified and fixed before Phase 2.** Fix: decouple via `Linear(n_features → d_model)` with d_model as a fixed hyperparameter.

---

## Resolved Decisions

| Question | Decision |
|---|---|
| ASOF_SLOT standardization | **16:45** and **19:30** |
| PARTIAL_AH always or earnings-only | **Earnings nights only** (calendar-driven) |
| P1 endpoint | **`ext_close(t+N)`** always |
| Training strategy | **Two separate models** (EXT_CLOSE, PARTIAL_AH) |

**Still open:** AH data source (polygon.io vs other) — must be resolved before Phase 0.

---

## Execution Plan

### Phase 0: Data infrastructure (prerequisite — blocks everything)

1. **Decide on after-hours data source.** Polygon.io recommended (full AH OHLCV, good historical depth). yfinance can fill recent data only and is unreliable for pre-2023 AH. Needs subscription/budget decision.
2. **Define `ext_features` schema:**
   - `ext_close` — 20:00 ET last trade price
   - `ret_after` — (ext_close / close_rth) - 1
   - `after_vol` — total AH volume
   - `after_vol_share` — after_vol / rth_vol
   - `after_range` — (AH high - AH low) / close_rth
   - `ext_last_1645` — last trade price as of 16:45 ET
   - `ret_after_sofar_1645` — (ext_last_1645 / close_rth) - 1
   - `ext_last_1930` — last trade price as of 19:30 ET
   - `ret_after_sofar_1930` — (ext_last_1930 / close_rth) - 1
   - `ext_data_incomplete` — flag (1 if AH data missing/partial)
3. **Build `backfill_ext_market.py`** — fetch historical AH data back to 2021 for all production tickers; write to disk cache under `cache/symbol/EXT_{symbol}`.
4. **Add nightly update** in `nightly_run.py` to refresh prior day's ext_features after 20:30 ET.

---

### Phase 1: Dark launch — merge without activating

5. Add `load_ext_features(symbol, date_range)` loader in `fetchBulkData.py`.
6. Merge ext_features into `df` in `fetch_all_data()` — columns present in TMP.csv but **not** added to any `selected_columns`.
7. Verify all TMP.csv files generate cleanly with no shape/merge errors.
8. Spot-check: confirm `ext_data_incomplete` is 0 for normal days and 1 for weekends/holidays (no AH session).

---

### Phase 2: EXT_CLOSE model variant

9. Add `prediction_mode` key to param structure: `rth` (default, existing behavior) / `ext_close` / `partial_ah`.
10. Modify label computation in `mainDeltafromToday.py`: when `prediction_mode=ext_close`, set P0 = `ext_close(t)`, P1 = `ext_close(t+N)`.
11. **Verify and fix d_model / n_features decoupling** in `trendAnalysisFromTodayNew.py` (§7.2 — add `Linear(n_features → d_model)` projection if not already present).
12. Add `ext_close` profile to `NVDA_param.py` as pilot ticker; add relevant ext_features to `selected_columns`.
13. Train NVDA ext_close model; compare F1 and label distribution vs RTH baseline.
14. If results competitive, roll out ext_close profiles to PLTR, APP, CRDO, INOD.

---

### Phase 3: PARTIAL_AH mode (earnings nights)

15. Add `ASOF_SLOT` config (`1645`, `1930`) to param/run configuration.
16. Modify label computation: P0 = `ext_last_<slot>(t)`, P1 = `ext_close(t+N)`.
17. Wire into `nightly_run.py`: detect earnings nights via `next_report_date == today`; trigger PARTIAL_AH inference for those tickers at the appropriate slot time.
18. Add fallback logic (FR5): if slot data missing, log warning and skip ticker for that slot — do **not** silently fall back to RTH close anchor.
19. Train PARTIAL_AH model variant per ticker using only earnings-night rows (expect thin dataset; may need pooled cross-ticker model).

---

### Phase 4: Operational hardening

20. Add cron triggers: **16:50 ET** and **19:35 ET** on earnings nights for PARTIAL_AH inference.
21. Add observability logging: mode, slot, count of missing/incomplete rows per run.
22. **Leakage audit**: backtest spot-check on earnings days — verify no artificial lift attributable to anchor mismatch (§8 AC3).
23. Update `PRD_ML_Model_Directional15d_UPDATED.md` to reference PARTIAL_AH as an additional inference mode.

---

## Dependencies / Risks

| Risk | Mitigation |
|---|---|
| AH data vendor cost/availability | Evaluate polygon.io free tier vs paid; yfinance as stopgap for recent data |
| Thin PARTIAL_AH training set (~20 rows/year) | Consider pooled cross-ticker model; evaluate minimum data threshold before enabling |
| d_model architecture break | Audit transformer before Phase 2; fix is straightforward (add Linear projection) |
| AH data quality (wide spreads, erroneous prints) | Use volume-weighted or median price for ext_last slots; flag outliers |

---

## Addendum 2026-03-16: Reconsideration — overnight_gap as alternative

**Challenge raised:** For a 15-day horizon, manual-execution, 1-2 trades/day strategy, full extended-hours infrastructure may add little predictive value because:

1. After-hours liquidity is thin — prices are noisy and spreads are wide.
2. Most overnight moves partially revert during regular hours by day 1-2.
3. By day 3-15 of the prediction horizon, the AH move from day 0 is likely irrelevant.
4. Professional daily models often ignore extended hours entirely.

**Simpler alternative that captures most overnight information:**

```
overnight_gap   = (open(T) - close(T-1)) / close(T-1)
```

This requires **zero new infrastructure** — `open` and `close` are already in the OHLCV fetch from Alpha Vantage. Rolling variants add further coverage:

```
overnight_gap_5d_mean  — persistent gap bias (momentum / news flow)
overnight_gap_5d_std   — gap volatility regime
overnight_gap_5d_abs   — magnitude regardless of direction
```

**Where the two approaches diverge:**

| Problem | overnight_gap | Full EXT_PRD |
|---|---|---|
| Better features for the 15d model | ✅ Zero cost, already in data | Overkill |
| Inference timing ON earnings night (16:30/19:30) | ❌ Not available until next open | ✅ Solves this |

**Revised recommendation:**

- **Add `overnight_gap` and rolling variants immediately** — cheap, no new vendor, high expected SHAP for gap-sensitive names like NVDA/CRDO.
- **Defer full EXT PRD to low priority.** The earnings-night inference timing is a workflow preference, not a model quality requirement. For manual execution at a 15-day horizon, waiting until next morning's open is almost always acceptable.
- **Revisit EXT PRD only if:** (a) overnight_gap SHAP shows strong signal and we want more granularity, OR (b) the workflow genuinely requires same-evening signals after earnings.
