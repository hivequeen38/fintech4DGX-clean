# Feature Engineering PRD — Additions for 1–3 Week Horizon Predictor (Items 4–6)

**Owner:** Milton Soong
**Date:** 2026-03-17
**Status:** Draft
**Reviewed by:** Claude / Milton Soong
**Review date:** 2026-03-17
**Scope:** Incorporate missing high-SHAP feature blocks and prioritize minimal-disruption upgrades to the existing feature set.
**Out of scope:** Re-architecting the model; switching data vendors; full options surface modeling unless noted.

---

## 4) High-SHAP Feature Blocks Missing from the Current Model

This section enumerates feature families that are commonly high-impact for **1–3 week** directional prediction and are currently absent (or only partially represented) in your pipeline.

### 4A) Earnings proximity and event gating (planned — should be included)
**Purpose:** Let the model condition the importance of revisions, options proxies, and price reaction on earnings timing.

**Features**
- `dte` — trading days to next earnings effective session
- `dse` — trading days since prior earnings effective session
- `earn_in_5` — `0 <= dte <= 5`
- `earn_in_10` — `0 <= dte <= 10`
- `earn_in_20` — `0 <= dte <= 20`

**Optional (recommended if your earnings feed supports it)**
- `is_earnings_day`
- `post_earn_day_1`, `post_earn_day_2`, `post_earn_day_3`
- `earnings_is_amc` / `earnings_is_bmo` (time-of-day flags)

> **[REVIEW — ALREADY IMPLEMENTED]** `dte`, `dse`, `earn_in_5/10/20`, and `surprisePercentage` are already in `NVDA_param.py` `selected_columns` (and all other tickers). The earnings calendar feed is the AV earnings endpoint already used in backfill. No action needed for the base features.
>
> **[REVIEW — OUTSTANDING]** `is_earnings_day`, `post_earn_day_1/2/3`, and `earnings_is_bmo/amc` are NOT yet computed. These are derivable from existing `dte`/`dse` in `analysisUtil.py` at zero infrastructure cost:
> ```python
> df['is_earnings_day'] = (df['dse'] == 0).astype(int)
> df['post_earn_day_1'] = (df['dse'] == 1).astype(int)
> df['post_earn_day_2'] = (df['dse'] == 2).astype(int)
> df['post_earn_day_3'] = (df['dse'] == 3).astype(int)
> ```
> AMC/BMO flags require earnings calendar to include time-of-day — check if AV endpoint provides this. If not, skip for now.
>
> **[REVIEW — LEAKAGE RISK]** `surprisePercentage` is already in `selected_columns` but the timing rule in Priority 2 (§5) applies here: it must only be non-null at/after the effective session. This is flagged as PRD-3 in backlog. **Must verify before relying on SHAP for this feature.**

---

### 4B) Extended-hours price discovery (planned — should be included)
**Purpose:** Capture the market's first repricing to earnings/guidance and major news that occurs in the **16:00–20:00 ET** window.

**Full-session features (as-of 20:00 ET)**
- `ext_close` — 20:00 ET close
- `ret_after = (ext_close / close_rth) - 1`
- `after_range = (after_high - after_low) / close_rth`
- `after_vol`
- `after_vol_share = after_vol / (rth_vol + after_vol)`

**Partial "as-of" snapshots (for predictions before 20:00 ET)**
For standardized snapshot slots (e.g., `16:30`, `19:30`):
- `ext_last_1630`, `ret_after_sofar_1630`, `after_range_sofar_1630`, `after_vol_sofar_1630`, `after_vol_share_sofar_1630`
- same set for `1930`

**Recommended metadata**
- `asof_slot` (e.g., 1630/1930/2000) and/or `minutes_since_16`

> **[REVIEW — DEPRIORITIZED per 2026-03-16 addendum]** Full EXT infrastructure deferred. Requires new data vendor (polygon.io or equivalent — not in current AV subscription), historical backfill to 2021, and nightly update job. Infrastructure cost outweighs expected lift for a 15-day horizon with manual execution.
>
> **[REVIEW — IMMEDIATE ALTERNATIVE IMPLEMENTED]** `overnight_gap` and rolling variants (`overnight_gap_5d_mean`, `overnight_gap_5d_std`, `overnight_gap_5d_abs`) were added to `analysisUtil.py` on 2026-03-17 and are active in all 5 ticker param files. This captures the same earnings/news shock signal at zero infrastructure cost using existing OHLCV.
>
> **Revisit EXT PRD only if:** overnight_gap SHAP is strong AND you need same-evening inference on earnings nights (not current workflow requirement).

---

### 4C) Relative returns (not just relative "trend")
**Purpose:** Explicitly represent stock performance vs sector/market over the same horizon. Often higher-signal than index levels.

**Features**
- `rel_ret_5_smh = ret_5d_stock - ret_5d_SMH`
- `rel_ret_10_spy = ret_10d_stock - ret_10d_SPY`
- `rel_ret_10_qqq = ret_10d_stock - ret_10d_QQQ`

(These can be computed using your existing price/lag framework.)

> **[REVIEW — ALREADY IMPLEMENTED]** `ret_5d_rel_SPY`, `ret_10d_rel_SPY`, `ret_5d_rel_SMH`, `ret_10d_rel_SMH` are computed in `analysisUtil.py` (lines 229–236) and are in `selected_columns` for all 5 tickers. No action needed.
>
> **[REVIEW — MINOR GAP]** `rel_ret_10_qqq` (vs QQQ, not SPY) is not yet included. `qqq_close` is already fetched, so this is a 1-line add to `analysisUtil.py`. Low priority unless SHAP on `ret_10d_rel_SPY` is saturated — QQQ and SPY are highly correlated for tech names and the marginal info is modest.

---

### 4D) Options-implied volatility (currently absent; optional upgrade)
**Purpose:** For 1–3 weeks, IV often becomes a top-tier signal. Your current put/call ratios are useful but coarser.

**If accessible via your data sources, add**
- `iv_atm_30d` (or nearest liquid expiry proxy)
- `iv_term_7_30 = iv_7d - iv_30d`
- `iv_term_30_90 = iv_30d - iv_90d`
- `rv10_over_iv30 = rv_10d / (iv_30d + eps)`

> **[REVIEW — PARTIALLY IMPLEMENTED]** `iv_30d`, `iv_skew_30d`, and `iv_term_ratio` (= `iv_7d / iv_30d`) are already backfilled and active in the `AAII_option_vol_ratio` profile for NVDA and PLTR. CRDO/APP/INOD IV backfill is complete as of 2026-03-16.
>
> **[REVIEW — NAMING DELTA]** The PRD names `iv_atm_30d` but the codebase uses `iv_30d`. These should be the same underlying metric — confirm with data source documentation to avoid confusion.
>
> **[REVIEW — NOT YET IMPLEMENTED]** `iv_term_30_90` (30d vs 90d spread) and `rv10_over_iv30` (realized vs implied ratio) are not yet computed. `rv10_over_iv30` is particularly actionable — `rv_20d` is already in `analysisUtil.py` and can be adapted to 10d; `iv_30d` is already fetched. Add both to `analysisUtil.py` if IV backfill is solid. **Priority: medium** — existing `iv_term_ratio` already captures the short-term event-vol signal.

---

### 4E) Short interest / crowding (currently absent; optional upgrade)
**Purpose:** Captures squeeze/demand-supply imbalance dynamics that can dominate 1–3 week moves.

**If accessible**
- `short_interest_pct_float`
- `days_to_cover`
- `borrow_fee` (if available)
- `short_interest_change` (Δ between reports)

> **[REVIEW — NOT IMPLEMENTED; LOW PRIORITY]** Short interest data is available from FINRA (biweekly settlement file) which is already fetched in `fetchBulkData.py`. However, FINRA data is reported biweekly (2-week lag), which significantly limits its signal freshness for a 1–3 week horizon. `borrow_fee` is not available from current data sources.
>
> **Recommendation:** Defer indefinitely. For names like NVDA/PLTR with high institutional ownership, short interest changes slowly and the biweekly lag makes it nearly useless for the horizon. Re-evaluate only if a real-time short interest feed (e.g., S3 Partners, IHS Markit) becomes available.

---

## 5) Minimal-Disruption Modifications (Priority Order)

This section lists recommended changes in the smallest steps that deliver the highest expected lift.

### Priority 1 — Add event + extended-hours block (highest payoff)
Add:
- `dte`, `dse`, `earn_in_5`, `earn_in_10`, `earn_in_20`
- `ext_close`, `ret_after`, `after_vol_share`, `after_range`
- Partial snapshots (at minimum one slot, recommended two):
  - `ext_last_1630`, `ret_after_sofar_1630`, `after_vol_share_sofar_1630`, `after_range_sofar_1630`
  - and optionally the `1930` slot
- Include `asof_slot` or `minutes_since_16` to reduce distribution shift between partial vs full snapshots.

> **[REVIEW]** Split this into two sub-priorities:
> - **P1a (done):** `dte/dse/earn_in_*` already in all param files. `overnight_gap` implemented as EXT alternative.
> - **P1b (deferred):** Full EXT block (`ext_close`, partial snapshots) is deferred per §4B review. Do not block on this — overnight_gap covers the signal at much lower cost for the 15d horizon.

---

### Priority 2 — Fix earnings leakage and availability timing
**Hard rule:** earnings result fields (e.g., `surprisePercentage`) must only appear **at/after** the effective earnings session.
Add:
- `surprise_is_missing` (or general `earnings_field_missing` flags)
- strict "effective session" mapping based on AMC/BMO where possible

> **[REVIEW — HIGH PRIORITY, UNRESOLVED]** This is logged as PRD-3 in the feature backlog. Currently `surprisePercentage` is populated as of the report date regardless of AMC/BMO timing, which means AMC earnings results appear available at the RTH close of the same day — a potential 1-day lookahead for same-day AMC earners.
>
> **Recommended fix path:**
> 1. Add `is_amc` flag from earnings calendar (if AV endpoint provides it; spot-check needed).
> 2. Shift `surprisePercentage` by +1 trading day for AMC reports in the training set.
> 3. Add `surprise_is_missing` sentinel (1 when no prior earnings result available).
>
> This is the **highest-risk open issue** in the current feature set from a leakage perspective. Should be resolved before relying on SHAP attribution for `surprisePercentage`.

---

### Priority 3 — Add explicit relative-return features
Add 2–3:
- `rel_ret_5_smh`
- `rel_ret_10_spy`
- `rel_ret_10_qqq`

> **[REVIEW — DONE]** `ret_5d_rel_SPY`, `ret_10d_rel_SPY`, `ret_5d_rel_SMH`, `ret_10d_rel_SMH` are implemented. `rel_ret_10_qqq` is the only gap; low priority (see §4C review).

---

### Priority 4 — Upgrade estimates features using your new AV "estimates" feed
Replace "level-only" reliance on `estEPS` with revisions/breadth/dispersion transforms, e.g.:
- `eps_rev_7_pct`, `eps_rev_30_pct`
- `eps_breadth_ratio_30`
- `eps_dispersion = (eps_high - eps_low) / (abs(eps_avg) + eps)`

> **[REVIEW — PARTIALLY IMPLEMENTED]** `eps_rev_7_pct`, `eps_rev_30_pct`, `eps_breadth_ratio_30`, `eps_breadth_ratio_7`, and `eps_rev_accel` are already in NVDA (and other tickers') `selected_columns` as of 2026-03-09/15. These were activated with a leakage monitor note.
>
> **[REVIEW — NOT YET IMPLEMENTED]** `eps_dispersion` (high−low / abs(avg)) is not computed. This captures analyst disagreement — high dispersion ahead of earnings often signals elevated price risk. Actionable if AV estimates endpoint provides `high_eps_estimate` and `low_eps_estimate` fields. Check data availability before adding.

---

## 6) Triage: Keep vs Reconsider (Redundancy Guidance)

This is not a directive to delete features immediately. It's a prioritization framework for later pruning if you need to reduce redundancy, improve stability, or simplify SHAP interpretation.

### Keep (core signal carriers)
- `daily_return`, `volatility`, `ATR`
- `volume`, `Volume_Oscillator`, `volume_volatility`
- `MACD*` and/or `RSI` (keep both if you prefer; one can be sufficient if pruning later)
- `VIXCLS`
- industry relative features: `rs_*`
- options proxies: `cp_sentiment_ratio`, `options_volume_ratio`
- **new adds:** `dte/dse/earn_in_*`, extended-hours features (`ext_close`, `ret_after`, `after_vol_share`, partial snapshots)

> **[REVIEW]** The "new adds" list here still references EXT features which are deferred. Replace with: **`overnight_gap` and rolling variants** as the implemented substitute, plus `post_earn_day_1/2/3` once added.
>
> RSI period was tuned from 20→14 (2026-03-16) and BBands from SMA→EMA. ATR period is default 14 — consistent with the 1–3 week horizon. Confirmed appropriate.

---

### Reconsider later (high redundancy; prune only if needed)
- Bollinger bands (`Real Upper/Middle/Lower`) **if** you already keep `ATR` + `volatility` + `high/low`
- Multiple overlapping market oscillators simultaneously (e.g., `SPY_stoch` + `calc_spy_oscillator` + `SPY_close`, and similarly for QQQ/VTWO)

**Rationale:** These tend to be alternate encodings of the same market regime signal. Keeping all is fine, but SHAP will often "spread" attribution across substitutes.

> **[REVIEW — AGREED]** This is the right approach: don't prune now, let SHAP attribution guide later. The overlapping oscillators (Stoch + calc_spy_oscillator + SPY_close) are a known redundancy. Once SHAP analysis is available post-overnight_gap backfill, revisit and prune low-SHAP substitutes.
>
> **[REVIEW — BBands]** BBands were kept and matype changed to EMA (2026-03-16). They do overlap with ATR+volatility but provide a different normalization (price-relative bands vs absolute ATR). Reasonable to keep for now; prune if SHAP shows near-zero attribution.

---

## Acceptance Criteria

1. Feature store includes all new fields listed in §4A–4C and extended-hours features in §4B.
2. Inference supports both:
   - end-of-extended-hours snapshot (20:00 ET), and
   - partial after-hours snapshots (standardized slots) for earnings-night predictions.
3. `surprisePercentage` (and any earnings-result fields) obey effective-session timing; leakage checks pass on known earnings dates.
4. Backtests show no artificial uplift attributable to lookahead; SHAP indicates increased importance of event-gated and extended-hours features in earnings-adjacent windows.

> **[REVIEW — AC UPDATE NEEDED]**
>
> - **AC1:** §4A and §4C are substantially done. §4B (EXT features) is deferred — AC1 should be amended to replace EXT block with `overnight_gap` as the accepted substitute.
> - **AC2:** EXT inference modes (20:00 / partial snapshots) are deferred. Remove from AC or mark as future/optional.
> - **AC3:** `surprisePercentage` leakage is the **highest-priority unresolved item**. This AC should be mandatory before any SHAP-based conclusions about earnings features.
> - **AC4:** Remains valid. SHAP analysis for `overnight_gap` should be run after the first full training cycle with the new feature (expected 2026-03-17 nightly run).
>
> **Revised ACs for current scope:**
> 1. `overnight_gap` and rolling variants present in all TMP.csv files and SHAP > 0 in at least 3 of 5 tickers.
> 2. `surprisePercentage` AMC timing fix implemented and leakage spot-check passes.
> 3. `eps_dispersion` evaluated for data availability; added if AV provides high/low consensus fields.
> 4. `post_earn_day_1/2/3` added to `analysisUtil.py` and param files.
