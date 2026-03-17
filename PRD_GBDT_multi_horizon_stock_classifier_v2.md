# PRD v2 — Multi-Horizon Stock Direction Classifier Using GBDT (LightGBM)

**Version**: 2.0
**Date**: 2026-03-02
**Status**: Draft for implementation

> **Changelog from v1**: All eight open issues from the v1 review have been resolved
> and locked. This document contains no open ambiguities — every decision is specified
> to the level needed to begin M1 without a follow-up design session.

---

## Issue Resolution Summary

| v1 Issue | Decision |
|----------|----------|
| Feature list undefined (`R^D`) | Specified in Section 6.1 (3-tier table from param files; Tier 2 fully populated 2026-03-03) |
| Hyperparameter ranges for 10k+ rows | Resized for ~570 training rows (Section 7.4) |
| Option B bucket label ambiguity | Choice 1 (single label per row) only; pooled rows excluded (Section 7.2) |
| Calibration method priority | Temperature scaling is default; isotonic gated to N≥300/class (Section 9) |
| Acceptance criteria unquantified | Numeric thresholds added (Section 15) |
| IV NaN vs 0.0 fill inconsistency | Preserve 0.0 fill in v1 for training-inference consistency; NaN migration in M6 (NFR1) |
| Model type independence | `model_type='lgbm'` is a fully independent factory instantiation — no architectural relationship to Transformer (Section 1.1) |
| Walk-forward deferred to M6 | 2-fold smoke test moved to M2; full 6-fold in M6 (Section 11.2) |

---

## 1. Purpose

Build a production-ready, multi-horizon stock-direction prediction system using
**LightGBM** to predict **Up / Flat / Down** for the next **1–15 trading days** for
each symbol in the active universe: CRDO, NVDA, PLTR, APP, INOD.

---

### 1.1 Model Type Independence

`model_type='lgbm'` is a new instantiation through the existing factory in
`mainDeltafromToday.main()`. It is completely independent of `model_type='transformer'`
and `model_type='trans_mz'` — separate model class, separate training loop, separate
artifact files, separate prediction CSV. No shared state across model types.

Which model types nightly_run.py calls is a scheduling decision, not an architectural
one. Running both Transformer and LGBM on the same night is fine. Running only LGBM
is fine. The models are orthogonal by construction.

---

## 2. Goals and Non-Goals

### 2.1 Goals
1. Predict 3-class direction (Up / Flat / Down) for horizons `h=1..15`.
2. **15 independent classifiers per symbol** as the production default.
   The 2-model bucket variant is an **M5 ablation only** — not a production path.
3. Improve iteration speed, debuggability, and baseline robustness vs Transformers.
4. Provide calibrated probabilities for ranking and long-only trade selection.
5. Provide SHAP-based leakage detection and feature attribution.
6. Integrate into nightly_run.py by M4.5.

### 2.2 Non-Goals
- Execution engine (order routing, smart order entry).
- Profitability guarantees.
- Sequence models (Transformer / LSTM / TCN) — out of scope here.
- CatBoost in v1 (backlogged to M5 alongside bucket ablation).
- Replacing Transformer in v1 (complement mode only).

---

## 3. Problem Statement

The active Transformer system trains 15 models per stock per param set (~30s/model,
~2.75h total). Class imbalance (Flat dominates) causes recurring UP class collapse
that requires manual intervention (tuning patience, label_smoothing, class weights).
SHAP analysis is slow on neural nets.

LightGBM on the same tabular feature set offers:
- Training time: ~5 min for all 5 symbols × 15 horizons on a single CPU core
- Native missing-value handling and built-in regularization
- Fast, reliable SHAP values via TreeExplainer
- Calibratable probabilities with a single temperature parameter per horizon

---

## 4. Data / Label Specification

### 4.1 Label definition

For each training row at date `t` and horizon `h`:

```
r_{t,h} = (close_{t+h} - close_t) / close_t

y_{t,h} = "up"   if r_{t,h} >  thr(h)
         "down"  if r_{t,h} < -thr(h)
         "flat"  otherwise

thr(h) = 0.03  for h <= 5
         0.05  for h >= 6

Integer encoding: down=0, flat=1, up=2
```

Consistent with existing Transformer label encoding in trendAnalysisFromTodayNew.py.

### 4.2 Symbol-specific threshold concern

The universe spans different volatility regimes. Fixed thresholds produce different
class distributions per symbol. **M1 deliverable**: class distribution report per
symbol × horizon (Up / Flat / Down counts in training set).

If any symbol × horizon has Up or Down < 50 training samples, the response is to
**increase class weights**, not to change the thresholds. Changing thresholds alters
what the model is predicting (a different economic question per symbol), and the
direction of the effect on minority-class counts is non-monotone for fat-tailed
return distributions. Weight adjustment is the correct lever:

- Default: `compute_class_weight('balanced', ...)` (Section 8.1)
- If Up or Down < 50 after balanced weighting: multiply `w_up` and `w_down` by an
  additional 2× and log `[LOW_SAMPLE_ALERT]`
- Thresholds `thr(h)=0.03/0.05` are fixed and symbol-invariant throughout v1

Document actual class counts per symbol × horizon in M1 before proceeding to M2.

### 4.3 Label alignment

The label for training row `t` requires `close_{t+h}`. The trailing `h` rows of
the full dataset have no valid label for horizon `h`. These rows are dropped per
horizon. Each of the 15 models trains on slightly fewer rows (h=15 drops 15 rows
from the tail). This is standard and not a leakage risk.

---

## 5. Architecture Decisions

| Question | Decision |
|----------|----------|
| Per-symbol vs cross-sectional pooling | **Per-symbol**. Each symbol has validated sector-specific features: NVDA+CRDO use semiconductor ETF alpha (rs_amd, rs_smh_trend); PLTR uses defense sector features (rs_ita, FDEFX); APP uses gaming/social ETF alpha (rs_gamr, rs_socl); INOD uses XLK sector proxy. Cross-symbol pooling would require a feature intersection that discards these signals. |
| Training config | **15 models per symbol** (one per horizon). 2-model bucket is M5 ablation only. |
| Option B if implemented | **Choice 1 only** (single label per row using representative horizon). Pooled-rows-with-horizon-id (Choice 2) is explicitly excluded — see Section 7.2. |
| Primary backend | **LightGBM**. CatBoost in M5. |

---

## 6. Data and Feature Interface

### 6.1 Canonical Feature Set

Features are organized in three tiers. The GBDT uses features directly from the
existing `{SYMBOL}_TMP.csv` files — same data source as the Transformer.

**No scaler is needed.** LightGBM's histogram-based splits are invariant to monotone
feature scaling. The Transformer's `robust_features` / `scaler_type` params are
Transformer-specific and do not apply.

#### Tier 1 — Core features (all 5 symbols, ~57 features)

| Group | Features |
|-------|----------|
| Price / volume | `adjusted close`, `daily_return`, `volume`, `Volume_Oscillator`, `volatility`, `VWAP`, `high`, `low`, `volume_volatility` |
| Company fundamentals | `EPS`, `estEPS`, `surprisePercentage`, `dte`, `dse`, `earn_in_5`, `earn_in_10`, `earn_in_20`, `totalRevenue`, `netIncome`, `eps_est_avg` |
| Idiosyncratic return | `ret_5d_rel_SPY`, `ret_10d_rel_SPY` |
| Realized volatility | `rv_10d`, `rv_20d`, `rv_term_ratio`, `vix_rv_ratio` |
| Momentum | `MACD_Signal`, `MACD`, `MACD_Hist`, `ATR`, `RSI`, `Real Upper Band`, `Real Middle Band`, `Real Lower Band` |
| Rates / FX | `interest`, `10year`, `2year`, `DTWEXBGS`, `DFEDTARU`, `BOGMBBM`, `jpy_close`, `twd_close` |
| Indices | `SPY_close`, `qqq_close`, `VTWO_close`, `SPY_stoch`, `calc_spy_oscillator`, `QQQ_stoch`, `VTWO_stoch` |
| Macro | `VIXCLS`, `DCOILWTICO`, `USEPUINDXD`, `UMCSENT`, `BSCICP02USM460S`, `DGORDER`, `PCU33443344`, `SAHMREALTIME`, `JTSJOL`, `GDP`, `FINRA_debit`, `Spread` |
| Metadata | `day_of_week`, `month`, `price_lag_1`, `price_lag_5`, `price_lag_15`, `price_change_1`, `price_change_5`, `price_change_15` |

#### Tier 2 — Symbol-specific sector features

NVDA and CRDO are both semiconductor names and share the same sector-peer feature block.
APP is in mobile/gaming application software. INOD is in broad technology/IoT.

| Symbol(s) | Features | Source |
|-----------|----------|--------|
| NVDA + CRDO | `ret_5d_rel_SMH`, `ret_10d_rel_SMH`, `rs_amd`, `rs_amd_trend`, `rs_intc_trend`, `rs_avgo_trend`, `rs_smh_trend` | Semiconductor sector alpha vs SMH ETF and peer stocks (AMD, INTC, AVGO) |
| PLTR | `rs_ita`, `rs_igv`, `rs_ita_trend`, `FDEFX`, `ADEFNO`, `IPDCONGD` | Defense/gov't IT sector alpha |
| APP | `rs_gamr`, `rs_gamr_trend`, `rs_socl`, `rs_socl_trend` | Gaming (GAMR ETF) and social media (SOCL ETF) sector alpha |
| INOD | `XLK_close` | Technology sector ETF (XLK) — sector regime proxy |

These features were selected through prior SHAP tuning passes on the Transformer and carry
directly into the GBDT. Expect another SHAP pass after M2 to confirm they remain relevant
for the GBDT's feature-selection dynamics.

#### Tier 3 — Options / IV features

**Tier 3a — Call/put ratio + options volume** (active for ALL 5 symbols):

| Feature | Description |
|---------|-------------|
| `cp_sentiment_ratio` | Call/put open-interest sentiment ratio |
| `options_volume_ratio` | Options volume vs 30-day average |

**Tier 3b — Implied Volatility** (NVDA + PLTR: active; APP + CRDO + INOD: pending backfill):

| Feature | Description | Status |
|---------|-------------|--------|
| `iv_30d` | Front-month OI-weighted IV (20–45 DTE) — overall vol regime | NVDA+PLTR active; APP+CRDO+INOD pending |
| `iv_skew_30d` | Put IV minus call IV (30d) — fear premium | same |
| `iv_term_ratio` | iv_7d / iv_30d — short-dated vol relative to 30d | same |

APP / CRDO / INOD already use `cp_sentiment_ratio` and `options_volume_ratio` (Tier 3a) in
the Transformer. Tier 3b IV features are 0.0-filled for those symbols until backfill
completes; enable per M6 migration note (Section 15, NFR1).

**Approximate feature counts per symbol**:
- NVDA: ~64 (57 Tier1 + 7 semiconductor + 5 options/IV)
- CRDO: ~64 (57 Tier1 + 7 semiconductor + 2 cp_ratio; IV pending)
- PLTR: ~63 (57 Tier1 + 6 defense + 5 options/IV — exact count pending PLTR_param.py audit)
- APP: ~63 (57 Tier1 + 4 gaming/social + 2 cp_ratio; IV pending)
- INOD: ~60 (57 Tier1 + 1 XLK + 2 cp_ratio; IV pending)

Log `D = len(feature_list)` in the run manifest (FR8) for every training run.

### 6.2 Output Schema (locked)

`{SYMBOL}_gbdt_15d_from_today_predictions.csv`:

```
date,        h,  P_down, P_flat, P_up,  edge,   score,  score_max, signal, model_type
2026-03-02,  1,  0.22,   0.61,   0.17, -0.05,   0.41,   0.35,      0,      lgbm
2026-03-02,  2,  0.18,   0.58,   0.24,  0.06,   0.41,   0.35,      0,      lgbm
...
2026-03-02, 15,  0.15,   0.51,   0.34,  0.19,   0.41,   0.35,      1,      lgbm
```

**Column legend**:

| Column | Per-row or per-date | Definition |
|--------|---------------------|------------|
| `date` | per-date | Inference date (market close) |
| `h` | per-row | Horizon (1–15 trading days) |
| `P_down` | per-row | Calibrated P(down) for horizon h |
| `P_flat` | per-row | Calibrated P(flat) for horizon h |
| `P_up` | per-row | Calibrated P(up) for horizon h |
| `edge` | per-row | `P_up - P_down` for horizon h |
| `score` | per-date (repeated) | Weighted-sum edge over h=6..15 (Section 12.2); primary ranking score |
| `score_max` | per-date (repeated) | `max(edge_h for h=6..15)`; alternative ranking score for comparison |
| `signal` | per-date (repeated) | `1` if `score > θ_symbol`, else `0`; trade filter |
| `model_type` | per-date (repeated) | Always `lgbm`; enables join with Transformer rows on `(symbol, date, h)` |

`score`, `score_max`, and `signal` are date-level scalars repeated across all 15 horizon
rows for the same date. This keeps the file flat (no multi-index) and compatible with the
existing Transformer CSV format.

The schema for the existing Transformer output (`{SYMBOL}_15d_from_today_predictions.csv`)
is unchanged. Both files coexist in complement mode.

---

## 7. Model Design

### 7.1 Why LightGBM fits this system

- Tabular mixed-scale features: histogram-based splits are scale-invariant.
- Small training N (~570 rows): generalizes better than Transformers at this size.
- Missing values: natively handled (learns optimal split direction for NaN).
- Fast SHAP: TreeExplainer is 10–50× faster than gradient-based SHAP for neural nets.
- Class imbalance: explicit class_weight per class per horizon, no focal loss complexity.

### 7.2 Training configuration — 15 models (production default)

One LightGBM 3-class classifier per horizon `h=1..15`, trained independently.
Each model uses the same feature matrix `X` with a different label vector `y_h`.
Class weights are computed independently per horizon. Temperature scalar fit
independently per horizon. Total: 5 symbols × 15 horizons = 75 models.

**2-model bucket variant (M5 ablation only — Choice 1)**:

If implemented, one label per row using the **representative horizon** of the bucket:
- Bucket 1–5: use `h=5` label with `thr=0.03`
- Bucket 6–15: use `h=10` label with `thr=0.05`

**Why Choice 2 (pooled rows) is excluded**: pooling 5 rows per date for bucket 1–5
means 5 rows share the same feature vector `X_t` but have correlated labels
(`r_{t,1}` is embedded in `r_{t,5}`). If cross-validation splits on rows instead of
dates, the same feature vector appears in both train and val folds — direct leakage.
Date-grouped splits fix the leakage but eliminate the apparent N inflation benefit.
Choice 1 is simpler and cleaner.

### 7.3 Artifact naming

All artifacts in `/workspace/model/`:

| Artifact | Filename |
|----------|----------|
| LightGBM model | `gbdt_{SYMBOL}_lgbm_reference_h{h:02d}.txt` |
| Temperature scalar | `gbdt_{SYMBOL}_lgbm_reference_h{h:02d}_temp.pkl` |
| Run manifest (JSON) | `gbdt_{SYMBOL}_lgbm_reference_manifest.json` |
| SHAP summary CSV | `gbdt_{SYMBOL}_lgbm_reference_shap.csv` |
| SHAP beeswarm PNG | `gbdt_{SYMBOL}_lgbm_reference_shap_h{h:02d}.png` |

Follows existing naming convention (`gbdt_` prefix mirrors `model_` prefix for
Transformer artifacts).

### 7.4 Hyperparameters — sized for ~570 training rows

The v1 PRD listed ranges calibrated for 10k+ row datasets. These are corrected here.

```python
LGBM_DEFAULTS = dict(
    objective             = 'multiclass',
    num_class             = 3,
    num_leaves            = 31,      # v1: 63-255 — too high for 570 rows (overfit)
    min_data_in_leaf      = 20,      # v1: 50-500 — 500 would force 1-leaf trees
    max_depth             = 6,       # ceiling; leaf-wise growth via num_leaves
    learning_rate         = 0.05,
    n_estimators          = 300,     # v1: 500-4000 — rely on early stopping, not high N
    feature_fraction      = 0.7,
    bagging_fraction      = 0.8,
    bagging_freq          = 5,
    lambda_l1             = 0.1,
    lambda_l2             = 1.0,
    early_stopping_rounds = 30,      # expect actual stopping at 80-200 iterations
    verbose               = -1,
    random_state          = 42,
)
```

Expected behavior: training stops at 80–200 iterations per model (~0.5–2s per model
on CPU). Total training time for 75 models: ~2–5 minutes.

Hyperparameter search space for optional Optuna tuning (NVDA first, then apply to
other symbols):

| Parameter | Search range |
|-----------|-------------|
| `num_leaves` | 15–63 |
| `min_data_in_leaf` | 10–50 |
| `learning_rate` | 0.02–0.10 |
| `lambda_l1` | 0.01–1.0 |
| `lambda_l2` | 0.1–10.0 |
| `feature_fraction` | 0.5–0.9 |

`n_estimators` is always determined by early stopping — never tuned as a fixed value.

---

## 8. Class Imbalance Handling

### 8.1 Default: per-horizon class weights

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_train_h)
class_weight = {0: weights[0], 1: weights[1], 2: weights[2]}
# Pass to LightGBM: lgb.train(..., params={..., 'class_weight': class_weight})
```

Do **not** use LightGBM's `is_unbalance=True` — it ignores per-class frequency
differences and is not suitable for 3-class with horizon-varying distributions.

Expected weights for h=5, NVDA-like distribution (Up≈20%, Flat≈62%, Down≈18%):
- `w_up ≈ 2.5×`, `w_down ≈ 2.8×`, `w_flat ≈ 0.8×`

### 8.2 UP class collapse guard

After training each model, check: if Up-class F1 on val set is 0, automatically:
1. Double the computed `w_up` and `w_down`
2. Halve `min_data_in_leaf`
3. Retrain once
4. Log `[IMBALANCE_ALERT]` regardless of outcome

### 8.3 Decision threshold tuning (required, not optional)

`argmax(P_calibrated)` will usually predict Flat due to residual imbalance. For
trading, the signal is `P(up)_h > θ_up`. Calibrate `θ_up` on the validation set:

```python
# Find smallest θ in [0.40, 0.90] such that:
#   Precision(Up | P(up) > θ) >= 0.50  (target precision for trade entry)
#   AND coverage >= 0.05  (at least 5% of val days trigger a signal)
# Fallback: θ = 0.65 with [LOW_COVERAGE_WARNING] if no θ satisfies both conditions
```

`θ_up` is saved per horizon in the run manifest and used at inference to set `signal`.

---

## 9. Calibration

### 9.1 Temperature scaling (default)

Temperature scaling fits one scalar `T` per model on the validation set:

```python
from scipy.optimize import minimize_scalar
from scipy.special import softmax

raw_logits_val = model.predict(X_val, raw_score=True)  # shape (N_val, 3)

def nll(T):
    probs = softmax(raw_logits_val / T, axis=1)
    return -np.mean(np.log(probs[np.arange(len(y_val)), y_val] + 1e-12))

result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
T_star = result.x
```

Expected range: `T_star ∈ [0.8, 2.5]`. Log and flag if `T_star < 0.8` (unusual
underconfidence — investigate model or label noise).

### 9.2 Isotonic regression (gated — will not trigger in v1)

Only consider isotonic if validation set has ≥ 300 samples **per class**. At
~195 val rows with Flat at 62%, Up and Down have ~38 and ~35 val samples
respectively. Isotonic is off the table for v1. Reserved for future use if data
grows.

### 9.3 Calibration quality metrics

Report per model:
- Brier score: uncalibrated vs calibrated (target: ≥ 5% reduction)
- Reliability diagram with 5 bins and Wilson 95% CI error bars (not 10 bins — too
  few points per bin at N=195)
- Expected Calibration Error (ECE): target < 0.08 post-calibration

---

## 10. Evaluation Plan

### 10.1 Offline metrics (per horizon h=1..15)
- Per-class F1 (Up / Flat / Down)
- Macro-F1
- Confusion matrix

Reported as: individual per-horizon table + bucket averages (h=1–5 and h=6–15).

### 10.2 Trading-aligned metrics (all required)

**Precision@K definition** (resolved from v1 ambiguity):

```
For each test day d:
  S_d = {symbols where signal=1}, ranked by score descending
  picks_d = top-min(|S_d|, K) symbols from S_d

  If |S_d| == 0: skip day — contributes 0 to numerator and 0 to denominator.

Precision@K = Σ_d hits(picks_d) / Σ_d |picks_d|

where hits(picks_d) = count of symbols in picks_d where r_{t,10} > 5%
```

Key edge cases:
- **0 signals on day d** → no trade; day excluded from P@K entirely. Do not force
  a trade just to fill K slots (that would require picking a `signal=0` symbol, which
  means no model agreement on direction).
- **|S_d| = 1, K = 2** → only 1 pick is made that day. The denominator increments
  by 1 (not 2). P@2 degrades gracefully to P@1 coverage on those days.
- **|S_d| ≥ K** → normal case: pick top-K by score.

**Coverage** (log alongside P@K):
```
coverage = (days with |S_d| >= 1) / total_test_days
```
If `coverage < 0.10`, P@K has < 10% of test days contributing picks — flag
`[LOW_COVERAGE_WARNING]` and report P@K with caveat (small N, high variance).

Practical baselines:
- **Random P@1 baseline**: 20% (1/5 symbols, uniform) — valid when coverage ≈ 100%
- **Always-Up P@1 baseline**: measured empirically on test period realized returns

**Down-class precision** (required in all regimes):
```
Down-precision@h = (predicted Down AND realized r_{t,h} < -thr(h)) / (predicted Down)
```
Report at h=5 and h=10. Target: ≥ 35%. Fallback acceptance path gate (Section 15).

**Rank IC**: Spearman correlation between `score` (weighted-sum edge, Section 12.2)
and `r_{t,10}` across all symbols × test dates. Target: IC > 0.05.

**Post-cost backtest (required in M3)**:

```python
cost_bps = 7  # one-way; 14bps round-trip
# Long if signal=1; hold for h=10 days; return = r_{t,10} - 14bps
# Sharpe = annualized mean(returns) / std(returns)
```

Report: daily Sharpe, max drawdown, turnover (% of days with a trade).

### 10.3 Baselines (all required before M2 gate)

| Baseline | Description | Key metric |
|----------|-------------|------------|
| Always-Flat | Predict Flat for every row | Sets macro-F1 floor (~0.25) |
| Prevalence-matched random | Sample from training class frequencies | Reference random |
| Momentum sign | Up if `price_change_5 > 0`, Down if < 0, else Flat | Simple heuristic |
| **Always-Up** | Predict Up for every row | **Sets Precision@1 ceiling in bull-market test period** |

The Always-Up baseline is critical: the test period (2024-04–2026-03) includes a
sustained bull run for NVDA, PLTR, and APP. A model without real UP predictive
signal can achieve high Precision@1 just by predicting Up constantly.

**Always-Up P@1 interpretation by regime**:

Always-Up P@1 is a pure function of realized returns in the test period — it measures
the difficulty of the environment, not the model. Three regimes and what they imply:

| Always-Up P@1 | Regime | Interpretation |
|---------------|--------|----------------|
| < 40% | Bear / choppy | Easy bar to beat. Model passes if P@1 > AU P@1. |
| 40–60% | Mixed | Standard bar. Model passes if P@1 > AU P@1. |
| > 60% | Bull-dominated | **Fallback path applies** (see below). |

**Fallback acceptance path when Always-Up P@1 > 60%**:

In a regime where > 60% of days end up > 5% for 10d, long-only Precision@1 is
regime-driven and not a valid differentiator. Apply two alternative criteria instead:

1. **Normalized precision**: `P@1_model - P@1_always_up >= -0.05`
   (model is within 5pp of always-up). This confirms the model is not *hurting*
   by filtering, and is capturing a high-quality subset of an already-high-hit-rate
   universe.

2. **Down-class precision** (required): when model predicts Down for h=10, the
   realized return must be `< -5%` in at least **35%** of cases. Always-Up cannot
   satisfy this criterion at all (it never predicts Down). A model with genuine
   directional signal should identify 35%+ of Down predictions correctly even in a
   bull market.

Both criteria 1 and 2 must hold for the fallback path to pass. If only criterion
1 holds, the model is riding the regime but not adding genuine two-sided signal.

**Down-class precision** is reported for every evaluation run regardless of regime,
as it captures signal that Precision@K (long-only) cannot.

### 10.4 SHAP Leakage Checklist (M4 deliverable)

For the top-10 SHAP features per symbol (by mean |SHAP| across h=5 and h=10):

| Feature | Source | Publication lag | As-of timestamp | Leakage risk |
|---------|--------|-----------------|-----------------|--------------|
| ... | ... | ... | end-of-day `t` or later? | None / Investigate / Block |

Known risks to pre-audit:
1. **AAII Spread**: published Thursdays post-close. Monday–Wednesday rows must use
   prior Thursday's value, not same-week Thursday.
2. **iv_30d / iv_skew_30d / iv_term_ratio**: Polygon end-of-day snapshot. Confirmed
   available at market close on date `t`. Safe.
3. **DTE / DSE**: pre-announced earnings dates. No look-ahead. Safe.
4. **eps_est_avg**: AV consensus estimate — verify it uses estimates available before
   earnings announcement, not the actual EPS from the report date.

Features flagged "Block": removed, model retrained, F1 delta recorded. If F1 drops
> 0.02, flag for manual review (may indicate a signal, not leakage). If F1 drops
< 0.005, block confirmed — the feature carried leakage, not signal.

---

## 11. Training / Validation Protocol

### 11.1 Fixed calendar splits

Splits are defined as calendar dates (not percentages) for regime interpretability.
Consistent with existing `training_set_size: 570, validation_set_size: 195` in param
files.

| Split | Calendar dates | Approx rows |
|-------|---------------|-------------|
| Train | `start_date` → 2023-07-15 | ~570 |
| Val | 2023-07-16 → 2024-04-01 | ~195 |
| Test | 2024-04-02 → today | ~475 |

`start_date` by symbol:
- NVDA / CRDO / APP / INOD: 2021-03-01
- PLTR: 2021-09-30 (IPO constraint; ~100 fewer training rows)

### 11.2 Walk-forward protocol

**M2 smoke test — 2 folds** (moved from M6 in v1):

| Fold | Train end | Val block | Test block |
|------|-----------|-----------|------------|
| 1 | 2022-12-31 | 2023-01–06 | 2023-07–12 |
| 2 | 2023-06-30 | 2023-07–12 | 2024-01–06 |

M2 gate: if macro-F1 (test) < 0.20 in both folds for any symbol, M3 is blocked
pending investigation (leakage audit, feature review).

**M6 full walk-forward — 6 folds**:

Starting from 2022-01-01 train-end, advance by 6-month expanding windows through
2025-06-30. Expect total training time < 30 minutes (75 models × 6 folds = 450
model fits at ~0.5–2s each).

---

## 12. Inference and Ranking

### 12.1 Daily inference procedure

For each symbol at market close:
1. Load today's feature row from `{SYMBOL}_TMP.csv` (same file Transformer uses)
2. Apply feature drift check: if any feature > 4σ from training mean, log
   `[DRIFT_ALERT]` but continue
3. Run `model_h.predict(X_today)` for h=1..15
4. Apply temperature scaling: `probs_h = softmax(logits_h / T_h_star)`
5. Compute `edge_h`, `score`, `signal` (Section 12.2–12.3)
6. Write to `{SYMBOL}_gbdt_15d_from_today_predictions.csv`

### 12.2 Ranking score

```python
# post-calibration probabilities
edge_h = P_up_h - P_down_h       # ∈ [-1, 1] for each h

# Weighted sum over long-horizon range (h=6..15)
w = {h: 1 + (h - 6) / 9 for h in range(6, 16)}   # w_6=1.0, w_15=2.0 (linear ramp)
raw_score = sum(w[h] * edge_h[h] for h in range(6, 16)) / sum(w.values())
score = (raw_score + 1) / 2      # normalize to [0, 1]
```

Rationale for h=6..15 weighting: primary use case is 1–3 week holds. Short horizons
(h=1..5) are included in the CSV for diagnostics but do not feed the ranking score.

Alternative: `score = max(edge_h for h=6..15)` — available as a column
`score_max` in the output CSV for comparison.

### 12.3 Trade filter

```
signal = 1 if score > θ_symbol else 0
```

`θ_symbol` is calibrated on the val set per symbol (Section 8.3) and stored in the
run manifest. Default fallback: `θ = 0.65`. The `signal` column in the output CSV
reflects the calibrated threshold.

---

## 13. NFR — Non-Functional Requirements

**NFR1 — No leakage + IV fill consistency**

Time-based splits only. No row shuffling before splitting.

**IV fill strategy**: the existing cp_ratio_backfill.py uses `fillna(0.0)` for
missing IV dates. In v1, this 0.0 fill is preserved in both training and inference
for consistency. If inference encounters a gap day (IV not yet available at run
time), it must apply the same `fillna(0.0)` — not leave NaN. LightGBM will see 0.0
and treat it identically to training rows with 0.0 IV.

**Migration to NaN (M6)**: when cp_ratio_backfill.py is updated to preserve NaN,
the GBDT training data must be regenerated with NaN values so training and inference
remain consistent. LightGBM handles NaN natively and correctly — this is the correct
long-term strategy. Do not switch in v1 mid-flight.

**NFR2 — Runtime**

- Training: ≤ 10 minutes for all 5 symbols × 15 horizons (75 models) on CPU
- Inference: ≤ 30 seconds for all 5 symbols × 15 horizons

**NFR3 — Maintainability**

Each symbol gets a `{SYMBOL}_gbdt_param.py`. Example skeleton for NVDA:

```python
# NVDA_gbdt_param.py

lgbm_reference = {
    # ── Shared with Transformer param (kept) ──────────────────────────
    "symbol":       "NVDA",
    "model_name":   "lgbm_reference",
    "start_date":   "2021-07-01",      # earliest training row
    "comment":      "NVDA GBDT — Tier1 + semiconductor sector + IV",
    "threshold":    0.05,              # labeling threshold thr(h)

    "selected_columns": [              # Tier 1 + Tier 2 + Tier 3 (Section 6.1)
        # ... feature names ...
    ],

    # ── GBDT-specific: calendar splits (replaces training_set_size) ───
    "train_end_date": "2024-03-31",    # train: start_date → train_end_date
    "val_end_date":   "2024-09-30",    # val:   train_end_date+1 → val_end_date
    # test window: val_end_date+1 → present (open-ended)

    # ── GBDT-specific: LightGBM hyperparameter overrides ─────────────
    "lgbm_params": {
        # symbol-specific overrides of LGBM_DEFAULTS (Section 7.4)
        # leave empty dict {} to use all defaults unchanged
    },

    # ── GBDT-specific: calibration method ────────────────────────────
    "calibration": "temperature",      # Section 9.1; "isotonic" reserved for M6+
}

# Identical to lgbm_reference but Tier 3b IV columns removed (0.0 fill risk)
lgbm_reference_base = {**lgbm_reference,
    "model_name": "lgbm_reference_base",
    "selected_columns": [  # Tier 1 + Tier 2 + Tier 3a (no iv_30d/skew/term_ratio)
        # ...
    ],
}
```

**Keys present in Transformer `{SYMBOL}_param.py` that are EXCLUDED from GBDT param**:

| Transformer key | Why excluded |
|-----------------|--------------|
| `window_size` | Transformer uses a 35-day input sequence; GBDT uses a single feature row |
| `target_size`, `num_zones` | Transformer output config; GBDT outputs per-horizon models |
| `batch_size`, `shuffle` | DataLoader concepts; GBDT trains on full dataset at once |
| `shuffle_splits`, `use_time_split` | Replaced by explicit `train_end_date` / `val_end_date` |
| `headcount`, `num_layers`, `dropout_rate`, `embedded_dim` | Transformer architecture params |
| `num_epochs` | Replaced by `n_estimators` + `early_stopping_rounds` in `lgbm_params` |
| `down_weight` | Transformer-specific scalar; replaced by `compute_class_weight('balanced')` |
| `scaler_type` | MinMax/Robust scaling; GBDT splits are scale-invariant (Section 7.1) |
| `training_set_size`, `validation_set_size` | Fixed-size Transformer splits; replaced by date-based splits |
| `l1_lambda`, `l2_weight_decay` | Transformer optimizer regularization; replaced by `lambda_l1`/`lambda_l2` in `lgbm_params` |
| `volatility_window`, `bband_time_period` | Feature computation params for TMP.csv generation; irrelevant here since GBDT reads pre-computed TMP.csv |
| `current_estEPS` | Runtime override used by Transformer inference to patch a single feature value; GBDT reads the actual value from TMP.csv |

Callable from: `mainDeltafromToday.main(NVDA_gbdt_param.lgbm_reference, model_type='lgbm')`

**NFR4 — Observability (required, not optional)**

- Log input data checksum (sha256 of `{SYMBOL}_TMP.csv`) at each training run
- Log prediction class distribution at inference (% Up / Flat / Down per symbol ×
  horizon)
- Feature drift check at inference: flag `[DRIFT_ALERT]` if any feature > 4σ from
  training mean. Common triggers: IV gap day (iv_30d = 0 when training mean ≠ 0),
  stale AAII data, missing macro release. Warn and continue — do not fail inference.

---

## 14. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Overfit at small N (~570 rows) | Dataset-appropriate hyperparams (Section 7.4); 2-fold smoke test gates M3 |
| UP class collapse | Per-horizon class weights + imbalance guard (Section 8.2); threshold tuning (Section 8.3) |
| Bull-market test bias (2024–2026) | Always-Up baseline required (Section 10.3); Rank IC as a bull-regime-robust metric |
| IV fill inconsistency (train 0.0 vs inference NaN) | Inference must apply same fillna(0.0) for IV gap days; documented in NFR1 |
| Option B pooled-rows leakage | Choice 2 excluded entirely; Choice 1 (single label per row) is the only bucket option |
| SHAP leakage signature | Leakage checklist in M4; block-and-retrain protocol |
| Calibration overfitting on small val set | Temperature scaling (1 parameter) is robust at N=195; isotonic is gated to N≥300/class |
| Inference crash on missing GBDT models | nightly_run.py gracefully skips GBDT inference if models not found; logs warning |

---

## 15. Acceptance Criteria (Quantified)

A training run is **accepted** and eligible for nightly deployment if all of the
following hold on the test set (or most recent walk-forward fold):

| Criterion | Minimum bar | Notes |
|-----------|-------------|-------|
| Up-class F1 (avg h=1..15) | ≥ 0.25 | Must detect Up events; F1=0 is unacceptable |
| Down-class F1 (avg h=1..15) | ≥ 0.20 | Directional; not just trend-following |
| Macro-F1 (avg h=1..15) | ≥ 0.32 | vs Always-Flat floor ≈ 0.25 |
| Precision@1 (h=10) — **primary path** | > Always-Up P@1 | Valid when Always-Up P@1 ≤ 60% |
| Precision@1 (h=10) — **fallback path** | See below | Active when Always-Up P@1 > 60% |
| Down-class precision (h=10) | ≥ 35% | Required in all regimes; fallback path gate |
| Brier reduction vs uncalibrated | ≥ 5% | Confirms calibration adds value |
| SHAP leakage audit | Passed (top-5 features all have valid as-of timestamps) | Required gate for production |

**Fallback path detail** (applies per-symbol when Always-Up P@1 > 60%):

When the test period is bull-dominated, the primary "beat Always-Up" bar is not
meaningful as a differentiator. The fallback path replaces the primary P@1 criterion
with both of:

1. `P@1_model - P@1_always_up >= -0.05` (within 5pp of always-up): model is not
   destroying precision by filtering — the curated subset is nearly as high-quality.
2. Down-class precision ≥ 35%: model has genuine two-sided signal that Always-Up
   fundamentally cannot provide.

If only criterion 1 holds and criterion 2 fails, the model is regime-riding with
no real Down signal. **Do not deploy** — retrain with stronger Down-class weighting
(Section 8.1) and recheck.

Always report Always-Up P@1 measured on the test period alongside model P@1, regardless
of which acceptance path is active.

On failure:
1. Log failure with metric values
2. Do **not** upload GBDT predictions to GCS
3. Continue uploading Transformer predictions (complement mode provides failover)
4. Root-cause before next training run

---

## 16. Implementation Milestones

### M1 — Pipeline skeleton + feature validation
- Data loader: reads `{SYMBOL}_TMP.csv`, selects feature columns
- Label generator: produces 15 label vectors; outputs class distribution table
  confirming ≥ 50 Up/Down per horizon per symbol
- Chronological splitter: calendar date splits (Section 11.1)
- Four baselines computed (All-Flat, All-Up, random, momentum-sign)
- `{SYMBOL}_gbdt_param.py` files created for all 5 symbols

### M2 — LightGBM 15-model + 2-fold walk-forward smoke test
- Train 75 models using LGBM_DEFAULTS (Section 7.4)
- Per-horizon class weights (Section 8.1)
- Early stopping on val set
- Test-set per-class F1 and macro-F1
- 2-fold walk-forward smoke test; gate M3 on result
- **M2 gate**: macro-F1 > 0.20 AND Up-class F1 > 0 in both smoke-test folds for
  all 5 symbols

### M3 — Calibration + trading metrics + post-cost backtest
- Temperature scaling per horizon (Section 9.1)
- Decision threshold `θ` calibration on val set (Section 8.3)
- Precision@1/2 with all 4 baselines including Always-Up
- Rank IC
- Post-cost backtest: Sharpe, drawdown, turnover (Section 10.2)
- Reliability diagrams (5-bin) + ECE

### M4 — SHAP analysis + leakage audit
- TreeExplainer SHAP on test set for all 5 symbols
- Top-20 feature importance per symbol (mean |SHAP| across h=5 and h=10)
- Leakage checklist table with block-and-retrain results (Section 10.4)
- Recompute M3 trading metrics after any blocked features

### M4.5 — nightly_run.py integration
- Add `model_type='lgbm'` calls to Phase 2 (training) for all 5 symbols
- Add `model_type='lgbm'` calls to Phase 1 (inference) for all 5 symbols
- Write `{SYMBOL}_gbdt_15d_from_today_predictions.csv`
- Update HTML report generator to render GBDT section alongside Transformer
- Upload GBDT artifacts to GCS alongside existing `.pth` models

### M5 — 2-model bucket variant + CatBoost comparison
- Implement 2-model bucket using Choice 1 label construction (Section 7.2)
- Implement CatBoost backend with same 15-model pipeline
- Ablation table: 15-model LightGBM vs 2-model LightGBM vs 15-model CatBoost

### M6 — Full walk-forward + production hardening
- 6-fold walk-forward evaluation (Section 11.2)
- Migrate IV fill: 0.0 → NaN (regenerate backfill, retrain, verify NFR1)
- Auto-compare new training run vs previous run; block deploy if acceptance
  criteria not met
- Drift monitoring: feature drift alerts in production inference (NFR4)

---

*Total models per full production run: 5 symbols × 15 horizons = 75 LightGBM models.*
*Estimated training time: 5–10 minutes on CPU. Inference: < 30 seconds.*
