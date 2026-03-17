# EDD — Multi-Horizon GBDT Stock Classifier

**Version**: 1.0
**Date**: 2026-03-03
**PRD Reference**: PRD_GBDT_multi_horizon_stock_classifier_v2.md
**Status**: Draft for review

---

## 1. Scope

This document translates every PRD v2 requirement into implementation-level
decisions: file layout, module interfaces, data flow, function signatures, error
handling, and test plan. It is the single reference for coding M1–M6.

Out of scope: Transformer model code, GCS upload logic (reused unchanged),
backfill scripts.

---

## 2. File Layout

### 2.1 New files

```
/workspace/
├── gbdt_pipeline.py              # all GBDT logic (single module for v1)
├── NVDA_gbdt_param.py
├── CRDO_gbdt_param.py
├── PLTR_gbdt_param.py
├── APP_gbdt_param.py
└── INOD_gbdt_param.py
```

### 2.2 Modified files

```
/workspace/
├── mainDeltafromToday.py         # add model_type='lgbm' branch in main()
└── nightly_run.py                # add lgbm Phase 1 (infer) + Phase 2 (train) calls
```

### 2.3 Artifacts written at runtime

```
/workspace/
└── {SYM}_gbdt_15d_from_today_predictions.csv   # one per symbol

/workspace/model/
├── gbdt_{SYM}_{model_name}_h{h:02d}.txt            # LightGBM text model
├── gbdt_{SYM}_{model_name}_h{h:02d}_temp.pkl       # temperature scalar (float)
├── gbdt_{SYM}_{model_name}_manifest.json            # full run record
├── gbdt_{SYM}_{model_name}_shap.csv                 # mean |SHAP| per feature × h
└── gbdt_{SYM}_{model_name}_shap_h{h:02d}.png        # beeswarm per horizon
```

---

## 3. Data Flow

```
{SYM}_TMP.csv
      │
      ▼
load_features(param)          → DataFrame [N_total × D]  (D = len(selected_columns)-1)
      │
      ▼
generate_labels(df, param)    → labels[h] = np.ndarray[N_total - h]   (h=1..15)
      │
      ▼
make_splits(df, labels, param)
      ├── X_train, y_train[h]  (dates: start_date → train_end_date)
      ├── X_val,   y_val[h]    (dates: train_end_date+1 → val_end_date)
      └── X_test,  y_test[h]   (dates: val_end_date+1 → today)
              │
              ▼  (loop h=1..15)
      train_one_horizon(X_tr, y_tr[h], X_val, y_val[h], param)
              ├── compute_class_weights(y_tr[h])
              ├── lgb.train(..., early_stopping)
              ├── [collapse guard: retrain if Up-F1_val == 0]
              └── → model_h, train_metrics_h
              │
              ▼
      fit_temperature(model_h, X_val, y_val[h])   → T_h
              │
              ▼
      calibrate_threshold(model_h, T_h, X_val, y_val[h])  → θ_h
              │
              ▼
      evaluate_horizon(model_h, T_h, θ_h, X_test, y_test[h], df_test)
              └── → eval_metrics_h
              │
              ▼  (after all 15 horizons)
      compute_trading_metrics(all_preds, df_test, param)
              └── → trading_metrics  (P@K, IC, down_precision, backtest)
              │
              ▼
      run_baselines(y_test, df_test)  → baseline_metrics
              │
              ▼
      check_acceptance(eval_metrics, trading_metrics, baseline_metrics)
              └── → passed: bool
              │
      ┌───────┴───────┐
      │               │
    passed          failed
      │               │
      ▼               ▼
  run_shap(...)    log failure, skip GCS upload
  write_manifest(...)
```

---

## 4. Module Specification: `gbdt_pipeline.py`

### 4.1 Constants

```python
LGBM_DEFAULTS = dict(
    objective             = 'multiclass',
    num_class             = 3,
    num_leaves            = 31,
    min_data_in_leaf      = 20,
    max_depth             = 6,
    learning_rate         = 0.05,
    n_estimators          = 300,
    feature_fraction      = 0.7,
    bagging_fraction      = 0.8,
    bagging_freq          = 5,
    lambda_l1             = 0.1,
    lambda_l2             = 1.0,
    early_stopping_rounds = 30,
    verbose               = -1,
    random_state          = 42,
)

# Label encoding (consistent with Transformer)
LABEL_MAP  = {'down': 0, 'flat': 1, 'up': 2}
LABEL_IMAP = {0: 'down', 1: 'flat', 2: 'up'}

THR_SHORT = 0.03   # h <= 5
THR_LONG  = 0.05   # h >= 6
```

---

### 4.2 `load_features(param: dict) -> pd.DataFrame`

**Purpose**: Load `{SYMBOL}_TMP.csv`, select the feature columns defined in
`param['selected_columns']`, apply IV 0.0 fill.

**Behavior**:
- Read `/workspace/{symbol}_TMP.csv` with `parse_dates=['date']`, set `date` as index
- Drop the `label` column (present in selected_columns for Transformer compat but unused here)
- Assert all other columns in `selected_columns` exist; raise `ValueError` listing any missing
- Fill IV columns (`iv_30d`, `iv_skew_30d`, `iv_term_ratio`) with `0.0` where null
  (NFR1: consistent with training-time fill even if backfill gaps exist at inference)
- Log `sha256(file contents)` to stdout for NFR4 observability
- Return DataFrame indexed by date, columns = features only (no label)

**Error handling**:
- Missing TMP.csv → `FileNotFoundError` with message including symbol name
- Column mismatch → `ValueError` listing missing column names (mirrors existing
  Transformer KeyError handling in `fetchBulkData.py`)

---

### 4.3 `generate_labels(df: pd.DataFrame, param: dict) -> dict[int, pd.Series]`

**Purpose**: For each horizon h=1..15, compute the 3-class label for every row t
where close_{t+h} exists.

**Behavior**:
```python
col = 'adjusted close'
for h in range(1, 16):
    thr = THR_SHORT if h <= 5 else THR_LONG
    fwd_return = df[col].shift(-h) / df[col] - 1
    labels = pd.Series('flat', index=df.index)
    labels[fwd_return >  thr] = 'up'
    labels[fwd_return < -thr] = 'down'
    labels[fwd_return.isna()] = np.nan   # trailing rows with no future close
    labels_h[h] = labels.map(LABEL_MAP)  # encoded as 0/1/2; NaN rows dropped later
```

**Outputs**: `dict` mapping `h → pd.Series` (same index as df, NaN for tail rows).

**M1 deliverable**: after generating labels, print a class distribution table:
```
Symbol: NVDA
h   |  down  |  flat  |  up   | total
--- | ------ | ------ | ----- | -----
1   |   112  |   390  |  108  |  610
...
```
Emit `[LOW_SAMPLE_ALERT]` if any (symbol, h) has Up or Down < 50 in train split.

---

### 4.4 `make_splits(df: pd.DataFrame, labels: dict, param: dict) -> tuple`

**Purpose**: Apply calendar date splits. Returns aligned (X, y) for train/val/test
per horizon.

**Behavior**:
```python
train_mask = df.index <= pd.Timestamp(param['train_end_date'])
val_mask   = (df.index > pd.Timestamp(param['train_end_date'])) & \
             (df.index <= pd.Timestamp(param['val_end_date']))
test_mask  = df.index > pd.Timestamp(param['val_end_date'])

X_train = df[train_mask].values.astype(np.float32)
X_val   = df[val_mask].values.astype(np.float32)
X_test  = df[test_mask].values.astype(np.float32)

# For each horizon, drop rows where label is NaN (trailing rows)
for h in range(1, 16):
    valid = labels[h].notna()
    y_train[h] = labels[h][train_mask & valid].values.astype(int)
    y_val[h]   = labels[h][val_mask   & valid].values.astype(int)
    y_test[h]  = labels[h][test_mask  & valid].values.astype(int)
    # Note: X_train rows must also be masked by valid for each h:
    X_train_h[h] = df[train_mask & valid].values.astype(np.float32)
    # (val and test: trailing rows are far from boundary, so valid ≈ all rows)
```

**Assertion**: no date appears in more than one split (checked on index overlap).
Raise `AssertionError` if violated — this would be a leakage bug.

**Returns**: `(X_splits, y_splits, date_splits)` where each is a dict keyed by h
and by split name ('train'/'val'/'test').

---

### 4.5 `compute_class_weights(y: np.ndarray) -> dict`

**Purpose**: Balanced per-class weights for LightGBM.

```python
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y)
return {0: float(weights[0]), 1: float(weights[1]), 2: float(weights[2])}
```

Called once per horizon inside `train_one_horizon`.

---

### 4.6 `train_one_horizon(X_train, y_train, X_val, y_val, param, h, feature_names) -> tuple`

**Purpose**: Train one LightGBM 3-class classifier for horizon h.

**Behavior**:
```python
params = {**LGBM_DEFAULTS, **param.get('lgbm_params', {})}
class_weight = compute_class_weights(y_train)
params['class_weight'] = class_weight

train_ds = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
val_ds   = lgb.Dataset(X_val,   label=y_val,   reference=train_ds)

callbacks = [lgb.early_stopping(params['early_stopping_rounds'], verbose=False),
             lgb.log_evaluation(-1)]

model = lgb.train(
    params, train_ds,
    num_boost_round = params['n_estimators'],
    valid_sets      = [val_ds],
    callbacks       = callbacks,
)
```

**Collapse guard** (Section 8.2):
```python
val_preds = model.predict(X_val).argmax(axis=1)
up_f1_val = f1_score(y_val, val_preds, labels=[2], average='macro', zero_division=0)

if up_f1_val == 0:
    params['class_weight'][0] *= 2   # down
    params['class_weight'][2] *= 2   # up
    params['min_data_in_leaf'] = max(5, params['min_data_in_leaf'] // 2)
    model = lgb.train(params, train_ds, num_boost_round=params['n_estimators'],
                      valid_sets=[val_ds], callbacks=callbacks)
    logging.warning(f'[IMBALANCE_ALERT] h={h}: Up-F1=0 on val, retrained with 2× weights')
```

**Returns**: `(model: lgb.Booster, metrics: dict)`
- `metrics` contains: `n_iterations`, `best_val_score`, `up_f1_val`, `class_weights`

---

### 4.7 `fit_temperature(model, X_val, y_val) -> float`

**Purpose**: Temperature scaling (Section 9.1).

```python
from scipy.optimize import minimize_scalar
from scipy.special import softmax

raw_logits = model.predict(X_val, raw_score=True)   # shape (N_val, 3)

def nll(T):
    probs = softmax(raw_logits / T, axis=1)
    return -np.mean(np.log(probs[np.arange(len(y_val)), y_val] + 1e-12))

result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
T_star = float(result.x)

if T_star < 0.8:
    logging.warning(f'[CALIBRATION_FLAG] T_star={T_star:.3f} < 0.8 — investigate model confidence')

return T_star
```

**Saved as**: `model/gbdt_{SYM}_{model_name}_h{h:02d}_temp.pkl` via `pickle.dump(T_star, f)`.

---

### 4.8 `calibrate_threshold(model, T, X_val, y_val, h) -> float`

**Purpose**: Find θ such that Up-class precision ≥ 0.50 with coverage ≥ 0.05 on val
(Section 8.3).

```python
raw_logits = model.predict(X_val, raw_score=True)
probs = softmax(raw_logits / T, axis=1)
p_up = probs[:, 2]

best_θ = 0.65   # fallback
for θ in np.linspace(0.40, 0.90, 51):
    mask = p_up > θ
    coverage = mask.mean()
    if coverage < 0.05:
        continue
    precision = (y_val[mask] == 2).mean()
    if precision >= 0.50:
        best_θ = float(θ)
        break
else:
    logging.warning(f'[LOW_COVERAGE_WARNING] h={h}: no θ achieves precision≥0.50 '
                    f'with coverage≥0.05. Using fallback θ=0.65')

return best_θ
```

---

### 4.9 `evaluate_horizon(model, T, θ, X_test, y_test, df_test_dates, h) -> dict`

**Purpose**: Full per-horizon metrics on test set.

**Returns dict**:
```python
{
    'h': h,
    'macro_f1':     float,
    'up_f1':        float,
    'down_f1':      float,
    'flat_f1':      float,
    'confusion':    np.ndarray (3×3),
    'brier_uncal':  float,
    'brier_cal':    float,
    'brier_reduction_pct': float,
    'ece':          float,          # 5-bin ECE
    'T_star':       float,
    'theta_up':     float,
    'down_precision': float,        # P(realized < -thr | predicted down)
}
```

**Down-class precision** (Section 10.2 + Section 15):
```python
down_preds = (p_up < (1 - θ)) & (probs[:, 0] > probs[:, 1])  # argmax=down
# Realized: actual label is 0 (down)
down_precision = (y_test[down_preds] == 0).mean() if down_preds.any() else np.nan
```

**Calibration metrics**:
- Brier: `sklearn.metrics.brier_score_loss` (one-vs-rest, averaged across classes)
- ECE: 5-bin equal-frequency bins on `p_up`, compare mean(p_up in bin) vs mean(y==2 in bin)

---

### 4.10 `compute_trading_metrics(all_preds: dict, df_full: pd.DataFrame, param: dict) -> dict`

**Purpose**: Precision@K, rank IC, down-class precision (aggregate), post-cost backtest.

**Inputs**:
- `all_preds`: `{h: DataFrame with columns [date, symbol, P_down, P_flat, P_up, edge, score, score_max, signal]}`
- `df_full`: raw prices DataFrame for computing realized returns

**Precision@K algorithm** (Section 10.2):
```python
results_p1, results_p2 = [], []
for date, group in test_preds.groupby('date'):
    S_d = group[group['signal'] == 1].sort_values('score', ascending=False)
    if len(S_d) == 0:
        continue   # no trade — skip day
    # Realized return for h=10 from this date
    for _, row in S_d.head(1).iterrows():   # P@1
        hit = realized_return(row['symbol'], date, h=10) > THR_LONG
        results_p1.append(int(hit))
    for _, row in S_d.head(2).iterrows():   # P@2
        hit = realized_return(row['symbol'], date, h=10) > THR_LONG
        results_p2.append(int(hit))

coverage = (len(results_p1)) / total_test_days
precision_at_1 = np.mean(results_p1) if results_p1 else np.nan
precision_at_2 = np.mean(results_p2) if results_p2 else np.nan
```

**Rank IC**:
```python
# Across all symbols × test dates, at h=10
rank_ic = spearmanr(scores_all, realized_returns_all).correlation
```

**Post-cost backtest** (Section 10.2):
```python
cost_bps = 7   # one-way
daily_returns = []
for date in test_dates:
    picks = top_K_signals(date, K=1)
    for sym in picks:
        r = realized_return(sym, date, h=10) - 14/10000  # round-trip / 10d
        daily_returns.append(r / 10)   # daily equivalent
sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
```

**Returns dict**: `{P@1, P@2, coverage, rank_IC, down_precision_agg, sharpe, max_dd, turnover, always_up_p1}`

---

### 4.11 `run_baselines(y_test, df_test, param) -> dict`

**Purpose**: Compute all 4 baselines (Section 10.3).

| Baseline | Implementation |
|----------|----------------|
| Always-Flat | `y_pred = np.ones_like(y_test)` (label 1) |
| Always-Up | `y_pred = np.full_like(y_test, 2)` |
| Prevalence-matched random | `np.random.choice([0,1,2], p=train_freq, size=N_test)` |
| Momentum-sign | `Up if price_change_5 > 0 else Down if < 0 else Flat` |

For Always-Up: also compute `always_up_p1` on the test period (fraction of test days
where 10d realized return > 5%) — this is the regime classifier for Section 15.

**Returns dict**: per-baseline macro-F1, Up-F1, Down-F1, plus `always_up_p1`.

---

### 4.12 `check_acceptance(eval_metrics, trading_metrics, baseline_metrics) -> bool`

**Purpose**: Apply Section 15 acceptance criteria.

```python
avg_up_f1   = np.mean([eval_metrics[h]['up_f1']   for h in range(1,16)])
avg_down_f1 = np.mean([eval_metrics[h]['down_f1'] for h in range(1,16)])
avg_macro   = np.mean([eval_metrics[h]['macro_f1'] for h in range(1,16)])
avg_brier_reduction = np.mean([eval_metrics[h]['brier_reduction_pct'] for h in range(1,16)])

always_up_p1 = baseline_metrics['always_up_p1']
model_p1     = trading_metrics['P@1']
down_prec    = trading_metrics['down_precision_agg']

# Primary / fallback path (Section 15)
if always_up_p1 <= 0.60:
    p1_ok = model_p1 > always_up_p1
else:
    p1_ok = (model_p1 - always_up_p1 >= -0.05) and (down_prec >= 0.35)

criteria = {
    'up_f1 >= 0.25':     avg_up_f1   >= 0.25,
    'down_f1 >= 0.20':   avg_down_f1 >= 0.20,
    'macro_f1 >= 0.32':  avg_macro   >= 0.32,
    'p1_criterion':      p1_ok,
    'down_prec >= 0.35': down_prec   >= 0.35,
    'brier_red >= 5%':   avg_brier_reduction >= 5.0,
}

passed = all(criteria.values())
if not passed:
    logging.error(f'[ACCEPTANCE_FAILED] {param["symbol"]} {param["model_name"]}')
    for k, v in criteria.items():
        logging.error(f'  {k}: {"PASS" if v else "FAIL"}')
return passed
```

SHAP leakage audit (final gate) is run manually in M4 — not automated in
`check_acceptance`. The manifest records `shap_audit_passed: null` until M4.

---

### 4.13 `run_shap(models, X_test, feature_names, param) -> None`

**Purpose**: SHAP beeswarm and feature importance CSV (M4).

```python
import shap

for h in [5, 10, 15]:   # representative horizons; full set optional
    explainer = shap.TreeExplainer(models[h])
    shap_vals = explainer.shap_values(X_test)   # list of 3 arrays (one per class)

    # Mean |SHAP| across classes for Up (index 2) — most relevant for trading
    mean_abs = np.abs(shap_vals[2]).mean(axis=0)
    shap_df = pd.DataFrame({'feature': feature_names, f'mean_abs_shap_h{h}': mean_abs})
    shap_df.sort_values(f'mean_abs_shap_h{h}', ascending=False, inplace=True)
    # Append to master SHAP CSV per symbol
    ...
    shap.summary_plot(shap_vals[2], X_test, feature_names=feature_names, show=False)
    plt.savefig(f'model/gbdt_{sym}_{model_name}_shap_h{h:02d}.png', dpi=150)
    plt.close()
```

---

### 4.14 `infer_today(models, Ts, thetas, param, df) -> pd.DataFrame`

**Purpose**: Daily inference — produce the output CSV row-set for today.

```python
today = df.index[-1]
X_today = df.iloc[[-1]].values.astype(np.float32)   # shape (1, D)

# Feature drift check
training_stats = load_manifest(param)['feature_stats']   # mean, std per feature
for i, feat in enumerate(feature_names):
    z = abs(X_today[0, i] - training_stats[feat]['mean']) / (training_stats[feat]['std'] + 1e-9)
    if z > 4:
        logging.warning(f'[DRIFT_ALERT] {feat}: z={z:.1f}')

rows = []
for h in range(1, 16):
    raw_logits = models[h].predict(X_today, raw_score=True)   # shape (1, 3)
    probs = softmax(raw_logits / Ts[h], axis=1)[0]
    edge = float(probs[2] - probs[0])
    rows.append({'date': today.date(), 'h': h,
                 'P_down': probs[0], 'P_flat': probs[1], 'P_up': probs[2],
                 'edge': edge})

df_out = pd.DataFrame(rows)

# Score and signal (date-level scalars, repeated across 15 rows)
edges = {r['h']: r['edge'] for _, r in df_out.iterrows()}
w = {h: 1 + (h - 6) / 9 for h in range(6, 16)}
raw_score = sum(w[h] * edges[h] for h in range(6, 16)) / sum(w.values())
score     = (raw_score + 1) / 2
score_max = max(edges[h] for h in range(6, 16))
signal    = int(score > thetas['symbol'])   # symbol-level θ from manifest

df_out['score']      = score
df_out['score_max']  = score_max
df_out['signal']     = signal
df_out['model_type'] = 'lgbm'
return df_out
```

**Log prediction class distribution** (NFR4):
```
[LGBM INFERENCE] NVDA today=2026-03-03
  h=1:  down=22% flat=61% up=17%
  ...
  h=15: down=15% flat=51% up=34%  ← signal=1
```

---

### 4.15 `write_manifest(param, models, Ts, thetas, eval_metrics, trading_metrics, feature_names) -> None`

**Purpose**: Save full run record to JSON (FR8).

```json
{
  "symbol": "NVDA",
  "model_name": "lgbm_reference",
  "run_timestamp": "2026-03-03T21:00:00",
  "data_sha256": "abc123...",
  "data_range": {"start": "2021-03-01", "end": "2026-03-02"},
  "feature_list": ["adjusted close", "daily_return", ...],
  "D": 64,
  "splits": {"train_end": "2024-03-31", "val_end": "2024-09-30"},
  "lgbm_defaults": { ... },
  "lgbm_overrides": {},
  "calibration": "temperature",
  "T_stars": {"1": 1.23, "2": 1.18, ...},
  "thetas": {"symbol": 0.68, "1": 0.65, "2": 0.67, ...},
  "always_up_p1": 0.64,
  "acceptance_path": "fallback",
  "acceptance_passed": true,
  "eval_metrics": { "1": {...}, ..., "15": {...} },
  "trading_metrics": { "P@1": 0.61, "P@2": 0.58, "coverage": 0.42, ... },
  "shap_audit_passed": null,
  "feature_stats": {"adjusted close": {"mean": 512.3, "std": 87.1}, ...}
}
```

---

### 4.16 Public entry points

```python
def train(param: dict) -> bool:
    """Full training pipeline. Returns True if acceptance passed."""
    df      = load_features(param)
    labels  = generate_labels(df, param)
    splits  = make_splits(df, labels, param)
    models, Ts, thetas, eval_metrics = {}, {}, {}, {}
    for h in range(1, 16):
        model, _ = train_one_horizon(..., h)
        T        = fit_temperature(model, ...)
        θ        = calibrate_threshold(model, T, ...)
        metrics  = evaluate_horizon(model, T, θ, ...)
        models[h], Ts[h], thetas[h], eval_metrics[h] = model, T, θ, metrics
        save_model(model, T, param, h)
    trading  = compute_trading_metrics(...)
    baselines= run_baselines(...)
    passed   = check_acceptance(eval_metrics, trading, baselines)
    if passed:
        run_shap(models, ...)
    write_manifest(...)
    return passed


def infer(param: dict) -> pd.DataFrame:
    """Daily inference. Returns prediction DataFrame."""
    df     = load_features(param)
    models = {h: load_model(param, h) for h in range(1, 16)}
    Ts     = {h: load_temperature(param, h) for h in range(1, 16)}
    thetas = load_manifest(param)['thetas']
    df_out = infer_today(models, Ts, thetas, param, df)
    write_predictions_csv(df_out, param)
    return df_out
```

---

## 5. Param File Schema (complete)

Each `{SYMBOL}_gbdt_param.py` exports two dicts: `lgbm_reference` and
`lgbm_reference_base` (no IV features — for symbols pending backfill).

```python
# Template — copy for each symbol, fill in symbol-specific values

lgbm_reference = {
    # ── Identity ──────────────────────────────────────────────────────
    "symbol":     "NVDA",
    "model_name": "lgbm_reference",
    "comment":    "NVDA GBDT reference — Tier1 + semiconductor + IV",

    # ── Labeling ──────────────────────────────────────────────────────
    "threshold": 0.05,         # passed to generate_labels(); THR_SHORT/LONG hardcoded

    # ── Feature list ──────────────────────────────────────────────────
    "selected_columns": [
        # MUST NOT include 'label'
        # Tier 1 core (all symbols)
        "adjusted close", "daily_return", "volume", "Volume_Oscillator",
        "volatility", "VWAP", "high", "low", "volume_volatility",
        "EPS", "estEPS", "surprisePercentage", "dte", "dse",
        "earn_in_5", "earn_in_10", "earn_in_20", "totalRevenue", "netIncome",
        "eps_est_avg",
        "ret_5d_rel_SPY", "ret_10d_rel_SPY",
        "rv_10d", "rv_20d", "rv_term_ratio", "vix_rv_ratio",
        "MACD_Signal", "MACD", "MACD_Hist", "ATR", "RSI",
        "Real Upper Band", "Real Middle Band", "Real Lower Band",
        "interest", "10year", "T10Y2Y", "M2SL", "DTWEXBGS",
        "DFEDTARU", "BOGMBBM", "eur_close",
        "SPY_close", "qqq_close", "VTWO_close", "SPY_stoch",
        "calc_spy_oscillator", "QQQ_stoch", "VTWO_stoch",
        "VIXCLS", "unemploy", "UMCSENT", "CPIAUCSL", "GDP",
        "BUSLOANS", "Spread",
        "day_of_week", "month",
        "price_lag_1", "price_lag_5", "price_lag_15",
        "price_change_1", "price_change_5", "price_change_15",
        # Tier 2 NVDA-specific (semiconductor sector)
        "ret_5d_rel_SMH", "ret_10d_rel_SMH",
        "rs_amd", "rs_amd_trend", "rs_intc_trend", "rs_avgo_trend", "rs_smh_trend",
        # Tier 3a (all symbols)
        "cp_sentiment_ratio", "options_volume_ratio",
        # Tier 3b (NVDA + PLTR only)
        "iv_30d", "iv_skew_30d", "iv_term_ratio",
    ],

    # ── Calendar splits ───────────────────────────────────────────────
    "start_date":     "2021-03-01",   # CRDO: "2022-01-27"; PLTR: "2021-09-30"
    "train_end_date": "2024-03-31",
    "val_end_date":   "2024-09-30",
    # test: 2024-10-01 → present (open-ended; uses all available rows in TMP.csv)

    # ── LightGBM overrides (empty = use LGBM_DEFAULTS) ───────────────
    "lgbm_params": {},

    # ── Calibration ───────────────────────────────────────────────────
    "calibration": "temperature",
}

# No IV features — use for APP / CRDO / INOD until backfill complete
lgbm_reference_base = {
    **lgbm_reference,
    "model_name": "lgbm_reference_base",
    "selected_columns": [
        c for c in lgbm_reference["selected_columns"]
        if c not in ("iv_30d", "iv_skew_30d", "iv_term_ratio")
    ],
}
```

**Per-symbol `start_date` and Tier 2 delta**:

| Symbol | `start_date` | Tier 2 additions vs template |
|--------|-------------|------------------------------|
| NVDA | 2021-03-01 | `ret_5d_rel_SMH`, `ret_10d_rel_SMH`, `rs_amd`, `rs_amd_trend`, `rs_intc_trend`, `rs_avgo_trend`, `rs_smh_trend` |
| CRDO | 2022-01-27 | same as NVDA (semiconductor peers); `jpy_close`, `twd_close` promoted from Tier 1 |
| PLTR | 2021-09-30 | `rs_ita`, `rs_igv`, `rs_ita_trend`, `FDEFX`, `ADEFNO`, `IPDCONGD` |
| APP | 2021-07-01 | `rs_gamr`, `rs_gamr_trend`, `rs_socl`, `rs_socl_trend` |
| INOD | 2021-07-01 | `XLK_close` |

---

## 6. Changes to `mainDeltafromToday.py`

Add an `lgbm` branch in `main()` after the existing `trans_mz` branch:

```python
# mainDeltafromToday.py  (addition only — no existing lines changed)

def main(param: dict, model_type: str = 'transformer', mode: str = 'train'):
    if model_type == 'transformer':
        # existing path — unchanged
        ...
    elif model_type == 'trans_mz':
        # existing path — unchanged
        ...
    elif model_type == 'lgbm':
        import gbdt_pipeline
        if mode == 'train':
            passed = gbdt_pipeline.train(param)
            return passed
        elif mode == 'infer':
            df_preds = gbdt_pipeline.infer(param)
            return df_preds
        else:
            raise ValueError(f'Unknown mode for lgbm: {mode}')
    else:
        raise ValueError(f'Unknown model_type: {model_type}')
```

No other changes to `mainDeltafromToday.py`.

---

## 7. Changes to `nightly_run.py`

### Phase 1 — Inference (after existing Transformer inference block)

```python
# After Phase 1 Transformer inference loop:
logging.info('=== GBDT INFERENCE ===')
for sym, gbdt_param_module in GBDT_SYMBOLS.items():
    param = gbdt_param_module.lgbm_reference
    try:
        mainDeltafromToday.main(param, model_type='lgbm', mode='infer')
    except FileNotFoundError:
        logging.warning(f'[GBDT] No trained models found for {sym} — skipping inference')
    except Exception as e:
        logging.error(f'[GBDT] Inference failed for {sym}: {e}')
```

### Phase 2 — Training (after existing Transformer training block)

```python
logging.info('=== GBDT TRAINING ===')
for sym, gbdt_param_module in GBDT_SYMBOLS.items():
    param = gbdt_param_module.lgbm_reference
    passed = mainDeltafromToday.main(param, model_type='lgbm', mode='train')
    if not passed:
        logging.error(f'[GBDT] Acceptance failed for {sym} — not uploading to GCS')
```

### Symbol registry

```python
# Top of nightly_run.py, alongside existing Transformer param imports:
import NVDA_gbdt_param, CRDO_gbdt_param, PLTR_gbdt_param, APP_gbdt_param, INOD_gbdt_param

GBDT_SYMBOLS = {
    'NVDA': NVDA_gbdt_param,
    'CRDO': CRDO_gbdt_param,
    'PLTR': PLTR_gbdt_param,
    'APP':  APP_gbdt_param,
    'INOD': INOD_gbdt_param,
}
```

### Skip flags

Both GBDT blocks are guarded by the existing `--skip-train` / `--skip-inference` flags —
no new flags needed.

---

## 8. Prediction CSV Writer

```python
def write_predictions_csv(df_out: pd.DataFrame, param: dict) -> None:
    sym  = param['symbol']
    path = f'/workspace/{sym}_gbdt_15d_from_today_predictions.csv'

    if os.path.exists(path):
        existing = pd.read_csv(path, parse_dates=['date'])
        date_str = str(df_out['date'].iloc[0])
        # Dedup: remove any existing rows for same (date, model_name)
        existing = existing[~(
            (existing['date'].astype(str) == date_str) &
            (existing['model_type'] == 'lgbm')
        )]
        combined = pd.concat([existing, df_out], ignore_index=True)
    else:
        combined = df_out

    combined.to_csv(path, index=False)
    logging.info(f'[GBDT] Written {len(df_out)} rows → {path}')
```

Uses the same deduplication pattern as `processDeltaFromTodayResults` in
`mainDeltafromToday.py` (matches on date + model_type, not just date).

---

## 9. Test Plan

All tests live in `/workspace/` and run with `python -m pytest /workspace/test_gbdt_*.py -v`.

### 9.1 `test_gbdt_labels.py`

| Test | What it checks |
|------|----------------|
| `test_label_thresholds_short` | h=3: return=+0.04 → up; return=+0.02 → flat; return=-0.04 → down |
| `test_label_thresholds_long` | h=8: return=+0.06 → up; return=+0.04 → flat; return=-0.06 → down |
| `test_label_encoding` | LABEL_MAP = {down:0, flat:1, up:2} |
| `test_label_tail_nans` | Last h rows of labels[h] are NaN |
| `test_label_lengths` | len(labels[h]) == N_total for all h (NaN for tail, not truncated) |

### 9.2 `test_gbdt_splits.py`

| Test | What it checks |
|------|----------------|
| `test_no_date_overlap` | train ∩ val == ∅, val ∩ test == ∅ |
| `test_chronological_order` | max(train dates) < min(val dates) < min(test dates) |
| `test_split_sizes` | train ≈ 570 rows, val ≈ 195 rows for NVDA |
| `test_no_shuffle` | rows in each split are in ascending date order |

### 9.3 `test_gbdt_calibration.py`

| Test | What it checks |
|------|----------------|
| `test_temperature_range` | T_star ∈ [0.1, 10.0] (optimizer bounds respected) |
| `test_temperature_reduces_nll` | NLL with T_star < NLL with T=1.0 on val set |
| `test_threshold_coverage` | If θ returned, coverage(val, θ) ≥ 0.05 |
| `test_threshold_fallback` | When no valid θ found, returns 0.65 and logs WARNING |

### 9.4 `test_gbdt_schema.py`

| Test | What it checks |
|------|----------------|
| `test_output_columns` | CSV has exactly: date, h, P_down, P_flat, P_up, edge, score, score_max, signal, model_type |
| `test_probs_sum_to_one` | P_down + P_flat + P_up ≈ 1.0 for every row (tolerance 1e-4) |
| `test_edge_definition` | edge == P_up - P_down for every row |
| `test_h_range` | h values are exactly {1, 2, ..., 15} for each date |
| `test_score_repeated` | score and score_max are identical for all 15 rows of the same date |
| `test_signal_binary` | signal ∈ {0, 1} |

### 9.5 Integration smoke test (NVDA only, h=10)

```python
# test_gbdt_integration.py
def test_nvda_h10_smoke():
    """Train NVDA lgbm_reference for h=10 only; assert M2 gate passes."""
    import NVDA_gbdt_param, gbdt_pipeline
    param = NVDA_gbdt_param.lgbm_reference
    df = gbdt_pipeline.load_features(param)
    labels = gbdt_pipeline.generate_labels(df, param)
    splits = gbdt_pipeline.make_splits(df, labels, param)
    model, metrics = gbdt_pipeline.train_one_horizon(
        splits['train']['X'], splits['train']['y'][10],
        splits['val']['X'],   splits['val']['y'][10],
        param, h=10, feature_names=list(df.columns)
    )
    assert metrics['up_f1_val'] > 0, 'Up-class F1 on val is 0 — collapse guard failed'
    assert metrics['macro_f1_val'] > 0.15
```

---

## 10. Milestone → Code Mapping

| Milestone | Deliverables | Files touched |
|-----------|-------------|---------------|
| **M1** | Data loader, label generator, splits, 4 baselines, class distribution report | `gbdt_pipeline.py` (§4.2–4.4, §4.11), all 5 `{SYM}_gbdt_param.py`, `test_gbdt_labels.py`, `test_gbdt_splits.py` |
| **M2** | 75 trained models, class weights, early stopping, collapse guard, 2-fold smoke test, M2 gate | `gbdt_pipeline.py` (§4.5–4.6, §4.16 `train()`), `test_gbdt_integration.py` |
| **M3** | Temperature scaling, θ calibration, P@K, rank IC, down-class precision, post-cost backtest, reliability diagrams | `gbdt_pipeline.py` (§4.7–4.10, §4.12), `test_gbdt_calibration.py`, `test_gbdt_schema.py` |
| **M4** | SHAP beeswarm + CSV, leakage checklist (manual), block-and-retrain | `gbdt_pipeline.py` (§4.13), update `manifest['shap_audit_passed']` |
| **M4.5** | nightly_run.py integration, inference CSV writer, HTML report GBDT section | `mainDeltafromToday.py` (§6), `nightly_run.py` (§7), `gbdt_pipeline.py` (§4.14–4.15, §8) |
| **M5** | 2-model bucket ablation (Choice 1), CatBoost comparison | `gbdt_pipeline.py` (new `train_bucket()` fn), no param file changes |
| **M6** | 6-fold walk-forward, NaN IV migration, Optuna tuning | `gbdt_pipeline.py` (new `walk_forward()` fn), update all `{SYM}_gbdt_param.py` `lgbm_params` |

---

## 11. Open Questions / Parking Lot

These are not blockers for M1 but should be resolved before M3:

1. **AAII Spread publication lag**: confirm current TMP.csv pipeline uses prior-Thursday
   value for Monday–Wednesday rows, not same-week. If not fixed, AAII Spread must be
   blocked in M4 leakage audit.

2. **PLTR Tier 2 feature audit**: `rs_ita`, `rs_igv`, `rs_ita_trend`, `FDEFX`,
   `ADEFNO`, `IPDCONGD` listed in PRD but PLTR_gbdt_param.py needs to confirm these
   columns exist in PLTR_TMP.csv before M1 completes.

3. **CRDO start_date**: CRDO IPO was 2022-01-27. That gives ~520 training rows vs
   ~570 for other symbols. The M1 class distribution report will show exact counts.
   If Up or Down < 50, the `[LOW_SAMPLE_ALERT]` will fire and 2× weight boost applies.

4. **`eps_est_avg` leakage risk**: PRD flags this for M4 audit. If AV returns the
   estimate *after* the earnings release date (not before), it must be blocked or
   shifted forward by 1 report cycle.
