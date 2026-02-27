# Multi-Horizon (15-day) Up/Flat/Down Transformer — Implementation Plan

**Goal**: Replace 15 separate models/heads (one per horizon) with a **single shared encoder** and a **multi-horizon classification head** that outputs predictions for horizons **h = 1..15** in one forward pass.

This plan assumes:
- **Daily (trading day) data** from 2021 onward
- **3-class labels**: `down / flat / up`
- Horizon-specific thresholds: **±3% for days 1–5**, **±5% for days 6–15**
- Long-only trading downstream (decisioning uses `P(up)` vs `P(down)`)

---

## 1) Define the Prediction Targets (Labels)

### 1.1 Forward return definition
For each date `t` and horizon `h`:

\[
r_{t,h} = \frac{P_{t+h} - P_t}{P_t}
\]

where `P_t` is the (adjusted) close at date `t`.

### 1.2 Horizon-specific thresholds
- If `h <= 5`: `thr = 0.03`
- If `h >= 6`: `thr = 0.05`

### 1.3 Class mapping
- `up` if `r_{t,h} >= thr`
- `down` if `r_{t,h} <= -thr`
- `flat` otherwise

Encode classes as integers — **matching the existing codebase**:
- `0 = flat/neutral`, `1 = UP`, `2 = DOWN`

> **[RESOLVED]** Keep existing encoding. `processPrediction.py`, `FocalLoss` class weights, and all label maps use `{0: '__', 1: 'UP', 2: 'DN'}`. The new multi-horizon model must use the same mapping. Define as a module-level constant in `trendAnalysisFromTodayNew.py`:
> ```python
> CLASS_LABELS = {0: '__', 1: 'UP', 2: 'DN'}  # 0=flat, 1=up, 2=down — DO NOT CHANGE
> ```

**Output label tensor per sample**: `y ∈ {0,1,2}^{15}`

---

## 2) Dataset Construction (No Leakage)

### 2.1 Input window
Choose a lookback window `L` trading days (typical: `L = 60, 90, 126, 252`).

Each sample uses feature history:
- `X_t = [x_{t-L+1}, ..., x_t]` with shape `(L, D)`

> **[RESOLVED]** Use **Option A**: all 60+ engineered features at every timestep. The existing DataFrame already has the full feature set correctly populated as-of each date — no additional pipeline engineering needed. `D` stays the same as today, `L` becomes a new param key (`lookback_window`, default 60). Data pipeline change: `raw_df.tail(1)` → `raw_df.tail(L)`, producing `(1, L, D)` instead of `(1, 1, D)`. Memory per stock: ~8MB — trivial.

> **[!] INOD DATA SIZE WARNING**: INOD has ~570 training samples total. With `L = 252`, usable samples drop to `570 - 252 - 15 ≈ 303`. After train/val/test splits that's roughly 180 train / 60 val / 60 test. That is marginal for a transformer. `L = 60` is safer for small-cap stocks with limited history. Suggest: use `L = 60` as the default and treat `L = 126/252` as optional experiments on larger-data stocks (NVDA, PLTR, APP).

### 2.2 Train/val/test split
Use **time-based splits** only:
- Train: 2021 → `T_train_end`
- Val: next block
- Test: most recent block

> **[RESOLVED]** `shuffle_splits=True` is a **hard error** for `model_type='multi_horizon_transformer'`. Add a `ValueError` guard at the top of the multi-horizon training entry point — silently inheriting look-ahead bias would produce misleadingly inflated validation numbers (documented: macro-F1 0.747 shuffled vs 0.18 honest). New param dicts for this model type must always set `shuffle_splits=False`.

Optionally do **walk-forward** evaluation (rolling origin) for robustness.

### 2.3 Standardization / normalization
To avoid "time level" leakage:
- Compute scalers **on train only**
- Apply to val/test

Prefer stationary transforms (returns, deltas) where appropriate. If you keep levels, standardize carefully.

> **[RESOLVED]** Scaler fitting logic is **unchanged** — fit on `(N_train_rows, D)` as today; `scaler.transform(window)` naturally handles the `(L, D)` window shape since transforms are applied column-wise. Naming convention from §10.5: `{SYM}_{model_name}_mh_scaler.joblib` (`_mh_` infix avoids collision during parallel-run period). `robust_features` two-scaler setup carries over unchanged.

### 2.4 Handling missing data
- Forward-fill only when valid and "as-of" correct
- Add missingness masks if missing is informative
- Drop samples where labels for horizons 1..15 are not available (near end of series)

> **[RESOLVED]** Label-dropping applies only in `build_training_dataset()`. The `prepare_inference_window()` function (§10.5) calls `raw_df.tail(L)` unconditionally — no label filtering. The boundary must be enforced in code: if label-dropping logic ever leaks into the data fetch or feature pipeline layer, it will silently truncate the most recent rows at inference time.

---

## 3) Model Architecture

### 3.1 Overview
A **single** transformer encoder processes `(L, D)` and produces a representation `z_t`.
A multi-horizon head outputs `(15, 3)` logits in one pass.

### 3.2 Encoder
- Input projection: `Linear(D → d_model)`
- Positional encoding (learned or sinusoidal)
- Transformer encoder stack: `N` layers, `n_heads`, dropout

> **[RESOLVED]** Use a **composable design**: define `MultiHorizonHead` as a standalone class, then build encoder-specific classes that own the head instance. This makes the head reusable across any future encoder (LSTM, GRU, CNN) without code duplication. `PositionalEncoding` is reused as-is from the existing codebase.
>
> ```python
> class MultiHorizonHead(nn.Module):
>     """Architecture-agnostic. Input: (batch, d_model). Output: (batch, 15, 3)."""
>     def __init__(self, d_model, num_horizons=15, num_classes=3):
>         self.fc = nn.Linear(d_model, num_horizons * num_classes)
>         self.num_horizons, self.num_classes = num_horizons, num_classes
>     def forward(self, z):   # z: (batch, d_model)
>         return self.fc(z).view(-1, self.num_horizons, self.num_classes)
>
> class MultiHorizonTransformer(nn.Module):  # transformer encoder + shared head
>     def __init__(self, ...):
>         self.input_projection = nn.Linear(input_dim, embedded_dim)
>         self.positional_encoding = PositionalEncoding(embedded_dim, dropout_rate)  # reused
>         self.transformer_encoder = nn.TransformerEncoder(...)
>         self.dropout = nn.Dropout(dropout_rate)
>         self.head = MultiHorizonHead(embedded_dim)
>
> # Future LSTM variant — same head, new encoder, no duplication:
> # class MultiHorizonLSTM(nn.Module):
> #     def __init__(self, ...):
> #         self.lstm = nn.LSTM(input_size=D, hidden_size=d_model, ...)
> #         self.head = MultiHorizonHead(d_model)
> ```
>
> Both register in `build_model()` under different `model_type` values. Training loop and inference pipeline only depend on the `(batch, 15, 3)` output contract — unchanged regardless of encoder type.

### 3.3 Pooling strategy (choose one)
- **Last token**: `z_t = h_L`
- **CLS token**: prepend `[CLS]` and use its output
- **Attention pooling**: learn a query vector to pool across time

Start with **last token** (`encoder_output[:, -1, :]`) — simplest baseline, zero extra parameters.

> **[RESOLVED]** Initial implementation uses **last-token pooling**: `z = encoder_output[:, -1, :]`. Our pre-engineered features already encode substantial historical context (rolling RSI, MACD, DTE/DSE, etc.), so the last position after self-attention over L days is a strong baseline. CLS token added to backlog as a future upgrade experiment.
>
> **Backlog B-MH1**: Benchmark CLS token pooling vs last-token. Add `pooling_strategy` param key (`'last'` default, `'cls'` optional). CLS prepends one learned `(1, 1, d_model)` embedding to the sequence before encoding.

### 3.4 Multi-horizon head (baseline)
- `logits_flat = Linear(d_model → 15*3)(z_t)`
- reshape to `logits ∈ R^{15×3}`

### 3.5 Horizon embedding (recommended upgrade)
Improves horizon conditioning (especially for 6–15):

For each horizon `h`:
- horizon embedding `e_h ∈ R^{d_h}`
- concatenate: `[z_t ; e_h]`
- small MLP → logits for that horizon

This yields `logits_h ∈ R^3` for each `h`, stacked to `(15, 3)`.

> **[RESOLVED]** §3.4 flat head is Week 1–2. §3.5 horizon embedding is **conditional Week 3** — only pursue if the §3.4 confusion matrix shows systematic underperformance on h=6–15 vs h=1–5. Adding horizon embeddings on top of a broken encoder obscures root cause; the diagnostic gate prevents that. Horizon embedding added to backlog as **Backlog B-MH2**.

---

## 4) Training Objective

### 4.1 Per-horizon cross-entropy
\[
\mathcal{L} = \sum_{h=1}^{15} w_h \cdot CE(\text{logits}_h, y_h)
\]

Start with `w_h = 1` for all horizons.
If your trading cadence is ~2–3 weeks, test modest up-weighting of horizons 6–15.

> **[RESOLVED]** Normalize loss by `num_horizons` — one line, not a hyperparameter: `L_total = sum(horizon_losses) / num_horizons`. Keeps loss in the same ~0.3–1.0 range as single-horizon training, making LR transfer from existing param dicts direct and loss curves interpretable. Gradient clipping at `max_norm=1.0` (already resolved) provides a secondary safety net.

### 4.2 Class imbalance handling (critical)
"Flat" usually dominates.

Options (start with A):
- **A) FocalLoss with per-horizon class weights** ← **USE THIS**
  Existing `FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0.1)` in `trendAnalysisFromTodayNew.py` handles both flat-dominance (via weights) and hard-example focus (via gamma). Apply per-horizon, sum across h.
  Compute class frequencies per horizon on the training set: `weights[h] = calculate_class_weight(train_df_h, num_classes=3)`.
- **B) Plain class-weighted CE** (fallback if focal causes instability)
- **C) Downsample flat** (keep all up/down, sample flat to a target ratio)

> **[RESOLVED]** Focal Loss promoted to Option A. Reuse `FocalLoss` class unchanged. The per-horizon application means we compute one `class_weights_tensor` per horizon (15 tensors total) and call `FocalLoss(weight=class_weights_h)(logits_h, labels_h)` for each `h`, then sum.

### 4.3 Regularization
- Dropout in encoder + head
- Weight decay
- Early stopping on validation metrics aligned with trading (see Section 6)

> **[RESOLVED]** Use `EarlyStopping(patience=15)` on **val loss** (normalized FocalLoss across 15 horizons). `TrendBasedStopping` monitors training loss — it detects divergence, not overfitting. `EarlyStopping` on val loss already exists in `analysisUtil.py` (`train_with_early_stopping`, line ~360) and computes both `avg_train_loss` and `avg_val_loss` per epoch — it needs the multi-horizon loss reshape added to its inner loop, then it's ready to use. Patience=15 (vs default 7) gives the model room to escape early noise with 15 horizon signals. Val macro-F1 stopping deferred to **Backlog B-MH3**.

---

## 5) Implementation Steps (Engineering Plan)

### Step 1 — Data pipeline
1. Generate `X` windows of shape `(N, L, D)`
2. Generate `Y` labels of shape `(N, 15)`
3. Verify:
   - no lookahead in features
   - label horizon alignment correct
   - correct handling of market holidays/trading days

**Unit tests**:
- For random sample date `t`, manually compute `r_{t,h}` and confirm label matches code.
- Confirm `X_t` ends at `t` (not `t+1`).

> **[RESOLVED]** Sequence windows introduce **no new leakage** — they stack rows already in the DataFrame. Price-derived features (RSI, MACD, Bollinger, volatility) use right-aligned `rolling()` — clean. CP ratio, AAII, FINRA debt, unemployment are fetched as daily/weekly/monthly as-of values — clean. DTE/DSE uses historical earnings dates, which is a pre-existing concern shared with the current single-timestep model. As-of audit for DTE/DSE tracked as **Backlog B-MH4** (pre-existing work, not a multi-horizon blocker).

### Step 2 — Model code
1. Implement encoder + pooling
2. Implement head (baseline, then horizon-embedding upgrade)
3. Output logits `(batch, 15, 3)`

> **[RESOLVED — Factory Integration]** `build_model()` is the correct hook. Register as `model_type='multi_horizon_transformer'`. Add an `elif` branch that instantiates `MultiHorizonTransformer(input_dim=input_dim, num_classes=num_classes, **{k: param[k] for k in ['headcount','num_layers','dropout_rate','embedded_dim','lookback_window']})`. New required param keys: `lookback_window` (int, e.g. 60), `pooling_strategy` (str, default `'last'`). Existing keys (`headcount`, `num_layers`, `dropout_rate`, `embedded_dim`) carry over unchanged — no param dict migration needed for existing stocks.

### Step 3 — Training loop
1. Compute CE per horizon, sum with weights
2. Add class weighting / focal loss
3. Train with:
   - **AdamW** `optim.AdamW(model.parameters(), lr=param['learning_rate'], weight_decay=param['l2_weight_decay'])`
   - CosineAnnealingLR (reuse existing scheduler pattern)
   - Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

> **[RESOLVED — AdamW]** Replace `optim.Adam` with `optim.AdamW` — drop-in change. AdamW correctly decouples weight decay from the adaptive gradient update. Apply to the multi-horizon model; the existing 15-model path can be migrated separately.

> **[RESOLVED — Gradient clipping]** Use `max_norm=1.0`. Add one line after `scaler.unscale_(optimizer)` in `analysisUtil.train_with_trend_based_stopping`:
> ```python
> torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
> ```
> This must be added to the training loop before `scaler.step(optimizer)`.

### Step 4 — Validation and checkpointing
Save checkpoints by **val metric**, not loss alone.

> **[RESOLVED — Checkpoint Format]** Use state_dict format for the multi-horizon model (existing 15-model `.pth` files unaffected). See §10.5 for full spec: `{'state_dict': ..., 'config': param, 'calib_temp': 1.0, 'train_date': ...}`. Loading path: `build_model(checkpoint['config'], ...)` → `load_state_dict(checkpoint['state_dict'])`. Avoids pickle version warnings and makes config introspection trivial.

---

## 6) Metrics (What "Good Enough" Looks Like)

Because you care about **tradability** and **1–2 trades/day**, track:

### 6.1 Core classification metrics (per horizon and bucket)
- Accuracy (mostly diagnostic; flat dominance makes it misleading)
- Macro-F1 (per horizon)
- Confusion matrix

### 6.2 Trading-aligned metrics (recommended)
Define per horizon:
- `P_up(h)` and `P_down(h)` from softmax
- `edge_h = P_up(h) - P_down(h)`

Track:
- **Precision@K** for `up` (K = 1 or 2 selections/day)
- **Hit rate when trading**: fraction of trades meeting threshold
- **Rank IC**: Spearman corr of `edge` vs realized forward return
- **Calibration**: Brier score + reliability curves for `P_up`

Bucket metrics:
- Days 1–5 (±3%)
- Days 6–15 (±5%)

> **[RESOLVED — Q11]** Use **per-stock time-series IC** (Option A): for each stock, compute Spearman correlation between `edge_h` and realized return across all test dates, then report mean ± std across the 5 stocks. Cross-sectional IC (correlation across stocks per day) is rejected — Spearman from N=5 is unreliable. Per-stock IC over ~60 test dates gives a meaningful sample size. Primary metric is mean IC across stocks; std reveals whether one outlier stock is driving results. Log both during evaluation.

---

## 7) Probability Calibration (Do This Before Trading)

### 7.1 Temperature scaling
Fit temperature `T` on validation set to calibrate softmax logits.
Optionally:
- one `T` per bucket (1–5 vs 6–15)
- or one `T` per horizon if you have enough validation data

### 7.2 Post-calibration checks
- Reliability curves for `P_up`
- Ensure predicted 0.7 means ~70% frequency in out-of-sample bins

> **[RESOLVED — Calibration in Inference Pipeline]** Fully specified in §10.5: `T` is stored in the checkpoint as `calib_temp` (default `1.0` = no-op). In `make_inference_multi_horizon()`, after loading the checkpoint: `logits = logits / T` before `softmax(logits, dim=2)`. Temperature fitting (on val set, post-training) updates `calib_temp` in the `.pth` file via `checkpoint['calib_temp'] = T_fitted; torch.save(checkpoint, path)` — no retraining required. Existing `make_prediciton_test()` / `run_daily_inference.py` are untouched.

---

## 8) Decisioning Across Horizons (Avoid Conflicts)

You will have 15 predictions per sample. Choose a single scalar score.

Good defaults:
- **Max edge in holding bucket**:
  `score = max(edge_h for h in 6..15)`
- **Weighted sum**:
  `score = Σ w_h * edge_h` with weights emphasizing 6–15

Long-only trade filter:
- Trade only if `score > θ` (threshold)
- Select top 1–2 names/day by score
- Add a kill switch (market regime) outside the model if desired

> **[RESOLVED — Q12]** Use **Option A** (separate file): keep `{SYM}_15d_from_today_predictions.csv` schema unchanged (argmax class labels only) — zero breakage to `get_historical_html.py`, `processPrediction.py`, and parallel-run HTML display. Write edge scores to `{SYM}_15d_edge_scores.csv` (one column per horizon, `edge_h = P_up(h) - P_down(h)`). The decisioning layer and IC metric computation load the edge file; existing HTML display loads the predictions file. Merge into a single file after the 15-model system is retired.

---

## 9) Backtest Integration (Minimal, Controlled)

### 9.1 Backtest assumptions
- Signals computed at close of day `t`
- Trades executed at next open (or next close), consistently

### 9.2 What to report
- Turnover
- Drawdowns
- Return vs SPY (and vs cash proxy)
- Sensitivity to transaction costs / slippage

### 9.3 Walk-forward
Use walk-forward training/retraining windows to reduce overfitting to 2021–present regime.

> **[RESOLVED — Q13]** Walk-forward is an **offline experiment only** (Option A). Daily retraining continues unchanged — multi-horizon reduces training jobs from 150 → 10, so compute pressure is lower, not higher. Walk-forward runs once (or occasionally) as a regime-generalization diagnostic: does the model trained on 2021→T stay reliable on T→T+3mo? Results inform architecture decisions but do not change the production retraining cadence. Periodic-retrain infrastructure (Option B) is a separate project, not a blocker for the multi-horizon model.

---

## 10) Deployment Notes

- One model artifact (encoder + multi-horizon head)
- Single inference pass per symbol/day
- Output: 15 horizon probability vectors
- Log:
  - inputs checksum
  - model version
  - calibration temperature(s)
  - prediction distributions

> **Training efficiency**: Currently 5 stocks × 2 param sets × 15 models = 150 training runs. Multi-horizon reduces this to 5 × 2 × 1 = 10 runs — **15× fewer training jobs**. This also reduces daily training wall-clock time proportionally.

See **Section 10.5** below for the full inference pipeline design.

---

## 10.5) Inference Pipeline Design

### Current flow (15-model)

```
run_daily_inference.py
  └─ for each stock:
       └─ for target_size in 1..15:
            └─ make_inference(param[target_size], incr_df)
                 ├─ load: model/model_{SYM}_{name}_fixed_noTimesplit_{h}.pth  (torch.load)
                 ├─ load: {SYM}_{name}_scaler.joblib
                 ├─ prep:  raw_df[-batch_size-h:] → scale → tensor (1, batch_size+h, D)
                 ├─ run:   model(tensor) → (1, batch_size+h, 3)
                 ├─ take:  last h predictions → argmax → ['UP'/'DN'/'__']
                 └─ write: {SYM}_15d_from_today_predictions.csv  (one row per horizon)
```

### New flow (multi-horizon model)

```
run_daily_inference.py
  └─ for each stock:
       └─ make_inference_multi_horizon(param, incr_df)
            ├─ load: model/model_{SYM}_{name}_mh_fixed_noTimesplit.pth
            │         (stored as: {'state_dict': ..., 'config': param, 'calib_temp': T})
            ├─ load: {SYM}_{name}_mh_scaler.joblib
            ├─ prep:  raw_df[-L:] → scale → tensor (1, L, D)
            ├─ run:   model(tensor) → (1, 15, 3)  logits
            ├─ calib: logits / T  (temperature scaling, T loaded from checkpoint)
            ├─ prob:  softmax(logits, dim=2) → (1, 15, 3)
            ├─ edge:  edge[h] = prob[h, 2] - prob[h, 0]  (P_up - P_down, class 2=UP, 0=flat)
            ├─ label: argmax(prob, dim=2) → (1, 15) → map via CLASS_LABELS
            └─ write: {SYM}_15d_from_today_predictions.csv  (all 15 rows, same format)
                      {SYM}_15d_edge_scores.csv             (optional: edge[h] per horizon)
```

### Key file/function changes

| File | Change |
|------|--------|
| `trendAnalysisFromTodayNew.py` | Add `make_inference_multi_horizon()` alongside existing `make_inference()` |
| `trendAnalysisFromTodayNew.py` | Add `make_prediction_multi_horizon()` alongside existing `make_prediciton_test()` |
| `run_daily_inference.py` | Detect `param.get('model_type') == 'multi_horizon_transformer'` → call new path |
| `processPrediction.py` | `process_prediction_results_test()` already accepts a list of labels — no change needed |
| `analysisUtil.py` | `train_with_trend_based_stopping()` unchanged; new multi-horizon training loop wraps it or is separate |

### Model artifact format (new)

```python
# Saving (at end of training):
torch.save({
    'state_dict': model.state_dict(),
    'config': param,                  # full param dict including lookback_window, etc.
    'calib_temp': 1.0,                # default = no calibration; update after temp scaling step
    'train_date': str(date.today()),
}, model_path)

# Loading (inference):
checkpoint = torch.load(model_path, map_location='cpu')
model = build_model(checkpoint['config'], input_dim=D, num_classes=3)
model.load_state_dict(checkpoint['state_dict'])
T = checkpoint.get('calib_temp', 1.0)
```

> This replaces `torch.save(model, path)` (full pickle) with state_dict format for the multi-horizon model only. Existing 15-model `.pth` files are unaffected.

### Feature tensor preparation

```python
def prepare_inference_window(raw_df, param, scaler):
    """
    Build (1, L, D) input tensor for multi-horizon inference.
    L = param['lookback_window']  (e.g. 60)
    D = len(param['selected_columns']) - 1  (exclude 'label')
    """
    L = param['lookback_window']
    cols = [c for c in param['selected_columns'] if c in raw_df.columns]
    window = raw_df[cols].tail(L).copy()
    if len(window) < L:
        raise ValueError(f"Not enough history: need {L} rows, got {len(window)}")
    features = scaler.transform(window)          # (L, D)
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, L, D)
    return tensor
```

### Scaler naming convention

| Type | Filename |
|------|----------|
| Existing 15-model | `{SYM}_{model_name}_scaler.joblib` |
| New multi-horizon | `{SYM}_{model_name}_mh_scaler.joblib` |

The `_mh_` infix prevents collision with existing scalers during the parallel-run period.

### Parallel-run strategy

During the transition period (≥2 weeks), both pipelines run each day:
- Old: 15 model files → P1–P15 written as `reference` rows (black in HTML)
- New: 1 model file → P1–P15 written as `multi_horizon` rows (new color, e.g. blue `#3b82f6`)

Add `multi_horizon` row type to `get_historical_html.py` color scheme. Compare prediction agreement before retiring old system.

---

## 11) Timeline (Practical Build Order)

1. **Week 1**: Data pipeline + label generation + unit tests
2. **Week 1–2**: Baseline multi-horizon head model + class-weighted CE
3. **Week 2**: Calibration + trading-aligned metrics + first walk-forward
4. **Week 3**: Horizon embedding + decisioning refinements
5. **Week 3–4**: Robust backtest + sensitivity + deployment packaging

> **[RESOLVED — Parallel Running]** Parallel-run strategy fully specified in §10.5. Both pipelines run simultaneously for ≥2 weeks: old 15-model system produces `reference`/`AAII_option_vol_ratio` rows (existing colors); new multi-horizon produces `multi_horizon` rows (blue `#3b82f6`). Add `multi_horizon` detection to `get_historical_html.py` color-code logic. Timeline updated: Week 3–4 includes parallel-run validation before retirement decision.

---

## 13) Validation Tests

### 13.1 Purpose

Verify that the multi-horizon model achieves competitive per-class F1 vs the three existing single-horizon baselines before going into production. Tests run against **NVDA** (most data, strongest baseline signal).

> **Note on test type**: This is a **slow integration test** (~15–30 min training), not a fast unit test. It must be tagged `@pytest.mark.slow` and excluded from the regular fast suite (`pytest -m "not slow"`). Fast in-memory unit tests (architecture, shape checks) live in `test_model_factory.py` and `test_dte_dse.py` as before.

---

### 13.2 Baseline Profiles to Compare

| Profile | Description | HTML color |
|---------|-------------|------------|
| `ref` | Reference (shuffled splits) | black |
| `ref_noshuf` | Reference (time-split, honest) | red `#dc2626` |
| `AAII_option_vol_ratio` | With AAII + CP ratio features | yellow `#fef08a` |
| `multi_horizon` | **New model under test** | blue `#3b82f6` |

The **primary comparison target is `ref_noshuf`** — it is the only baseline that uses honest time-based splits, matching the multi-horizon model's `shuffle_splits=False` guard.

---

### 13.3 Model Isolation Requirement (Critical)

**The validation test MUST NOT overwrite any production model files.**

Production models live in `/workspace/model/model_{SYM}_{name}_fixed_noTimesplit_{1-15}.pth`. The nightly training run reads and overwrites these files. The validation test must be completely isolated from this directory.

**Isolation rules:**
- All test model artifacts are saved to a **pytest-managed temp directory** (`tmp_path` fixture or `tempfile.mkdtemp()`), never to `/workspace/model/`
- Scalers used during evaluation are **loaded read-only** from their production paths — never re-fitted and never written back
- The test reads the production scalers (`{SYM}_{name}_scaler.joblib`) but saves its own scaler to the temp dir under a different name (`{SYM}_mh_test_scaler.joblib`)
- All temp files are automatically cleaned up when the test session ends (pytest `tmp_path` handles this)
- The test does **not** call `torch.save(model, path)` to `/workspace/model/` under any code path

**Implementation guard** (add at top of test training call):
```python
assert not model_path.startswith('/workspace/model/'), \
    f"Test must not write to production model dir: {model_path}"
```

---

### 13.4 Test Specification

**File**: `test_multi_horizon_nvda_baseline.py`

**Training setup**:
- Symbol: `NVDA`
- Date range: same start date as existing NVDA models → last row with available data
- `shuffle_splits=False` (enforced by the `ValueError` guard)
- `lookback_window=60` (default)
- All other params from the NVDA multi-horizon param dict
- **Model saved to `tmp_path` only** — never to `/workspace/model/`

**Baseline evaluation**:
- Load existing production models from `/workspace/model/` (read-only)
- Run inference on the same held-out test split used by the multi-horizon model
- Extract per-class F1 for each of the 3 baseline profiles

**Multi-horizon evaluation**:
- Train in temp dir, evaluate on same test split
- Extract per-class F1 from test split only (not train/val)
- Report three classification classes: `flat (0)`, `UP (1)`, `DOWN (2)`
- Report two horizon buckets: `h=1–5` (±3% threshold) and `h=6–15` (±5% threshold)
- Report macro-F1 as aggregate

**Output format** (logged to stdout and saved as `nvda_mh_baseline_comparison.csv`):

```
model           | bucket  | F1_flat | F1_up | F1_down | macro_F1
----------------|---------|---------|-------|---------|----------
ref             | h=1-5   |  ...    |  ...  |   ...   |   ...
ref             | h=6-15  |  ...    |  ...  |   ...   |   ...
ref_noshuf      | h=1-5   |  ...    |  ...  |   ...   |   ...
ref_noshuf      | h=6-15  |  ...    |  ...  |   ...   |   ...
AAII_cp_ratio   | h=1-5   |  ...    |  ...  |   ...   |   ...
AAII_cp_ratio   | h=6-15  |  ...    |  ...  |   ...   |   ...
multi_horizon   | h=1-5   |  ...    |  ...  |   ...   |   ...
multi_horizon   | h=6-15  |  ...    |  ...  |   ...   |   ...
```

---

### 13.5 Acceptance Criteria

**Hard pass/fail**:
- [ ] Multi-horizon `macro_F1` on `h=1–5` ≥ `ref_noshuf` macro_F1 on the same bucket
- [ ] Multi-horizon `macro_F1` on `h=6–15` ≥ `ref_noshuf` macro_F1 on the same bucket

**Soft targets** (investigate if missed, not automatic blockers):
- `F1_up` ≥ 0.25 in both buckets (UP class is hardest, but should be detectable)
- `F1_down` ≥ 0.25 in both buckets
- `F1_flat` ≥ 0.50 in both buckets (flat dominates, should be easiest)

**If multi-horizon underperforms `ref_noshuf`**:
1. Inspect confusion matrix — is one class collapsing to all-flat?
2. Check class weights computation per horizon — is imbalance being handled?
3. Try increasing `patience` (early stopping) or reducing `lookback_window`
4. Do NOT proceed to parallel-run deployment until criteria pass

---

### 13.6 How to Run

```bash
# Fast unit tests only (no training) — run any time
pytest /workspace/test_*.py -v -m "not slow"

# Full validation test (requires training — ~15-30 min on CPU, ~5 min on GPU)
pytest /workspace/test_multi_horizon_nvda_baseline.py -v -m slow -s

# Both
pytest /workspace/test_*.py -v
```

---

## 12) Checklist (Done/Not Done)

- [ ] Labels correct for ±3% (h=1..5) and ±5% (h=6..15)
- [ ] No feature leakage (as-of discipline)
- [ ] One-pass logits `(15,3)`
- [ ] Flat imbalance handled
- [ ] Metrics include Precision@K and calibration
- [ ] Backtest uses consistent execution timing
- [ ] Walk-forward validation performed
- [ ] Deployment logs model + calibration versioning

- [ ] `CLASS_LABELS` constant defined and used everywhere (encoding: 0=flat, 1=UP, 2=DN)
- [ ] Old 15-model system runs in parallel for ≥2 weeks before retirement
- [ ] `build_model()` factory updated with `model_type='multi_horizon_transformer'`
- [ ] `make_inference_multi_horizon()` + `make_prediction_multi_horizon()` implemented
- [ ] `run_daily_inference.py` dispatches to new path based on `model_type`
- [ ] Scaler saved as `{SYM}_{name}_mh_scaler.joblib`; model as `{SYM}_{name}_mh_fixed_noTimesplit.pth`
- [ ] Model artifact uses state_dict format with `calib_temp` field
- [ ] AdamW replaces Adam for multi-horizon training
- [ ] Gradient clipping `max_norm=1.0` added to training loop
- [ ] Multi-horizon HTML row color (blue `#3b82f6`) added to `get_historical_html.py`
- [ ] `test_multi_horizon_nvda_baseline.py` written and passing (macro-F1 ≥ ref_noshuf on both h=1-5 and h=6-15)
- [ ] Validation test saves models to `tmp_path` only — confirmed it does not touch `/workspace/model/`
- [ ] `nvda_mh_baseline_comparison.csv` generated and reviewed before parallel-run deployment

---

## Backlog

| ID | Item |
|----|------|
| B-MH1 | Benchmark CLS token pooling vs last-token. Add `pooling_strategy` param key (`'last'` default, `'cls'` optional). |
| B-MH2 | Horizon embedding: add per-horizon learned offset to encoder output before head. Only pursue if confusion matrix shows systematic underperformance on h=6-15 vs h=1-5. |
| B-MH3 | Val macro-F1 stopping: replace val-loss EarlyStopping with F1-based stopping criterion. |
| B-MH4 | As-of audit for DTE/DSE features (pre-existing concern shared with single-horizon models). |
| B-MH5 | `inference()` support for `model_type="trans_mz"` in `mainDeltafromToday.py`. Requires: load state-dict checkpoint, run single forward pass on today's scaled features, decode argmax to p1-p15, feed into `processDeltaFromTodayResults`. |
