# PRD — Multi-Horizon Stock Direction Classifier Using GBDT (LightGBM/CatBoost)

## 1. Purpose
Build a **production-ready, multi-horizon** stock-direction prediction system using **Gradient-Boosted Decision Trees (GBDT)**—primarily **LightGBM** and/or **CatBoost**—to predict **Up / Flat / Down** for the next **15 trading days**.  
This PRD replaces (or complements) a Transformer-based approach with a model family that is typically more suitable for **tabular, engineered, cross-sectional, mixed-scale** feature sets.

---

## 2. Goals and Non-Goals

### 2.1 Goals
1. **Predict 3-class direction** (Up/Down/Flat) for horizons `h=1..15` trading days.
2. Support two training configurations:
   - **15-model approach**: one classifier per horizon.
   - **2-model bucket approach**: one model for horizons 1–5, one model for horizons 6–15.
   - **3-model bucket approach**: one model for horizons 1–5, one model for horizons 6–10, one for 11-15.
3. Improve **iteration speed**, **debuggability**, and **baseline robustness** relative to Transformers.
4. Provide **calibrated probabilities** suitable for downstream ranking and trade selection.
5. Provide interpretable diagnostics (feature importance, SHAP) to support leakage detection and model governance.

### 2.2 Non-Goals
- Not designing a full execution engine (order routing, smart order execution).
- Not guaranteeing profitability; objective is to build a statistically sound predictor and evaluation harness.
- Not building a deep learning sequence model in this PRD (Transformer/TCN/LSTM are out of scope here).

---

## 3. Users / Stakeholders
- **Primary user**: the system owner (you) running daily predictions and selecting (at most) 1–2 trades/day, over the next 15 day prediction horizon
- **Secondary stakeholders**: future collaborators who need repeatable training, evaluation, and audit trails.

---

## 4. Problem Statement
Current approach uses a Transformer model and (previously) ran multiple separate predictions. For **engineered tabular features** and **moderate dataset sizes** (daily bars since 2021), GBDT methods often:
- outperform or match deep models,
- train faster,
- provide stronger interpretability and leakage detection tools,
- yield well-behaved probabilities after calibration.

We need a GBDT-based multi-horizon system designed for:
- 3-class outputs (Up/Flat/Down),
- horizon-specific thresholds (±3% for days 1–5, ±5% for days 6–15),
- strong class imbalance (Flat dominates),
- time-series constraints (no leakage, time-based splits).

---

## 5. Requirements

### 5.1 Functional Requirements
**FR1 — Labeling**
- Produce labels `y_{t,h} ∈ {down, flat, up}` for horizons `h=1..15` using:
  - forward returns from close `t` to close `t+h`,
  - threshold `thr(h)=0.03` for `h<=5`, `thr(h)=0.05` for `h>=6`.

**FR2 — Model training**
- Support:
  - **15 separate classifiers** (one per horizon), OR
  - **2 bucketed classifiers**: `(1–5)` and `(6–15)`. or 
  - **3 bucketed classifiers**: `(1–5), (6-10), and `(11–15)`. or 
- Support both **LightGBM** and **CatBoost** training backends.

**FR3 — Class imbalance handling**
- Provide one or more of:
  - class weights (per-horizon preferred),
  - downsampling of Flat,
  - threshold tuning for operating point.

**FR4 — Probability calibration**
- Provide calibration of predicted probabilities:
  - baseline: temperature scaling or isotonic calibration on validation set,
  - optional per-horizon calibration (if enough data).

**FR5 — Explainability**
- Provide:
  - feature importance (gain/split),
  - SHAP values on validation/test,
  - per-horizon SHAP summaries (15-model) or per-bucket (2-model).

**FR6 — Evaluation**
- Provide:
  - per-horizon metrics (macro-F1, per-class F1, confusion matrix),
  - bucket-level metrics (1–5 and 6–15),
  - trading-aligned metrics: Precision@K (K=1,2), hit rate conditional on trading, rank IC.

**FR7 — Inference**
- Daily inference for each symbol produces:
  - `P(up), P(flat), P(down)` per horizon (15-model) or per bucket (2-model).
- Expose a single scalar score for ranking, e.g.:
  - `edge = P(up) - P(down)` and
  - `score = max(edge_h for h=6..15)` or weighted sum.

**FR8 — Reproducibility**
- Every training run must save:
  - data range,
  - feature list and transformations,
  - hyperparameters,
  - random seeds,
  - calibration parameters,
  - metrics and plots.

---

### 5.2 Non-Functional Requirements
**NFR1 — No leakage**
- Strict time-based splitting.
- Feature alignment “as-of” timestamp, especially for non-daily sources.

**NFR2 — Runtime**
- Training should complete in minutes per horizon on a typical workstation.
- Daily inference should run in seconds per basket.

**NFR3 — Maintainability**
- Modular pipeline:
  - data → features → labels → train → calibrate → evaluate → predict.

**NFR4 — Observability**
- Logging:
  - input data checksums,
  - prediction distributions,
  - drift monitoring hooks (optional).

---

## 6. Data and Feature Interface

### 6.1 Inputs
- Daily feature vector per symbol per date:
  - `X_t ∈ R^D` (engineered features, cross-sectional aggregates, macro proxies as-of t)

### 6.2 Outputs
- For each symbol/date:
  - **15-model**: `P_{t,h}(down/flat/up)` for `h=1..15`
  - **2-model**: `P_{t,bucket}(down/flat/up)` for buckets `(1–5)` and `(6–15)`
- Plus:
  - ranking score(s) for trade selection.

### 6.3 Label generation
For each `t` and horizon `h`:
\[
r_{t,h} = \frac{P_{t+h} - P_t}{P_t}
\]
and threshold by horizon as specified.

---

## 7. Model Design

### 7.1 Why GBDT is the baseline-best fit
- Strong on tabular/mixed-scale features and non-linear interactions.
- Less data-hungry than Transformers for daily financial data.
- Quick iteration for ablation and leakage detection.
- Better interpretability (feature importances, SHAP).
- Often produces probabilities that calibrate well with simple post-processing.

### 7.2 Training configurations

#### Option A — 15 Models (Recommended for best accuracy and control)
- Train one 3-class classifier per horizon `h`.
- Advantages:
  - horizon-specific patterns captured cleanly,
  - per-horizon class weights and calibration.
- Costs:
  - 15 trainings, but each is fast.

#### Option B — 2 Models (Recommended for simplicity / fast experimentation)
- One classifier for horizons 1–5 (±3% band),
- One classifier for horizons 6–15 (±5% band).
- Implementation:
  - either label using bucket threshold and predict “bucket direction,”
  - or train on pooled samples across horizons within bucket with horizon id as a feature.

### 7.3 Algorithms
- Primary: **LightGBM multiclass** (`objective=multiclass`)
- Secondary: **CatBoost multiclass** (handles categorical variables, robust defaults)

### 7.4 Hyperparameter baseline (starting points)
**LightGBM**
- `num_leaves`: 63–255
- `max_depth`: -1 or 6–12
- `learning_rate`: 0.02–0.1
- `n_estimators`: 500–4000 (with early stopping)
- `min_data_in_leaf`: 50–500
- `feature_fraction`: 0.6–0.9
- `bagging_fraction`: 0.6–0.9
- `bagging_freq`: 1–10
- `lambda_l1/l2`: tune for stability

**CatBoost**
- `depth`: 6–10
- `learning_rate`: 0.03–0.1
- `iterations`: 1000–6000 (with early stopping)
- `l2_leaf_reg`: 3–30

---

## 8. Imbalance Strategy (Flat Dominates)

### 8.1 Default approach
- Use **class weights** derived from training frequencies.
- Prefer **per-horizon weights** in the 15-model approach.

### 8.2 Alternatives
- Downsample Flat to a target ratio vs Up+Down.
- Tune decision thresholds post-calibration for precision-biased trading.

---

## 9. Calibration

### 9.1 Calibration method
- Baseline: **isotonic regression** or **temperature scaling** on validation set.
- If calibration data is limited:
  - calibrate per bucket (1–5 vs 6–15), not per horizon.

### 9.2 What “good calibration” means
- Reliability curves show predicted 0.7 corresponds to ~70% empirical frequency.
- Brier score improves vs uncalibrated model.

---

## 10. Evaluation Plan

### 10.1 Offline metrics (per horizon or per bucket)
- Per-class F1 (Up/Flat/Down)
- Macro-F1
- Confusion matrices

### 10.2 Trading-aligned metrics (must-have)
- Precision@K (K=1,2 daily picks)
- Hit rate conditional on trading (meets ±3% or ±5% threshold)
- Rank IC:
  - score vs realized forward returns
- Post-cost backtest metrics:
  - turnover, drawdown, Sharpe (optional but recommended)

### 10.3 Baselines
- Prevalence-matched random
- Always-Flat
- Simple heuristic (e.g., last-5-day momentum sign)

Model must beat these baselines out-of-sample to proceed.

---

## 11. Training / Validation Protocol

### 11.1 Time-based splits
- Train: 2021 → `T_train_end`
- Val: next segment
- Test: final segment

### 11.2 Walk-forward (recommended)
Repeat:
- train on expanding window,
- validate on next block,
- test on subsequent unseen blocks.

This reduces “one-regime” overfitting.

---

## 12. Inference and Ranking

### 12.1 Probability outputs
For each symbol/day produce:
- 15-model: `P(up/down/flat)` per horizon
- 2-model: `P(up/down/flat)` per bucket

### 12.2 Ranking score
Define:
- `edge = P(up) - P(down)`

Common scoring choices:
- `score = max(edge_h for h=6..15)` (for 2–3 week holding)
- `score = Σ w_h * edge_h` (weighted sum across horizons)

### 12.3 Trade filters (long-only)
- Only consider trades where `score > θ`
- Pick top 1–2 symbols/day by score
- Optional risk filters (e.g., VIX regime) applied outside model

---

## 13. Deliverables

1. **Training pipeline**:
   - data loader, label generator, split manager, model trainer, evaluator
2. **Model artifacts**:
   - saved LightGBM/CatBoost models per horizon or bucket
   - calibration models/parameters
3. **Reports**:
   - metrics dashboard (CSV/JSON + plots)
   - SHAP summaries
   - ablation results (feature groups)
4. **Daily inference script**:
   - outputs predictions + ranking scores per symbol

---

## 14. Risks and Mitigations

### Risk: Data leakage (especially macro / fundamentals)
- Mitigation: strict “as-of” joins + release-date alignment; time-only splits.

### Risk: Flat class overwhelms learning
- Mitigation: class weights, flat downsampling, precision-focused thresholding.

### Risk: Regime instability (2021–present)
- Mitigation: walk-forward evaluation and periodic retraining.

### Risk: Misleading F1 vs tradability
- Mitigation: track Precision@K and calibrated edge; evaluate post-cost backtest.

---

## 15. Acceptance Criteria

A run is acceptable if, out-of-sample (test or walk-forward):
1. Beats **Always-Flat** and prevalence-matched random on **macro-F1** and **Up/Down F1**.
2. Achieves improved **Precision@1/2** for the actionable class (Up) vs baselines.
3. Shows usable calibration (reliability curve improved; Brier down).
4. Produces stable feature attributions (no obvious leakage signatures).

---

## 16. Implementation Milestones

1. **M1 — Pipeline skeleton** (data, labels, splits, baseline metrics)
2. **M2 — LightGBM 15-model** with class weights + early stopping
3. **M3 — Calibration + trading-aligned metrics**
4. **M4 — SHAP analysis + leakage audit**
5. **M5 — 2-model bucket variant + comparison**
6. **M6 — Walk-forward evaluation + deployment packaging**
