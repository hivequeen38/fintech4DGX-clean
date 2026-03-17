# PRD — Multi-Horizon Stock Direction Classifier Using GBDT (LightGBM/CatBoost)
> **[REVIEWED 2026-03-02 by Claude]** — Inline comments marked with `► COMMENT:` blocks.
> Comments cover: design concerns, data-size risks, label construction ambiguities,
> integration gaps, and acceptance-criteria weaknesses.

---

## 1. Purpose
Build a **production-ready, multi-horizon** stock-direction prediction system using **Gradient-Boosted Decision Trees (GBDT)**—primarily **LightGBM** and/or **CatBoost**—to predict **Up / Flat / Down** for the next **15 trading days**.
This PRD replaces (or complements) a Transformer-based approach with a model family that is typically more suitable for **tabular, engineered, cross-sectional, mixed-scale** feature sets.

> ► **COMMENT (Scope/Positioning):** "Replaces or complements" is deliberately left open, but the integration architecture needs to be decided before M1. Two distinct paths:
> - **Complement**: GBDT outputs are a separate signal alongside Transformer probabilities. Both feed a meta-ranker or ensemble. Requires a unified inference output schema so nightly_run.py can call both.
> - **Replace**: Transformer training is dropped, GBDT takes over both training and inference slots in nightly_run.py. Simpler operationally.
>
> Leaving this ambiguous until M6 risks building two parallel inference pipelines that diverge in their output format. **Recommended**: decide before M1 and lock the output schema (column names, file paths, GCS upload format) so nightly_run.py Phase 1 can route to either backend without a rewrite.

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

> ► **COMMENT (Goal 2):** Three training configurations are listed as co-equal options but they impose different trade-offs on labeling (see FR1 comment below). Suggest picking one as the "production default" now (recommend 15-model for precision) and treating the others as ablation variants. Keeping all three active simultaneously multiplies the artifact management burden by 3×.

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

> ► **COMMENT (Dataset size):** "Moderate dataset sizes (daily bars since 2021)" translates to roughly **1,100–1,300 rows per symbol** (2021-01-01 through 2026-03-01 ≈ 5 years × 252 trading days). After the time-based split (e.g., 70/15/15), you get **~770 training rows, ~195 val, ~195 test per symbol**.
>
> GBDT is indeed more data-efficient than Transformers, but 770 rows is still *small* relative to the typical GBDT sweet spot (10k+ rows). This has several implications:
> - `min_data_in_leaf=50–500` (Section 7.4) — the lower bound (50) may be more appropriate here; 500 would force nearly-flat trees on 770 rows.
> - `n_estimators=500–4000` (Section 7.4) — with so few rows, aggressive early stopping is critical to prevent overfit. Don't treat 4000 as an upper bound to reach; the default should cap lower (e.g., 300–500) and rely on early stopping.
> - Class imbalance is *amplified* at small N: Up/Down events at ±3% threshold may be only 80–150 training examples each for short-horizon labels.
>
> Consider whether **cross-symbol pooling** (all 5 stocks combined → ~3,850 training rows) would produce better generalization for the GBDT, at the cost of symbol-specific feature drift. The PRD is silent on this architecture choice — it must be made explicit.

---

## 5. Requirements

### 5.1 Functional Requirements
**FR1 — Labeling**
- Produce labels `y_{t,h} ∈ {down, flat, up}` for horizons `h=1..15` using:
  - forward returns from close `t` to close `t+h`,
  - threshold `thr(h)=0.03` for `h<=5`, `thr(h)=0.05` for `h>=6`.

> ► **COMMENT (FR1 — Fixed thresholds across symbols):** The same ±3%/±5% thresholds are applied to all symbols, but the active universe spans very different volatility regimes:
> - INOD (small-cap micro): daily ATR can be 5–10% — ±3% is hardly Flat for this stock.
> - NVDA/PLTR (large-cap high-beta): daily ATR ~2–4% — ±3% over 5 days is plausible but tight.
> - CRDO: mid-cap, moderate volatility.
>
> Fixed thresholds will produce dramatically different Up/Flat/Down distributions across symbols. For example, INOD might have 40% Up and 35% Down with ±3% (almost no Flat), while a lower-vol stock might be 20% Up / 60% Flat / 20% Down. This means:
> 1. Class weights trained on one symbol's distribution won't transfer if you ever pool data.
> 2. Acceptance criteria ("beats Always-Flat") are much easier for INOD than for low-vol symbols.
>
> **Recommendation**: Either (a) use symbol-specific thresholds derived from realized volatility (e.g., 0.5× 30-day realized vol), or (b) document the expected class distribution per symbol before finalizing the thresholds. This should be done in M1.
>
> ► **COMMENT (FR1 — Label construction for bucket models):** For Option B (2-model bucket), the PRD says "label using bucket threshold and predict bucket direction." This is ambiguous. There are at least two interpretations:
> 1. **Single label per row**: for a given date `t`, compute the 5-day return `r_{t,5}` and label it. Train one model. At inference, apply to today's row.
> 2. **Pooled rows**: for each date `t`, create 5 training rows (h=1,2,3,4,5), each with its own horizon-specific label but the same feature vector, adding `h` as a feature. Labels for adjacent horizons are then **highly correlated** (r_{t,1} is embedded in r_{t,2}). This inflates the apparent training set size without adding real information and can produce overly optimistic CV scores.
>
> Interpretation 1 is cleaner but loses within-bucket horizon specificity. Interpretation 2 has the correlation problem. The PRD must specify which interpretation is intended.

**FR2 — Model training**
- Support:
  - **15 separate classifiers** (one per horizon), OR
  - **2 bucketed classifiers**: `(1–5)` and `(6–15)`. or
  - **3 bucketed classifiers**: `(1–5), (6-10), and `(11–15)`. or
- Support both **LightGBM** and **CatBoost** training backends.

> ► **COMMENT (FR2 — Multi-task alternative missing):** The PRD only considers independent classifiers or bucketed classifiers. A third option worth evaluating is **multi-output / multi-task LightGBM** (not natively supported, but achievable via stacked label columns + softmax head, or a single LightGBM with `num_class=45` i.e., 15 horizons × 3 classes). Multi-task would let the model share a feature representation across horizons — adjacent horizons share signal (h=1 return is a component of h=5 return), and this structure is thrown away by training 15 independent models.
>
> This is not a blocking issue for M1, but should be an explicit future option in the roadmap.

**FR3 — Class imbalance handling**
- Provide one or more of:
  - class weights (per-horizon preferred),
  - downsampling of Flat,
  - threshold tuning for operating point.

> ► **COMMENT (FR3):** With the current Transformer system, the most persistent failure mode has been **UP class collapse** (F1_up → 0), despite class weights. For GBDT, the same risk exists. The PRD should specify:
> 1. A **minimum acceptable Up/Down F1** (e.g., ≥ 0.25) as a training-time diagnostic, not just a post-hoc metric. If GBDT also collapses UP, training should be flagged and retried with stronger imbalance correction.
> 2. **Decision threshold tuning** should be a first-class citizen, not just an "alternative." At inference, using `argmax(P)` on an imbalanced model will almost always predict Flat. The trading signal should be derived from `P(up) > θ_up` where `θ_up` is tuned on the validation set to achieve a target precision (e.g., Precision@1 ≥ 0.55).

**FR4 — Probability calibration**
- Provide calibration of predicted probabilities:
  - baseline: temperature scaling or isotonic calibration on validation set,
  - optional per-horizon calibration (if enough data).

> ► **COMMENT (FR4):** With ~195 validation rows and a 3-class problem, **isotonic regression will overfit badly** — isotonic regression is non-parametric and requires enough points to estimate a monotone function reliably (rule of thumb: 100+ per class, so 300+ total for 3 classes). With ~195 validation rows and, say, 30 Up events in val, isotonic regression on Up probabilities has extremely high variance.
>
> **Recommended order of preference for this dataset size**:
> 1. **Temperature scaling** (1 parameter, fits on tiny val sets, low overfit risk)
> 2. **Platt scaling** (2 parameters, still low risk)
> 3. Isotonic regression only if val set grows beyond ~500 rows per class
>
> The PRD should demote isotonic to "only if calibration data is sufficient" and promote temperature scaling as the default.

**FR5 — Explainability**
- Provide:
  - feature importance (gain/split),
  - SHAP values on validation/test,
  - per-horizon SHAP summaries (15-model) or per-bucket (2-model).

> ► **COMMENT (FR5 — SHAP for leakage detection):** This is the most valuable use case for SHAP in this system. In the Transformer-based pipeline, SHAP analysis (shap_analysis.log) was run for NVDA. The GBDT equivalent would be much faster and more reliable.
>
> Specific leakage signatures to watch for:
> - Any feature that ranks in top-3 SHAP *and* contains any look-ahead (e.g., next-period volume, same-day sentiment published after close)
> - `DTE`/`DSE` (days-to/since-earnings) features: high SHAP might indicate the model learned earnings-event patterns rather than directional signal — not leakage, but worth flagging
> - IV features (iv_30d, iv_skew_30d, iv_term_ratio): if these have very high SHAP on short horizons, verify that IV was available at market close on date `t` (not published after close)
>
> **Recommend adding a "SHAP leakage checklist" table to M4 deliverables**: for each top-10 SHAP feature, document (a) data source, (b) publication timing, (c) whether it's as-of `t` or `t+1`.

**FR6 — Evaluation**
- Provide:
  - per-horizon metrics (macro-F1, per-class F1, confusion matrix),
  - bucket-level metrics (1–5 and 6–15),
  - trading-aligned metrics: Precision@K (K=1,2), hit rate conditional on trading, rank IC.

> ► **COMMENT (FR6 — Precision@K ambiguity):** "Precision@K (K=1,2 daily picks)" is underspecified. Does it mean:
> - Top-K stocks per day from the 5-symbol basket? (K=1 means "best-ranked stock" each day)
> - K trades total over the test period?
>
> If it's the former (more likely), on a 5-symbol basket the precision@1 baseline from random is 20% (1/5). This should be stated explicitly so the acceptance criteria (Section 15) can be made concrete.
>
> **Also note**: "Rank IC" (rank information coefficient) is the Spearman correlation between `edge_h` score and realized forward returns. This is a continuous-output metric that doesn't care about the Up/Flat/Down classification. It can be positive even when all predictions are "Flat" (if the flat-class probabilities are slightly ordered correctly). It should complement, not substitute for, per-class F1.

**FR7 — Inference**
- Daily inference for each symbol produces:
  - `P(up), P(flat), P(down)` per horizon (15-model) or per bucket (2-model).
- Expose a single scalar score for ranking, e.g.:
  - `edge = P(up) - P(down)` and
  - `score = max(edge_h for h=6..15)` or weighted sum.

> ► **COMMENT (FR7 — Score definition):** `score = max(edge_h for h=6..15)` picks the single best horizon's edge. This is optimistic and noisy (one outlier horizon with high P(up) due to sampling noise can dominate). A weighted sum across horizons (e.g., linear ramp weighting longer horizons more) is more stable. **Recommend**: use the weighted sum as the default, expose `max` as a variant, and include a comparison in M3/FR6 evaluation.
>
> Also: after calibration, `edge = P(up) - P(down)` is a valid signed probability. But before calibration, raw GBDT softmax outputs can be poorly scaled (P(up) can be > 0.5 even for near-random predictions on imbalanced data because softmax is relative). **Enforce: compute `edge` only from post-calibration probabilities.**

**FR8 — Reproducibility**
- Every training run must save:
  - data range,
  - feature list and transformations,
  - hyperparameters,
  - random seeds,
  - calibration parameters,
  - metrics and plots.

> ► **COMMENT (FR8 — Integration with existing naming conventions):** The current system saves models to `/workspace/model/model_{SYMBOL}_{model_name}_fixed_noTimesplit_{1-15}.pth`. GBDT artifacts (`.txt` for LightGBM, `.cbm` for CatBoost) need a parallel naming scheme. Suggest:
> - `/workspace/model/gbdt_{SYMBOL}_{model_name}_h{horizon}.{ext}` for 15-model
> - `/workspace/model/gbdt_{SYMBOL}_{model_name}_bucket{bucket}.{ext}` for 2-model
>
> Calibration models (temperature scalars / isotonic pickles) should live alongside: `gbdt_{SYMBOL}_{model_name}_h{horizon}_calib.pkl`
>
> The run manifest (data range, features, hyperparams, seeds) should be saved as JSON alongside: `gbdt_{SYMBOL}_{model_name}_run_manifest.json`

---

### 5.2 Non-Functional Requirements
**NFR1 — No leakage**
- Strict time-based splitting.
- Feature alignment "as-of" timestamp, especially for non-daily sources.

> ► **COMMENT (NFR1 — Specific leakage risks in this pipeline):** Beyond the standard "use time-only splits" guidance, there are specific leakage risks in the current feature set:
> 1. **AAII sentiment**: published weekly (Thursday after close). If today is Monday and AAII survey from last Thursday is used, that's correct. But if the pipeline uses the *current week's* survey (published Thursday) for Monday–Wednesday predictions, that's a 3-day look-ahead.
> 2. **Implied volatility (iv_30d)**: option IV from the end-of-day snapshot should be available at market close. Verify that the backfill source (Polygon/Alpha Vantage) provides end-of-day IV *on* the trading day, not the next morning.
> 3. **Earnings dates (DTE/DSE)**: confirmed safe (these are pre-announced). But DSE (days-since-earnings) includes the earnings day itself — verify that same-day earnings surprises are not embedded in the feature.
> 4. **Forward-fill of missing values**: if IV is missing on a given day and forward-filled from the previous day, that's fine. But if *back-filled* (filled from a future day), that's leakage. The cp_ratio_backfill.py code uses `fillna(0.0)` for missing IV — safe for the model but worth checking whether zero-IV is a meaningful imputation vs NaN-masking.

**NFR2 — Runtime**
- Training should complete in minutes per horizon on a typical workstation.
- Daily inference should run in seconds per basket.

> ► **COMMENT (NFR2):** With 770 training rows and LightGBM, training 15 models for 1 symbol will take **5–30 seconds** total on a single CPU core. For 5 symbols × 15 models = 75 models, the whole training pass should be **under 5 minutes** even without GPU. This is a major advantage over the current Transformer pipeline (~2.75 hours). No GPU needed for LightGBM baseline; GPU support is available in LightGBM via `device='gpu'` but the speedup at this data size is marginal.

**NFR3 — Maintainability**
- Modular pipeline:
  - data → features → labels → train → calibrate → evaluate → predict.

> ► **COMMENT (NFR3):** The modular pipeline description is correct, but given the existing codebase structure (`fetchBulkData.py`, `trendAnalysisFromTodayNew.py`, `mainDeltafromToday.py`, `nightly_run.py`), the GBDT pipeline should be a **new module** (e.g., `gbdt_pipeline.py` or `trendAnalysisGBDT.py`) that is callable from `mainDeltafromToday.main()` via `model_type='lgbm'` or `model_type='catboost'` — consistent with the existing `model_type` dispatch added for `trans_mz`. This avoids creating a parallel entry point that drifts from nightly_run.py.

**NFR4 — Observability**
- Logging:
  - input data checksums,
  - prediction distributions,
  - drift monitoring hooks (optional).

> ► **COMMENT (NFR4):** "Drift monitoring hooks (optional)" should be elevated to recommended for a production system. The most lightweight useful check is: compare the mean and std of each feature in today's inference row vs the training set distribution. Flag (don't fail) if any feature is >3σ from training mean. This catches data pipeline issues (e.g., IV backfill gap, missing sentiment) before they silently corrupt predictions.

---

## 6. Data and Feature Interface

### 6.1 Inputs
- Daily feature vector per symbol per date:
  - `X_t ∈ R^D` (engineered features, cross-sectional aggregates, macro proxies as-of t)

> ► **COMMENT (Section 6.1 — Feature list is completely absent):** This section is critically underspecified. The feature set `X_t ∈ R^D` does not list any features. For the GBDT to be implementable, the feature list must be defined. The existing system uses (per `NVDA_param.py` AAII_option_vol_ratio selected_columns):
> - Price/volume technicals (RSI, MACD, Bollinger bands, ATR, OBV, etc.)
> - Sentiment (AAII bull/bear ratio, options volume ratio, cp_sentiment_ratio)
> - IV features: iv_30d, iv_skew_30d, iv_term_ratio
> - Structural: DTE (days-to-earnings), DSE (days-since-earnings)
> - CP ratio: cp_volume_ratio, cp_oi_ratio (some commented out)
>
> The GBDT PRD should either (a) inherit this exact feature set from the existing param files, or (b) define a new canonical feature list. Leaving it as `R^D` means M1 cannot start without a separate feature spec conversation.
>
> Additionally, note that LightGBM handles missing values natively (splits on NaN), while CatBoost has its own NaN handling. The zero-fill approach used by cp_ratio_backfill.py (filling missing IV with 0.0) may actually hurt GBDT — LightGBM would interpret 0.0 as a valid observed value, not as "missing," and could learn spurious patterns. **Recommend: preserve NaN for missing IV values and let LightGBM handle them natively**.

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

> ► **COMMENT (Section 6.3 — Label alignment):** `P_{t+h}` is the **closing price h trading days after date t**. To avoid leakage, when training on date `t`, the label `y_{t,h}` requires knowing price at `t+h`. This is fine for historical training. For inference on today (`t=today`), no label exists by definition — but the *feature vector* `X_t` is computed from data available at close of `t`. Confirm that no feature in `X_t` uses prices or volumes from after close of `t`.
>
> Also: the formula uses closing price `P_t` as both the start of the forward return and the feature-generation anchor. Make sure the feature pipeline (RSI, MACD, etc.) also uses close-of-`t`, not open-of-`t+1` or intraday prices.

---

## 7. Model Design

### 7.1 Why GBDT is the baseline-best fit
- Strong on tabular/mixed-scale features and non-linear interactions.
- Less data-hungry than Transformers for daily financial data.
- Quick iteration for ablation and leakage detection.
- Better interpretability (feature importances, SHAP).
- Often produces probabilities that calibrate well with simple post-processing.

> ► **COMMENT (Section 7.1):** All five points are correct. One additional advantage worth stating: **GBDT is more robust to irrelevant features**. The Transformer system requires careful `selected_columns` tuning (as seen in NVDA_param.py / PLTR_param.py). LightGBM's built-in feature selection (via `feature_fraction` and gain-based importance) means you can safely include a larger initial feature set and let SHAP tell you what's actually useful — reducing the manual feature curation burden.

### 7.2 Training configurations

#### Option A — 15 Models (Recommended for best accuracy and control)
- Train one 3-class classifier per horizon `h`.
- Advantages:
  - horizon-specific patterns captured cleanly,
  - per-horizon class weights and calibration.
- Costs:
  - 15 trainings, but each is fast.

> ► **COMMENT (Option A):** Agreed this is the right default. One practical concern: with 15 independent models, **you lose inter-horizon consistency**. It's possible for the model to predict Up at h=5 but Down at h=4, which is economically incoherent (if price is up 3% in 5 days, it almost certainly didn't fall 5% in 4 days). This is not a blocking issue for v1 but is worth flagging in reports when it occurs. A future isotonic constraint (ensure predictions are monotone in some sense across horizons) could be added in M5/M6.

#### Option B — 2 Models (Recommended for simplicity / fast experimentation)
- One classifier for horizons 1–5 (±3% band),
- One classifier for horizons 6–15 (±5% band).
- Implementation:
  - either label using bucket threshold and predict "bucket direction,"
  - or train on pooled samples across horizons within bucket with horizon id as a feature.

> ► **COMMENT (Option B — Label ambiguity, see also FR1 comment):** The "either/or" implementation choice has fundamentally different semantics:
>
> **Choice 1** (single label per row, e.g., h=5 return for bucket 1–5): loses h=1,2,3,4 signal entirely. Simpler but wasteful.
>
> **Choice 2** (pooled rows, horizon_id as feature): 5× more training rows for bucket 1–5, 10× for bucket 6–15 — but those rows are not independent (5 rows from same date share the same feature vector `X_t`, only labels differ). Cross-validation must be done by date, not by row, or you'll see massive validation set leakage (the same feature vector appears in both train and val folds). This is a **critical implementation constraint** that must be documented.
>
> **Recommendation**: For M2, implement Option A (15 models) first. Implement Option B only in M5 as an explicit ablation, and default to Choice 1 (median-horizon label per bucket) to avoid the pooling pitfall.

### 7.3 Algorithms
- Primary: **LightGBM multiclass** (`objective=multiclass`)
- Secondary: **CatBoost multiclass** (handles categorical variables, robust defaults)

> ► **COMMENT (Section 7.3 — Categorical variables):** The existing feature set includes `daily_sentiment` (string: 'bullish', 'bearish', 'neutral', 'no_trades'). LightGBM requires this to be integer-encoded (label encoding or one-hot). CatBoost can ingest it as-is if marked as a categorical column. The data loading code must handle this differently for each backend. Flag this as a preprocessing concern in M1.
>
> Also: `horizon_id` if used in Option B is a numeric-but-ordinal feature. Both LightGBM and CatBoost will treat it as numeric, which is correct.

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

> ► **COMMENT (Hyperparameter ranges — dataset-size mismatch):** The ranges listed are calibrated for medium-large datasets (10k–1M rows). For ~770 training rows, several adjustments are needed:
> - `num_leaves`: 63–255 → **15–63**. With 770 rows, 255 leaves means each leaf has ~3 rows — extreme overfit. Restrict to 15–63.
> - `min_data_in_leaf`: 50–500 → **20–100**. At 770 rows, min_data=500 would force a 1-leaf tree (entire dataset in one leaf).
> - `n_estimators`: 500–4000 → **100–500 with early stopping on val set**. 4000 trees × small dataset = guaranteed overfit.
> - `max_depth`: Keep 6 as a reasonable ceiling, or use `num_leaves` as the primary constraint (LightGBM's leaf-wise growth means `max_depth` is secondary when `num_leaves` is set).
>
> **Recommended starter config for this dataset size**:
> ```python
> lgb_params = dict(
>     objective='multiclass', num_class=3,
>     num_leaves=31, min_data_in_leaf=30,
>     learning_rate=0.05, n_estimators=300,
>     feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
>     lambda_l1=0.1, lambda_l2=1.0,
>     early_stopping_rounds=30,
> )
> ```

**CatBoost**
- `depth`: 6–10
- `learning_rate`: 0.03–0.1
- `iterations`: 1000–6000 (with early stopping)
- `l2_leaf_reg`: 3–30

> ► **COMMENT (CatBoost ranges):** Same size concern: `iterations=1000–6000` is large for 770 rows. With early stopping, CatBoost will self-regulate, but set `early_stopping_rounds=50` and expect to stop at 200–400 iterations in practice. `depth=6–10` is OK for CatBoost (it uses symmetric trees, so depth is less prone to overfit than LightGBM's leaf-wise growth). Recommend depth=6 as the default.

---

## 8. Imbalance Strategy (Flat Dominates)

### 8.1 Default approach
- Use **class weights** derived from training frequencies.
- Prefer **per-horizon weights** in the 15-model approach.

> ► **COMMENT (Section 8.1):** For LightGBM multiclass, class weights are specified via `class_weight` parameter (dict mapping class label to weight) or `is_unbalance=True` (automatic). **Do not use `is_unbalance=True` for 3-class multiclass** — it applies equal weighting within each class regardless of per-horizon distribution, which may not be correct. Use explicit `class_weight={0: w_down, 1: w_flat, 2: w_up}` computed from training label frequencies per horizon.
>
> **Expected class distributions** (rough estimate, h=1, NVDA-style):
> - Up (>3%): ~15–20% of training days
> - Down (<-3%): ~15–20%
> - Flat: ~60–70%
>
> Inverse-frequency weights: Up≈4×, Down≈4×, Flat≈1×. This should approximately match what the Transformer pipeline was doing.

### 8.2 Alternatives
- Downsample Flat to a target ratio vs Up+Down.
- Tune decision thresholds post-calibration for precision-biased trading.

> ► **COMMENT (Section 8.2):** Threshold tuning is the most practically useful technique for trading applications where you care about precision of the "Up" signal, not overall accuracy. After calibration, find `θ_up` on the validation set such that `Precision(Up | P(up) > θ_up) ≥ 0.55` (or whatever target precision is acceptable for trade entry). This reduces coverage (fewer trade signals) but improves hit rate. **This should be elevated to a first-class deliverable in M3**, not listed as an alternative.

---

## 9. Calibration

### 9.1 Calibration method
- Baseline: **isotonic regression** or **temperature scaling** on validation set.
- If calibration data is limited:
  - calibrate per bucket (1–5 vs 6–15), not per horizon.

> ► **COMMENT (Section 9.1 — Revisit priority order):** As noted in FR4 comments, isotonic regression should not be the baseline here — temperature scaling should. The conditional "if calibration data is limited" will *always* be true for per-horizon calibration with 195 validation rows. Suggest rewriting this section as:
>
> - **Default (recommended)**: Temperature scaling per horizon (1 parameter per horizon, 15 parameters total). Fit on val set. Generalizes well even at low N.
> - **Fallback**: Per-bucket temperature scaling (1 parameter per bucket, 2–3 parameters total) if per-horizon val sets are too small after class-stratified splitting.
> - **Advanced (only if val N > 500 per class)**: Isotonic regression per bucket.

### 9.2 What "good calibration" means
- Reliability curves show predicted 0.7 corresponds to ~70% empirical frequency.
- Brier score improves vs uncalibrated model.

> ► **COMMENT (Section 9.2):** With small N, reliability curves (typically 10 bins) will have very noisy bins. Use fewer bins (5–7) and report confidence intervals on the empirical frequency (Wilson interval). A reliability curve with 10-bin isotonic is nearly meaningless at 195 total validation points. The Brier score improvement is a more robust summary statistic for this sample size.

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

> ► **COMMENT (Section 10.2 — "Optional but recommended" Sharpe):** The Sharpe and post-cost metrics should be elevated to **required** rather than optional, especially since the system targets 1–2 live trades/day. Without a post-cost backtest, it's impossible to distinguish a predictive signal from one that's predictive but too costly to exploit (high-turnover h=1 signals often fall into this category). A simple vectorized backtest (long if `score > θ`, hold for h days, subtract 5–10bps one-way cost) takes 20 lines of pandas code and should be in M3.

### 10.3 Baselines
- Prevalence-matched random
- Always-Flat
- Simple heuristic (e.g., last-5-day momentum sign)

> ► **COMMENT (Section 10.3 — Baseline coverage):** Good selection. Add one more: **Always-Up** baseline. Given that the stocks in the universe (NVDA, PLTR, APP, CRDO, INOD) are high-momentum names, "always predict Up" may have surprisingly high Precision@1 during bull runs. Knowing the Always-Up precision sets a concrete bar that the model must beat *without* simply predicting Up for everything.

Model must beat these baselines out-of-sample to proceed.

---

## 11. Training / Validation Protocol

### 11.1 Time-based splits
- Train: 2021 → `T_train_end`
- Val: next segment
- Test: final segment

> ► **COMMENT (Section 11.1 — Split proportions unspecified):** The PRD doesn't specify what fraction of data goes to each split. For ~1,200 rows, common options:
> - 70/15/15: ~840/180/180 rows. Val and test are thin (especially for per-class calibration).
> - 80/10/10: ~960/120/120. Test is very thin — 120 days ≈ 6 months of trading, only 1–2 regime regimes.
> - 60/20/20: ~720/240/240. Better calibration/test N but less training data.
>
> **Recommendation for M1**: define the split as a fixed calendar date (e.g., train through 2024-06-30, val 2024-07-01 to 2024-12-31, test 2025-01-01 onward). This is more interpretable than percentage splits and makes it easy to reason about what market regimes are in each split.

### 11.2 Walk-forward (recommended)
Repeat:
- train on expanding window,
- validate on next block,
- test on subsequent unseen blocks.

This reduces "one-regime" overfitting.

> ► **COMMENT (Section 11.2 — Walk-forward complexity):** Walk-forward is the right approach for production but significantly increases implementation complexity. With 1,200 rows and, say, 6-month blocks, you'd get ~6 walk-forward folds from 2021 to 2025. Each fold requires retraining all 15 models × 5 symbols = 75 models. Total: 75 × 6 = 450 training runs per full walk-forward evaluation. At ~5 minutes per full training pass, that's 37.5 minutes — still fast enough for a single evaluation run.
>
> **However**: the PRD lists walk-forward as M6 (the last milestone). Consider moving at least a **2-fold walk-forward** to M2 to validate that the model doesn't simply overfit the single train/test split. If the model collapses UP class or shows near-zero Out-of-sample F1 on 2 consecutive folds, it's better to know that at M2 than at M6.

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

> ► **COMMENT (Section 12.2 — Score integration with existing HTML report):** The current system outputs predictions to `{SYMBOL}_15d_from_today_predictions.csv` and renders them to HTML (`stock_trends.html`) uploaded to GCS. The GBDT inference must produce the same (or compatible) output format. Specifically:
> - Column schema of `{SYMBOL}_15d_from_today_predictions.csv` must be preserved
> - `edge` score per horizon should be added as a new column alongside the existing class predictions
> - The HTML render function must be updated to display calibrated probabilities
>
> This integration is missing from the PRD and should be added as a sub-deliverable in M6 (or earlier if the GBDT replaces the Transformer).

### 12.3 Trade filters (long-only)
- Only consider trades where `score > θ`
- Pick top 1–2 symbols/day by score
- Optional risk filters (e.g., VIX regime) applied outside model

> ► **COMMENT (Section 12.3 — θ calibration):** `θ` is the decision threshold for entering a trade. The PRD doesn't specify how `θ` is set or validated. This threshold directly determines the precision-recall trade-off of the trading system:
> - Low `θ`: more trades, lower precision, higher turnover
> - High `θ`: fewer trades, higher precision, risk of no signal on many days
>
> **θ must be calibrated on the validation set** (not test set, to avoid look-ahead bias) by optimizing for a business metric (e.g., maximize Precision@1 subject to at least 1 signal per week). Document this process in M3.

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

> ► **COMMENT (Section 13 — Missing deliverable: param files):** The existing system uses symbol-specific param files (`NVDA_param.py`, `PLTR_param.py`, etc.) to define `selected_columns`, `model_name`, hyperparameters. The GBDT system should follow the same pattern: `NVDA_gbdt_param.py` (or extend the existing param files with a `lgbm_reference` dict). This makes hyperparameter auditing, A/B comparison, and rollback straightforward — the same way `mz_reference_v2` was added to `PLTR_param.py` without touching production config.

---

## 14. Risks and Mitigations

### Risk: Data leakage (especially macro / fundamentals)
- Mitigation: strict "as-of" joins + release-date alignment; time-only splits.

### Risk: Flat class overwhelms learning
- Mitigation: class weights, flat downsampling, precision-focused thresholding.

### Risk: Regime instability (2021–present)
- Mitigation: walk-forward evaluation and periodic retraining.

### Risk: Misleading F1 vs tradability
- Mitigation: track Precision@K and calibrated edge; evaluate post-cost backtest.

> ► **COMMENT (Section 14 — Missing risks):**
>
> **Risk: Small dataset → GBDT overfits training split**
> - Mitigation: Restrict `num_leaves` and `min_data_in_leaf` to dataset-appropriate ranges (see Section 7.4 comments). Monitor train vs val loss curves. If val loss starts diverging before early stopping triggers, reduce model capacity.
>
> **Risk: 5-symbol universe is too concentrated for cross-sectional evaluation**
> - Precision@1 computed across 5 symbols daily has very high variance. A bad prediction for INOD (small-cap, high-vol) can dominate metrics. Consider reporting metrics separately for large-cap (NVDA, PLTR, APP) and small-cap (CRDO, INOD) subsets.
>
> **Risk: Inconsistent IV coverage breaks inference feature vector**
> - If IV backfill has a gap on a given day, the inference feature vector has NaN where the training set had 0.0 (from the zero-fill). This will cause LightGBM to treat it differently. Either keep NaN throughout (training + inference) or keep 0.0 throughout — but do not mix strategies.

---

## 15. Acceptance Criteria

A run is acceptable if, out-of-sample (test or walk-forward):
1. Beats **Always-Flat** and prevalence-matched random on **macro-F1** and **Up/Down F1**.
2. Achieves improved **Precision@1/2** for the actionable class (Up) vs baselines.
3. Shows usable calibration (reliability curve improved; Brier down).
4. Produces stable feature attributions (no obvious leakage signatures).

> ► **COMMENT (Section 15 — Criteria need quantification):** The criteria as written are directional ("beats", "improved", "usable") but not quantified. This makes it impossible to objectively gate a model as "pass/fail" at milestone review. Recommend adding numeric thresholds:
>
> | Criterion | Minimum bar |
> |-----------|-------------|
> | Up-class F1 (per horizon avg) | ≥ 0.25 |
> | Macro-F1 (per horizon avg) | ≥ 0.30 |
> | Precision@1 (daily, val set) | ≥ 0.50 (vs random ~0.20 for 5-symbol basket) |
> | Brier score reduction vs uncalibrated | ≥ 5% |
> | No SHAP leakage signature | Top-5 features all have valid as-of timestamps |
>
> These numbers are provisional and should be validated against the Always-Flat and Always-Up baselines on actual data in M1. Adjust thresholds upward once baseline performance is measured.
>
> ► **COMMENT (Section 15 — "Beats Always-Flat on macro-F1" is a very low bar):** Always-Flat achieves macro-F1 ≈ (0 + F1_flat + 0) / 3 = F1_flat / 3. If Flat is 60% of the data, F1_flat ≈ 0.75 (high recall, 100% precision), so Always-Flat macro-F1 ≈ 0.25. A model that predicts mostly Flat with occasional UP can beat this trivially. **The real bar should be Up-class F1 > 0.25 AND Down-class F1 > 0.20**, since those are the trading-relevant classes.

---

## 16. Implementation Milestones

1. **M1 — Pipeline skeleton** (data, labels, splits, baseline metrics)
2. **M2 — LightGBM 15-model** with class weights + early stopping
3. **M3 — Calibration + trading-aligned metrics**
4. **M4 — SHAP analysis + leakage audit**
5. **M5 — 2-model bucket variant + comparison**
6. **M6 — Walk-forward evaluation + deployment packaging**

> ► **COMMENT (Section 16 — Milestone sequencing):**
>
> The milestone order is logical but has a few gaps:
>
> 1. **M1 should include feature list finalization.** The PRD currently doesn't specify `R^D`. Before writing any pipeline skeleton code, the feature list must be agreed upon (see Section 6.1 comment). Add to M1: "Define canonical feature set per symbol; document expected column schema of input DataFrame."
>
> 2. **M2 should include a walk-forward smoke test (2 folds).** See Section 11.2 comment — catching overfit early is better than at M6.
>
> 3. **CatBoost is listed in FR2 and Section 7.3 but has no milestone.** If the plan is LightGBM-first (correct strategy), CatBoost should appear as an optional M5 variant alongside the 2-model bucket comparison.
>
> 4. **Integration with nightly_run.py is in M6 ("deployment packaging").** This is late — it means you'd have a complete GBDT pipeline with no way to run it nightly until the final milestone. Suggest adding a **M3.5 — basic nightly_run.py integration** step (GBDT as a new `model_type='lgbm'` route in mainDeltafromToday.main()) so the pipeline is smoke-testable in the nightly loop early.
>
> **Revised milestone suggestion**:
> - M1: Pipeline skeleton + **feature list finalization** + baseline metrics + data checksums
> - M2: LightGBM 15-model + class weights + early stopping + **2-fold walk-forward smoke test**
> - M3: Calibration + threshold tuning + trading-aligned metrics + **basic post-cost backtest**
> - M4: SHAP analysis + leakage audit + feature ablations
> - M4.5: nightly_run.py integration (model_type='lgbm' route, HTML report output)
> - M5: 2-model bucket variant + CatBoost comparison
> - M6: Full walk-forward evaluation + deployment hardening + observability (drift monitoring)

---

*End of reviewed PRD. Summary of top issues requiring decisions before M1:*
1. **Feature list**: What is `R^D`? Define the canonical GBDT feature set from existing param files.
2. **Symbol-specific vs pooled training**: Per-symbol models (current approach) or cross-sectional pool?
3. **GBDT output format**: Must be compatible with existing `{SYMBOL}_15d_from_today_predictions.csv` schema and nightly_run.py upload.
4. **Complement vs Replace Transformer**: Decide architecture before M1 to avoid dual-pipeline drift.
5. **Hyperparameter ranges**: Adjust all LightGBM defaults down to match the ~770-row training set.
6. **IV NaN strategy**: Pick one (NaN or 0.0) consistently across training and inference.
