# PRD: ML Model --- 15-Day Directional Path (Hard Labels)

## 1. Purpose

Build an ML model that produces **hard-label directional forecasts** for
a fixed set of tickers. The model supports a **daily cadence** workflow
for a **T+1 execution plan** (manual execution), with a forecast horizon
of **15 trading days**.

The model must: - Provide daily directional guidance (Up / Flat / Down)
for each of the next 15 trading days. - Support downstream portfolio
construction by `trade_daemon`. - Be evaluated primarily against **SPY
total return**, with a goal of **beating SPY by a margin net of
costs**. - Demonstrate positive **beta-adjusted alpha** vs SPY (and
optionally QQQ).

------------------------------------------------------------------------

## 2. Scope

### In Scope

-   Daily feature update using data available up to **close(T)**.
-   Hard-label forecasts for **T+1..T+15**.
-   Deterministic output artifacts consumable by `trade_daemon`.
-   Walk-forward backtesting and performance evaluation.

### Out of Scope

-   Intraday prediction.
-   Execution tactics.
-   Short selling, leverage, options.

------------------------------------------------------------------------

## 3. Universe

Primary universe: - NVDA - PLTR - CRDO - APP - TSM - SPY (benchmark;
optionally modeled asset)

Notes: - Strategy is intentionally tech/semi concentrated. - Defensive
parking uses Fidelity core cash **SPAXX** (implicit cash position).

------------------------------------------------------------------------

## 4. Definitions

-   **T**: current trading day.
-   **T+1**: next trading day.
-   **Horizon**: 15 trading days.
-   **Hard label**: categorical output in {UP, FLAT, DOWN}.

Return definition: - Returns are computed close-to-close from close(T)
to close(T+k).

------------------------------------------------------------------------

## 5. Expected Behavior

### 5.1 Label Definition

For each ticker and horizon day k:

-   Days 1--5:
    -   UP if return ≥ +3%
    -   DOWN if return ≤ −3%
    -   FLAT otherwise
-   Days 6--15:
    -   UP if return ≥ +5%
    -   DOWN if return ≤ −5%
    -   FLAT otherwise

Thresholds must remain fixed unless formally versioned.

### 5.2 No Look-Ahead

-   Features at inference must only use data up to close(T).
-   Labels must never use future data beyond their training cutoff.

### 5.3 Determinism

Given identical data snapshot, feature snapshot, and configuration,
outputs must be reproducible.

------------------------------------------------------------------------

## 6. Data & Features

### 6.1 Minimum Inputs

Per ticker: - OHLCV daily bars - Corporate action adjusted prices -
Calendar features (if used)

Optional: - Earnings proximity flags - Analyst revisions - News
sentiment (timestamp verified)

### 6.2 Feature Update

Feature update recomputes predictors using data through close(T).\
This is separate from model inference.

------------------------------------------------------------------------

## 7. Training Policy

### 7.1 Regime Assumption

Training window begins in **2021** (AI-era regime hypothesis).\
Reduced history implies lower statistical power; model complexity must
remain controlled.

### 7.2 Retraining Cadence

-   Daily inference required.
-   Default retrain: weekly or monthly.
-   Daily retraining allowed only if empirically justified.

### 7.3 Walk-Forward Evaluation

Minimum folds: - 2021--2022 → test 2023 - 2021--2023 → test 2024 -
2021--2024 → test 2025 - 2021--2025 → test 2026 YTD

------------------------------------------------------------------------

## 8. Output Contract

### 8.1 Output Artifact

Machine-readable JSON per run date.

Each ticker has its own model instance (per-ticker model allowed).

Required fields: - run_date - asof_market_close - predictions (15
labels) - model_metadata (per ticker allowed) - data_snapshot_id - code
version hash

### 8.2 Derived Scoring (Recency-Weighted)

Downstream scoring uses recency-weighted labels:

Let y_k ∈ {+1, 0, −1}.

Weighted score: S_w = Σ exp(−λ(k−1)) \* y_k\
Default λ in range 0.10--0.25 (configurable, versioned).

Near-term score: S5 = Σ y_k for k=1..5

Model may emit these; otherwise trade_daemon computes them.

------------------------------------------------------------------------

## 9. Non-Functional Requirements

-   Nightly runtime \< 30 minutes.
-   Fail closed on missing data.
-   Deterministic logging.
-   Drift monitoring.

### 9.1 Audit Retention Policy

Retain: - Model outputs - Feature snapshots (hash) - Configuration -
Model version + git SHA - Data snapshot identifiers

Retention minimum: **7 years** (or indefinitely if storage permits).

------------------------------------------------------------------------

## 10. Backtesting Requirements

Backtest must match live timing: - Signal formed at close(T) - Execution
assumed at open(T+1)

Performance evaluation must include: - CAGR vs SPY - Beta vs SPY -
Regression alpha vs SPY - Sharpe and Sortino - Max drawdown - Worst
day - Turnover and cost drag

------------------------------------------------------------------------

## 11. Acceptance Criteria

Model acceptable only if OUT-OF-SAMPLE:

1.  CAGR ≥ SPY CAGR + 3% (minimum hurdle; configurable).
2.  Regression alpha \> 0 (preferably ≥ 2% annualized).
3.  Performance stable across walk-forward folds.
4.  No excessive reliance on single sub-period.
5.  Worst-day behavior consistent with portfolio risk tolerance.
6.  Turnover compatible with ≤2 trades/day downstream.

Classification metrics (secondary): - Balanced directional accuracy. -
Confusion matrix stability. - Label distribution stability over time.

------------------------------------------------------------------------

## 12. Future Extensions

-   Regime indicators (volatility/rates).
-   Confidence/probability outputs.
-   Recency-weighted training.
-   Cross-ticker feature sharing (if justified).

------------------------------------------------------------------------

Generated on 2026-02-26
