"""NVDA LightGBM GBDT param file.

Feature list mirrors NVDA_param.AAII_option_vol_ratio.selected_columns
(minus 'label') — already validated through Transformer SHAP tuning passes.

lgbm_reference      : Tier1 + semiconductor sector + IV (all active)
lgbm_reference_base : same without Tier 3b IV features (use if IV backfill
                      for this symbol is reset or models are retrained cold)
"""

lgbm_reference = {
    # ── Identity ──────────────────────────────────────────────────────
    "symbol":     "NVDA",
    "model_name": "lgbm_reference",
    "comment":    "NVDA GBDT — Tier1 + semiconductor sector + IV (backfill complete 2026-03-01)",

    # ── Labeling ──────────────────────────────────────────────────────
    "threshold": 0.05,   # passed for reference; THR_SHORT/LONG hardcoded in pipeline

    # ── Feature list (no 'label') ─────────────────────────────────────
    "selected_columns": [
        # --- Tier 1: Price / volume ---
        "adjusted close",
        "daily_return",
        "volume",
        "Volume_Oscillator",
        "volatility",
        "VWAP",
        "high",
        "low",
        "volume_volatility",
        # --- Tier 1: Company fundamentals ---
        "EPS",
        "estEPS",
        "surprisePercentage",
        "dte",
        "dse",
        "earn_in_5",
        "earn_in_10",
        "earn_in_20",
        "totalRevenue",
        "netIncome",
        # --- Tier 1: Idiosyncratic return vs market ---
        "ret_5d_rel_SPY",
        "ret_10d_rel_SPY",
        # --- Tier 2: Idiosyncratic return vs semiconductor sector ---
        "ret_5d_rel_SMH",
        "ret_10d_rel_SMH",
        # --- Tier 1: Realized volatility ---
        "rv_10d",
        "rv_20d",
        "rv_term_ratio",
        "vix_rv_ratio",
        # --- Tier 1: Analyst estimates ---
        "eps_est_avg",
        # --- Tier 1: Momentum ---
        "MACD_Signal",
        "MACD",
        "MACD_Hist",
        "ATR",
        "RSI",
        "Real Upper Band",
        "Real Middle Band",
        "Real Lower Band",
        # --- Tier 1: Rates / FX ---
        "interest",
        "10year",
        "2year",
        "DTWEXBGS",
        "DFEDTARU",
        "BOGMBBM",
        "jpy_close",
        "twd_close",
        # --- Tier 1: Indices ---
        "SPY_close",
        "qqq_close",
        "VTWO_close",
        "SPY_stoch",
        "calc_spy_oscillator",
        "QQQ_stoch",
        "VTWO_stoch",
        # --- Tier 1: Macro ---
        "VIXCLS",
        "DCOILWTICO",
        "USEPUINDXD",
        "UMCSENT",
        "BSCICP02USM460S",
        "DGORDER",
        "PCU33443344",
        "SAHMREALTIME",
        "JTSJOL",
        "GDP",
        "FINRA_debit",
        "Spread",
        # --- Tier 1: Metadata ---
        "day_of_week",
        "month",
        "price_lag_1",
        "price_lag_5",
        "price_lag_15",
        "price_change_1",
        "price_change_5",
        "price_change_15",
        # --- Tier 2: Semiconductor sector peers ---
        "rs_amd",
        "rs_amd_trend",
        "rs_intc_trend",
        "rs_avgo_trend",
        "rs_smh_trend",
        # --- Tier 3a: Options flow (all symbols) ---
        "cp_sentiment_ratio",
        "options_volume_ratio",
        # --- Tier 3b: Implied Volatility (NVDA backfill complete) ---
        "iv_30d",
        "iv_skew_30d",
        "iv_term_ratio",
    ],

    # ── Calendar splits ───────────────────────────────────────────────
    "start_date":     "2021-03-01",
    "train_end_date": "2024-03-31",
    "val_end_date":   "2024-09-30",
    # test: 2024-10-01 → present

    # ── LightGBM overrides ────────────────────────────────────────────
    "lgbm_params": {
        "num_leaves":        47,   # balanced complexity for band architecture
        "min_data_in_leaf":  15,   # guard against leaf overfitting in stacked dataset
        "lambda_l2":         3.0,  # stronger regularisation for more data
        "n_estimators":      500,
        "feature_fraction":  0.6,
        "bagging_fraction":  0.75,
    },

    # ── Per-ticker acceptance overrides ───────────────────────────────
    # short_dn_f1: h=1 structurally sparse (few 1-day >5% moves) → relax
    "acceptance_overrides": {
        "short_dn_f1": 0.08,
    },

    # ── Calibration ───────────────────────────────────────────────────
    "calibration": "temperature",
}

# Without Tier 3b IV — fallback if IV columns are unavailable
lgbm_reference_base = {
    **lgbm_reference,
    "model_name": "lgbm_reference_base",
    "selected_columns": [
        c for c in lgbm_reference["selected_columns"]
        if c not in ("iv_30d", "iv_skew_30d", "iv_term_ratio")
    ],
}
