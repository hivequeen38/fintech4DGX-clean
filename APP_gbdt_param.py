"""APP LightGBM GBDT param file.

Feature list mirrors APP_param.AAII_option_vol_ratio.selected_columns
(minus 'label') — already validated through Transformer SHAP tuning passes.

lgbm_reference      : Tier1 + gaming/social sector + Tier3a options + IV
                      (backfill complete 2026-03-06)
lgbm_reference_base : same without Tier 3b IV columns (fallback if IV unavailable)
"""

lgbm_reference = {
    # ── Identity ──────────────────────────────────────────────────────
    "symbol":     "APP",
    "model_name": "lgbm_reference",
    "comment":    "APP GBDT — Tier1 + gaming/social sector + options + IV (backfill complete 2026-03-06)",

    # ── Labeling ──────────────────────────────────────────────────────
    "threshold": 0.05,

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
        "T10Y2Y",
        "M2SL",
        "DTWEXBGS",
        "DFEDTARU",
        "BOGMBBM",
        "eur_close",
        # --- Tier 1: Indices ---
        "SPY_close",
        "qqq_close",
        "VTWO_close",
        "SPY_stoch",
        "calc_spy_oscillator",
        "QQQ_stoch",
        # --- Tier 1: Macro ---
        "VIXCLS",
        "unemploy",
        "UMCSENT",
        "CPIAUCSL",
        "GDP",
        "BUSLOANS",
        "RSAFS",
        "Spread",
        # --- Tier 1: Metadata ---
        "day_of_week",
        "month",
        "price_lag_1",
        "price_lag_5",
        "price_lag_15",
        "price_change_1",
        "price_change_5",
        # --- Tier 2: Gaming / social media ETF sector ---
        "rs_gamr",
        "rs_socl",
        "rs_gamr_trend",
        "rs_socl_trend",
        # --- Tier 3a: Options flow ---
        "cp_sentiment_ratio",
        "options_volume_ratio",
        # --- Tier 3b: Implied Volatility (APP backfill complete 2026-03-06) ---
        "iv_30d",
        "iv_skew_30d",
        "iv_term_ratio",
    ],

    # ── Calendar splits ───────────────────────────────────────────────
    "start_date":     "2021-07-01",
    "train_end_date": "2024-03-31",
    "val_end_date":   "2024-09-30",

    # ── LightGBM overrides ────────────────────────────────────────────
    "lgbm_params": {
        "num_leaves":        31,
        "min_data_in_leaf":  15,
        "lambda_l2":         3.0,
        "feature_fraction":  0.55,
        "bagging_fraction":  0.7,
    },

    # ── Per-ticker acceptance overrides ───────────────────────────────
    "acceptance_overrides": {
        "up_f1":    0.22,
        "macro_f1": 0.26,
    },

    # ── Calibration ───────────────────────────────────────────────────
    "calibration": "temperature",
}

lgbm_reference_base = {
    **lgbm_reference,
    "model_name": "lgbm_reference_base",
    "selected_columns": [
        c for c in lgbm_reference["selected_columns"]
        if c not in ("iv_30d", "iv_skew_30d", "iv_term_ratio")
    ],
}
