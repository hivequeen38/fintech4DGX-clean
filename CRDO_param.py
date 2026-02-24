AAII_option_vol_ratio = {
    "symbol": "CRDO",
    "model_name": "AAII_option_vol_ratio",
    "start_date": "2022-01-27",
    # "end_date":"2024-03-01",
    "current_estEPS": 0.45,
    # "current_unemploy": 4.3,
    "comment": "None",
    "threshold": 0.05,
    "selected_columns": [

       "label", #64 features
        ##################################
        # company Stock fundamentals
        ##################################
        "adjusted close",
        "daily_return",     # this is good
        "volume",
        # "volume_norm",    # regular volume still have better F1
        'Volume_Oscillator',
        "volatility",
        "VWAP",           # Volume Weighted Average Price (10/5/24 VWAP > Volume)
        "high",           # high low is slightly underperforming
        "low",
        'volume_volatility',
        # "relative_volume",
        # "relative_close",
        # "relative_high",
        # "relative_low",

        ##################################
        # company fundamentals
        ##################################
        'EPS', 
        'estEPS',
        'surprisePercentage',
        'dte',
        'dse',
        'earn_in_5',
        'earn_in_10',
        'earn_in_20',
        'totalRevenue',
        'netIncome',
        
        ##################################
        # Stock Momentum 
        ##################################
        "MACD_Signal",
        "MACD",
        "MACD_Hist",
        "ATR",
        "RSI",
        # 'RSI_overbought',
        # 'RSI_oversold',
        # 'RSI_bullish_divergence',
        # 'RSI_bearish_divergence',
        # 'RSI_momentum_strength',
        'Real Upper Band',  # bollinger band (taken out as not that useful)
        'Real Middle Band',
        'Real Lower Band',

        ##################################
        # Monetary market
        ##################################
        "interest",
        "10year",
        "2year",
        # "3month",
        'DTWEXBGS',         # DOLLAR INDEX (10/5/24 this not as good as using eur_close)
        'DFEDTARU',         # federal fund rate upper limit
        'BOGMBBM',          # Monetary Base; Reserve Balances
        # 'eur_close',
        'jpy_close',
        'twd_close',
        # 'USALOLITONOSTSAM', # Leading Indicators: Composite Leading Indicator: Normalised for United States (in 5/9/24, no data since Jan 24)

        ##################################
        # Other Indeces
        ##################################
        "SPY_close",
        "qqq_close",
        "VTWO_close",
        'SPY_stoch',        #the appromiate for SnP Oscillator using slowD (slightly better than SPY_close)
        'calc_spy_oscillator',  # self calculated SnP500 oscillator based on SPY
        'QQQ_stoch',      #the appromiate for NASDAQ Oscillator using slowD (very close but in combo not as good as using QQQ_close)
        'VTWO_stoch',       #the appromiate for Russell Oscillator using slowD (much better)
  
        ##################################
        # General Market fundamentals
        ##################################
        "VIXCLS",           # VIX from FRED
        # "VIXCLS_Y",         # VIX from YFinance
        "DCOILWTICO",       # WTI oil price
        # "unemploy",         # as of 5/9, still no satat for 5/1
        # "FEDTARMDLR",       # Longer Run FOMC Summary of Economic Projections for the Fed Funds Rate, Median (probably not useful (in 5/9/24, last data point is 3/20/24)
        "USEPUINDXD",       # Economic Policy Uncertainty Index for United States
        'UMCSENT',          # consumer sentiment (removed, slightly worse then base (no update since 3/1/24)
        # 'BSCICP03USM665S',  # sentiment OECD   (at 5/9/24, no data since Jan 24)
        'BSCICP02USM460S',   # Business Tendency Surveys (Manufacturing): Confidence Indicators: Composite Indicators: National Indicator for United States 
        'DGORDER',          # DURABLE GOODS ORDER
        # 'CORESTICKM159SFRBATL',     # Sticky Price Consumer Price Index less Food and Energy (no update in 5 mo, I think this is dead)
        # 'PCE',              # No data since 3/1
        'PCU33443344',      # Producer Price Index by Industry: Semiconductor and Other Electronic Component Manufacturing 
        'SAHMREALTIME',     # sahm unemployment (seems to work better than unemploy)
        'JTSJOL',           # non farm job opening
        'GDP',              # Add GDP
        'FINRA_debit',      # finra debit
        # 'Bullish',          # AAII Bullish percentage
        # 'Bearish',          # AAII Bearish percentage
        'Spread',           # AAII Bullish-Bearish Spread

        ##################################
        # other meta data
        #
        'day_of_week',
        'month',
        'price_lag_1',
        'price_lag_5',
        'price_lag_15',
        'price_change_1',
        'price_change_5',
        'price_change_15',

        #################################
        # Industry Specific
        #################################
        'rs_amd',
        # 'rs_intc',
        # 'rs_avgo',
        'rs_amd_trend',
        'rs_intc_trend',
        'rs_avgo_trend',
        # 'rs_sox_trend_short',
        # 'rs_sox_trend_med',
        # 'rs_sox_trend_long',
        # 'rs_sox_volatility',
        'rs_smh_trend',
        # 'TSMC_close',
        # 'tsm_price_change_1',
        # 'rs_sp500_trend',
        # 'rs_nasdaq_trend',
        #################################
        # call put data
        #################################
        # 'cp_volume_ratio',
        # 'cp_oi_ratio',
        'cp_sentiment_ratio',
        'options_volume_ratio'
        ],
    'robust_features': [    # volatile features like volume, returns, etc, should be here (this for using robust_scaler)
        "daily_return",     # this is good
        "volume",
        # "volume_norm",    # regular volume still have better F1
        "VWAP",             # Volume Weighted Average Price (10/5/24 VWAP > Volume)
        "VIXCLS",           # VIX from FRED
        "MACD_Signal",
        "MACD",
        "MACD_Hist",
        "ATR",
        "RSI",
        'Real Upper Band',  # bollinger band (taken out as not that useful)
        'Real Middle Band',
        'Real Lower Band',
        'price_change_1',
        'price_change_5',
        'price_change_15',
    ],
    "window_size": 35,
    "target_size": 15,
    "num_zones": 15,
    "test_size": 0.2,
    "validation_size": 0.25,
    "batch_size": 128,      # increased from 32 for better GPU utilization
    "shuffle": False,       # even though turning it on improves things, should not do it for time series data
    "use_time_split": False,   # True: TimeSeriesSplit(n_splits=3); False: single chronological split
    "headcount": 8,         
    "num_layers": 2,            # 2 MUCH BETTER THAN 3
    "dropout_rate": 0.1,
    "learning_rate": 0.0005,
    "num_epochs": 200,
    "down_weight": 1.5,        # tried 1.25 to 1.75. 1,5 still the best
    "scaler_type": 'MinMax',    # MinMax is the best
    "volatility_window": 13,     # 13 is best among [4, 7, 10, 13, 16]
    "bband_time_period": 20,    # Bollinger band time period (changing it doesn't seem to make a dent)
    "training_set_size": 570,
    "validation_set_size": 195,
    "l1_lambda": 1e-6,          # This better than 0.5e-6
    "l2_weight_decay": 1.6e-5,  # 1.6 seems best
    "embedded_dim": 128        # increased from 64 for model capacity
}   # AAII_option_vol_ratio

AAII_reference = {
    "symbol": "CRDO",
    "model_name": "AAII",
    "start_date": "2022-01-27",
    # "end_date":"2024-03-01",
    "current_estEPS": 0.36,
    # "current_unemploy": 4.3,
    "comment": "None",
    "threshold": 0.05,
    "selected_columns": [

       "label", #64 features
        ##################################
        # company Stock fundamentals
        ##################################
        "adjusted close",
        "daily_return",     # this is good
        "volume",
        # "volume_norm",    # regular volume still have better F1
        'Volume_Oscillator',
        "volatility",
        "VWAP",           # Volume Weighted Average Price (10/5/24 VWAP > Volume)
        "high",           # high low is slightly underperforming
        "low",
        'volume_volatility',
        # "relative_volume",
        # "relative_close",
        # "relative_high",
        # "relative_low",

        ##################################
        # company fundamentals
        ##################################
        'EPS', 
        'estEPS',
        'surprisePercentage',
        'totalRevenue',
        'netIncome',
        
        ##################################
        # Stock Momentum 
        ##################################
        "MACD_Signal",
        "MACD",
        "MACD_Hist",
        "ATR",
        "RSI",
        # 'RSI_overbought',
        # 'RSI_oversold',
        # 'RSI_bullish_divergence',
        # 'RSI_bearish_divergence',
        # 'RSI_momentum_strength',
        'Real Upper Band',  # bollinger band (taken out as not that useful)
        'Real Middle Band',
        'Real Lower Band',

        ##################################
        # Monetary market
        ##################################
        "interest",
        "10year",
        "2year",
        # "3month",
        'DTWEXBGS',         # DOLLAR INDEX (10/5/24 this not as good as using eur_close)
        'DFEDTARU',         # federal fund rate upper limit
        'BOGMBBM',          # Monetary Base; Reserve Balances
        # 'eur_close',
        'jpy_close',
        'twd_close',
        # 'USALOLITONOSTSAM', # Leading Indicators: Composite Leading Indicator: Normalised for United States (in 5/9/24, no data since Jan 24)

        ##################################
        # Other Indeces
        ##################################
        "SPY_close",
        "qqq_close",
        "VTWO_close",
        'SPY_stoch',        #the appromiate for SnP Oscillator using slowD (slightly better than SPY_close)
        'calc_spy_oscillator',  # self calculated SnP500 oscillator based on SPY
        'QQQ_stoch',      #the appromiate for NASDAQ Oscillator using slowD (very close but in combo not as good as using QQQ_close)
        'VTWO_stoch',       #the appromiate for Russell Oscillator using slowD (much better)
  
        ##################################
        # General Market fundamentals
        ##################################
        "VIXCLS",           # VIX from FRED
        # "VIXCLS_Y",         # VIX from YFinance
        "DCOILWTICO",       # WTI oil price
        # "unemploy",         # as of 5/9, still no satat for 5/1
        # "FEDTARMDLR",       # Longer Run FOMC Summary of Economic Projections for the Fed Funds Rate, Median (probably not useful (in 5/9/24, last data point is 3/20/24)
        "USEPUINDXD",       # Economic Policy Uncertainty Index for United States
        'UMCSENT',          # consumer sentiment (removed, slightly worse then base (no update since 3/1/24)
        # 'BSCICP03USM665S',  # sentiment OECD   (at 5/9/24, no data since Jan 24)
        'BSCICP02USM460S',   # Business Tendency Surveys (Manufacturing): Confidence Indicators: Composite Indicators: National Indicator for United States 
        'DGORDER',          # DURABLE GOODS ORDER
        # 'CORESTICKM159SFRBATL',     # Sticky Price Consumer Price Index less Food and Energy (no update in 5 mo, I think this is dead)
        # 'PCE',              # No data since 3/1
        'PCU33443344',      # Producer Price Index by Industry: Semiconductor and Other Electronic Component Manufacturing 
        'SAHMREALTIME',     # sahm unemployment (seems to work better than unemploy)
        'JTSJOL',           # non farm job opening
        'GDP',              # Add GDP
        'FINRA_debit',      # finra debit
        # 'Bullish',          # AAII Bullish percentage
        # 'Bearish',          # AAII Bearish percentage
        'Spread',           # AAII Bullish-Bearish Spread

        ##################################
        # other meta data
        #
        'day_of_week',
        'month',
        'price_lag_1',
        'price_lag_5',
        'price_lag_15',
        'price_change_1',
        'price_change_5',
        'price_change_15',

        #################################
        # Industry Specific
        #################################
        'rs_amd',
        # 'rs_intc',
        'rs_avgo',
        'rs_amd_trend',
        'rs_intc_trend',
        'rs_avgo_trend',
        # 'rs_sox_trend_short',
        # 'rs_sox_trend_med',
        # 'rs_sox_trend_long',
        # 'rs_sox_volatility',
        'rs_smh_trend',
        # 'TSMC_close',
        # 'tsm_price_change_1',
        # 'rs_sp500_trend',
        # 'rs_nasdaq_trend',
        #################################
        # call put data
        #################################
        # 'cp_volume_ratio',
        # 'cp_oi_ratio',
        'cp_sentiment_ratio',
        ],
    'robust_features': [    # volatile features like volume, returns, etc, should be here (this for using robust_scaler)
        "daily_return",     # this is good
        "volume",
        # "volume_norm",    # regular volume still have better F1
        "VWAP",             # Volume Weighted Average Price (10/5/24 VWAP > Volume)
        "VIXCLS",           # VIX from FRED
        "MACD_Signal",
        "MACD",
        "MACD_Hist",
        "ATR",
        "RSI",
        'Real Upper Band',  # bollinger band (taken out as not that useful)
        'Real Middle Band',
        'Real Lower Band',
        'price_change_1',
        'price_change_5',
        'price_change_15',
    ],
    "window_size": 35,
    "target_size": 15,
    "num_zones": 15,
    "test_size": 0.2,
    "validation_size": 0.25,
    "batch_size": 128,      # increased from 32 for better GPU utilization
    "shuffle": False,       # even though turning it on improves things, should not do it for time series data
    "use_time_split": False,   # True: TimeSeriesSplit(n_splits=3); False: single chronological split
    "headcount": 8,         
    "num_layers": 2,            # 2 MUCH BETTER THAN 3
    "dropout_rate": 0.1,
    "learning_rate": 0.0005,
    "num_epochs": 200,
    "down_weight": 1.5,        # tried 1.25 to 1.75. 1,5 still the best
    "scaler_type": 'MinMax',    # MinMax is the best
    "volatility_window": 13,     # 13 is best among [4, 7, 10, 13, 16]
    "bband_time_period": 20,    # Bollinger band time period (changing it doesn't seem to make a dent)
    "training_set_size": 570,
    "validation_set_size": 195,
    "l1_lambda": 1e-6,          # This better than 0.5e-6
    "l2_weight_decay": 1.6e-5,  # 1.6 seems best
    "embedded_dim": 128        # increased from 64 for model capacity
}   # AAII_reference

reference = {
    "symbol": "CRDO",
    "model_name": "ref",
    "start_date": "2022-01-27",
    # "end_date":"2024-03-01",
    "current_estEPS": 0.45,
    # "current_unemploy": 4.3,
    "comment": "None",
    "threshold": 0.05,
    "selected_columns": [

       "label", # 64 features
        ##################################
        # company Stock fundamentals
        ##################################
        "adjusted close",
        "daily_return",     # this is good
        "volume",
        # "volume_norm",    # regular volume still have better F1
        'Volume_Oscillator',
        "volatility",
        "VWAP",           # Volume Weighted Average Price (10/5/24 VWAP > Volume)
        "high",           # high low is slightly underperforming
        "low",
        'volume_volatility',
        # "relative_volume",
        # "relative_close",
        # "relative_high",
        # "relative_low",

        ##################################
        # company fundamentals
        ##################################
        'EPS', 
        'estEPS',
        'surprisePercentage',
        'dte',
        'dse',
        'earn_in_5',
        'earn_in_10',
        'earn_in_20',
        'totalRevenue',
        'netIncome',
        
        ##################################
        # Stock Momentum 
        ##################################
        "MACD_Signal",
        "MACD",
        "MACD_Hist",
        "ATR",
        "RSI",
        # 'RSI_overbought',
        # 'RSI_oversold',
        # 'RSI_bullish_divergence',
        # 'RSI_bearish_divergence',
        # 'RSI_momentum_strength',
        'Real Upper Band',  # bollinger band (taken out as not that useful)
        'Real Middle Band',
        'Real Lower Band',

        ##################################
        # Monetary market
        ##################################
        "interest",
        "10year",
        "2year",
        "3month",
        'DTWEXBGS',         # DOLLAR INDEX (10/5/24 this not as good as using eur_close)
        'DFEDTARU',         # federal fund rate upper limit
        'BOGMBBM',          # Monetary Base; Reserve Balances
        # 'eur_close',
        'jpy_close',
        'twd_close',
        # 'USALOLITONOSTSAM', # Leading Indicators: Composite Leading Indicator: Normalised for United States (in 5/9/24, no data since Jan 24)

        ##################################
        # Other Indeces
        ##################################
        "SPY_close",
        "qqq_close",
        "VTWO_close",
        'SPY_stoch',        #the appromiate for SnP Oscillator using slowD (slightly better than SPY_close)
        'calc_spy_oscillator',  # self calculated SnP500 oscillator based on SPY
        'QQQ_stoch',      #the appromiate for NASDAQ Oscillator using slowD (very close but in combo not as good as using QQQ_close)
        'VTWO_stoch',       #the appromiate for Russell Oscillator using slowD (much better)
  
        ##################################
        # General Market fundamentals
        ##################################
        "VIXCLS",           # VIX from FRED
        # "VIXCLS_Y",         # VIX from YFinance
        "DCOILWTICO",       # WTI oil price
        # "unemploy",         # as of 5/9, still no satat for 5/1
        "FEDTARMDLR",       # Longer Run FOMC Summary of Economic Projections for the Fed Funds Rate, Median (probably not useful (in 5/9/24, last data point is 3/20/24)
        # "USEPUINDXD",       # Economic Policy Uncertainty Index for United States
        'UMCSENT',          # consumer sentiment (removed, slightly worse then base (no update since 3/1/24)
        # 'BSCICP03USM665S',  # sentiment OECD   (at 5/9/24, no data since Jan 24)
        # 'BSCICP02USM460S',   # Business Tendency Surveys (Manufacturing): Confidence Indicators: Composite Indicators: National Indicator for United States 
        'DGORDER',          # DURABLE GOODS ORDER
        'CORESTICKM159SFRBATL',     # Sticky Price Consumer Price Index less Food and Energy (no update in 5 mo, I think this is dead)
        # 'PCE',              # No data since 3/1
        'PCU33443344',      # Producer Price Index by Industry: Semiconductor and Other Electronic Component Manufacturing 
        'SAHMREALTIME',     # sahm unemployment (seems to work better than unemploy)
        'JTSJOL',           # non farm job opening
        'GDP',              # Add GDP
        'FINRA_debit',      # finra debit

        ##################################
        # other meta data
        #
        'day_of_week',
        'month',
        'price_lag_1',
        'price_lag_5',
        'price_lag_15',
        'price_change_1',
        'price_change_5',
        'price_change_15',

        #################################
        # Industry Specific
        #################################
        'rs_amd',
        'rs_intc',
        'rs_avgo',
        'rs_amd_trend',
        'rs_intc_trend',
        'rs_avgo_trend',
        # 'rs_sox_trend_short',
        # 'rs_sox_trend_med',
        # 'rs_sox_trend_long',
        # 'rs_sox_volatility',
        'rs_smh_trend',
        # 'TSMC_close',
        # 'tsm_price_change_1',
        # 'rs_sp500_trend',
        # 'rs_nasdaq_trend',
    ],
    'robust_features': [    # volatile features like volume, returns, etc, should be here (this for using robust_scaler)
        "daily_return",     # this is good
        "volume",
        # "volume_norm",    # regular volume still have better F1
        "VWAP",             # Volume Weighted Average Price (10/5/24 VWAP > Volume)
        "VIXCLS",           # VIX from FRED
        "MACD_Signal",
        "MACD",
        "MACD_Hist",
        "ATR",
        "RSI",
        'Real Upper Band',  # bollinger band (taken out as not that useful)
        'Real Middle Band',
        'Real Lower Band',
        'price_change_1',
        'price_change_5',
        'price_change_15',
    ],
    "window_size": 35,
    "target_size": 15,
    "num_zones": 15,
    "test_size": 0.2,
    "validation_size": 0.25,
    "batch_size": 128,      # increased from 32 for better GPU utilization
    "shuffle": False,       # even though turning it on improves things, should not do it for time series data
    "shuffle_splits": True,    # True: train_test_split with shuffle (look-ahead bias, inflated F1); False: chronological honest split
    "use_time_split": False,   # True: TimeSeriesSplit(n_splits=3); False: single chronological split
    "headcount": 8,
    "num_layers": 2,            # 2 MUCH BETTER THAN 3
    "dropout_rate": 0.1,
    "learning_rate": 0.0005,
    "num_epochs": 200,
    "down_weight": 1.5,        # tried 1.25 to 1.75. 1,5 still the best
    "scaler_type": 'MinMax',    # MinMax is the best
    "volatility_window": 13,     # 13 is best among [4, 7, 10, 13, 16]
    "bband_time_period": 20,    # Bollinger band time period (changing it doesn't seem to make a dent)
    "training_set_size": 570,
    "validation_set_size": 195,
    "l1_lambda": 1e-6,          # This better than 0.5e-6
    "l2_weight_decay": 1.6e-5,  # 1.6 seems best
    "embedded_dim": 128        # increased from 64 for model capacity
}   # CRDO_ref_param

# Identical to reference but uses chronological (no-shuffle) train/test split — honest out-of-sample evaluation
reference_no_shuffle = {**reference, "model_name": "ref_noshuf", "shuffle_splits": False}