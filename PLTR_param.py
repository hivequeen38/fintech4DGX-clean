AAII_option_vol_ratio = {
    "symbol": "PLTR",
    "model_name": "AAII_option_vol_ratio",
    "start_date": "2021-09-30",
    # "end_date":"2024-10-14",
    "current_estEPS": 0.14, # per seeking alpha 
    # "current_unemploy": 4.3,
    "next_report_date": "2025-08-04",
    "comment": "None",
    "threshold": 0.05,
    "selected_columns": [
       "label",

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

        ##################################
        # company fundamentals
        ##################################
        'EPS', 
        'estEPS',
        'surprisePercentage',
        'totalRevenue',
        'netIncome',
        # 'days_to_report',
        
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
        # 'jpy_close',
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
        "USEPUINDXD",       # Economic Policy Uncertainty Index for United States
        'UMCSENT',          # consumer sentiment (removed, slightly worse then base (no update since 3/1/24)
        # 'BSCICP03USM665S',  # sentiment OECD   (at 5/9/24, no data since Jan 24)
        'BSCICP02USM460S',   # Business Tendency Surveys (Manufacturing): Confidence Indicators: Composite Indicators: National Indicator for United States 
        'DGORDER',          # DURABLE GOODS ORDER
        'CORESTICKM159SFRBATL',     # Sticky Price Consumer Price Index less Food and Energy (no update in 5 mo, I think this is dead)
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

        ##################################
        # stock specific features
        #
        'rs_ita',
        'rs_igv',
        'rs_ita_trend',
        # 'rs_igv_trend',
        'FDEFX',          # National Defense Consumption Expenditures and Gross Investment
        'ADEFNO',         # Manufacturers New Orders: Defense Capital Goods
        'IPDCONGD',      # Industrial Production: Consumer Goods
        #################################
        # call put data
        #################################
        # 'cp_volume_ratio',
        # 'cp_oi_ratio',
        'cp_sentiment_ratio',
        'options_volume_ratio',
    ],
    "window_size": 35,
    "target_size": 15,
    "test_size": 0.2,
    "validation_size": 0.25,
    "batch_size": 32,       # best amongst 25, 32, and 40
    "shuffle": False,       # even though turning it on improves things, should not do it for time series data
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
    "embedded_dim": 64         # default to 64
}   # end PLTR_AAII_ref_param

AAII_reference_test = {
    "symbol": "PLTR",
    "model_name": "AAII",
    "start_date": "2021-09-30",
    # "end_date":"2024-10-14",
    "current_estEPS": 0.14, # per seeking alpha 
    # "current_unemploy": 4.3,
    "comment": "None",
    "threshold": 0.05,
    "selected_columns": [
       "label",

        ##################################
       # company Stock fundamentals
       ##################################
        "adjusted close",
        "daily_return",     # this is good
        "volume",           
        "volume_norm",    # regular volume still have better F1
        'Volume_Oscillator',
        "volatility",
        "VWAP",           # Volume Weighted Average Price (10/5/24 VWAP > Volume)
        "high",           # high low is slightly underperforming
        "low",

        ##################################
        # company fundamentals
        ##################################
        'EPS', 
        # 'estEPS',   # whack SPAH pass 1
        'surprisePercentage',
        # 'totalRevenue',   # whack SPAH pass 1  
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
        # "interest",         # whack SPAH pass 1
        "10year",
        "2year",
        "3month",
        'DTWEXBGS',         # DOLLAR INDEX (10/5/24 this not as good as using eur_close)
        'DFEDTARU',         # federal fund rate upper limit
        'BOGMBBM',          # Monetary Base; Reserve Balances
        'eur_close',
        'jpy_close',
        'USALOLITONOSTSAM', # Leading Indicators: Composite Leading Indicator: Normalised for United States (in 5/9/24, no data since Jan 24)
      

        ##################################
        # Other Indeces
        ##################################
        # "SPY_close",            # whack SPAH pass 1  
        "qqq_close",
        # "VTWO_close",       # whack SPAH pass 1
        'SPY_stoch',        #the appromiate for SnP Oscillator using slowD 
        'calc_spy_oscillator',  # self calculated SnP500 oscillator based on SPY
        'QQQ_stoch',      #the appromiate for NASDAQ Oscillator using slowD (very close but in combo not as good as using QQQ_close)
        'VTWO_stoch',       #the appromiate for Russell Oscillator using slowD (much better)
  
        ##################################
        # General Market fundamentals
        ##################################
         "VIXCLS",           # VIX from FRED
        # "VIXCLS_Y",         # VIX from YFinance
        "DCOILWTICO",       # WTI oil price
        # "unemploy",         # as of 5/9, still no satat for 5/1 # whack SPAH pass 1
        "FEDTARMDLR",       # Longer Run FOMC Summary of Economic Projections for the Fed Funds Rate, Median (probably not useful (in 5/9/24, last data point is 3/20/24)
        "USEPUINDXD",       # Economic Policy Uncertainty Index for United States
        # 'UMCSENT',          # consumer sentiment (removed, slightly worse then base (no update since 3/1/24)    # whack SPAH pass 1
        # 'BSCICP03USM665S',  # sentiment OECD   (at 5/9/24, no data since Jan 24)
        'BSCICP02USM460S',   # Business Tendency Surveys (Manufacturing): Confidence Indicators: Composite Indicators: National Indicator for United States 
        'DGORDER',          # DURABLE GOODS ORDER
        # 'CORESTICKM159SFRBATL',     # Sticky Price Consumer Price Index less Food and Energy (no update in 5 mo, I think this is dead)  # whack SPAH pass 1
        'PCE',              # No data since 3/1
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

        ##################################
        # stock specific features
        #
        'rs_ita',
        'rs_igv',
        'rs_ita_trend',     
        'rs_igv_trend',
        'FDEFX',          # National Defense Consumption Expenditures and Gross Investment
        'ADEFNO',         # Manufacturers New Orders: Defense Capital Goods
        'IPDCONGD',      # Industrial Production: Consumer Goods
        #################################
        # call put data
        #################################
        'cp_volume_ratio',
        # 'cp_oi_ratio',
        'cp_sentiment_ratio',
    ],
    "window_size": 35,
    "target_size": 15,
    "test_size": 0.2,
    "validation_size": 0.25,
    "batch_size": 32,       # best amongst 25, 32, and 40
    "shuffle": False,       # even though turning it on improves things, should not do it for time series data
    "headcount": 8,         
    "num_layers": 2,            # 2 MUCH BETTER THAN 3
    "dropout_rate": 0.1,
    "learning_rate": 0.0005,
    "num_epochs": 200,
    "down_weight": 1.5,        # tried 1.25 to 1.75. 1,5 still the best
    "scaler_type": 'MinMax',    # MinMax is the best
    "volatility_window": 13,     # 13 is best among [4, 7, 10, 13, 16]
    "bband_time_period": 20,    # Bollinger band time period 
    "training_set_size": 570,
    "validation_set_size": 195,
    "l1_lambda": 1e-6,          # This better than 0.5e-6
    "l2_weight_decay": 1.6e-5,  # 1.6 seems best
    "embedded_dim": 64         # default to 64
}   # end PLTR_AAII_ref_param

AAII_Min_features = {
    "symbol": "PLTR",
    "start_date": "2021-09-30",
    # "end_date":"2024-10-14",
    "current_estEPS": 0.14, # per seeking alpha 
    # "current_unemploy": 4.3,
    "comment": "None",
    "threshold": 0.05,
    "selected_columns": [
       "label",

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
        # "10year",
        # "2year",
        # "3month",
        "T10Y2Y",           # 10 year to 2 year yield spread
        'DTWEXBGS',         # DOLLAR INDEX (10/5/24 this not as good as using eur_close)
        # 'DFEDTARU',         # federal fund rate upper limit
        # 'BOGMBBM',          # Monetary Base; Reserve Balances
        # 'eur_close',
        # 'jpy_close',
        # 'USALOLITONOSTSAM', # Leading Indicators: Composite Leading Indicator: Normalised for United States (in 5/9/24, no data since Jan 24)
      

        ##################################
        # Other Indeces
        ##################################
        "SPY_close",
        "qqq_close",
        "VTWO_close",
        # 'SPY_stoch',        #the appromiate for SnP Oscillator using slowD (slightly better than SPY_close)
        # 'calc_spy_oscillator',  # self calculated SnP500 oscillator based on SPY
        # 'QQQ_stoch',      #the appromiate for NASDAQ Oscillator using slowD (very close but in combo not as good as using QQQ_close)
        # 'VTWO_stoch',       #the appromiate for Russell Oscillator using slowD (much better)
  
        ##################################
        # General Market fundamentals
        ##################################
        "VIXCLS",           # VIX from FRED
        # "DCOILWTICO",       # WTI oil price
        # "unemploy",         # as of 5/9, still no satat for 5/1
        # "FEDTARMDLR",       # Longer Run FOMC Summary of Economic Projections for the Fed Funds Rate, Median (probably not useful (in 5/9/24, last data point is 3/20/24)
        # "USEPUINDXD",       # Economic Policy Uncertainty Index for United States
        # 'UMCSENT',          # consumer sentiment (removed, slightly worse then base (no update since 3/1/24)
        # 'BSCICP03USM665S',  # sentiment OECD   (at 5/9/24, no data since Jan 24)
        # 'DGORDER',          # DURABLE GOODS ORDER
        # 'CORESTICKM159SFRBATL',     # Sticky Price Consumer Price Index less Food and Energy (no update in 5 mo, I think this is dead)
        # 'PCE',              # No data since 3/1
        # 'PCU33443344',      # Producer Price Index by Industry: Semiconductor and Other Electronic Component Manufacturing 
        'SAHMREALTIME',     # sahm unemployment (seems to work better than unemploy)
        'JTSJOL',           # non farm job opening
        # 'GDP',              # Add GDP
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

        ##################################
        # stock specific features
        #
        'rs_ita',
        'rs_igv',
        'rs_ita_trend',
        'rs_igv_trend',
        'FDEFX',          # National Defense Consumption Expenditures and Gross Investment
        'ADEFNO',         # Manufacturers New Orders: Defense Capital Goods
        'IPDCONGD',      # Industrial Production: Consumer Goods
    ],
    "window_size": 35,
    "target_size": 15,
    "test_size": 0.2,
    "validation_size": 0.25,
    "batch_size": 32,       # best amongst 25, 32, and 40
    "shuffle": False,       # even though turning it on improves things, should not do it for time series data
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
    "embedded_dim": 64         # default to 64
}   # end AAII

AAII_reference = {
    "symbol": "PLTR",
    "model_name": "AAII",
    "start_date": "2021-09-30",
    # "end_date":"2024-10-14",
    "current_estEPS": 0.14, # per seeking alpha 
    # "current_unemploy": 4.3,
    "comment": "None",
    "threshold": 0.05,
    "selected_columns": [
       "label",

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
        # 'jpy_close',
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
        "USEPUINDXD",       # Economic Policy Uncertainty Index for United States
        'UMCSENT',          # consumer sentiment (removed, slightly worse then base (no update since 3/1/24)
        # 'BSCICP03USM665S',  # sentiment OECD   (at 5/9/24, no data since Jan 24)
        'BSCICP02USM460S',   # Business Tendency Surveys (Manufacturing): Confidence Indicators: Composite Indicators: National Indicator for United States 
        'DGORDER',          # DURABLE GOODS ORDER
        'CORESTICKM159SFRBATL',     # Sticky Price Consumer Price Index less Food and Energy (no update in 5 mo, I think this is dead)
        'PCE',              # No data since 3/1
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

        ##################################
        # stock specific features
        #
        'rs_ita',
        'rs_igv',
        'rs_ita_trend',
        # 'rs_igv_trend',
        'FDEFX',          # National Defense Consumption Expenditures and Gross Investment
        'ADEFNO',         # Manufacturers New Orders: Defense Capital Goods
        'IPDCONGD',      # Industrial Production: Consumer Goods
        #################################
        # call put data
        #################################
        # 'cp_volume_ratio',
        # 'cp_oi_ratio',
        'cp_sentiment_ratio',
    ],
    "window_size": 35,
    "target_size": 15,
    "test_size": 0.2,
    "validation_size": 0.25,
    "batch_size": 32,       # best amongst 25, 32, and 40
    "shuffle": False,       # even though turning it on improves things, should not do it for time series data
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
    "embedded_dim": 64         # default to 64
}   # end PLTR_AAII_ref_param


reference = {
    "symbol": "PLTR",
    "model_name": "ref",
    "start_date": "2021-09-30",
    # "end_date":"2024-10-14",
    "current_estEPS": 0.14, # per seeking alpha 
    # "current_unemploy": 4.3,
    "comment": "None",
    "threshold": 0.05,
    "selected_columns": [
       "label",

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
        "3month",
        'DTWEXBGS',         # DOLLAR INDEX (10/5/24 this not as good as using eur_close)
        'DFEDTARU',         # federal fund rate upper limit
        'BOGMBBM',          # Monetary Base; Reserve Balances
        # 'eur_close',
        # 'jpy_close',
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
        "USEPUINDXD",       # Economic Policy Uncertainty Index for United States
        'UMCSENT',          # consumer sentiment (removed, slightly worse then base (no update since 3/1/24)
        # 'BSCICP03USM665S',  # sentiment OECD   (at 5/9/24, no data since Jan 24)
        'BSCICP02USM460S',   # Business Tendency Surveys (Manufacturing): Confidence Indicators: Composite Indicators: National Indicator for United States 
        'DGORDER',          # DURABLE GOODS ORDER
        'CORESTICKM159SFRBATL',     # Sticky Price Consumer Price Index less Food and Energy (no update in 5 mo, I think this is dead)
        'PCE',              # No data since 3/1
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

        ##################################
        # stock specific features
        #
        'rs_ita',
        'rs_igv',
        'rs_ita_trend',
        'rs_igv_trend',
        'FDEFX',          # National Defense Consumption Expenditures and Gross Investment
        'ADEFNO',         # Manufacturers New Orders: Defense Capital Goods
        'IPDCONGD',      # Industrial Production: Consumer Goods
    ],
    "window_size": 35,
    "target_size": 15,
    "test_size": 0.2,
    "validation_size": 0.25,
    "batch_size": 32,       # best amongst 25, 32, and 40
    "shuffle": False,       # even though turning it on improves things, should not do it for time series data
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
    "embedded_dim": 64         # default to 64
}   # end PLTR_ref_param