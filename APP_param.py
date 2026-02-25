AAII_option_vol_ratio = {
    "symbol": "APP",
    "model_name": "AAII_option_vol_ratio",
    "start_date": "2021-07-01",
    "end_date":"2024-11-05",
    "current_estEPS": 2.86, # per seeking alpha 
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
        # Idiosyncratic return (stock alpha vs market)
        ##################################
        'ret_5d_rel_SPY',   # stock 5d return minus SPY 5d return
        'ret_10d_rel_SPY',  # stock 10d return minus SPY 10d return

        ##################################
        # Analyst estimate features (EPS consensus, AV EARNINGS_ESTIMATES)
        ##################################
        'eps_est_avg',            # AV consensus average EPS estimate (upcoming quarter, raw AV units)
        # 'eps_rev_30_pct',       # BACKLOG: leakage risk — forward-filled end-of-quarter revision stat
        # 'eps_rev_7_pct',        # BACKLOG: leakage risk — forward-filled end-of-quarter revision stat
        # 'eps_breadth_ratio_30', # BACKLOG: leakage risk — forward-filled end-of-quarter revision stat
        # 'eps_dispersion',       # BACKLOG: leakage risk — forward-filled end-of-quarter revision stat

        ##################################
        # Stock Momentum 
        ##################################
        "MACD_Signal",
        "MACD",
        "MACD_Hist",
        "ATR",
        "RSI",
        'Real Upper Band',  # bollinger band (taken out as not that useful)
        'Real Middle Band',
        'Real Lower Band',

        ##################################
        # Monetary market
        ##################################
        "interest",
        "10year",
        # "2year",
        # "3month",
        "T10Y2Y",           # 10 year to 2 year yield spread
        'M2SL',             # M2 money supply
        'DTWEXBGS',         # DOLLAR INDEX (10/5/24 this not as good as using eur_close)
        'DFEDTARU',         # federal fund rate upper limit
        'BOGMBBM',          # Monetary Base; Reserve Balances
        'eur_close',
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
        # 'VTWO_stoch',       #the appromiate for Russell Oscillator using slowD (much better)
  
        ##################################
        # General Market fundamentals
        ##################################
         "VIXCLS",           # VIX from FRED
        # "VIXCLS_Y",         # VIX from YFinance
        # "DCOILWTICO",       # WTI oil price
        "unemploy",         # as of 5/9, still no satat for 5/1
        # "FEDTARMDLR",       # Longer Run FOMC Summary of Economic Projections for the Fed Funds Rate, Median (probably not useful (in 5/9/24, last data point is 3/20/24)
        # "USEPUINDXD",       # Economic Policy Uncertainty Index for United States
        'UMCSENT',          # consumer sentiment (removed, slightly worse then base (no update since 3/1/24)
        # 'BSCICP03USM665S',  # sentiment OECD   (at 5/9/24, no data since Jan 24)
        # 'BSCICP02USM460S',   # Business Tendency Surveys (Manufacturing): Confidence Indicators: Composite Indicators: National Indicator for United States 
        # 'DGORDER',          # DURABLE GOODS ORDER
        # 'CORESTICKM159SFRBATL',     # Sticky Price Consumer Price Index less Food and Energy (no update in 5 mo, I think this is dead)
        # 'PCE',              # No data since 3/1
        'CPIAUCSL',         # Consumer Price Index for All Urban Consumers: All Items
        # 'PCU33443344',      # Producer Price Index by Industry: Semiconductor and Other Electronic Component Manufacturing 
        # 'SAHMREALTIME',     # sahm unemployment (seems to work better than unemploy)
        # 'JTSJOL',           # non farm job opening
        'GDP',              # Add GDP
        'BUSLOANS',       # business & commercial loans
        'RSAFS',          # advance retail sales, correlation with ad spending
        # 'FINRA_debit',      # finra debit
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
        # 'price_change_15',

        ##################################
        # APP specific features
        #
        'rs_gamr',
        'rs_socl',
        'rs_gamr_trend',
        'rs_socl_trend',
        # 'rs_u',
        # 'rs_ttwo',
        # 'rs_apps',
        # 'rs_u_trend',
        # 'rs_ttwo_trend',
        # 'rs_apps_trend',
        # 'cp_volume_ratio',
        'cp_sentiment_ratio',
        'options_volume_ratio',
        ##################################
        # Implied Volatility features
        ##################################
        'iv_30d',        # front-month OI-weighted IV (20-45 DTE): overall vol regime
        'iv_skew_30d',   # put IV minus call IV (30d): fear premium
        'iv_term_ratio', # iv_7d / iv_30d: >1 = short-dated vol elevated = event risk
    ],
     "window_size": 35,
    "target_size": 15,
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
}   # referenceAAII_option_vol_ratio

AAII_Min_features = {
    "symbol": "APP",
    "start_date": "2021-07-01",
    "end_date":"2024-11-05",
    "current_estEPS": 1.77, # per seeking alpha 
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

        ##################################
        # company fundamentals
        ##################################
        'EPS', 
        'estEPS',
        'surprisePercentage',
        'totalRevenue',
        'netIncome',
        

        ##################################
        # Idiosyncratic return (stock alpha vs market)
        ##################################
        'ret_5d_rel_SPY',   # stock 5d return minus SPY 5d return
        'ret_10d_rel_SPY',  # stock 10d return minus SPY 10d return

        ##################################
        # Analyst estimate features (EPS consensus, AV EARNINGS_ESTIMATES)
        ##################################
        'eps_est_avg',            # AV consensus average EPS estimate (upcoming quarter, raw AV units)
        # 'eps_rev_30_pct',       # BACKLOG: leakage risk — forward-filled end-of-quarter revision stat
        # 'eps_rev_7_pct',        # BACKLOG: leakage risk — forward-filled end-of-quarter revision stat
        # 'eps_breadth_ratio_30', # BACKLOG: leakage risk — forward-filled end-of-quarter revision stat
        # 'eps_dispersion',       # BACKLOG: leakage risk — forward-filled end-of-quarter revision stat

        ##################################
        # Stock Momentum 
        ##################################
        "MACD_Signal",
        "MACD",
        "MACD_Hist",
        "ATR",
        "RSI",
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
        # 'M2SL',             # M2 money supply
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
        'SPY_stoch',        #the appromiate for SnP Oscillator using slowD (slightly better than SPY_close)
        # 'calc_spy_oscillator',  # self calculated SnP500 oscillator based on SPY
        'QQQ_stoch',      #the appromiate for NASDAQ Oscillator using slowD (very close but in combo not as good as using QQQ_close)
        # 'VTWO_stoch',       #the appromiate for Russell Oscillator using slowD (much better)
  
        ##################################
        # General Market fundamentals
        ##################################
         "VIXCLS",           # VIX from FRED
        # "VIXCLS_Y",         # VIX from YFinance
        # "DCOILWTICO",       # WTI oil price
        "unemploy",         # as of 5/9, still no satat for 5/1
        # "FEDTARMDLR",       # Longer Run FOMC Summary of Economic Projections for the Fed Funds Rate, Median (probably not useful (in 5/9/24, last data point is 3/20/24)
        # "USEPUINDXD",       # Economic Policy Uncertainty Index for United States
        'UMCSENT',          # consumer sentiment (removed, slightly worse then base (no update since 3/1/24)
        # 'BSCICP03USM665S',  # sentiment OECD   (at 5/9/24, no data since Jan 24)
        # 'DGORDER',          # DURABLE GOODS ORDER
        # 'CORESTICKM159SFRBATL',     # Sticky Price Consumer Price Index less Food and Energy (no update in 5 mo, I think this is dead)
        # 'PCE',              # No data since 3/1
        'CPIAUCSL',         # Consumer Price Index for All Urban Consumers: All Items
        # 'PCU33443344',      # Producer Price Index by Industry: Semiconductor and Other Electronic Component Manufacturing 
        # 'SAHMREALTIME',     # sahm unemployment (seems to work better than unemploy)
        # 'JTSJOL',           # non farm job opening
        # 'GDP',              # Add GDP
        'BUSLOANS',       # business & commercial loans
        'RSAFS',          # advance retail sales, correlation with ad spending
        # 'FINRA_debit',      # finra debit
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
        # APP specific features
        #
        'rs_gamr',
        'rs_socl',
        'rs_gamr_trend',
        'rs_socl_trend',
        # 'rs_u',
        # 'rs_ttwo',
        # 'rs_apps',
        # 'rs_u_trend',
        # 'rs_ttwo_trend',
        # 'rs_apps_trend',
    ],
     "window_size": 35,
    "target_size": 15,
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
}   # AAII_Min_features

AAII_reference = {
    "symbol": "APP",
    "model_name": "AAII",
    "start_date": "2021-07-01",
    "end_date":"2024-11-05",
    "current_estEPS": 2.86, # per seeking alpha 
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

        ##################################
        # company fundamentals
        ##################################
        'EPS', 
        'estEPS',
        'surprisePercentage',
        'totalRevenue',
        'netIncome',
        

        ##################################
        # Idiosyncratic return (stock alpha vs market)
        ##################################
        'ret_5d_rel_SPY',   # stock 5d return minus SPY 5d return
        'ret_10d_rel_SPY',  # stock 10d return minus SPY 10d return

        ##################################
        # Analyst estimate features (EPS consensus, AV EARNINGS_ESTIMATES)
        ##################################
        'eps_est_avg',            # AV consensus average EPS estimate (upcoming quarter, raw AV units)
        # 'eps_rev_30_pct',       # BACKLOG: leakage risk — forward-filled end-of-quarter revision stat
        # 'eps_rev_7_pct',        # BACKLOG: leakage risk — forward-filled end-of-quarter revision stat
        # 'eps_breadth_ratio_30', # BACKLOG: leakage risk — forward-filled end-of-quarter revision stat
        # 'eps_dispersion',       # BACKLOG: leakage risk — forward-filled end-of-quarter revision stat

        ##################################
        # Stock Momentum 
        ##################################
        "MACD_Signal",
        "MACD",
        "MACD_Hist",
        "ATR",
        "RSI",
        'Real Upper Band',  # bollinger band (taken out as not that useful)
        'Real Middle Band',
        'Real Lower Band',

        ##################################
        # Monetary market
        ##################################
        "interest",
        "10year",
        # "2year",
        # "3month",
        "T10Y2Y",           # 10 year to 2 year yield spread
        'M2SL',             # M2 money supply
        'DTWEXBGS',         # DOLLAR INDEX (10/5/24 this not as good as using eur_close)
        'DFEDTARU',         # federal fund rate upper limit
        'BOGMBBM',          # Monetary Base; Reserve Balances
        'eur_close',
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
        # 'VTWO_stoch',       #the appromiate for Russell Oscillator using slowD (much better)
  
        ##################################
        # General Market fundamentals
        ##################################
         "VIXCLS",           # VIX from FRED
        # "VIXCLS_Y",         # VIX from YFinance
        # "DCOILWTICO",       # WTI oil price
        "unemploy",         # as of 5/9, still no satat for 5/1
        # "FEDTARMDLR",       # Longer Run FOMC Summary of Economic Projections for the Fed Funds Rate, Median (probably not useful (in 5/9/24, last data point is 3/20/24)
        # "USEPUINDXD",       # Economic Policy Uncertainty Index for United States
        'UMCSENT',          # consumer sentiment (removed, slightly worse then base (no update since 3/1/24)
        # 'BSCICP03USM665S',  # sentiment OECD   (at 5/9/24, no data since Jan 24)
        # 'BSCICP02USM460S',   # Business Tendency Surveys (Manufacturing): Confidence Indicators: Composite Indicators: National Indicator for United States 
        # 'DGORDER',          # DURABLE GOODS ORDER
        # 'CORESTICKM159SFRBATL',     # Sticky Price Consumer Price Index less Food and Energy (no update in 5 mo, I think this is dead)
        'PCE',              # No data since 3/1
        'CPIAUCSL',         # Consumer Price Index for All Urban Consumers: All Items
        # 'PCU33443344',      # Producer Price Index by Industry: Semiconductor and Other Electronic Component Manufacturing 
        # 'SAHMREALTIME',     # sahm unemployment (seems to work better than unemploy)
        # 'JTSJOL',           # non farm job opening
        'GDP',              # Add GDP
        'BUSLOANS',       # business & commercial loans
        'RSAFS',          # advance retail sales, correlation with ad spending
        # 'FINRA_debit',      # finra debit
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
        # 'price_change_15',

        ##################################
        # APP specific features
        #
        'rs_gamr',
        'rs_socl',
        'rs_gamr_trend',
        'rs_socl_trend',
        # 'rs_u',
        # 'rs_ttwo',
        # 'rs_apps',
        # 'rs_u_trend',
        # 'rs_ttwo_trend',
        # 'rs_apps_trend',
        # 'cp_volume_ratio',
        'cp_sentiment_ratio'
    ],
     "window_size": 35,
    "target_size": 15,
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
}   # reference

reference = {
    "symbol": "APP",
    "model_name": "ref",
    "start_date": "2021-07-01",
    "end_date":"2024-11-05",
    "current_estEPS": 2.86, # per seeking alpha 
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
        # Idiosyncratic return (stock alpha vs market)
        ##################################
        'ret_5d_rel_SPY',   # stock 5d return minus SPY 5d return
        'ret_10d_rel_SPY',  # stock 10d return minus SPY 10d return

        ##################################
        # Analyst estimate features (EPS consensus, AV EARNINGS_ESTIMATES)
        ##################################
        'eps_est_avg',            # AV consensus average EPS estimate (upcoming quarter, raw AV units)
        # 'eps_rev_30_pct',       # BACKLOG: leakage risk — forward-filled end-of-quarter revision stat
        # 'eps_rev_7_pct',        # BACKLOG: leakage risk — forward-filled end-of-quarter revision stat
        # 'eps_breadth_ratio_30', # BACKLOG: leakage risk — forward-filled end-of-quarter revision stat
        # 'eps_dispersion',       # BACKLOG: leakage risk — forward-filled end-of-quarter revision stat

        ##################################
        # Stock Momentum 
        ##################################
        "MACD_Signal",
        "MACD",
        "MACD_Hist",
        "ATR",
        "RSI",
        'Real Upper Band',  # bollinger band (taken out as not that useful)
        'Real Middle Band',
        'Real Lower Band',

        ##################################
        # Monetary market
        ##################################
        "interest",
        "10year",
        # "2year",
        # "3month",
        "T10Y2Y",           # 10 year to 2 year yield spread
        'M2SL',             # M2 money supply
        'DTWEXBGS',         # DOLLAR INDEX (10/5/24 this not as good as using eur_close)
        'DFEDTARU',         # federal fund rate upper limit
        'BOGMBBM',          # Monetary Base; Reserve Balances
        'eur_close',
        'jpy_close',
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
        # 'VTWO_stoch',       #the appromiate for Russell Oscillator using slowD (much better)
  
        ##################################
        # General Market fundamentals
        ##################################
         "VIXCLS",           # VIX from FRED
        # "VIXCLS_Y",         # VIX from YFinance
        # "DCOILWTICO",       # WTI oil price
        "unemploy",         # as of 5/9, still no satat for 5/1
        # "FEDTARMDLR",       # Longer Run FOMC Summary of Economic Projections for the Fed Funds Rate, Median (probably not useful (in 5/9/24, last data point is 3/20/24)
        # "USEPUINDXD",       # Economic Policy Uncertainty Index for United States
        'UMCSENT',          # consumer sentiment (removed, slightly worse then base (no update since 3/1/24)
        # 'BSCICP03USM665S',  # sentiment OECD   (at 5/9/24, no data since Jan 24)
        # 'BSCICP02USM460S',   # Business Tendency Surveys (Manufacturing): Confidence Indicators: Composite Indicators: National Indicator for United States 
        # 'DGORDER',          # DURABLE GOODS ORDER
        # 'CORESTICKM159SFRBATL',     # Sticky Price Consumer Price Index less Food and Energy (no update in 5 mo, I think this is dead)
        'PCE',              # No data since 3/1
        'CPIAUCSL',         # Consumer Price Index for All Urban Consumers: All Items
        # 'PCU33443344',      # Producer Price Index by Industry: Semiconductor and Other Electronic Component Manufacturing 
        # 'SAHMREALTIME',     # sahm unemployment (seems to work better than unemploy)
        # 'JTSJOL',           # non farm job opening
        'GDP',              # Add GDP
        'BUSLOANS',       # business & commercial loans
        'RSAFS',          # advance retail sales, correlation with ad spending
        # 'FINRA_debit',      # finra debit
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
        # APP specific features
        #
        'rs_gamr',
        'rs_socl',
        'rs_gamr_trend',
        'rs_socl_trend',
        # 'rs_u',
        # 'rs_ttwo',
        # 'rs_apps',
        # 'rs_u_trend',
        # 'rs_ttwo_trend',
        # 'rs_apps_trend',
        #################################
        # call put data
        #################################
        # 'cp_volume_ratio',
        # 'cp_oi_ratio',
    ],
     "window_size": 35,
    "target_size": 15,
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
}   # reference

# Identical to reference but uses chronological (no-shuffle) train/test split — honest out-of-sample evaluation
reference_no_shuffle = {**reference, "model_name": "ref_noshuf", "shuffle_splits": False}