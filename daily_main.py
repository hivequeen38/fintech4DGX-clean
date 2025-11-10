import mainDeltafromToday
import NVDA_param
import PLTR_param
import APP_param
import META_param
import MSTR_param
import INOD_param

'''
This is the daily run entry, (either by hand, or eventually via timed activation)
All the different stock param will be here, then call into the mainDeltaFromToday() 
'''

SPY_param = {
    "symbol": "SPY",
    "start_date": "2019-01-02",
    "end_date":"2024-10-14",
    # "current_estEPS": 0.09,
    # "current_unemploy": 4.3,
    "comment": "None",
    "threshold": 0.015,
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
        # 'EPS', 
        # 'estEPS',
        # 'surprisePercentage',
        # 'totalRevenue',
        # 'netIncome',
        
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
        "2year",
        "3month",
        'DTWEXBGS',         # DOLLAR INDEX (10/5/24 this not as good as using eur_close)
        'DFEDTARU',         # federal fund rate upper limit
        'BOGMBBM',          # Monetary Base; Reserve Balances
        'eur_close',
        'jpy_close',
        # 'USALOLITONOSTSAM', # Leading Indicators: Composite Leading Indicator: Normalised for United States (in 5/9/24, no data since Jan 24)
      

        ##################################
        # Other Indeces
        ##################################
        # "SPY_close",
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
    ],
    "window_size": 35,
    "target_size": 15,
    "test_size": 0.2,
    "validation_size": 0.25,
    "batch_size": 32,       # best amongst 25, 32, and 40
    "shuffle": False,       # even though turning it on improves things, should not do it for time series data
    "headcount": 10,         
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
    "l1_lambda": 1.5e-6,          # This better than 0.5e-6
    "l2_weight_decay": 5.6e-5,  # 5.6 seems best
    "embedded_dim": 50,         # default to 70 NEED TO BE DIVISIBLE BY NUM OF HEADS
    "n_splits": 5,              # 5 is best
    "n_gap": 15,                # match the prediction horizon
    "n_test_size": 15,          # match prediction horizon
}

######################################
# MAKE SURE to change end date!
######################################

# first do NVDA
#!!! Remember to change date
def NVDA_ref_main(end_date_str: str):
    param = NVDA_param.reference
    param['end_date'] = end_date_str
    mainDeltafromToday.main(param, input_comment='(ref)')
    return

def NVDA_main(end_date_str: str):
    param = NVDA_param.AAII_reference
    param['end_date'] = end_date_str
    mainDeltafromToday.main(param, input_comment='(+AAII)')  
    return

def NVDA_mz_main(end_date_str: str):
    param = NVDA_param.reference
    param['end_date'] = end_date_str
    mainDeltafromToday.mz_main(param, input_comment='(MZ reference)')  
    return

def PLTR_ref_main(end_date_str: str):
    param = PLTR_param.reference
    param['end_date'] = end_date_str
    mainDeltafromToday.main(param, input_comment='(ref)')
    return

def PLTR_main(end_date_str: str):
    param = PLTR_param.AAII_reference
    param['end_date'] = end_date_str
    mainDeltafromToday.main(param, input_comment='(+AAII)')
    return

def APP_ref_main(end_date_str: str):
    param = APP_param.reference
    param['end_date'] = end_date_str
    mainDeltafromToday.main(param, input_comment='(ref)')
    return

def APP_main(end_date_str: str):
    param = APP_param.AAII_reference
    param['end_date'] = end_date_str
    mainDeltafromToday.main(param, input_comment='(+AAII)')
    return

def INOD_ref_main(end_date_str: str):
    param = INOD_param.reference
    param['end_date'] = end_date_str
    mainDeltafromToday.main(param, input_comment='(ref)')
    return

def INOD_main(end_date_str: str):
    param = INOD_param.AAII_reference
    param['end_date'] = end_date_str
    mainDeltafromToday.main(param, input_comment='(+AAII)')
    return

def META_ref_main(end_date_str: str):
    param = META_param.reference
    param['end_date'] = end_date_str
    mainDeltafromToday.main(param, input_comment='(ref)')
    return

def META_main(end_date_str: str):
    param = META_param.AAII_reference
    param['end_date'] = end_date_str
    mainDeltafromToday.main(param, input_comment='(+AAII)')
    return

def MSTR_ref_main(end_date_str: str):
    param = MSTR_param.reference
    param['end_date'] = end_date_str
    mainDeltafromToday.main(param, input_comment='(ref)')
    return

def MSTR_main(end_date_str: str):
    param = MSTR_param.AAII_reference
    param['end_date'] = end_date_str
    mainDeltafromToday.main(param, input_comment='(+AAII)')
    return

def NVDA_inference(end_date_str: str, load_cache: bool = True):
    param = NVDA_param.reference
    param['end_date'] = end_date_str
    mainDeltafromToday.inference(param, input_comment='inference from last Save model', load_cache=False)
    return

def PLTR_inference(end_date_str: str):
    PLTR_param['end_date'] = end_date_str
    mainDeltafromToday.inference(PLTR_param, input_comment='inference from Save Model')
    return

def SPY_main(end_date_str: str):
    SPY_param['end_date'] = end_date_str
    mainDeltafromToday.main(SPY_param, input_comment='New Training, threshold=0.015')
    return
