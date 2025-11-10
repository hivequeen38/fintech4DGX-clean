
import pandas as pd
from pandas import DataFrame
import numpy as np
import os
import logging      
import time
from datetime import datetime
import trendConfig
import trendAnalysisBlackBox
import trendAnalysisFromTodayNew

print("All libraries loaded for "+ __file__)



param = {
    "symbol": "NVDA",
    "start_date": "2021-03-01",
    "end_date":"2024-10-14",
    "current_estEPS": 0.74,
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
        'DGORDER',          # DURABLE GOODS ORDER
        'CORESTICKM159SFRBATL',     # Sticky Price Consumer Price Index less Food and Energy (no update in 5 mo, I think this is dead)
        # 'PCE',              # No data since 3/1
        'PCU33443344',      # Producer Price Index by Industry: Semiconductor and Other Electronic Component Manufacturing 
        'SAHMREALTIME',     # sahm unemployment (seems to work better than unemploy)
        'JTSJOL',           # non farm job opening
        'GDP',              # Add GDP
        'FINRA_debit',      # finra debit
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
}

# fetch closing price & date from one of the intermediate result files
def fetchDateAndClosing(param: dict[str]):
    symbol = param['symbol']
    file_path = symbol + '_TMP.csv'
    if os.path.isfile(file_path):
        # file exist
        df = pd.read_csv(file_path)
    else:
        print('>>> ERROR, file '+ file_path + ' Not found!')
        return None, None
    
    # Get the last row of the DataFrame
    last_row = df.iloc[-1]

    # Fetch the values of 'date' and 'close' columns
    # last_date = last_row['date']
    # last_close = last_row['close']

    # use the date from param 'last_date'
    # !!! Debug this tomorrow
    last_date = param['end_date']
    last_close = df.loc[df['date'] == last_date, 'adjusted close'].values[0]
    return last_date, last_close

# process result files
def processDeltaFromTodayResults( symbol: str, incr_df: DataFrame, dateStr: str, closingPrice: float, comment: str):
    file_path = symbol+ "_" + "15d_from_today_predictions.csv"
    num_of_days = 15
    df: DataFrame

    if os.path.isfile(file_path):
        # file exist
        df = pd.read_csv(file_path,index_col=False)
        # df['date'] = pd.to_datetime(df['date'])  # Convert to datetime if it's not already
    else:
        columns = []
        columns.insert(0, 'date')
        columns.insert(1,'close')
        columns.extend([f'p{i}' for i in range(1, num_of_days + 1)])
        columns.insert(16, 'comment')
        df = DataFrame(columns= columns)
        df = df.reset_index(drop=True)

    df.loc[len(df)] = np.nan  # This appends a row filled with NaNs at the end

    # Step 2: Set values for specific columns in the new row
    df.loc[len(df)-1, 'date'] = dateStr  
    df.loc[len(df)-1, 'close'] = closingPrice 
    input_col = ['p1', 'p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15']
    for target_item in input_col:
        df.loc[len(df)-1, target_item] = incr_df.iloc[0][target_item] 
    
    df.loc[len(df)-1, 'comment'] = comment 

    # now store this back to disk
    df.to_csv(file_path, index=False, header=True) # we want to save the date index

    print(">>> Final result delta from today")
    print(df)


#########################
# PROCESS END DATE
def processEndDate ( target_date: str, load_cache: bool = True):
    param['end_date'] = target_date
    run_date = datetime.now().strftime('%Y-%m-%d %H:%M')
    main(run_date= run_date, input_comment='now processing end_date= ' + target_date, load_cache = load_cache)

def process_first_5_days(param, incr_df, turn_random_on: bool):
    target_size_elements = [1,2,3,4,5]
    for target_item in target_size_elements:
        param["target_size"] = target_item
        param["comment"]='FIXED New baseline with 40 features= ' + str(target_item)
        # trendAnalysisBlackBox.analyze_trend(trendConfig.config, param, target_item, incr_df, turn_random_on, True)
        trendAnalysisFromTodayNew.analyze_trend(trendConfig.config, param, target_item, incr_df, turn_random_on, True)
    
def process_last_10_days(param, incr_df, turn_random_on: bool):
    target_size_elements = [6,7,8,9,10,11,12,13,14,15]
        # for first 5 days at 3%, then rest of the 15 days at 5%

    for target_item in target_size_elements:
        param["target_size"] = target_item
        param["comment"]='FIXED New baseline with 40 features= ' + str(target_item)
        # trendAnalysisBlackBox.analyze_trend(trendConfig.config, param, target_item, incr_df, turn_random_on, True)
        
        
        trendAnalysisFromTodayNew.analyze_trend(trendConfig.config, param, target_item, incr_df, turn_random_on, True)
    

###########################
# MAIN LOOP enters here
def main(run_date=None, input_comment=None, load_cache=True):
    logging.basicConfig(level=logging.INFO)

    if (run_date is None):
        run_date = datetime.now().strftime('%Y-%m-%d %H:%M')

    if (input_comment is None):
        input_comment = ''

    # load today's data
    # trendAnalysisBlackBox.load_data_to_cache(trendConfig.config, param)
    if load_cache == True:
        trendAnalysisFromTodayNew.load_data_to_cache(trendConfig.config, param)
    

    # FIRST DO FIXED SEED
    #
    input_col = ['date', 'close', 'p1', 'p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15', 'comment']
    incr_df = pd.DataFrame(columns=input_col)
    incr_df = incr_df.reset_index(drop=True)
    
    # for first 5 days at 3%, then rest of the 15 days at 5%

    param["threshold"] = 0.03
    process_first_5_days(param, incr_df, False)

    param["threshold"] = 0.05
    process_last_10_days(param, incr_df, False)


    # at this point there are 15x individual result files that's been updated. 
    # grab the current date and the closing price from the first one 
    # NVDA_1d_predictions_test.csv
    #
    comment = 'Refactored make_prediction_test(Fixed seed)' + ' Rundate= ' + str(run_date) + ' ' + input_comment

    dateStr, closing_price = fetchDateAndClosing(param)
    processDeltaFromTodayResults(param["symbol"], incr_df, dateStr, closing_price, comment)

    # RANDOM SEED
    #
    input_col = ['date', 'close', 'p1', 'p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15']
    incr_df = pd.DataFrame(columns=input_col)
    incr_df = incr_df.reset_index(drop=True)
    
    target_size_elements = [1,2,3,4,5]
    # for first 5 days at 3%, then rest of the 15 days at 5%

    param["threshold"] = 0.03
    process_first_5_days(param, incr_df, True)

    param["threshold"] = 0.05
    process_last_10_days(param, incr_df, True)


    # at this point there are 15x individual result files that's been updated. 
    # grab the current date and the closing price from the first one 
    # NVDA_1d_predictions_test.csv
    #
    comment = 'Refactored make_prediction_test (Random seed)'  + ' Rundate= ' + str(run_date) + ' ' + input_comment

    dateStr, closing_price = fetchDateAndClosing(param)
    processDeltaFromTodayResults(param["symbol"], incr_df, dateStr, closing_price, comment)

# mainline entrance
main()
