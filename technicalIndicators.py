import pandas as pd
import numpy as np
from pandas import DataFrame
from datetime import datetime
import matplotlib.pyplot as plt
from alpha_vantage.techindicators import TechIndicators
import fetchBulkData


################################################################################################
# When run this programe, it will get an overview of the last batch of buy and sell signals
# then it will go into an infinite loop where it will
# 1. wake up right after market close, 
# 2. calculate buy sell signals again
# 3. check latest signals, if they are from today, send alert via SMS
#

config = {
    "alpha_vantage": {
        "key": "H896AQT2GYE4ZO8Z", # Claim your free API key here: https://www.alphavantage.co/support/#api-key
        "symbol": "SPY",
        "outputsize": "full",
        "key_adjusted_close": "5. adjusted close",
        "key_volume": "6. volume",
        "key_high": "2. high",
        "key_low": "3. low",
        "SPY_symbol": "SPY",
        "url": "https://www.alphavantage.co/query"
    },
    "data": {
        "window_size": 20,      # I thhnk this is the sequence size
        "train_split_size": 0.92,
    },
    "plots": {
        "xticks_interval": 90, # show a date every 90 days
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "num_lstm_layers": 1,   # change to single layer
        "lstm_size": 36,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu", # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    },
    "eodhd": {

    }
}

buy_sell_str = ['Sell', 'Hold','Buy']
rsi_signal_str = ['Oversold', 'Hold','Overbought']

# get the closing price
def get_ma_for_symbol( symbol: str, type: str, output_df: DataFrame)-> DataFrame:
    function = "TIME_SERIES_DAILY_ADJUSTED"  # Fetch daily time series data
    datatype = "json"  # You can also fetch CSV, but JSON is easier to parse in Python
    outputsize = "full" #fetch all the historical data
    params = {
        "function": function,
        "apikey": config["alpha_vantage"]["key"],
        "datatype": datatype,
        "outputsize": outputsize,
        "symbol": symbol,
        "ascend": False
    } 
    close_df = fetchBulkData.fetch_feed(config["alpha_vantage"]["url"], params)
    close_df.rename(columns={'4. close': 'open'}, inplace=True)
    # Now get just the close column
    close_df.drop(['open','high','low', 'adjusted close', 'volume'], axis='columns', inplace=True)

    # Define the window for the moving averages
    short_window = 10
    long_window = 100

    #########################################################################
    # now get ma (of a variety of type) info from Alpha Vantage
    #
    ti = TechIndicators(key=config["alpha_vantage"]["key"])
    ma_data_short: []
    ma_data_long: []
    ma_meta_data: []

    match type:
        case 'SMA':
            ma_data_short, ma_meta_data = ti.get_sma(symbol, interval= 'daily', time_period=short_window, series_type= 'open' )
            ma_data_long, sma_meta_data = ti.get_sma(symbol, interval= 'daily', time_period=long_window, series_type= 'open' )

        case 'EMA':
            ma_data_short, ma_meta_data = ti.get_ema(symbol, interval= 'daily', time_period=short_window, series_type= 'open' )
            ma_data_long, sma_meta_data = ti.get_ema(symbol, interval= 'daily', time_period=long_window, series_type= 'open' )

        case 'WMA':
            ma_data_short, ma_meta_data = ti.get_wma(symbol, interval= 'daily', time_period=short_window, series_type= 'open' )
            ma_data_long, sma_meta_data = ti.get_wma(symbol, interval= 'daily', time_period=long_window, series_type= 'open' )

        case 'DEMA':
            ma_data_short, ma_meta_data = ti.get_dema(symbol, interval= 'daily', time_period=short_window, series_type= 'open' )
            ma_data_long, sma_meta_data = ti.get_dema(symbol, interval= 'daily', time_period=long_window, series_type= 'open' )

        case 'TEMA':
            ma_data_short, ma_meta_data = ti.get_tema(symbol, interval= 'daily', time_period=short_window, series_type= 'open' )
            ma_data_long, sma_meta_data = ti.get_tema(symbol, interval= 'daily', time_period=long_window, series_type= 'open' )
        
        case 'TRIMA':
            ma_data_short, ma_meta_data = ti.get_trima(symbol, interval= 'daily', time_period=short_window, series_type= 'open' )
            ma_data_long, sma_meta_data = ti.get_trima(symbol, interval= 'daily', time_period=long_window, series_type= 'open' )

        case 'KAMA':
            ma_data_short, ma_meta_data = ti.get_kama(symbol, interval= 'daily', time_period=short_window, series_type= 'open' )
            ma_data_long, sma_meta_data = ti.get_kama(symbol, interval= 'daily', time_period=long_window, series_type= 'open' )

        case 'T3':
            ma_data_short, ma_meta_data = ti.get_t3(symbol, interval= 'daily', time_period=short_window, series_type= 'open' )
            ma_data_long, sma_meta_data = ti.get_t3(symbol, interval= 'daily', time_period=long_window, series_type= 'open' )


    # Extracting date and atr
    # Extracting 'Technical Analysis: (x)MA' part
    # Convert to DataFrame
    ma_short = pd.DataFrame([(date, values[type]) for date, values in ma_data_short.items()], columns=['date', 'Short_MA'])

    # Extracting date and atr
    # Extracting 'Technical Analysis: SMA' part
    # Convert to DataFrame
    ma_long = pd.DataFrame([(date, values[type]) for date, values in ma_data_long.items()], columns=['date', 'Long_MA'])

    df: DataFrame = pd.DataFrame()

    # rename the column head so they can be merged
    df = pd.merge(ma_short, ma_long, on='date', how='inner')
    df = pd.merge(df, close_df, on='date', how='inner')

    # sort to ascend, and only keep last x data poiint
    df = df.head(500)
    df.sort_values(by='date', ascending=True, inplace=True)


    # Create a column to hold the buy/sell signals
    df['Signal'] = 0.0

    # Create a signal when the short moving average crosses the long moving average
    df['Signal'][short_window:] = np.where(df['Short_MA'][short_window:] > df['Long_MA'][short_window:], 1.0, 0.0)

    # Generate trading orders
    df['Position'] = df['Signal'].diff()

    #DEBUG
    df.to_csv(symbol+'_'+ type+' '+'ti_debug.csv', index=True) # we want to save the date index

    # filter out the buy signal and sell signal, and identify them
    #
    signal_df = df[df['Position'].isin([1, -1])]

    # print('==All the buy/sell signals for (' + symbol+'), TYPE=' +type+ ':')
    # print(signal_df)

    print('==> ' + symbol+ ' Last signal')
    print(signal_df.iloc[-1]['date'] + ' '+ 'singal=' +  str(signal_df.iloc[-1]['Position']))
    buy_sell_signal_str_index = int(signal_df.iloc[-1]['Position'])+1
    signal_str = buy_sell_str[buy_sell_signal_str_index]
    date_str = signal_df.iloc[-1]['date']

    new_row = pd.DataFrame({'Symbol': symbol, 'Date': date_str, 'Buy/Sell': signal_str , 'Details': 'signals from ' + type + ' crossover'}, index=[0])
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    # print('==> In routine')
    # print(output_df)

    return output_df
          

#########################################################################
# now get RSI info from Alpha Vantage
#
def get_rsi(symbol: str, output_df: DataFrame)-> DataFrame:
    ti = TechIndicators(key=config["alpha_vantage"]["key"])
    rsi_data, rsi_meta_data = ti.get_rsi(symbol, interval='daily', time_period=20, series_type='open')
    # Extracting date and atr
    # Convert to DataFrame
    df = pd.DataFrame([(date, values['RSI']) for date, values in rsi_data.items()], 
                    columns=['date', 'RSI'])
    # Create a column to hold the buy/sell signals
    df['Signal'] = 0.0

    # Convert 'RSI' column to numeric (float), handling non-numeric values
    df['RSI'] = pd.to_numeric(df['RSI'], errors='coerce')
    
    # sort to ascend, and only keep last x data poiint
    df = df.head(500)
    df.sort_values(by='date', ascending=True, inplace=True)

    # Set the 'signal' column based on the conditions
    df['signal'] = np.where(df['RSI'] > 70, 1, np.where(df['RSI'] < 30, -1, 0))
    filtered_df = df[df['signal'] != 0]
    # print('==> RSI Data')
    # print(filtered_df)

    print('==> ' + symbol+ ' Last signal from RSI')
    print(filtered_df.iloc[-1]['date'] + ' '+ 'singal=' +  str(filtered_df.iloc[-1]['signal']))
    rsi_signal_str_index = int(filtered_df.iloc[-1]['signal'])+1
    signal_str = rsi_signal_str[rsi_signal_str_index]
    date_str = filtered_df.iloc[-1]['date']

    new_row = pd.DataFrame({'Symbol': symbol, 'Date': date_str, 'Buy/Sell': signal_str , 'Details': 'signals from RSI= '+ str(filtered_df.iloc[-1]['RSI'])}, index=[0])
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    return output_df


#########################################################################
# now get MACD info from Alpha Vantage
#
def get_macd(symbol: str, output_df: DataFrame)-> DataFrame:
    ti = TechIndicators(key=config["alpha_vantage"]["key"])
    macd_data, rsi_meta_data = ti.get_macd(symbol, interval='daily', series_type='open')

    # Extracting MACD data into a DataFrame
    df = pd.DataFrame(macd_data).T
    df = df.apply(pd.to_numeric)  # Convert data from string to numeric

    # Set the name of the index
    df.index.name = 'date'

    # Now, if you want to reset the index to make 'date' a column
    df.reset_index(inplace=True)
     
    # sort to ascend, and only keep last x data poiint
    df = df.head(500)
    df.sort_values(by='date', ascending=True, inplace=True)

    # first the histogram
    df['Hist_Signal'] = 0
    df['Hist_Signal'][1:] = np.where(df['MACD_Hist'][1:] > 0, 1, np.where(df['MACD_Hist'][1:] < 0, -1, 0))
    hist_crossover_df = df[df['Hist_Signal'] != 0]

    print('==> ' + symbol+ ' Last MACD Histogram 0 crossover signal')
    print(hist_crossover_df.iloc[-1]['date'] + ' '+ 'Hist singal=' +  str(hist_crossover_df.iloc[-1]['Hist_Signal']))
    buy_sell_signal_str_index = int(hist_crossover_df.iloc[-1]['Hist_Signal'])+1
    signal_str = buy_sell_str[buy_sell_signal_str_index]
    date_str = hist_crossover_df.iloc[-1]['date']

    new_row = pd.DataFrame({'Symbol': symbol, 'Date': date_str, 'Buy/Sell': signal_str , 'Details': 'signals from MACD Histogram crossover='+ str(hist_crossover_df.iloc[-1]['MACD_Hist'])}, index=[0])
    output_df = pd.concat([output_df, new_row], ignore_index=True)

    # now handle the signal zero line cross over
    df['Signal_Line_Signal'] = 0
    df['Signal_Line_Signal'] = np.where(df['MACD'] > df['MACD_Signal'], 1, np.where(df['MACD'] < df['MACD_Signal'], -1, 0))
    signal_crossover_df = df[df['Signal_Line_Signal'] != 0]

    print('==> ' + symbol+ ' Last MACD Signal line crossover signal')
    print(signal_crossover_df.iloc[-1]['date'] + ' '+ 'Hist singal=' +  str(signal_crossover_df.iloc[-1]['Signal_Line_Signal']))
    buy_sell_signal_str_index = int(signal_crossover_df.iloc[-1]['Signal_Line_Signal'])+1
    signal_str = buy_sell_str[buy_sell_signal_str_index]
    date_str = signal_crossover_df.iloc[-1]['date']

    new_row = pd.DataFrame({'Symbol': symbol, 'Date': date_str, 'Buy/Sell': signal_str , 'Details': 'signals from MACD signal crossover='+ str(signal_crossover_df.iloc[-1]['MACD'])+ ' & signal=' + str(signal_crossover_df.iloc[-1]['MACD_Signal'])}, index=[0])
    output_df = pd.concat([output_df, new_row], ignore_index=True)

    return output_df

# Main flow starts here
# 
stocks_to_watch = {
    "NVDA": {
        "relevance_cutoff": 0.33,
        "last_timestamp": "20240129T1331"
    }, 
    "META": {
         "relevance_cutoff": 0.50,
        "last_timestamp": "20240129T1331"
    },
    "NOW": {
         "relevance_cutoff": 0.50,
        "last_timestamp": "20240129T1331"
    },
    "MSFT": {
         "relevance_cutoff": 0.50,
        "last_timestamp": "20240129T1331"
    },
    "SMCI": {
         "relevance_cutoff": 0.50,
        "last_timestamp": "20240129T1331"
    },
    "GOOG": {
         "relevance_cutoff": 0.50,
        "last_timestamp": "20240129T1331"
    },
    "AMZN": {
         "relevance_cutoff": 0.50,
        "last_timestamp": "20240129T1331"
    },
    "ADBE": {
         "relevance_cutoff": 0.50,
        "last_timestamp": "20240129T1331"
    },
    "QCOM": {
         "relevance_cutoff": 0.50,
        "last_timestamp": "20240129T1331"
    }
}

moving_average_types = {
    'SMA',
    'EMA',
    'WMA',
    'DEMA',
    'TEMA',
    'TRIMA',
    'KAMA',
    'T3'
}


####
# Main program starts here
#
# Define column names
columns = ['Symbol', 'Date', 'Buy/Sell', 'Details']

# Create an empty DataFrame with these columns
output_df = pd.DataFrame(columns=columns)

# iterate all the stocks and get the SMA
#
for symbol, details in stocks_to_watch.items():
    output_df = get_macd(symbol, output_df)
    output_df = get_rsi(symbol, output_df)
    for type in moving_average_types:
        output_df = get_ma_for_symbol(symbol, type, output_df)
        # print('DEBUG: current state of output_df, symbol=' + symbol + ' type='+ type)
        # print(output_df)

print('==> Latest buy/sell signals')
print(output_df)

# get just today's stuff
# Convert 'date' column to datetime type if it's not already
output_df['Date'] = pd.to_datetime(output_df['Date'])

# Get today's date (assuming you're running this on 2024-02-26)
today = datetime.now().strftime('%Y-%m-%d')

# Filter the DataFrame to only include rows with today's date
df_filtered = output_df[output_df['Date'] == today]

print('==>Today Buy/Sell signals')
print(df_filtered)
      

# Dump the latest batch buy/sell signal information into a file
output_df.to_csv('ti_last entries.csv', index=True) # we want to save the date index


