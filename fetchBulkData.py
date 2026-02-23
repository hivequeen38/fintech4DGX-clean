import yfinance as yf
import datetime
from pandas import DataFrame
import requests
import pandas as pd
import numpy as np
import calendar
from pandas.tseries.offsets import Week
from io import BytesIO
import pandas_datareader as pdr
import os
import time
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from dateutil.relativedelta import relativedelta

def find_next_trading_day_no_day_of_week(input_df: DataFrame, df: DataFrame):
    '''
    This call assume there is NO next_day_of_week col,
    will just go to the next trading day from the date
    '''
    # !!! Right now it assumes there is a next_trading_day col. 
    # need to refactor that out!!!

    # 3. use routine to find the next matching trading day (or equal)
    masterDate_df = df[['date']]
    masterDate_df.loc[:, 'date'] = pd.to_datetime(masterDate_df['date'])

    # Shift the 'reportDate' to the next available 'date' in master_df
    input_df['shifted_date'] = input_df['date'].apply(
        lambda x: masterDate_df[masterDate_df['date'] >= x]['date'].min()
    )
    input_df['shifted_date'] = input_df['shifted_date'].dt.strftime('%Y-%m-%d')
    input_df.drop(columns=['date'], inplace=True)
    # Step 2: Rename 'shifted_date' to 'date'
    input_df.rename(columns={'shifted_date': 'date'}, inplace=True)
    return input_df

def find_next_trading_day(input_df: DataFrame, df: DataFrame):
    '''
    This call assume there is a next_day_of_week col
    '''
    # !!! Right now it assumes there is a next_trading_day col. 
    # need to refactor that out!!!

    # 3. use routine to find the next matching trading day (or equal)
    masterDate_df = df[['date']]
    masterDate_df.loc[:, 'date'] = pd.to_datetime(masterDate_df['date'])

    # !!! depends on when this is run, the latest date that is a day of week might not be in there
  
    # Shift the 'reportDate' to the next available 'date' in master_df
    input_df['shifted_date'] = input_df['next_day_of_week'].apply(
        lambda x: masterDate_df[masterDate_df['date'] >= x]['date'].min()
    )
    input_df['shifted_date'] = input_df['shifted_date'].dt.strftime('%Y-%m-%d')
    # rename the shifted date as the new date
    input_df.drop(columns=['date', 'next_day_of_week'], inplace=True)

    # Step 2: Rename 'shifted_date' to 'date'
    input_df.rename(columns={'shifted_date': 'date'}, inplace=True)
    return input_df

def calculate_date_offset_middle_of_month(input_df: DataFrame, df: DataFrame):
    # 1. Convert the date column to datetime
    input_df['date'] = pd.to_datetime(input_df['date'])
    
    # Function to get the 15th of the same month
    def get_15th_of_month(date):
        return pd.Timestamp(date.year, date.month, 15)

    # 2. Apply the function to the 'date' column and create a new column '15th_of_month'
    input_df['15th_of_month'] = input_df['date'].apply(get_15th_of_month)
    input_df.drop(columns=['date'], inplace=True)

    # Step 2: Rename '15th_of_month' to 'date'
    input_df.rename(columns={'15th_of_month': 'date'}, inplace=True)

    # 3. Now the 15th might fall on a non trading day, so have to find ge same/next trading day via DF
    input_df = find_next_trading_day_no_day_of_week(input_df, df)
    return input_df

def calculate_date_offset_two_months(input_df: DataFrame, df: DataFrame):   #!!! No one calls this
    # 1. Convert the date column to datetime
    # input_df['date'] = pd.to_datetime(input_df['date'])
    
    # # Function to shift the date by 2 months
    # def shift_by_2_months(date):
    #     return date + relativedelta(months=2)

    # # Apply the function to the 'date' column and create a new column 'shifted_date'
    # input_df['shifted_date'] = input_df['date'].apply(shift_by_2_months)   
    # input_df.drop(columns=['date'], inplace=True)

    # # Step 2: Rename '15th_of_month' to 'date'
    # input_df.rename(columns={'shifted_date': 'date'}, inplace=True)
 

    # # 3. Now the 15th might fall on a non trading day, so have to find ge same/next trading day via DF
    # input_df = find_next_trading_day(input_df, df)
    return calculate_date_offset_num_months(input_df, df, 2)

def calculate_date_offset_num_months(input_df: DataFrame, df: DataFrame, num_of_months: int):
    # 1. Convert the date column to datetime
    input_df['date'] = pd.to_datetime(input_df['date'])
    
    # Function to shift the date by 2 months
    def shift_by_x_months(date):
        return date + relativedelta(months=num_of_months)

    # Apply the function to the 'date' column and create a new column 'shifted_date'
    input_df['shifted_date'] = input_df['date'].apply(shift_by_x_months)   
    input_df.drop(columns=['date'], inplace=True)

    # Step 2: Rename '15th_of_month' to 'date'
    input_df.rename(columns={'shifted_date': 'date'}, inplace=True)
 

    # 3. Now the 15th might fall on a non trading day, so have to find ge same/next trading day via DF
    input_df = find_next_trading_day_no_day_of_week(input_df, df)
    return input_df

def calculate_date_offset(input_df: DataFrame, df: DataFrame, offset_type: str, days_of_week: int, is_last_day_of_week: bool):
    ''' Input df is the new data structure with a date col and a col of data
        df is the master df
        offset_type is month, quarter, week, or day
        days of week is to offset to the next <<weekday>> (e.g. unemployment is always out on a friday)
        a -1 means do not adjust for days of week
        '''

    # 1. use the relativedelta to find the calendar day off set (which might be in a weekend)

    # Convert the date column to datetime
    input_df['date'] = pd.to_datetime(input_df['date'])

    # Shift each date by one month using apply and relativedelta
    if (offset_type == 'month'):
        input_df['date'] = input_df['date'].apply(lambda x: x + relativedelta(months=1))
    elif (offset_type == 'day'):
        input_df['date'] = input_df['date'].apply(lambda x: x + relativedelta(days=1))
    elif (offset_type == 'week'):
        input_df['date'] = input_df['date'].apply(lambda x: x + relativedelta(weeks=1))
    else:
        input_df['date'] = input_df['date'].apply(lambda x: x + relativedelta(months=3))


    # Func to find the next specified day of week after the shifted date
    def next_day_of_week(date):
        # Check if the date is already a matching day of week
        if date.weekday() == days_of_week:  
            return date
        else:
            # Add offset to get to the next Friday
            return date + Week(weekday=days_of_week)
        
    # Function to get the last day of week of the month
    def get_last_day_of_week(date):
        # Get the last day of the month
        _, last_day = calendar.monthrange(date.year, date.month)
        
        # Create a date object for the last day of the month
        last_day_of_month = pd.Timestamp(date.year, date.month, last_day)
        
        # Find the last Friday before or on the last day of the month
        last_day_of_week = last_day_of_month - pd.offsets.Week(weekday=days_of_week)
        return last_day_of_week

    # Apply the function to get the next day of week (if desired)
    if (days_of_week != -1):
            if (is_last_day_of_week):
                input_df['next_day_of_week'] = input_df['date'].apply(get_last_day_of_week)
            else:
                input_df['next_day_of_week'] = input_df['date'].apply(next_day_of_week)
    else:
        input_df['next_day_of_week'] = input_df['date']

    # 3. use routine to find the next matching trading day (or equal)
    input_df = find_next_trading_day(input_df, df)
    return input_df

def calculate_date_offset_month_week_days_of_week(input_df: DataFrame, df: DataFrame, num_of_months: int, num_of_weeks: int, days_of_week: int):
    ''' Input df is the new data structure with a date col and a col of data
        df is the master df
        Enter number of months/weeks to offset, then find the matching days of week of that week.
        '''

    # 1. use the relativedelta to find the calendar day off set (which might be in a weekend)

    # Convert the date column to datetime
    input_df['date'] = pd.to_datetime(input_df['date'])

    # Shift each date by n month using apply and relativedelta
    input_df['date'] = input_df['date'].apply(lambda x: x + relativedelta(months=num_of_months))
    input_df['date'] = input_df['date'].apply(lambda x: x + relativedelta(weeks=num_of_weeks))

    # Func to find the next specified day of week after the shifted date (or the same one if already landed on it)
    def next_day_of_week(date):
        # Check if the date is already a matching day of week
        if date.weekday() == days_of_week:  
            return date
        else:
            # Add offset to get to the next day of week
            return date + Week(weekday=days_of_week)

    # Apply the function to get the next day of week (if desired)
    input_df['next_day_of_week'] = input_df['date'].apply(next_day_of_week)
    
    # 3. use routine to find the next matching trading day (or equal)
    input_df = find_next_trading_day(input_df, df)
    return input_df

# import alpha_vantage.fundamentaldata as av_fund

##################################
#   this will 
#   1. read in a config yaml file
#   2. read the different exchange imformation and what to fetch from them
#   3. Fetch them in
#   4. stitch the data together (each feed is a column)
#   ASSUMPTION- date stamp are in the same format coming in
#   RETURN the result df
def merge_feed_data_frame (sourceDf: any, newDf: any):
    ''' Function merge the newDF to the sourceDF'''
    # add newdf columns to the sourcedf columns, lining up using the date index
    sourceDf = pd.merge(sourceDf, newDf, on='date', how='left')
    return sourceDf

##################################################################################################
# give all the parameter, fetch a single feed
# will return a df that is arranged in descending order WITH timestamp as index
#   symbol might not be a symbol but something like interest or treasury yield for some feeds
#
def fetch_feed(base_url: str, params: dict[str, str], config):
    ''' funciton fetch a single specified feed '''
    # check if file exist, if it does load from file instead
    symbol = params['symbol']
    file_path = symbol+'.csv'
    df= pd.DataFrame()

    ts = TimeSeries(key=config["alpha_vantage"]["key"])

    data = ts.get_daily_adjusted(symbol,outputsize= 'full')
    
    # Extract the 'Time Series FX (Daily)' part of the data
    time_series = data[0]

    # Create a DataFrame from the extracted data
    df = pd.DataFrame({
        'date': time_series.keys(),
        'open': [float(values["1. open"]) for values in time_series.values()],
        'high': [float(values["2. high"]) for values in time_series.values()],
        'low': [float(values["3. low"]) for values in time_series.values()],
        'close': [float(values["4. close"]) for values in time_series.values()],
        'adjusted close': [float(values["5. adjusted close"]) for values in time_series.values()],
        'volume': [float(values["6. volume"]) for values in time_series.values()]
    })

    # # Sort the DataFrame by date in ascending order (oldest date at the top)
    # spy_df.sort_index(inplace=True)

    # Assuming 'date' is the name of your date column
    df.sort_values(by='date', ascending=True, inplace=True)

    if params.get('end_date') is not None:
        if df.iloc[-1]['date'] < params['end_date']:
            currentTime = datetime.now().strftime('%Y-%m-%d %H:%M')
            print("At " + currentTime + " - No data for " + symbol + " after " + params['end_date'])
            exit()
    return df

####################################################
# Function to fetch Alpha Vantage Time Series Data
#
def get_Alpha_Vantage_timeseries_Data( ts: TimeSeries, df: DataFrame, feed_name: str, column_name: str, field_name: str):
    data = ts.get_daily_adjusted(feed_name,outputsize= 'full')
    
    # Extract the 'Time Series FX (Daily)' part of the data
    time_series = data[0]

    # Create a DataFrame from the extracted data
    new_df = pd.DataFrame({
        'date': time_series.keys(),
        column_name: [float(values[field_name]) for values in time_series.values()]
    })

    # Assuming 'date' is the name of your date column
    new_df.sort_values(by='date', ascending=True, inplace=True)
    df = merge_feed_data_frame(df, new_df)
    return (df)

################################################
# Function to fetch FRED data
def get_FRED_Data( df: DataFrame, feed_name: str, start_date_timestamp: str):
    '''
        date_offset_type
        none= do not offset
        month+ one month forward
        week, quarter
    '''
    # Do stuff
    data_source = 'fred'
    new_df = pdr.DataReader(feed_name, data_source, start_date_timestamp)
    
    # Convert the index into a column
    new_df = new_df.reset_index()             # this will get the date index into its own column
    new_df.rename(columns={'DATE': 'date'}, inplace=True)    # date col heading has the wrong case
    new_df['date'] = new_df['date'].dt.strftime('%Y-%m-%d')

    # now due to the fact that some of the fred data change can happen on non trading days, those data might get lost
    # we need to
    # 1. create a date df with every day (trading or non-trading from start till now)
    # 2. merge in data to that date df
    # 3. fill forward.
    # 4. THEN merge it into the master df
    #

    # Get today's date
    currentDateTime = datetime.date.today()
    end_date = currentDateTime.strftime('%Y-%m-%d')

    # Generate a date range
    date_range = pd.date_range(start=start_date_timestamp, end=end_date, freq='D')

    # Create a DataFrame with the 'date' column
    date_df = pd.DataFrame(date_range, columns=['date'])
    date_df['date'] = date_df['date'].dt.strftime('%Y-%m-%d')
    new_df = merge_feed_data_frame(date_df, new_df)
    new_df.ffill(inplace=True)

    df = merge_feed_data_frame(df, new_df)
    return df

################################################
# Function to fetch FRED data
def get_FRED_data_with_date_offset( df: DataFrame, feed_name: str, start_date_timestamp: str, offset_type: str, day_of_week: int, is_last_day_of_week: bool):
    '''
        date_offset_type
        none= do not offset
        month+ one month forward
        week, quarter
    '''
    # Do stuff
    data_source = 'fred'
    new_df = pdr.DataReader(feed_name, data_source, start_date_timestamp)
    
    # Convert the index into a column
    new_df = new_df.reset_index()             # this will get the date index into its own column
    new_df.rename(columns={'DATE': 'date'}, inplace=True)    # date col heading has the wrong case
    new_df['date'] = new_df['date'].dt.strftime('%Y-%m-%d')

    # New DF now contains a date col and a value column that needs to be time shifted
    # shift DOWN (i.e. data date of 7/1/24 is not known by public unitl 8/1)
    #
    new_df = calculate_date_offset(new_df, df, offset_type, day_of_week, is_last_day_of_week)

    # at this time new_df has dates that exist in master df and can be safely merged w/o other adjustment for weekend
    df = merge_feed_data_frame(df, new_df)
    return df

def get_FRED_data_month_week_days_of_week( df: DataFrame, feed_name: str, start_date_timestamp: str, month: int, week: int, days_of_week: int):
    '''
        For GDP, offset by 5 months, 1 week, then find the next friday
    '''
    # Do stuff
    data_source = 'fred'
    new_df = pdr.DataReader(feed_name, data_source, start_date_timestamp)
    
    # Convert the index into a column
    new_df = new_df.reset_index()             # this will get the date index into its own column
    new_df.rename(columns={'DATE': 'date'}, inplace=True)    # date col heading has the wrong case
    new_df['date'] = new_df['date'].dt.strftime('%Y-%m-%d')

    # New DF now contains a date col and a value column that needs to be time shifted
    # shift DOWN (i.e. data date of 7/1/24 is not known by public unitl 8/1)
    #
    new_df = calculate_date_offset_month_week_days_of_week(new_df, df, month, week, days_of_week )

    # at this time new_df has dates that exist in master df and can be safely merged w/o other adjustment for weekend
    df = merge_feed_data_frame(df, new_df)
    return df

def get_yfinance_data( df: DataFrame, symbol: str, start_date_timestamp: str, column_name: str):
    # Construct the ticker for the VIX
    ticker = yf.Ticker(symbol)

    # Get today's date
    today = datetime.date.today().strftime('%Y-%m-%d')

    # Use the history method to fetch the data
    data = ticker.history(start=start_date_timestamp, end=today)
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'date', 'Close': column_name}, inplace=True)
    data['date'] = data['date'].dt.strftime('%Y-%m-%d')
    data= data[['date', column_name]]
    df = merge_feed_data_frame(df, data)
    return df

def get_bitcoin_price_data(df: DataFrame, start_date_timestamp: str, column_name: str = 'btc_price'):
    # Construct the ticker for Bitcoin
    ticker = yf.Ticker('BTC-USD')

    # Get today's date
    today = datetime.date.today().strftime('%Y-%m-%d')

    # Get historical data
    data = ticker.history(start=start_date_timestamp, end=today, interval='1d')
    
    # Process historical data
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'date', 'Close': column_name}, inplace=True)
    data['date'] = data['date'].dt.strftime('%Y-%m-%d')
    data = data[['date', column_name]]
    
    # Get current price and add it as today's row
    current_price = ticker.fast_info['lastPrice']
    today_row = pd.DataFrame({'date': [today], column_name: [current_price]})
    
    # Concatenate historical data with today's price
    data = pd.concat([data, today_row], ignore_index=True)
    
    # Merge with existing dataframe
    df = merge_feed_data_frame(df, data)
    return df

def add_rsi_signals(df, rsi_column='RSI', price_column='adjusted close',
                    overbought=70, oversold=30, 
                    divergence_lookback=10, sensitivity=3):
    """
    Add additional RSI signals to existing DataFrame with RSI values
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing RSI and price data
    rsi_column : str
        Name of the RSI column in the DataFrame
    price_column : str
        Name of the price column in the DataFrame
    overbought : int
        Overbought threshold
    oversold : int
        Oversold threshold
    divergence_lookback : int
        Number of periods to look back for divergence
    sensitivity : int
        Number of consecutive periods needed to confirm signal
    
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with additional signal columns
    """
    # Create copy of DataFrame to avoid modifying original
    results = df.copy()
    
    # Convert RSI and price columns to numeric
    results[rsi_column] = pd.to_numeric(results[rsi_column], errors='coerce')
    results[price_column] = pd.to_numeric(results[price_column], errors='coerce')

    # Add overbought/oversold signals with confirmation
    results['RSI_overbought'] = 0
    results['RSI_oversold'] = 0
    
    # Require 'sensitivity' number of consecutive periods above/below thresholds
    for i in range(sensitivity, len(results)):
        if all(results[rsi_column].iloc[i-sensitivity:i] > overbought):
            results.loc[results.index[i], 'RSI_overbought'] = 1
        if all(results[rsi_column].iloc[i-sensitivity:i] < oversold):
            results.loc[results.index[i], 'RSI_oversold'] = 1
    
    # Detect RSI divergence
    results['RSI_bullish_divergence'] = 0
    results['RSI_bearish_divergence'] = 0
    
    for i in range(divergence_lookback, len(results)):
        # Price making lower low but RSI making higher low (bullish)
        if (results[price_column].iloc[i] < results[price_column].iloc[i-divergence_lookback] and 
            results[rsi_column].iloc[i] > results[rsi_column].iloc[i-divergence_lookback] and 
            results[rsi_column].iloc[i] < oversold):
            results.loc[results.index[i], 'RSI_bullish_divergence'] = 1
            
        # Price making higher high but RSI making lower high (bearish)
        if (results[price_column].iloc[i] > results[price_column].iloc[i-divergence_lookback] and 
            results[rsi_column].iloc[i] < results[rsi_column].iloc[i-divergence_lookback] and 
            results[rsi_column].iloc[i] > overbought):
            results.loc[results.index[i], 'RSI_bearish_divergence'] = 1
    
    # Add momentum strength indicator
    results['RSI_momentum_strength'] = abs(50 - results[rsi_column])
    
    return results

# Example usage
# df_with_signals = add_rsi_signals(df, 
#                                  rsi_column='RSI',
#                                  price_column='Close',
#                                  overbought=70,
#                                  oversold=30,
#                                  divergence_lookback=10,
#                                  sensitivity=3)


def prepare_rsi_features(df, normalize=True, window_sizes=[5, 10, 20]):
    """
    Prepare RSI-based features for transformer model with advanced feature engineering
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing RSI signals
    normalize : bool
        Whether to normalize numerical features
    window_sizes : list
        List of window sizes for rolling statistics
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with engineered features ready for transformer input
    """
    features = df.copy()
    
    # 1. Basic RSI signals (binary features don't need normalization)
    features['RSI_overbought'] = features['RSI_overbought'].astype(int)
    features['RSI_oversold'] = features['RSI_oversold'].astype(int)
    features['RSI_bullish_divergence'] = features['RSI_bullish_divergence'].astype(int)
    features['RSI_bearish_divergence'] = features['RSI_bearish_divergence'].astype(int)
    
    # 2. Create additional derived features
    # Distance from overbought/oversold levels
    features['RSI_dist_from_overbought'] = 70 - features['RSI']
    features['RSI_dist_from_oversold'] = features['RSI'] - 30
    
    # RSI momentum features
    features['RSI_momentum'] = features['RSI'].diff()
    features['RSI_acceleration'] = features['RSI_momentum'].diff()
    
    # Rolling statistics for RSI and momentum strength
    for window in window_sizes:
        # RSI volatility
        features[f'RSI_volatility_{window}d'] = features['RSI'].rolling(window).std()
        
        # Momentum strength trends
        features[f'RSI_momentum_strength_{window}d_mean'] = features['RSI_momentum_strength'].rolling(window).mean()
        features[f'RSI_momentum_strength_{window}d_std'] = features['RSI_momentum_strength'].rolling(window).std()
        
        # Trend strength indicators
        features[f'RSI_trend_strength_{window}d'] = abs(
            features['RSI'].rolling(window).mean() - features['RSI']
        )
    
    # 3. Normalize numerical features if requested
    if normalize:
        numerical_columns = [
            'RSI_momentum_strength',
            'RSI_dist_from_overbought',
            'RSI_dist_from_oversold',
            'RSI_momentum',
            'RSI_acceleration'
        ] + [col for col in features.columns if any(f'_{w}d' in col for w in window_sizes)]
        
        for col in numerical_columns:
            if col in features.columns:
                mean = features[col].mean()
                std = features[col].std()
                features[col] = (features[col] - mean) / (std + 1e-8)
    
    # 4. Handle missing values from rolling calculations
    features = features.bfill()
    
    return features

def find_missing_dates(master_df, input_df):
    """
    Find dates that are present in master_df but missing from input_df.
    
    Parameters:
    master_df (pandas.DataFrame): Master DataFrame with complete date entries
    input_df (pandas.DataFrame): Input DataFrame that might be missing dates
    date_column (str): Name of the date column in both DataFrames
    
    Returns:
    pandas.DataFrame: DataFrame containing only the missing dates
    """
    # Convert date columns to sets for efficient comparison
    master_dates = set(master_df['date'])
    input_dates = set(input_df['date'])
    
    # Find dates that are in master but not in input
    missing_dates = master_dates - input_dates
    
    # Create a DataFrame with missing dates
    missing_df = master_df[master_df['date'].isin(missing_dates)].copy()
    
    # Sort the missing dates to maintain the same order as master_df
    missing_df = missing_df.sort_values(by='date')
    
    return missing_df

def get_historical_cp_ratios(symbol, dates_df, api_key):
    """
    Get C/P ratios for dates provided in DataFrame
    """
    
    results = []
    
    for date_str in dates_df['date']:
        # date_str = date.strftime('%Y-%m-%d')
        try:
            print(f"Fetching data for {date_str}")
            
            response = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "HISTORICAL_OPTIONS",
                    "symbol": symbol,
                    "date": date_str,
                    "apikey": api_key
                }
            )
            
            data = response.json()
            
            if "data" in data and data["data"]:
                df = pd.DataFrame(data["data"])
                
                # Convert volume and open_interest to numeric
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce')
                
                daily_totals = df.groupby('type').agg({
                    'volume': 'sum',
                    'open_interest': 'sum'
                }).reset_index()
                
                call_data = daily_totals[daily_totals['type'] == 'call'].iloc[0]
                put_data = daily_totals[daily_totals['type'] == 'put'].iloc[0]
                
                cp_volume_ratio = call_data['volume'] / put_data['volume'] if put_data['volume'] != 0 else float('inf')
                cp_oi_ratio = call_data['open_interest'] / put_data['open_interest'] if put_data['open_interest'] != 0 else float('inf')
                
                results.append({
                    'date': date_str,
                    'call_volume': float(call_data['volume']),
                    'put_volume': float(put_data['volume']),
                    'call_oi': float(call_data['open_interest']),
                    'put_oi': float(put_data['open_interest']),
                    'cp_volume_ratio': float(cp_volume_ratio),
                    'cp_oi_ratio': float(cp_oi_ratio)
                })
            
            # Sleep to respect API rate limits
            time.sleep(12)
            
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.set_index('date', inplace=True)
        results_df.sort_index(inplace=True)

    results_df.reset_index(inplace=True)
    return results_df

def get_historical_cp_ratios_with_sentiments_OLD(symbol, dates_df, api_key):
    """
    Get C/P ratios for dates provided in DataFrame with improved handling of edge cases
    """
    # First create an empty DataFrame with the correct structure and types
    results_df = pd.DataFrame(columns=[
        'date', 'call_volume', 'put_volume', 'call_oi', 'put_oi',
        'cp_volume_ratio', 'cp_oi_ratio', 'daily_sentiment',
        'bullish_volume', 'bearish_volume'
    ])
    
    # Set proper dtypes upfront
    results_df = results_df.astype({
        'date': 'object',
        'call_volume': 'float64',
        'put_volume': 'float64',
        'call_oi': 'float64',
        'put_oi': 'float64',
        'cp_volume_ratio': 'float64',
        'cp_oi_ratio': 'float64',
        'daily_sentiment': 'object',
        'bullish_volume': 'float64',
        'bearish_volume': 'float64'
    })
    
    for date_str in dates_df['date']:
        try:
            print(f"Fetching data for {date_str}")
            
            response = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "HISTORICAL_OPTIONS",
                    "symbol": symbol,
                    "date": date_str,
                    "apikey": api_key
                }
            )
            
            data = response.json()
            
            # Initialize row with zeros and default values
            row = {
                'date': date_str,
                'call_volume': 0.0,
                'put_volume': 0.0,
                'call_oi': 0.0,
                'put_oi': 0.0,
                'cp_volume_ratio': 0.0,
                'cp_oi_ratio': 0.0,
                'daily_sentiment': 'no_trades',
                'bullish_volume': 0.0,
                'bearish_volume': 0.0
            }
            
            if "data" in data and data["data"]:
                df = pd.DataFrame(data["data"])
                
                if not df.empty:
                    # Convert numeric columns
                    for col in ['volume', 'open_interest', 'bid', 'ask', 'last']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    
                    # Calculate midpoint
                    df['mid'] = ((df['bid'] + df['ask']) / 2).fillna(df['last'])
                    
                    # Calculate sentiment
                    df['sentiment'] = df.apply(lambda x: 
                        'no_trade' if x['volume'] == 0 or pd.isna(x['last']) else
                        'bullish' if (x['last'] >= x['mid'] and x['type'] == 'call') or 
                                    (x['last'] < x['mid'] and x['type'] == 'put') else
                        'bearish', axis=1)
                    
                    # Calculate totals by type
                    type_totals = df.groupby('type').agg({
                        'volume': 'sum',
                        'open_interest': 'sum'
                    }).fillna(0)
                    
                    # Extract values safely
                    row['call_volume'] = float(type_totals.loc['call', 'volume']) if 'call' in type_totals.index else 0.0
                    row['put_volume'] = float(type_totals.loc['put', 'volume']) if 'put' in type_totals.index else 0.0
                    row['call_oi'] = float(type_totals.loc['call', 'open_interest']) if 'call' in type_totals.index else 0.0
                    row['put_oi'] = float(type_totals.loc['put', 'open_interest']) if 'put' in type_totals.index else 0.0
                    
                    # Calculate ratios
                    if row['put_volume'] > 0:
                        row['cp_volume_ratio'] = row['call_volume'] / row['put_volume']
                    if row['put_oi'] > 0:
                        row['cp_oi_ratio'] = row['call_oi'] / row['put_oi']
                    
                    # Calculate sentiment volumes
                    row['bullish_volume'] = float(df[df['sentiment'] == 'bullish']['volume'].sum())
                    row['bearish_volume'] = float(df[df['sentiment'] == 'bearish']['volume'].sum())
                    
                    # Determine overall sentiment
                    if row['bullish_volume'] == 0 and row['bearish_volume'] == 0:
                        row['daily_sentiment'] = 'no_trades'
    
                    elif row['bullish_volume'] > row['bearish_volume']:
                        row['daily_sentiment'] = 'bullish'
                    elif row['bearish_volume'] > row['bullish_volume']:
                        row['daily_sentiment'] = 'bearish'
                    else:
                        row['daily_sentiment'] = 'neutral'
            
            # Append row to results DataFrame
            results_df.loc[len(results_df)] = row
            time.sleep(12)
            
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            results_df.loc[len(results_df)] = row
            continue
    
    # Force fill any remaining NaN values
    float_cols = ['call_volume', 'put_volume', 'call_oi', 'put_oi', 
                  'cp_volume_ratio', 'cp_oi_ratio', 'bullish_volume', 'bearish_volume']
    
    results_df[float_cols] = results_df[float_cols].fillna(0.0)
    results_df['daily_sentiment'] = results_df['daily_sentiment'].fillna('no_trades')
    
    # Set index and sort
    if not results_df.empty:
        results_df.set_index('date', inplace=True)
        results_df.sort_index(inplace=True)
        results_df.reset_index(inplace=True)
    
    # Final check to ensure no NaN values
    results_df = results_df.fillna({
        'call_volume': 0.0,
        'put_volume': 0.0,
        'call_oi': 0.0,
        'put_oi': 0.0,
        'cp_volume_ratio': 0.0,
        'cp_oi_ratio': 0.0,
        'daily_sentiment': 'no_trades',
        'bullish_volume': 0.0,
        'bearish_volume': 0.0
    })
    
    print(results_df.to_string())  # This will show all values
    print(results_df.dtypes)  # This will show the data types of each column
    return results_df

def get_historical_cp_ratios_with_sentiments(symbol, dates_df, api_key):
    """
    Get C/P ratios for dates provided in DataFrame with improved handling of edge cases
    Also calculates options volume ratio (total options volume / stock volume)
    """
    # First create an empty DataFrame with the correct structure and types
    results_df = pd.DataFrame(columns=[
        'date', 'call_volume', 'put_volume', 'call_oi', 'put_oi',
        'cp_volume_ratio', 'cp_oi_ratio', 'daily_sentiment',
        'bullish_volume', 'bearish_volume', 'stock_volume', 'options_volume_ratio'
    ])
    
    # Set proper dtypes upfront
    results_df = results_df.astype({
        'date': 'object',
        'call_volume': 'float64',
        'put_volume': 'float64',
        'call_oi': 'float64',
        'put_oi': 'float64',
        'cp_volume_ratio': 'float64',
        'cp_oi_ratio': 'float64',
        'daily_sentiment': 'object',
        'bullish_volume': 'float64',
        'bearish_volume': 'float64',
        'stock_volume': 'float64',
        'options_volume_ratio': 'float64'
    })
    
    for date_str in dates_df['date']:
        try:
            print(f"Fetching data for {date_str}")
            
            # Get options data
            options_response = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "HISTORICAL_OPTIONS",
                    "symbol": symbol,
                    "date": date_str,
                    "apikey": api_key
                }
            )
            
            options_data = options_response.json()
            
            # Get stock data for the same date to get volume
            stock_response = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "TIME_SERIES_DAILY",
                    "symbol": symbol,
                    "outputsize": "compact",
                    "apikey": api_key,
                    "datatype": "json"
                }
            )
            
            stock_data = stock_response.json()
            
            # Initialize row with zeros and default values
            row = {
                'date': date_str,
                'call_volume': 0.0,
                'put_volume': 0.0,
                'call_oi': 0.0,
                'put_oi': 0.0,
                'cp_volume_ratio': 0.0,
                'cp_oi_ratio': 0.0,
                'daily_sentiment': 'no_trades',
                'bullish_volume': 0.0,
                'bearish_volume': 0.0,
                'stock_volume': 0.0,
                'options_volume_ratio': 0.0
            }
            
            # Extract stock volume if available
            if "Time Series (Daily)" in stock_data and date_str in stock_data["Time Series (Daily)"]:
                row['stock_volume'] = float(stock_data["Time Series (Daily)"][date_str]["5. volume"])
            
            if "data" in options_data and options_data["data"]:
                df = pd.DataFrame(options_data["data"])
                
                if not df.empty:
                    # Convert numeric columns
                    for col in ['volume', 'open_interest', 'bid', 'ask', 'last']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    
                    # Calculate midpoint
                    df['mid'] = ((df['bid'] + df['ask']) / 2).fillna(df['last'])
                    
                    # Calculate sentiment
                    df['sentiment'] = df.apply(lambda x: 
                        'no_trade' if x['volume'] == 0 or pd.isna(x['last']) else
                        'bullish' if (x['last'] >= x['mid'] and x['type'] == 'call') or 
                                    (x['last'] < x['mid'] and x['type'] == 'put') else
                        'bearish', axis=1)
                    
                    # Calculate totals by type
                    type_totals = df.groupby('type').agg({
                        'volume': 'sum',
                        'open_interest': 'sum'
                    }).fillna(0)
                    
                    # Extract values safely
                    row['call_volume'] = float(type_totals.loc['call', 'volume']) if 'call' in type_totals.index else 0.0
                    row['put_volume'] = float(type_totals.loc['put', 'volume']) if 'put' in type_totals.index else 0.0
                    row['call_oi'] = float(type_totals.loc['call', 'open_interest']) if 'call' in type_totals.index else 0.0
                    row['put_oi'] = float(type_totals.loc['put', 'open_interest']) if 'put' in type_totals.index else 0.0
                    
                    # Calculate ratios
                    if row['put_volume'] > 0:
                        row['cp_volume_ratio'] = row['call_volume'] / row['put_volume']
                    if row['put_oi'] > 0:
                        row['cp_oi_ratio'] = row['call_oi'] / row['put_oi']
                    
                    # Calculate sentiment volumes
                    row['bullish_volume'] = float(df[df['sentiment'] == 'bullish']['volume'].sum())
                    row['bearish_volume'] = float(df[df['sentiment'] == 'bearish']['volume'].sum())
                    
                    # Determine overall sentiment
                    if row['bullish_volume'] == 0 and row['bearish_volume'] == 0:
                        row['daily_sentiment'] = 'no_trades'
                    elif row['bullish_volume'] > row['bearish_volume']:
                        row['daily_sentiment'] = 'bullish'
                    elif row['bearish_volume'] > row['bullish_volume']:
                        row['daily_sentiment'] = 'bearish'
                    else:
                        row['daily_sentiment'] = 'neutral'
                    
                    # Calculate total options volume and options volume ratio
                    total_options_volume = row['call_volume'] + row['put_volume']
                    if row['stock_volume'] > 0:
                        row['options_volume_ratio'] = total_options_volume / row['stock_volume']
            
            # Append row to results DataFrame
            results_df.loc[len(results_df)] = row
            time.sleep(12)
            
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            results_df.loc[len(results_df)] = row
            continue
    
    # Force fill any remaining NaN values
    float_cols = ['call_volume', 'put_volume', 'call_oi', 'put_oi', 
                  'cp_volume_ratio', 'cp_oi_ratio', 'bullish_volume', 'bearish_volume',
                  'stock_volume', 'options_volume_ratio']
    
    results_df[float_cols] = results_df[float_cols].fillna(0.0)
    results_df['daily_sentiment'] = results_df['daily_sentiment'].fillna('no_trades')
    
    # Set index and sort
    if not results_df.empty:
        results_df.set_index('date', inplace=True)
        results_df.sort_index(inplace=True)
        results_df.reset_index(inplace=True)
    
    # Final check to ensure no NaN values
    results_df = results_df.fillna({
        'call_volume': 0.0,
        'put_volume': 0.0,
        'call_oi': 0.0,
        'put_oi': 0.0,
        'cp_volume_ratio': 0.0,
        'cp_oi_ratio': 0.0,
        'daily_sentiment': 'no_trades',
        'bullish_volume': 0.0,
        'bearish_volume': 0.0,
        'stock_volume': 0.0,
        'options_volume_ratio': 0.0
    })
    
    print(results_df.to_string())  # This will show all values
    print(results_df.dtypes)  # This will show the data types of each column
    return results_df

def get_historical_cp_ratios_with_sentiments_with_retry(symbol, missing_dates, api_key, max_retries=20, wait_time=300):
    """
    Gets historical CP ratios with sentiments with retry logic when data isn't ready.
    
    Parameters:
    - symbol: Stock symbol
    - missing_dates: Dates to fetch data for
    - api_key: API key for the data source
    - max_retries: Maximum number of retry attempts (default 20)
    - wait_time: Time to wait between retries in seconds (default 300 = 5 minutes)
    
    Returns:
    - DataFrame with CP ratios and sentiment data
    """
    for attempt in range(max_retries + 1):  # +1 to include the initial attempt
        # Get the data
        result_df = get_historical_cp_ratios_with_sentiments(symbol, missing_dates, api_key)
        
        # Check if data is ready (bullish and bearish volumes are not all zeros)
        if not ((result_df['bullish_volume'] == 0) & (result_df['bearish_volume'] == 0)).all():
            # Data is ready, return it
            return result_df
            
        # If this was the last attempt, break out of the loop
        if attempt == max_retries:
            break
            
        # Log the retry attempt
        print(f"Attempt {attempt+1}/{max_retries}: Data not ready for {symbol}. Waiting {wait_time/60} minutes before retry.")
        
        # Wait before the next attempt
        time.sleep(wait_time)
    
    # If we get here, we've exhausted all retries
    print(f">>>WARNING<<<: After {max_retries} attempts, sentiment data still not available for {symbol}. Proceeding with zeros.")
    return result_df

def get_historical_cp_ratios_with_sentiments_new_OLD(symbol, dates_df, api_key):
    """
    Get C/P ratios for dates provided in DataFrame with improved handling of edge cases
    Continues from where it left off if a saved file exists
    """
    file_path = f"{symbol}-cp_ratios_sentiment_w_volume.csv"
    
    # First create an empty DataFrame with the correct structure and types
    results_df = pd.DataFrame(columns=[
        'date', 'call_volume', 'put_volume', 'call_oi', 'put_oi',
        'cp_volume_ratio', 'cp_oi_ratio', 'daily_sentiment',
        'bullish_volume', 'bearish_volume'
    ])
    
    # Set proper dtypes upfront
    results_df = results_df.astype({
        'date': 'object',
        'call_volume': 'float64',
        'put_volume': 'float64',
        'call_oi': 'float64',
        'put_oi': 'float64',
        'cp_volume_ratio': 'float64',
        'cp_oi_ratio': 'float64',
        'daily_sentiment': 'object',
        'bullish_volume': 'float64',
        'bearish_volume': 'float64'
    })
    
    # Check if we already have saved results
    if os.path.exists(file_path):
        # Load existing results
        results_df = pd.read_csv(file_path)
        
        # Find dates that haven't been processed yet
        processed_dates = set(results_df['date'])
        all_dates = set(dates_df['date'])
        remaining_dates = sorted(list(all_dates - processed_dates))
        
        print(f"Found existing file with {len(results_df)} entries")
        print(f"Continuing with {len(remaining_dates)} remaining dates")
        
        # Update dates_df to only include remaining dates
        dates_df = pd.DataFrame({'date': remaining_dates})
    else:
        print(f"Starting fresh - no existing file found")
    
    for date_str in dates_df['date']:
        try:
            print(f"Fetching data for {date_str}")
            
            response = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "HISTORICAL_OPTIONS",
                    "symbol": symbol,
                    "date": date_str,
                    "apikey": api_key
                }
            )
            
            data = response.json()
            
            # Initialize row with zeros and default values
            row = {
                'date': date_str,
                'call_volume': 0.0,
                'put_volume': 0.0,
                'call_oi': 0.0,
                'put_oi': 0.0,
                'cp_volume_ratio': 0.0,
                'cp_oi_ratio': 0.0,
                'daily_sentiment': 'no_trades',
                'bullish_volume': 0.0,
                'bearish_volume': 0.0
            }
            
            if "data" in data and data["data"]:
                df = pd.DataFrame(data["data"])
                
                if not df.empty:
                    # Convert numeric columns
                    for col in ['volume', 'open_interest', 'bid', 'ask', 'last']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    
                    # Calculate midpoint
                    df['mid'] = ((df['bid'] + df['ask']) / 2).fillna(df['last'])
                    
                    # Calculate sentiment
                    df['sentiment'] = df.apply(lambda x: 
                        'no_trade' if x['volume'] == 0 or pd.isna(x['last']) else
                        'bullish' if (x['last'] >= x['mid'] and x['type'] == 'call') or 
                                    (x['last'] < x['mid'] and x['type'] == 'put') else
                        'bearish', axis=1)
                    
                    # Calculate totals by type
                    type_totals = df.groupby('type').agg({
                        'volume': 'sum',
                        'open_interest': 'sum'
                    }).fillna(0)
                    
                    # Extract values safely
                    row['call_volume'] = float(type_totals.loc['call', 'volume']) if 'call' in type_totals.index else 0.0
                    row['put_volume'] = float(type_totals.loc['put', 'volume']) if 'put' in type_totals.index else 0.0
                    row['call_oi'] = float(type_totals.loc['call', 'open_interest']) if 'call' in type_totals.index else 0.0
                    row['put_oi'] = float(type_totals.loc['put', 'open_interest']) if 'put' in type_totals.index else 0.0
                    
                    # Calculate ratios
                    if row['put_volume'] > 0:
                        row['cp_volume_ratio'] = row['call_volume'] / row['put_volume']
                    if row['put_oi'] > 0:
                        row['cp_oi_ratio'] = row['call_oi'] / row['put_oi']
                    
                    # Calculate sentiment volumes
                    row['bullish_volume'] = float(df[df['sentiment'] == 'bullish']['volume'].sum())
                    row['bearish_volume'] = float(df[df['sentiment'] == 'bearish']['volume'].sum())
                    
                    # Determine overall sentiment
                    if row['bullish_volume'] == 0 and row['bearish_volume'] == 0:
                        row['daily_sentiment'] = 'no_trades'
                    elif row['bullish_volume'] > row['bearish_volume']:
                        row['daily_sentiment'] = 'bullish'
                    elif row['bearish_volume'] > row['bullish_volume']:
                        row['daily_sentiment'] = 'bearish'
                    else:
                        row['daily_sentiment'] = 'neutral'
            
            # Append row to results DataFrame
            results_df.loc[len(results_df)] = row
            
            # Save intermediate results after each successful fetch
            results_df.to_csv(file_path, index=False)
            print(f"Saved intermediate results: {len(results_df)} total entries")
            
            time.sleep(12)
            
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            results_df.loc[len(results_df)] = row
            continue
    
    # Force fill any remaining NaN values
    float_cols = ['call_volume', 'put_volume', 'call_oi', 'put_oi', 
                  'cp_volume_ratio', 'cp_oi_ratio', 'bullish_volume', 'bearish_volume']
    
    results_df[float_cols] = results_df[float_cols].fillna(0.0)
    results_df['daily_sentiment'] = results_df['daily_sentiment'].fillna('no_trades')
    
    # Set index and sort
    if not results_df.empty:
        results_df.set_index('date', inplace=True)
        results_df.sort_index(inplace=True)
        results_df.reset_index(inplace=True)
    
    # Final check to ensure no NaN values
    results_df = results_df.fillna({
        'call_volume': 0.0,
        'put_volume': 0.0,
        'call_oi': 0.0,
        'put_oi': 0.0,
        'cp_volume_ratio': 0.0,
        'cp_oi_ratio': 0.0,
        'daily_sentiment': 'no_trades',
        'bullish_volume': 0.0,
        'bearish_volume': 0.0
    })
    
    # only keep the 4 columns (Note: you had 'call_volumne' with typo, I fixed it)
    columns = ['date', 'call_volume', 'put_volume', 'cp_volume_ratio', 'cp_oi_ratio', 'bullish_volume', 'bearish_volume']
    results_df = results_df[columns]

    print(results_df.to_string())  # This will show all values
    print(results_df.dtypes)  # This will show the data types of each column
    return results_df



def clean_cp_ratio_file(symbol):
    """
    Clean up the existing CP ratio file by removing duplicates
    """
    file_path = f"{symbol}-cp_ratios_sentiment_w_volume.csv"
    
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Nothing to clean.")
        return
    
    # Read the file
    try:
        df = pd.read_csv(file_path)
        original_count = len(df)
        
        # Check if the file has headers
        if 'date' not in df.columns and df.shape[1] == 10:
            # File doesn't have headers, assign them
            df.columns = [
                'date', 'call_volume', 'put_volume', 'call_oi', 'put_oi',
                'cp_volume_ratio', 'cp_oi_ratio', 'daily_sentiment',
                'bullish_volume', 'bearish_volume'
            ]
        
        # Drop duplicates
        df = df.drop_duplicates(subset=['date'])
        cleaned_count = len(df)
        
        # Save cleaned file
        df.to_csv(file_path, index=False)
        
        print(f"Cleaned {file_path}:")
        print(f"  Original rows: {original_count}")
        print(f"  After deduplication: {cleaned_count}")
        print(f"  Removed {original_count - cleaned_count} duplicate rows")
        
        return df
    except Exception as e:
        print(f"Error cleaning file: {e}")
        return None

def get_historical_cp_ratios_with_sentiments_new(symbol, dates_df, api_key):
    """
    Get C/P ratios for dates provided in DataFrame with improved handling of edge cases
    Continues from where it left off if a saved file exists
    """
    file_path = f"{symbol}-cp_ratios_sentiment_w_volume.csv"
    
    # Clean up the existing file first to remove duplicates
    if os.path.exists(file_path):
        print("First cleaning the existing file to remove duplicates...")
        clean_cp_ratio_file(symbol)
    
    # First create an empty DataFrame with the correct structure and types
    results_df = pd.DataFrame(columns=[
        'date', 'call_volume', 'put_volume', 'call_oi', 'put_oi',
        'cp_volume_ratio', 'cp_oi_ratio', 'daily_sentiment',
        'bullish_volume', 'bearish_volume'
    ])
    
    # Set proper dtypes upfront
    results_df = results_df.astype({
        'date': 'object',
        'call_volume': 'float64',
        'put_volume': 'float64',
        'call_oi': 'float64',
        'put_oi': 'float64',
        'cp_volume_ratio': 'float64',
        'cp_oi_ratio': 'float64',
        'daily_sentiment': 'object',
        'bullish_volume': 'float64',
        'bearish_volume': 'float64'
    })
    
    # Check if we already have saved results
    if os.path.exists(file_path):
        # Load existing results
        existing_df = pd.read_csv(file_path)
        
        # Find dates that haven't been processed yet
        processed_dates = set(existing_df['date'])
        all_dates = set(dates_df['date'])
        remaining_dates = sorted(list(all_dates - processed_dates))
        
        print(f"Found existing file with {len(existing_df)} entries")
        print(f"Continuing with {len(remaining_dates)} remaining dates")
        
        # Start with the de-duplicated existing data
        results_df = existing_df.copy()
        
        # Update dates_df to only include remaining dates
        dates_to_process_df = pd.DataFrame({'date': remaining_dates})
    else:
        print(f"Starting fresh - no existing file found")
        dates_to_process_df = dates_df.copy()
    
    # Track if we added any new data
    new_data_added = False
    
    # Only process the remaining dates
    for date_str in dates_to_process_df['date']:
        try:
            print(f"Fetching data for {date_str}")
            
            response = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "HISTORICAL_OPTIONS",
                    "symbol": symbol,
                    "date": date_str,
                    "apikey": api_key
                }
            )
            
            data = response.json()
            
            # Initialize row with zeros and default values
            row = {
                'date': date_str,
                'call_volume': 0.0,
                'put_volume': 0.0,
                'call_oi': 0.0,
                'put_oi': 0.0,
                'cp_volume_ratio': 0.0,
                'cp_oi_ratio': 0.0,
                'daily_sentiment': 'no_trades',
                'bullish_volume': 0.0,
                'bearish_volume': 0.0
            }
            
            if "data" in data and data["data"]:
                df = pd.DataFrame(data["data"])
                
                if not df.empty:
                    # Convert numeric columns
                    for col in ['volume', 'open_interest', 'bid', 'ask', 'last']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    
                    # Calculate midpoint
                    df['mid'] = ((df['bid'] + df['ask']) / 2).fillna(df['last'])
                    
                    # Calculate sentiment
                    df['sentiment'] = df.apply(lambda x: 
                        'no_trade' if x['volume'] == 0 or pd.isna(x['last']) else
                        'bullish' if (x['last'] >= x['mid'] and x['type'] == 'call') or 
                                    (x['last'] < x['mid'] and x['type'] == 'put') else
                        'bearish', axis=1)
                    
                    # Calculate totals by type
                    type_totals = df.groupby('type').agg({
                        'volume': 'sum',
                        'open_interest': 'sum'
                    }).fillna(0)
                    
                    # Extract values safely
                    row['call_volume'] = float(type_totals.loc['call', 'volume']) if 'call' in type_totals.index else 0.0
                    row['put_volume'] = float(type_totals.loc['put', 'volume']) if 'put' in type_totals.index else 0.0
                    row['call_oi'] = float(type_totals.loc['call', 'open_interest']) if 'call' in type_totals.index else 0.0
                    row['put_oi'] = float(type_totals.loc['put', 'open_interest']) if 'put' in type_totals.index else 0.0
                    
                    # Calculate ratios
                    if row['put_volume'] > 0:
                        row['cp_volume_ratio'] = row['call_volume'] / row['put_volume']
                    if row['put_oi'] > 0:
                        row['cp_oi_ratio'] = row['call_oi'] / row['put_oi']
                    
                    # Calculate sentiment volumes
                    row['bullish_volume'] = float(df[df['sentiment'] == 'bullish']['volume'].sum())
                    row['bearish_volume'] = float(df[df['sentiment'] == 'bearish']['volume'].sum())
                    
                    # Determine overall sentiment
                    if row['bullish_volume'] == 0 and row['bearish_volume'] == 0:
                        row['daily_sentiment'] = 'no_trades'
                    elif row['bullish_volume'] > row['bearish_volume']:
                        row['daily_sentiment'] = 'bullish'
                    elif row['bearish_volume'] > row['bullish_volume']:
                        row['daily_sentiment'] = 'bearish'
                    else:
                        row['daily_sentiment'] = 'neutral'
            
            # Add as a new row - we've already filtered out processed dates
            results_df.loc[len(results_df)] = row
            new_data_added = True
            
            # Save after processing each date
            if new_data_added:
                # For safety, explicitly drop duplicates before saving
                results_df = results_df.drop_duplicates(subset=['date'])
                results_df.to_csv(file_path, index=False)
                print(f"Saved data: {len(results_df)} total unique entries")
            
            time.sleep(12)
            
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            continue
    
    # Force fill any remaining NaN values
    float_cols = ['call_volume', 'put_volume', 'call_oi', 'put_oi', 
                  'cp_volume_ratio', 'cp_oi_ratio', 'bullish_volume', 'bearish_volume']
    
    results_df[float_cols] = results_df[float_cols].fillna(0.0)
    results_df['daily_sentiment'] = results_df['daily_sentiment'].fillna('no_trades')
    
    # Set index and sort
    if not results_df.empty:
        # First save a sorted version
        results_df_sorted = results_df.sort_values(by='date')
        results_df_sorted.to_csv(file_path, index=False)
        
        # Now continue with processing
        results_df.set_index('date', inplace=True)
        results_df.sort_index(inplace=True)
        results_df.reset_index(inplace=True)
    
    # Final check to ensure no NaN values
    results_df = results_df.fillna({
        'call_volume': 0.0,
        'put_volume': 0.0,
        'call_oi': 0.0,
        'put_oi': 0.0,
        'cp_volume_ratio': 0.0,
        'cp_oi_ratio': 0.0,
        'daily_sentiment': 'no_trades',
        'bullish_volume': 0.0,
        'bearish_volume': 0.0
    })
    
    # only keep the required columns
    columns = ['date', 'call_volume', 'put_volume', 'cp_volume_ratio', 'cp_oi_ratio', 'bullish_volume', 'bearish_volume']
    if all(col in results_df.columns for col in columns):
        results_df = results_df[columns]
    
    print(results_df.to_string())  # This will show all values
    print(results_df.dtypes)  # This will show the data types of each column
    return results_df

import time
from requests.exceptions import RequestException  # Adjust this import based on what exceptions your code might throw

def fetch_with_retry(currency_from, currency_to, config, max_retries=10, sleep_time=300):
    """
    Fetch exchange rate data with retry logic
    
    Args:
        currency_from: Source currency code (e.g., 'TWD')
        currency_to: Target currency code (e.g., 'USD')
        max_retries: Maximum number of retry attempts
        sleep_time: Time to sleep between retries in seconds (default: 300 = 5 minutes)
    
    Returns:
        Exchange rate data if successful
    
    Raises:
        Exception: If all retry attempts fail
    """
    fx_data = ForeignExchange(key=config["alpha_vantage"]["key"])
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Attempt to fetch the exchange rate data
            exchange_rate = fx_data.get_currency_exchange_daily(
                currency_from, currency_to, outputsize='full'
            )
            # If successful, return the data
            return exchange_rate
            
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                # If we've reached the maximum retries, propagate the error
                raise Exception(f"Failed to fetch {currency_from}/{currency_to} exchange rate after {max_retries} attempts: {str(e)}")
            
            print(f"Attempt {retry_count} failed: {str(e)}. Retrying in {sleep_time/60} minutes...")
            time.sleep(sleep_time)  # Sleep for the specified time (default 5 minutes)
    
    # This should never be reached due to the raise in the exception handler
    # but including it for completeness
    raise Exception(f"Failed to fetch {currency_from}/{currency_to} exchange rate after {max_retries} attempts")

def compute_dte_dse_features(df, eff_sessions, symbol):
    """
    Add DTE/DSE earnings-proximity features to df (in-place copy).

    Parameters
    ----------
    df : DataFrame
        Must have a 'date' column (string 'YYYY-MM-DD'), sorted ascending.
        df['date'] is used as the trading calendar.
    eff_sessions : list of datetime-like
        Sorted effective earnings dates (next trading day after each report).
        Include the upcoming report date (from param['next_report_date']) if known.
    symbol : str
        Ticker symbol — used only for the warning message.

    Returns
    -------
    DataFrame with five new columns appended:
        dte        – trading days to next earnings (999 = none known)
        dse        – trading days since last earnings (NaN if no prior)
        earn_in_5  – 1 if dte in [0, 5]
        earn_in_10 – 1 if dte in [0, 10]
        earn_in_20 – 1 if dte in [0, 20]
    """
    import numpy as np

    df = df.copy()
    td = pd.to_datetime(df['date']).reset_index(drop=True)
    n = len(td)

    td_pos = {d: i for i, d in enumerate(td)}
    eff = pd.DatetimeIndex(pd.to_datetime(list(eff_sessions))).sort_values()

    dte_arr = np.full(n, 999, dtype=float)
    dse_arr = np.full(n, np.nan, dtype=float)

    for i in range(n):
        d = td.iloc[i]

        # DTE: first effective session >= today
        nj = eff.searchsorted(d, side='left')
        if nj < len(eff):
            nday = eff[nj]
            if nday in td_pos:
                dte_arr[i] = td_pos[nday] - i
            else:
                # Earnings beyond price history — approximate via calendar days
                dte_arr[i] = max(0, round((nday - d).days * 252 / 365))

        # DSE: most recent effective session <= today
        lj = eff.searchsorted(d, side='right') - 1
        if lj >= 0:
            lday = eff[lj]
            if lday in td_pos:
                dse_arr[i] = i - td_pos[lday]
            else:
                dse_arr[i] = max(0, round((d - lday).days * 252 / 365))

    sentinel_count = int((dte_arr == 999).sum())
    if sentinel_count > 0:
        print(f"  ⚠️  [{symbol}] {sentinel_count} row(s) have no known upcoming earnings "
              f"(dte=999). Add 'next_report_date' to {symbol}_param.py to fix.")

    df['dte']        = dte_arr
    df['dse']        = dse_arr
    df['earn_in_5']  = ((df['dte'] >= 0) & (df['dte'] <= 5)).astype(int)
    df['earn_in_10'] = ((df['dte'] >= 0) & (df['dte'] <= 10)).astype(int)
    df['earn_in_20'] = ((df['dte'] >= 0) & (df['dte'] <= 20)).astype(int)
    return df


def fetch_next_report_date(symbol, api_key):
    """
    Try to fetch the next upcoming earnings date for `symbol` from live sources.

    Priority:
      1. Alpha Vantage EARNINGS_CALENDAR (CSV endpoint, free tier)
      2. yfinance ticker.calendar
    Returns a pd.Timestamp, or None if both fail.
    """
    # 1. Alpha Vantage EARNINGS_CALENDAR
    try:
        url = (
            f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR"
            f"&symbol={symbol}&horizon=3month&apikey={api_key}"
        )
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and resp.content:
            from io import StringIO
            df_ec = pd.read_csv(StringIO(resp.text))
            if 'symbol' in df_ec.columns and 'reportDate' in df_ec.columns:
                df_ec = df_ec[df_ec['symbol'] == symbol].copy()
                if not df_ec.empty:
                    df_ec['reportDate'] = pd.to_datetime(df_ec['reportDate'])
                    upcoming = df_ec[
                        df_ec['reportDate'] >= pd.Timestamp.today().normalize()
                    ].sort_values('reportDate')
                    if not upcoming.empty:
                        nrd = upcoming.iloc[0]['reportDate']
                        print(f">  [AV] Next earnings for {symbol}: {nrd.date()}")
                        return nrd
    except Exception as e:
        print(f">  [AV] EARNINGS_CALENDAR fetch failed for {symbol}: {e}")

    # 2. yfinance
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        if cal is not None:
            # yfinance >= 0.2 returns a dict; older returns a DataFrame
            if isinstance(cal, dict) and 'Earnings Date' in cal:
                dates = cal['Earnings Date']
                if dates:
                    nrd = pd.to_datetime(dates[0])
                    print(f">  [yfinance] Next earnings for {symbol}: {nrd.date()}")
                    return nrd
            elif hasattr(cal, 'loc') and 'Earnings Date' in cal.index:
                nrd = pd.to_datetime(cal.loc['Earnings Date'].iloc[0])
                print(f">  [yfinance] Next earnings for {symbol}: {nrd.date()}")
                return nrd
    except Exception as e:
        print(f">  [yfinance] earnings fetch failed for {symbol}: {e}")

    return None


################################################
# return all the merged data into a single DF
#
def fetch_all_data(config, param):
    """Function get ALL the relevant strings."""

    symbol = param['symbol']
    base_url = config["alpha_vantage"]["url"]
    api_key = config["alpha_vantage"]["key"]

    print('api_key= ', api_key)
    df= pd.DataFrame()

    print('Now processing symbol= ', symbol)
    function = "TIME_SERIES_DAILY_ADJUSTED"  # Fetch daily time series data
    datatype = "json"  # You can also fetch CSV, but JSON is easier to parse in Python
    outputsize = "full" #fetch all the historical data
    params = {
        "function": function,
        "apikey": api_key,
        "datatype": datatype,
        "outputsize": outputsize,
        "symbol": symbol,
        "ascend": True
    }
    print('>Get stock feed for symbol= ', symbol)
    df = fetch_feed(base_url, params, config)


    ####################
    # GET FINRA DATA
    #
    # URL of the .XLS file

    print('>Get FINRA Data')
    url = 'https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx'

    try:
        # Fetch the content of the URL
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors

        # Verify the content is an Excel file
        content_type = response.headers.get('Content-Type')
        if content_type != 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            raise ValueError(f"Expected an Excel file, but got Content-Type: {content_type}")

        # Read the Excel file into a DataFrame using 'openpyxl' engine
        finra_df = pd.read_excel(BytesIO(response.content), engine='openpyxl')

        # Display the DataFrame
        # print(finra_df.head())

    except Exception as e:
        print(f"An error occurred getting FINRA data: {e}")

    finra_df.rename(columns={'Year-Month': 'date', "Debit Balances in Customers' Securities Margin Accounts": 'FINRA_debit'}, inplace=True)
    # Convert 'date' column to datetime; day defaults to 1
    # Convert the 'date' column to datetime format
    finra_df['date'] = pd.to_datetime(finra_df['date'], format='%Y-%m')
    # Keep only the 'date' and 'FINRA_debit' columns
    finra_df =  finra_df[['date', 'FINRA_debit']]
    finra_df.sort_values(by='date', ascending=True, inplace=True)

    # Filter the DataFrame to eliminate rows before the target date
    start_date_timestamp = param['start_date']

    target_date = pd.to_datetime(start_date_timestamp)       
    finra_df = finra_df[finra_df['date'] >= target_date]

    # finra is released middle of next month
    finra_df = calculate_date_offset_num_months(finra_df, df, 1)
    finra_df = calculate_date_offset_middle_of_month(finra_df, df)
    df = merge_feed_data_frame(df, finra_df)

    #####################
    # get AAII survey data

    # print('>Get AAII Data')
    # url = 'https://www.aaii.com/files/surveys/sentiment.xls'

    # filepath = 'sentiment.xls'

    # try:
    #     if filepath:
    #         aaii_df = pd.read_excel(filepath, engine='xlrd', header=3)

    #         # OR Method 2: Drop rows where date conversion fails
    #         aaii_df = aaii_df[pd.to_datetime(aaii_df['Date'], errors='coerce').notna()]    

    #         # Reset the index
    #         aaii_df = aaii_df.reset_index(drop=True)
    #     else:
    #         # Fetch the content of the URL
    #         response = requests.get(url)
    #         response.raise_for_status()  # Check for HTTP errors

    #         # Verify the content is an Excel file
    #         content_type = response.headers.get('Content-Type')
    #         if content_type != 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
    #             raise ValueError(f"Expected an Excel file, but got Content-Type: {content_type}")

    #         # Read the Excel file into a DataFrame using 'openpyxl' engine
    #         aaii_df = pd.read_excel(BytesIO(response.content), engine='openpyxl')

    # except Exception as e:
    #     print(f"An error occurred getting AAII data: {e}")

    # aaii_df.rename(columns={'Date': 'date'}, inplace=True)
    # # Convert 'date' column to datetime; day defaults to 1
    # # Convert the 'date' column to datetime format
    # aaii_df['date'] = pd.to_datetime(aaii_df['date'], format='%Y-%m-%d')
    # # Keep only the 'date' and 'Bullish', 'Neutral', & 'Bearish' columns
    # aaii_df =  aaii_df[['date', 'Bullish', 'Bearish', 'Spread']]
    # aaii_df.sort_values(by='date', ascending=True, inplace=True)

    # # Filter the DataFrame to eliminate rows before the target date
    # start_date_timestamp = param['start_date']

    # target_date = pd.to_datetime(start_date_timestamp)       
    # aaii_df = aaii_df[aaii_df['date'] >= target_date]
    # aaii_df['date'] = aaii_df['date'].dt.strftime('%Y-%m-%d')   # needed before a merge
    # df = merge_feed_data_frame(df, aaii_df)

    # get AAII survey data
    print('>Get AAII Data')
    url = 'https://www.aaii.com/files/surveys/sentiment.csv'  # Changed URL to CSV

    filepath = 'sentiment.csv'  # Changed file extension to CSV

    # Use pandas read_csv instead of read_excel
    # Since CSV already has a header row, don't skip rows with header parameter
    aaii_df = pd.read_csv(filepath)
    
    # Check column names to debug
    # print(f"Columns in the dataframe: {aaii_df.columns.tolist()}")
    
    # Make sure 'Date' exists before renaming
    # if 'Date' in aaii_df.columns:
    #     aaii_df.rename(columns={'Date': 'date'}, inplace=True)
    # else:
    #     # If 'Date' doesn't exist, try to find a date column
    #     date_columns = [col for col in aaii_df.columns if 'date' in col.lower()]
    #     if date_columns:
    #         # Use the first column that might be a date
    #         aaii_df.rename(columns={date_columns[0]: 'date'}, inplace=True)
    #     else:
    #         # If no date column found, let's assume the first column is the date
    #         aaii_df.rename(columns={aaii_df.columns[0]: 'date'}, inplace=True)
    #         print(f"No date column found. Using {aaii_df.columns[0]} as date.")
    
    # # Drop rows where date conversion fails
    # aaii_df = aaii_df[pd.to_datetime(aaii_df['date'], errors='coerce').notna()]    

    # Reset the index
    aaii_df = aaii_df.reset_index(drop=True)

    # except Exception as e:
    #     print(f"An error occurred getting AAII data: {e}")
    #     raise  # Re-raise to see the full traceback

    # Convert the 'date' column to datetime format - adjusting for MM-DD-YY format
    aaii_df['date'] = pd.to_datetime(aaii_df['date'], format='%m-%d-%y', errors='coerce')
    # print(f"Date column after conversion: {aaii_df['date'].head()}")

    # Check if required columns exist
    required_columns = ['Bullish', 'Bearish', 'Spread']
    missing_columns = [col for col in required_columns if col not in aaii_df.columns]
    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        print(f"Available columns: {aaii_df.columns.tolist()}")
        # Try to find similar column names
        for missing_col in missing_columns:
            potential_matches = [col for col in aaii_df.columns if missing_col.lower() in col.lower()]
            if potential_matches:
                print(f"Potential matches for '{missing_col}': {potential_matches}")
                # Use the first match
                aaii_df.rename(columns={potential_matches[0]: missing_col}, inplace=True)

    # Keep only the required columns, but check if they exist first
    cols_to_keep = ['date']
    for col in required_columns:
        if col in aaii_df.columns:
            cols_to_keep.append(col)

    aaii_df = aaii_df[cols_to_keep]
    aaii_df.sort_values(by='date', ascending=True, inplace=True)

    # Filter the DataFrame to eliminate rows before the target date
    start_date_timestamp = param['start_date']

    target_date = pd.to_datetime(start_date_timestamp)       
    aaii_df = aaii_df[aaii_df['date'] >= target_date]
    aaii_df['date'] = aaii_df['date'].dt.strftime('%Y-%m-%d')   # needed before a merge
    df = merge_feed_data_frame(df, aaii_df)
    
    ####################################
    # now fetch the interest rate
    # check if file exist, if it does load from file instead
    #  (switch from using FRED which is where AV get there data from anyway)

    print('>Get Interest rate')
    

    df = get_FRED_Data(df, 'DFF', start_date_timestamp)
    df.rename(columns={'DFF': 'interest'},inplace=True)

  
    ##############################################################
    # now fetch the SPY rate (use SPY to approximate SP500 index 
    # check if file exist, if it does load from file instead
    #
    print('>Get SPY data')
    ts = TimeSeries(key=config["alpha_vantage"]["key"])

    df = get_Alpha_Vantage_timeseries_Data( ts, df, 'SPY', 'SPY_close', '5. adjusted close')

    if symbol == 'NVDA' or symbol == 'SMCI' or symbol == 'CRDO' or symbol == 'TSM' or symbol == 'ANET' or symbol == 'ALAB' or symbol == 'TSLA':
        print('>Get AMD data')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])

        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'AMD', 'AMD_close', '5. adjusted close')

        print('>Get Intel data')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])

        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'INTC', 'INTC_close', '5. adjusted close')

        print('>Get Broadcom data')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])

        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'AVGO', 'AVGO_close', '5. adjusted close')

        print('>Get SMH (semi conductor ETF) data')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])

        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'SMH', 'SMH_close', '5. adjusted close')

        print('>Get TSMC data')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])

        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'TSM', 'TSMC_close', '5. adjusted close')

        ####################
        # Chip equipment maker
        #
        print('>Get Chip equipment maker data (ASML)')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'ASML', 'ASML_close', '5. adjusted close')

        print('>Get Chip equipment maker data (AMAT)')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'AMAT', 'AMAT_close', '5. adjusted close')

        print('>Get Chip equipment maker data (Lam research)')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'LRCX', 'LRCX_close', '5. adjusted close')

        ####################
        # AI chip customers
        #
        print('>Get AI chip customer data (MSFT)')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'MSFT', 'MSFT_close', '5. adjusted close')

        print('>Get AI chip customer data (META)')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'META', 'META_close', '5. adjusted close')

        print('>Get AI chip customer data (GOOGLE)')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'GOOGL', 'GOOGL_close', '5. adjusted close')

        print('>Get AI chip customer data (CRM)')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'CRM', 'CRM_close', '5. adjusted close')

        ####################################
        # Get Bittorrent data
        #
        # print('>Get BTC data')
        # df = get_bitcoin_price_data(df, start_date_timestamp, 'btc_price')



    if symbol == 'PLTR':
        print('>Get ITA (defense & aerospace ETF) data')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'ITA', 'ITA_close', '5. adjusted close')

        print('>Get IGV (software sector) data')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'IGV', 'IGV_close', '5. adjusted close')

        print('>Get FDEFX (National Defense Consumption Expenditures and Gross Investment) data')
        df = get_FRED_data_month_week_days_of_week(df, 'FDEFX', start_date_timestamp, 1, 5, 2)

        print('>Get ADEFNO (Manufacturers New Orders: Defense Capital Goods) data')
        df = get_FRED_data_month_week_days_of_week(df, 'ADEFNO', start_date_timestamp, 2, 1, 0)
        
        print('>Get IPDCONGD (Industrial Production: Durable Consumer Goods) data')
        df = get_FRED_data_month_week_days_of_week(df, 'IPDCONGD', start_date_timestamp, 2, 3, 3)

        ####################
        # GET AI MARKET RELATED data
        #
        print('>Get AI Market data (BOTZ: Global X Robotics & AI ETF)')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'BOTZ', 'BOTZ_close', '5. adjusted close')

        print('>Get AI Market data (AIQ: Global X Artificial Intelligence ETF)')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'AIQ', 'AIQ_close', '5. adjusted close')

        print('>Get AI Market data (HNQ: ROBO Global Artificial Intelligence ETF)')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'THNQ', 'THNQ_close', '5. adjusted close')
        

    if symbol == 'APP':
        print('>Get GAMR (Wedbush ETFMG Video Game Tech ETF) data')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'GAMR', 'GAMR_close', '5. adjusted close')

        print('>Get SOCL (Global X Social Media ETF (includes mobile advertising) data')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'SOCL', 'SOCL_close', '5. adjusted close')

        print('>Get U (unity ironsource) data')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'U', 'U_close', '5. adjusted close')

        print('>Get ttwo (Take-Two Chartboost) data')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'TTWO', 'TTWO_close', '5. adjusted close')

        print('>Get APPS (Digital Turbine) data')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'APPS', 'APPS_close', '5. adjusted close')

    if symbol == 'META':
        print('>Get SOCL (Global X Social Media ETF (includes mobile advertising) data')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'SOCL', 'SOCL_close', '5. adjusted close')

        print('>Get XLC (Communication Services Select Sector SPDR Fund) data')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'XLC', 'XLC_close', '5. adjusted close')

    # if symbol == 'MSTR':
        ####################################
        # Get Bittorrent data
        #
        # print('>Get BTC data')
        # df = get_bitcoin_price_data(df, start_date_timestamp, 'btc_price')

    if symbol == 'INOD':
        ####################################
        # Get XLK data (technology sector)
        #
        print('>Get XLK (Technology Select Sector SPDR Fund) data')
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        df = get_Alpha_Vantage_timeseries_Data( ts, df, 'XLK', 'XLK_close', '5. adjusted close')

    ##############################################################
    # now fetch the QQQ rate (use QQQ to approximate Nasdaq 
    # check if file exist, if it does load from file instead
    #
    print('>Get QQQ rate')
    df = get_Alpha_Vantage_timeseries_Data( ts, df, 'QQQ', 'qqq_close', '5. adjusted close')

    ##############################################################
     # now fetch the VTWO rate (use VTWO to approximate Russell 2000 
    # check if file exist, if it does load from file instead
    #
    print('>Get VTWO (Russell 2000) rate')
    df = get_Alpha_Vantage_timeseries_Data( ts, df, 'VTWO', 'VTWO_close', '5. adjusted close')

    ####################################
    # now fetch the 10 year treasury rate
    #  !!! Note treasury should not shift since that data is avail real time
    #
    # df = get_FRED_data_with_date_offset(df, 'DGS10', start_date_timestamp, 'day', -1, False)
    # df.rename(columns={'DGS10': 'DGS10_S'}, inplace=True)

    print('>Get 10 yrs treasury rate')
    df = get_FRED_Data( df, 'DGS10', start_date_timestamp)
    df.rename(columns={'DGS10': '10year'}, inplace=True)
    
    ####################################
    # now fetch the 2 year treasury rate
    #
    # df = get_FRED_data_with_date_offset(df, 'DGS2', start_date_timestamp, 'day', -1, False)
    # df.rename(columns={'DGS2': 'DG2_S'}, inplace=True)
    print('>Get 2 yrs treasury rate')
    df = get_FRED_Data( df, 'DGS2', start_date_timestamp)
    df.rename(columns={'DGS2': '2year'}, inplace=True)

    ####################################
    # now fetch the 3 months treasury rate
    # 
    # df = get_FRED_data_with_date_offset(df, 'DGS3MO', start_date_timestamp, 'day', -1, False)
    # df.rename(columns={'DGS3MO': 'DGS3MO_S'}, inplace=True)
   
    print('>Get 3 mo treasury rate')
    df = get_FRED_Data( df, 'DGS3MO', start_date_timestamp)
    df.rename(columns={'DGS3MO': '3month'}, inplace=True)

    print('>Get 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity')
    df = get_FRED_Data( df, 'T10Y2Y', start_date_timestamp)

    ###############
    # GET VIX from FRED
    #
    print('>Get VIX from FRED')
    
    df = get_FRED_Data( df, 'VIXCLS', start_date_timestamp)
    # df = get_yfinance_data(df, '^VIX',start_date_timestamp, 'VIXCLS_Y' )

    ####################################
    # now fetch the unemployment
    # 
    # df = get_FRED_Data( df, 'UNRATE', start_date_timestamp)
    # note unemployment rate is always announced on a friday (which is 4 in day of week)
    #
    print('>Get Unemployment rate')
    
    df = get_FRED_data_with_date_offset(df, 'UNRATE', start_date_timestamp, 'month', 4, False)
    # df = get_FRED_Data(df, 'UNRATE', start_date_timestamp)
    df.rename(columns={'UNRATE': 'unemploy'}, inplace=True)

    ###
    # now get consumer sentiment from FRED
    #   Assume updated on last friday of the month
    #
    # df = get_FRED_Data(df, 'UMCSENT', start_date_timestamp)
    print('>Get UMCSENT (consumer sentiment)')
    
    df = get_FRED_data_with_date_offset(df, 'UMCSENT', start_date_timestamp, 'month', 4, True)

    ####################################
    # get busness and comercial loans from FRED (BUSLOANS)
    print('>Get BUSLOANS (consumer sentiment)')
    # df = get_FRED_data_with_date_offset(df, 'BUSLOANS', start_date_timestamp, 'month', 4, True)
    df = get_FRED_data_month_week_days_of_week(df, 'BUSLOANS', start_date_timestamp, 2, 1, 4)

    ####################################
    # get Advance Retail Sales: Retail Trade and Food Services from FRED (RSAFS)
    print('>Get RSAFS (Advance Retail Sales: Retail Trade and Food Services)')
    # df = get_FRED_data_with_date_offset(df, 'BUSLOANS', start_date_timestamp, 'month', 4, True)
    df = get_FRED_data_month_week_days_of_week(df, 'RSAFS', start_date_timestamp, 1, 3, 3)

    ####################################
    # get Advance Retail Sales: Retail Trade and Food Services from FRED (RSAFS)
    print('>Get M2SL (M2 Money Supply )')
    # df = get_FRED_data_with_date_offset(df, 'BUSLOANS', start_date_timestamp, 'month', 4, True)
    df = get_FRED_data_month_week_days_of_week(df, 'M2SL', start_date_timestamp, 1, 4, 1)

    ####################################
    # get 30-Year Fixed Rate Mortgage Average in the US  (MORTGAGE30US)
    print('>Get MORTGAGE30US (30-Year Fixed Rate Mortgage Average in the US)')
    # df = get_FRED_data_with_date_offset(df, 'BUSLOANS', start_date_timestamp, 'month', 4, True)
    df = get_FRED_Data(df, 'MORTGAGE30US', start_date_timestamp)


    #########################################################################
    # now get MACD info from Alpha Vantage
    #
    print('>Get MACD for symbol= ', symbol)
    ti = TechIndicators(key=config["alpha_vantage"]["key"])

    # use the BTC stock
    macd_data, MACD_meta_data = ti.get_macdext(symbol, interval='daily', series_type='close',
                    fastperiod=None, slowperiod=None, signalperiod=None, fastmatype=None,
                    slowmatype=None, signalmatype=None)
    # Extracting date and MACD_Signal
    extracted_macd = [{'date': date, 'MACD_Signal': values['MACD_Signal']} 
             for date, values in macd_data.items()]
    # Creating a DataFrame
    macd_signal_df = pd.DataFrame(extracted_macd)
    df = merge_feed_data_frame(df, macd_signal_df)

    # Extracting date and MACD
    extracted_macd = [{'date': date, 'MACD': values['MACD']} 
             for date, values in macd_data.items()]
    # Creating a DataFrame
    macd_df = pd.DataFrame(extracted_macd)
    df = merge_feed_data_frame(df, macd_df)

    # Extracting date and MACD Histogram
    extracted_macd = [{'date': date, 'MACD_Hist': values['MACD_Hist']} 
             for date, values in macd_data.items()]
    # Creating a DataFrame
    macd_hist_df = pd.DataFrame(extracted_macd)
    df = merge_feed_data_frame(df, macd_hist_df)
    
    #########################################################################
    # now get ATR info from Alpha Vantage
    #
    
    
    # ti = TechIndicators(key=config["alpha_vantage"]["key"])
    print('>Get ATR for symbol= ', symbol)
    
    atr_data, atr_meta_data = ti.get_atr(symbol, interval='daily', time_period=14)
    # Extracting date and atr
    # Extracting 'Technical Analysis: ATR' part
    # Convert to DataFrame
    atr_df = pd.DataFrame([(date, values['ATR']) for date, values in atr_data.items()], 
                    columns=['date', 'ATR'])

    # Convert 'Date' to datetime and sort
    # df['Date'] = pd.to_datetime(df['Date'])
    atr_df.sort_values(by='date', inplace=True)

    # Set 'Date' as the index
    atr_df.set_index('date', inplace=True)
    df = merge_feed_data_frame(df, atr_df)

    #########################################################################
    # now get RSI info from Alpha Vantage
    #
    # ti = TechIndicators(key=config["alpha_vantage"]["key"])

    print('>Get RSI for symbol= ', symbol)
    
    rsi_data, rsi_meta_data = ti.get_rsi(symbol, interval='daily', time_period=20, series_type='close')
    # Extracting date and atr
    # Extracting 'Technical Analysis: ATR' part
    # Convert to DataFrame
    rsi_df = pd.DataFrame([(date, values['RSI']) for date, values in rsi_data.items()], 
                    columns=['date', 'RSI'])

    # Convert 'Date' to datetime and sort
    # df['Date'] = pd.to_datetime(df['Date'])
    rsi_df.sort_values(by='date', inplace=True)

    # Set 'Date' as the index
    rsi_df.set_index('date', inplace=True)
    df = merge_feed_data_frame(df, rsi_df)

    # now calculate derived RSI signals
    # print('>Get derived RSI signals for symbol= ', symbol)
    # df_with_signals = add_rsi_signals(df, 
    #                              rsi_column='RSI',
    #                              price_column='adjusted close',
    #                              overbought=70,
    #                              oversold=30,
    #                              divergence_lookback=10,
    #                              sensitivity=3)
    # transformed_features = prepare_rsi_features(df_with_signals)

    # rsi_columns = ['date', 'RSI_overbought', 'RSI_oversold', 'RSI_bullish_divergence', 'RSI_bearish_divergence', 'RSI_momentum_strength'] # start with the core RSI signals, there are more
    # df_cleaned = transformed_features[rsi_columns]
    # df = merge_feed_data_frame(df, df_cleaned)

    ########################################################################
    # now get Stochastic Oscillator (for SnP Oscilator by using SPY) from Alpha Vantage
    #
    # ti = TechIndicators(key=config["alpha_vantage"]["key"])
    print('>Get SnP Oscilator by using SPY')
    
    stoch_data, stoch_meta_data = ti.get_stoch('SPY', interval='daily', fastkperiod=0,
                  slowkperiod=0, slowdperiod=None, slowkmatype=0, slowdmatype=0)
    
    extracted_stoch = [{'date': date, 'SPY_stoch': values['SlowK']} 
             for date, values in stoch_data.items()]
    # Creating a DataFrame
    stoch_df = pd.DataFrame(extracted_stoch)
    df = merge_feed_data_frame(df, stoch_df)


    #########################################################################
    # now get Stochastic Oscillator (for NASDAQ Oscilator by using QQQ) from Alpha Vantage
    #

    print('>Get SnP Oscilator by using QQQ')
   
    stoch_data, stoch_meta_data = ti.get_stoch('QQQ', interval='daily', fastkperiod=None,
                  slowkperiod=None, slowdperiod=None, slowkmatype=0, slowdmatype=0)
    extracted_stoch = [{'date': date, 'QQQ_stoch': values['SlowK']} 
             for date, values in stoch_data.items()]
    # Creating a DataFrame
    stoch_df = pd.DataFrame(extracted_stoch)
    df = merge_feed_data_frame(df, stoch_df)

    #########################################################################
    # now get Stochastic Oscillator (for Russell Oscilator by using VTWO) from Alpha Vantage
    #
    print('>Get SnP Oscilator by using VTWO (Russell)')
  
    stoch_data, stoch_meta_data = ti.get_stoch('VTWO', interval='daily', fastkperiod=None,
                  slowkperiod=None, slowdperiod=None, slowkmatype=0, slowdmatype=0)
    extracted_stoch = [{'date': date, 'VTWO_stoch': values['SlowK']} 
             for date, values in stoch_data.items()]
    # Creating a DataFrame
    stoch_df = pd.DataFrame(extracted_stoch)
    df = merge_feed_data_frame(df, stoch_df)


    #########################################################################
    # now get dollar index
    # !! This is real tie too so should not date shift
    #
    print('>Get dollar index')
  
    df = get_FRED_Data(df, 'DTWEXBGS', start_date_timestamp)
    
    #########################################################################
    # Get WTI oil price
    # !! This is real tie too so should not date shift
    #
    print('>Get WTI')
    df = get_FRED_Data(df, 'DCOILWTICO', start_date_timestamp)
    
    #########################################################################
    # now get 
    # Business Tendency Surveys (Manufacturing): Confidence Indicators: 
    # Composite Indicators: OECD Indicator for United States
    #   BSCICP03USM665S
    # !!! No update since 4/10/24 (Still no update as of 9/9/24, I think this is DEAD)
    # !!!! This shouldbe replaced by BSCICP02USM460S TBD
    #
    print('>Get BSCICP03USM665S (Business Tendency Surveys (Manufacturing): Confidence Indicators)')
  
    df = get_FRED_data_with_date_offset(df, 'BSCICP03USM665S', start_date_timestamp, 'quarter', -1, False)
    # df = get_FRED_Data(df, 'BSCICP03USM665S', start_date_timestamp)

    #########################################################################
    # now get Longer Run FOMC Summary of Economic Projections for the Fed Funds Rate, Median
    #
    # no need to shift data as announcement date is future facing and not previous period
    #
    print('>Get Run FOMC Summary of Economic Projections for the Fed Funds Rate, Median')
 
    df = get_FRED_Data(df, 'FEDTARMDLR', start_date_timestamp)

    #########################################################################
    # now get  Manufacturers' New Orders: Durable Goods Longer
    # July data is released on 9/4 (likely last day then shifted due to labor day
    # will assume a 2 mo shift)
    #
    # df = get_FRED_Data(df, 'DGORDER', start_date_timestamp)
    print('>Get DGORDER (Manufacturers New Orders: Durable Goods)')
 
    df = get_FRED_data_month_week_days_of_week( df, 'DGORDER', start_date_timestamp, 2, 0, 2 )
    
    #########################################################################
    # now get  Economic Policy Uncertainty Index for United States 
    #
    print('>Get USEPUINDXD (Economic Policy Uncertainty Index for US)')
 
    df = get_FRED_Data(df, 'USEPUINDXD', start_date_timestamp)
    # df = get_FRED_data_with_date_offset(df, 'USEPUINDXD', start_date_timestamp, 'day', -1, False)

    #########################################################################
    # now get Leading Indicators: Composite Leading Indicator: Normalised for United States 
    #  USALOLITONOSTSAM
    #   !!! Leave it alone, no update since April 24 as of sept 24
    #
    print('>Get USALOLITONOSTSAM (Composite Leading Indicator: Normalised for United States)')
    df = get_FRED_Data(df, 'USALOLITONOSTSAM', start_date_timestamp)

    #########################################################################
    # Business Tendency Surveys (Manufacturing): Confidence Indicators: Composite Indicators: National Indicator for United States (BSCICP02USM460S)
    #
    print('>Get BSCICP02USM460S (Business Tendency Surveys (Manufacturing): Confidence Indicators)')
    df = get_FRED_data_with_date_offset(df, 'BSCICP02USM460S', start_date_timestamp, 'month', -1, False)
    # df = get_FRED_Data(df, 'BSCICP02USM460S', start_date_timestamp)
    
    #########################################################################
    # now get Federal Funds Target Range - Upper Limit  
    # DFEDTARU
    #   ! Leave alone, same day update
    #
    print('>Get DFEDTARU (Federal Funds Target Range - Upper Limit)')
    df = get_FRED_Data(df, 'DFEDTARU', start_date_timestamp)


     #########################################################################
    # now get Sticky Price Consumer Price Index less Food and Energy 
    # CORESTICKM159SFRBATL
    # last updated 9/11 (assume 2nd tuesday of the following month)
    #
    print('>Get CORESTICKM159SFRBATL (Sticky Price Consumer Price Index less Food and Energy)')
    df = get_FRED_data_month_week_days_of_week(df, 'CORESTICKM159SFRBATL', start_date_timestamp, 1, 1, 2 )
    # df = get_FRED_Data(df, 'CORESTICKM159SFRBATL', start_date_timestamp)

    print('>Get CPIAUCSL (Consumer Price Index for All Urban Consumers: All Items in U.S. City Average)')
    df = get_FRED_data_month_week_days_of_week(df, 'CPIAUCSL', start_date_timestamp, 1, 1, 2 )

     #########################################################################
    # now get Personal Consumption Expenditures 
    # PCE
    #
    # df = get_FRED_data_with_date_offset(df, 'PCE', start_date_timestamp, 'month', -1, False)
    print('>Get PCE (Personal Consumption Expenditures)')
    df = get_FRED_Data(df, 'PCE', start_date_timestamp)

    # print('>>> after get_fred for PCE, date type =' + str(df['date'].dtype))

    #########################################################################
    # now get Bollinger Bands info from Alpha Vantage
    #
    print('>Get Bollinger Bands for symbol= ', symbol)
    ti = TechIndicators(key=config["alpha_vantage"]["key"])
    if params.get('bband_time_period') is not None:
        time_period = params['bband_time_period']
    else:
        time_period = 20    # default to 20
    data, bband_meta_data = ti.get_bbands(symbol, interval='daily', time_period= time_period, series_type= 'close', nbdevup=None, nbdevdn=None, matype=None)
    
    # Convert the nested dictionary to DataFrame
    bband_df = pd.DataFrame.from_dict(data, orient='index')  # Keys become index, columns are the inner keys
    # Optional: Convert index to datetime to handle it as time series data
    bband_df.index = pd.to_datetime(bband_df.index)

    # Convert datetime index to string
    bband_df.index = bband_df.index.strftime('%Y-%m-%d')

    # Convert to DataFrame with dates as index
    bband_df.sort_index(inplace=True)  # Sort by date
    
    # Resetting the index to make the date a column
    bband_df.reset_index(inplace=True)
    bband_df.rename(columns={'index': 'date'}, inplace=True)  # Rename the column to 'date'

    df = merge_feed_data_frame(df, bband_df)

    # only get the EPS related stuff if they are present in the param 
    #########
    # get quarterly income
    # Note Income statement dates do NOT line up with market open days
    # !!! IMPORTANT this has to be gotten BEFORE the EPS dates
    # One needs to 
    # 1. line up quarterEndDate to the closest report Date
    # 2. Combione these data,
    # 3. then look for the next trade date and line them up.
    #
    if params.get('symbol') != 'QQQ':
        fd = FundamentalData(key=config["alpha_vantage"]["key"])
        income_data = fd.get_income_statement_quarterly(symbol)
        # Convert the list of lists into a DataFrame, using the first row as column names
        # Extract the array part (assuming it's the first element)
        data_array = income_data[0]

        # Convert to DataFrame
        income_df = pd.DataFrame(data_array)

        # If the first row represents column names
        income_df.reset_index(drop=True, inplace=True)

        # change fiscal date ending to reporting date
        # income_df.rename(columns={'reportedDate': 'date'}, inplace=True)
        # income_df.rename(columns={'fiscalDateEnding': 'date'}, inplace=True)
        income_df = income_df.filter(items=['fiscalDateEnding', 'netIncome', 'totalRevenue'])
        
        # Sort the DataFrame by date in ascending order)
        income_df.sort_values(by='fiscalDateEnding', ascending=True, inplace=True)
        # income_df.to_csv('income_df.csv', index=False)

        # We now have a sorted income_df that has the above three columns


        #########################################################################
        # Get EPS data
        #
        print('>Get EPS data for symbol= ', symbol)
        attempt = 0
        max_attempts = 12
        while attempt < max_attempts:
            params = {
                    "apikey": api_key,
                    "function": 'EARNINGS',
                    "datatype": 'json',
                    "symbol": symbol
                }
            try:
                response = requests.get(base_url, params=params, timeout=100)
                if response.status_code == 200:
                    data = response.json()
                    # data = json.loads(json_data)

                    # Check if quarterlyEarnings key exists
                    if 'quarterlyEarnings' in data and data['quarterlyEarnings']:
                    # Extract annual earnings data
                        earnings_data = data['quarterlyEarnings']

                        # Create DataFrame
                        eps_df = pd.DataFrame(earnings_data)

                        # Rename columns if necessary
                        eps_df.rename(columns={'reportedEPS': 'EPS', 'estimatedEPS': 'estEPS'}, inplace=True)
                        eps_df = eps_df.filter(items=['reportedDate', 'EPS', 'estEPS', 'surprisePercentage'])
                        
                        # Sort the DataFrame by date in ascending order)
                        eps_df.sort_values(by='reportedDate', ascending=True, inplace=True)
                        # eps_df.to_csv('eps_df.csv', index=False)
                        break   # break out of the loop
                    else:
                        print(f"Attempt {attempt + 1}: 'quarterlyEarnings' data not yet available. Waiting for 300 seconds...")

                else:
                    print(f"Attempt {attempt + 1}: Error {response.status_code}: {response.text}")

            except Exception as e:
                print(f"Attempt {attempt + 1}: Exception occurred: {str(e)}")
    
            # Wait before retrying
            attempt += 1
            if attempt < max_attempts:
                time.sleep(300)
        # print(eps_df)

        # Ensure the date fields are in datetime format and normalize them
        eps_df['reportedDate'] = pd.to_datetime(eps_df['reportedDate'])
        income_df['fiscalDateEnding'] = pd.to_datetime(income_df['fiscalDateEnding'])
        eps_df['reportedDate'] = eps_df['reportedDate'].dt.normalize()
        income_df['fiscalDateEnding'] = income_df['fiscalDateEnding'].dt.normalize()

        # First we have to line up income_df quarterEndDate to the next reportedDate
        #   - Define a new reportedDat in the income df, line that up to the next reportedDate in the eps_df
        #
        income_df['reportedDate'] = income_df['fiscalDateEnding'].apply(
            lambda x: eps_df[eps_df['reportedDate'] > x]['reportedDate'].min()
        )
        income_df['reportedDate'] = pd.to_datetime(income_df['reportedDate'])

        # Now the income has a new matching reportedDate that matches when that financial statement matches the 
        # EPS data
        # we now merge these to one df
        #

        combined_finance_df = pd.merge(eps_df, income_df, on='reportedDate', how='left')

        ### now we are doing the reportDate shift work, shift each entry one entry down to the next trading day
        # Create a new DataFrame with the 'date' column
        #
        masterDate_df = df[['date']]
        masterDate_df.loc[:, 'date'] = pd.to_datetime(masterDate_df['date'])

        # Shift the 'reportDate' to the next available 'date' in master_df
        combined_finance_df['shifted_reportedDate'] = combined_finance_df['reportedDate'].apply(
            lambda x: masterDate_df[masterDate_df['date'] > x]['date'].min()
        )
        # At this point we have a combined _finance_df that has all the income/eps data lined up to the NEXT tradding day
        #
        combined_finance_df.rename(columns={'shifted_reportedDate': 'date'}, inplace=True)

        # conver that date back to object which is required for a merge
        combined_finance_df['date'] = combined_finance_df['date'].dt.strftime('%Y-%m-%d')

        # Only keep the following columns
        combined_finance_df = combined_finance_df.filter(items=['date', 'EPS', 'estEPS', 'surprisePercentage', 'netIncome', 'totalRevenue'])

        ### Adjust estEPS
        # RATIONALE: At any given moment, the estEPS is known to investors, and it is only reported in the NEXT period of qtr report
        # So we want to shift it down (-1) one
        # then use current_estEPS from param file to fill in the last entry
        #
        combined_finance_df['estEPS'] = combined_finance_df['estEPS'].shift(-1)
        # combined_finance_df['estEPS'].iloc[-1] = param['current_estEPS']
        combined_finance_df.loc[combined_finance_df.index[-1], 'estEPS'] = param['current_estEPS']

        # now perform the same exercise for total revenue ONLY if that field exist
        if 'estimated_revenue' in param:
            combined_finance_df['totalRevenue'] = combined_finance_df['totalRevenue'].shift(-1)
            combined_finance_df.loc[combined_finance_df.index[-1], 'totalRevenue'] = param['estimated_revenue']

        # Merge with existing dataframe
        df = merge_feed_data_frame(df, combined_finance_df)

        # ── DTE / DSE: trading-day distance to next/last earnings ─────────────
        # Effective sessions: the shifted_reportedDate values already computed
        # above (next trading day after each report, since AV gives date-only so
        # we conservatively assume AMC).  These are the 'date' values in
        # combined_finance_df rows that have EPS data.
        eff_sessions = (
            pd.to_datetime(combined_finance_df.loc[combined_finance_df['EPS'].notna(), 'date'])
            .sort_values()
            .tolist()
        )

        # Determine the next report date.
        # Priority: param override → live fetch (AV then yfinance) → +3 months estimate
        next_report_date_param = param.get('next_report_date')
        if next_report_date_param:
            nrd = pd.to_datetime(next_report_date_param)
            print(f">Next earnings for {symbol}: {nrd.date()} (from param — manual override)")
        else:
            nrd = fetch_next_report_date(symbol, config["alpha_vantage"]["key"])
            if nrd is None:
                if eff_sessions:
                    last_known = pd.to_datetime(eff_sessions[-1])
                    nrd = last_known + pd.DateOffset(months=3)
                    print(f">Next earnings for {symbol}: {nrd.date()} "
                          f"(estimated — last known + 3 months; live fetch unavailable).")
                else:
                    print(f">Warning: No next earnings date found for {symbol}. DTE will be 999.")

        if nrd is not None:
            df_dates = pd.to_datetime(df['date'])
            # Next trading day after the report date (AMC assumption)
            candidates = df_dates[df_dates > nrd]
            if not candidates.empty:
                eff_sessions.append(candidates.min())
            else:
                # Report is beyond our price history; add raw date for approximation
                eff_sessions.append(nrd)
            eff_sessions = sorted(set(eff_sessions))

        df = compute_dte_dse_features(df, eff_sessions, symbol)
        print(f">DTE/DSE added for {symbol}. Latest dte={df['dte'].iloc[-1]:.0f}, dse={df['dse'].iloc[-1]:.0f}")
        # ── end DTE/DSE ────────────────────────────────────────────────────────
    
    #########################################################################
    # now get Monetary Base; Reserve Balances
    # BOGMBBM
    # this seems to be release on the last tuesday of each month
    #
    print('>Get BOGMBBM data (Monetary Base; Reserve Balances)')
    df = get_FRED_Data( df, 'BOGMBBM', start_date_timestamp)
    df.rename(columns={'BOGMBBM': 'BOGMBBM_unshifted'}, inplace=True)
    df = get_FRED_data_with_date_offset( df, 'BOGMBBM', start_date_timestamp, 'month', 1, True)

    #########################################################################
    # get foreign exchange data
    # 
    print('>Get foreign exchange data')
    
    try:
        eur_exchange_rate = fetch_with_retry('EUR', 'USD', config,  max_retries=3, sleep_time=300)
        # Process your data here

        # Only process the data if fetch was successful
        if eur_exchange_rate is not None:
            # Extract the 'Time Series FX (Daily)' part of the data
            eur_time_series = eur_exchange_rate[0]
        else:
            print("Warning: EUR exchange rate data could not be fetched")
    except Exception as e:
        print(f"Error: {e}")
    # eur_exchange_rate = fx_data.get_currency_exchange_daily('EUR', 'USD', outputsize='full')

    # # Extract the 'Time Series FX (Daily)' part of the data
    # eur_time_series = eur_exchange_rate[0]

    # Create a DataFrame from the extracted data
    eur_df = pd.DataFrame({
        'date': eur_time_series.keys(),
        'eur_close': [float(values["4. close"]) for values in eur_time_series.values()]
    })

    eur_df.sort_values(by='date', ascending=True, inplace=True)

    df = merge_feed_data_frame(df, eur_df)

    # now get JPY
    try:
        jpy_exchange_rate = fetch_with_retry('JPY', 'USD', config, max_retries=3, sleep_time=300)
        # Process your data here
    except Exception as e:
        print(f"Error: {e}")
    # jpy_exchange_rate = fx_data.get_currency_exchange_daily('JPY', 'USD', outputsize='full')

    # Extract the 'Time Series FX (Daily)' part of the data
    jpy_time_series = jpy_exchange_rate[0]

    # Create a DataFrame from the extracted data
    jpy_df = pd.DataFrame({
        'date': jpy_time_series.keys(),
        'jpy_close': [float(values["4. close"]) for values in jpy_time_series.values()]
    })

    jpy_df.sort_values(by='date', ascending=True, inplace=True)

    df = merge_feed_data_frame(df, jpy_df)

    # now get TWD
    try:
        twd_exchange_rate = fetch_with_retry('TWD', 'USD', config, max_retries=3, sleep_time=300)
        # Process your data here
    except Exception as e:
        print(f"Error: {e}")
    
    # twd_exchange_rate = fx_data.get_currency_exchange_daily('TWD', 'USD', outputsize='full')

    # Extract the 'Time Series FX (Daily)' part of the data
    twd_time_series = twd_exchange_rate[0]

    # Create a DataFrame from the extracted data
    twd_df = pd.DataFrame({
        'date': twd_time_series.keys(),
        'twd_close': [float(values["4. close"]) for values in twd_time_series.values()]
    })

    twd_df.sort_values(by='date', ascending=True, inplace=True)

    df = merge_feed_data_frame(df, twd_df)

    
    ####################################################
    # fetch the sahm data
    #
    print('>Get SAHMREALTIME (SAHM Real Time) data')

    df = get_FRED_data_with_date_offset(df, 'SAHMREALTIME', start_date_timestamp, 'month', 4, False)
    
    ####################################################
    # fetch the non farm job opening (JTSJOL)
    # Assume release on 1st thur of everymonth
    #
    print('>Get JTSJOL (non farm job opening) data')
    df = get_FRED_data_with_date_offset(df, 'JTSJOL', start_date_timestamp, 'month', 3, False)
    # df = get_FRED_Data(df, 'JTSJOL', start_date_timestamp)

    #########################################################################
    # now get Producer Price Index by Industry: Semiconductor and Other Electronic Component Manufacturing 
    # PCU33443344
    #   july data is out 8/13 (Tues). Will shift one month, one week, and then tuesday
    #
    print('>Get PCU33443344 (Producer Price Index by Industry: Semiconductor and Other Electronic Component Manufacturing)')
    df = get_FRED_data_month_week_days_of_week(df, 'PCU33443344', start_date_timestamp, 1, 1, 1 )

    ####################################################
    # fetch the GDP data
    # This seems to be released 2mo (end of month in from the date after a quarter shift)
    # 
    # df = get_FRED_Data(df, 'GDP', start_date_timestamp)
    # df.rename(columns={'GDP': 'GDP_unshifted'}, inplace=True)
    print('>Get GDP data')
    df = get_FRED_data_month_week_days_of_week( df, 'GDPC1', start_date_timestamp, 4, 4, 3 )
    df.rename(columns={'GDPC1': 'GDP'}, inplace=True)

    ####################
    # Get data from Yahopo finance
    # !!! These are all one day delay
    #
    # df = get_yfinance_data( df, 'PMI', start_date_timestamp, 'PMI')

    # df = get_yfinance_data( df, '^GSPC', start_date_timestamp, 'SP500')
    # df = get_yfinance_data( df, '^IXIC', start_date_timestamp, 'NASDAQ')
    # df = get_yfinance_data( df, '^RUT', start_date_timestamp, 'RUSSELL')
    # print('>Get SOX data')
    # df = get_yfinance_data( df, '^SOX', start_date_timestamp, 'SOX')
    
    # First, clean the existing file
    clean_cp_ratio_file(symbol)

    ####################################
    # get call/put ratio data
    # ASSUMPTION: for each symbol, there is already a preprocessed <symbol>_cp_ratios.csv file
    # this daily code will read that in, figure out what is missing from end date, and just fill those in
    #
    if 'cp_sentiment_ratio' in param['selected_columns'] and 'options_volume_ratio' not in param['selected_columns']:
        # In this case use the old code without options ratio
        # only process cp ratio if this stock is configured to use it
        print('>Get Call/Put ratio data (OLD WAY)')
        file_path = symbol + '-cp_ratios_sentiment.csv'
        if not os.path.isfile(file_path):
            print(f'Warning: {file_path} not found. cp_sentiment_ratio will be 0.')
            df['cp_sentiment_ratio'] = 0.0
        if os.path.isfile(file_path):
            # file exist
            df_cp_ratios = pd.read_csv(file_path)

            start_date = param['start_date']
            df = df[df['date'] >= start_date]   # whack all older dates before the desired start date
            df = df[df['date'] <= param['end_date']]

            # now we have to figure out the dates in master df
            missing_dates = find_missing_dates(df, df_cp_ratios)  # Returns None (no missing dates in range)
            
            if not missing_dates.empty:
                print('>> calculate new missing cp-ratios')
                # result_df = get_historical_cp_ratios_with_sentiments(symbol, missing_dates, api_key)
                result_df = get_historical_cp_ratios_with_sentiments_new(symbol, missing_dates, api_key)
                result_df_copy = result_df.copy()

                columns = ['date', 'cp_volume_ratio','cp_oi_ratio', 'bullish_volume', 'bearish_volume']
                result_df = result_df[columns]

                # append the result to the df_cp_ratio and write id back to disk
                new_cp_df = pd.concat([df_cp_ratios, result_df], ignore_index=True)
                new_cp_df.to_csv(file_path, index=False)

                # fetch it again to rid of formating error that causes NaN
                df_cp_ratios = pd.read_csv(file_path)

            # now merge in the ceombined df 
            df = merge_feed_data_frame(df, df_cp_ratios)

            # Calculate cp_sentiment_ratio with special handling for edge cases
            print('>> calculate cp_sentiment_ratio')

            def calculate_sentiment_ratio(row):
                bullish = row['bullish_volume']
                bearish = row['bearish_volume']
                
                # Case 1: Both are zero - sentiment is neutral
                if bullish == 0 and bearish == 0:
                    return 0.0
                
                # Case 2: Bullish is zero but bearish is not - extremely bearish
                elif bullish == 0 and bearish > 0:
                    return 0.0
                
                # Case 3: Bullish is positive but bearish is zero - extremely bullish
                elif bullish > 0 and bearish == 0:
                    return 3.0
                
                # Standard case: Normal ratio calculation
                else:
                    return bullish / bearish

            # Apply the function to calculate the sentiment ratio
            df['cp_sentiment_ratio'] = df.apply(calculate_sentiment_ratio, axis=1)

            # Optional: Round the ratio to 2 decimal places for readability
            df['cp_sentiment_ratio'] = df['cp_sentiment_ratio'].round(2)
        else:
            print('>>>'+ file_path + ' file does not exist' )
    else: 
        if 'cp_sentiment_ratio' in param['selected_columns'] and 'options_volume_ratio' in param['selected_columns']:
            #in this case use the new code with options ratio
            print('>Get Call/Put ratio data (new WAY with options volume ratio)')
            file_path = symbol + '-cp_ratios_sentiment_w_volume.csv'
            if not os.path.isfile(file_path):
                print(f'')
                print(f'⚠️ ⚠️ ⚠️  ALERT: {file_path} not found!')
                print(f'⚠️  cp_sentiment_ratio and options_volume_ratio will be set to 0.0')
                print(f'⚠️  This will result in missing feature data for the model!')
                print(f'')
                df['cp_sentiment_ratio'] = 0.0
                df['options_volume_ratio'] = 0.0
            if os.path.isfile(file_path):
                # file exist
                df_cp_ratios = pd.read_csv(file_path)

                start_date = param['start_date']
                df = df[df['date'] >= start_date]   # whack all older dates before the desired start date
                df = df[df['date'] <= param['end_date']]

                # now we have to figure out the dates in master df
                missing_dates = find_missing_dates(df, df_cp_ratios)  # Returns None (no missing dates in range)
                
                if not missing_dates.empty:
                    print('>> calculate new missing cp-ratios')
                    # result_df = get_historical_cp_ratios_with_sentiments(symbol, missing_dates, api_key)
                    result_df = get_historical_cp_ratios_with_sentiments_new(symbol, missing_dates, api_key)
                    result_df_copy = result_df.copy()

                    columns = ['date', 'call_volume', 'put_volume', 'cp_volume_ratio','cp_oi_ratio', 'bullish_volume', 'bearish_volume']
                    result_df = result_df[columns]

                    # append the result to the df_cp_ratio and write id back to disk
                    new_cp_df = pd.concat([df_cp_ratios, result_df], ignore_index=True)
                    new_cp_df.to_csv(file_path, index=False)

                    # fetch it again to rid of formating error that causes NaN
                    df_cp_ratios = pd.read_csv(file_path)

                # now merge in the ceombined df 
                df = merge_feed_data_frame(df, df_cp_ratios)

                # Calculate cp_sentiment_ratio with special handling for edge cases
                print('>> calculate cp_sentiment_ratio')

                def calculate_sentiment_ratio(row):
                    bullish = row['bullish_volume']
                    bearish = row['bearish_volume']
                    
                    # Case 1: Both are zero - sentiment is neutral
                    if bullish == 0 and bearish == 0:
                        return 0.0
                    
                    # Case 2: Bullish is zero but bearish is not - extremely bearish
                    elif bullish == 0 and bearish > 0:
                        return 0.0
                    
                    # Case 3: Bullish is positive but bearish is zero - extremely bullish
                    elif bullish > 0 and bearish == 0:
                        return 3.0
                    
                    # Standard case: Normal ratio calculation
                    else:
                        return bullish / bearish

                # Apply the function to calculate the sentiment ratio
                df['cp_sentiment_ratio'] = df.apply(calculate_sentiment_ratio, axis=1)

                # Optional: Round the ratio to 2 decimal places for readability
                df['cp_sentiment_ratio'] = df['cp_sentiment_ratio'].round(2)

                # calculate volume ratio
                df ['options_volume'] = df['call_volume'] + df['put_volume']
                df['options_volume_ratio'] = df['options_volume'] / df['volume']

                # VALIDATION: Alert if options_volume_ratio is all zeros
                non_zero_count = (df['options_volume_ratio'] > 0).sum()
                total_count = len(df)
                if non_zero_count == 0:
                    print(f"⚠️  ALERT: options_volume_ratio is ALL ZEROS for {symbol}! This indicates missing CP ratio data.")
                elif non_zero_count < total_count * 0.5:
                    print(f"⚠️  WARNING: options_volume_ratio has {non_zero_count}/{total_count} non-zero values for {symbol} (less than 50%)")
                else:
                    print(f"✓ options_volume_ratio validation OK: {non_zero_count}/{total_count} non-zero values for {symbol}")
    # df.to_csv('debug_post_netIncome_merge.csv')

    # Create an unfragmented copy of the DataFrame
    df = df.copy()
    return df