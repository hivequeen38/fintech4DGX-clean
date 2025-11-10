##########################################################
# A Refactored version of the fetchBulkData

import requests
import pandas as pd
from pandas import DataFrame
import datetime
import io
import pandas_datareader as pdr
import os
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.cryptocurrencies import CryptoCurrencies

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
def fetch_feed(base_url: str, params: dict[str, str]):
    ''' funciton fetch a single specified feed '''
    # check if file exist, if it does load from file instead
    symbol = params['symbol']
    file_path = symbol+'.csv'
    df= pd.DataFrame()

    # if os.path.isfile(file_path): DO AWAY with reading from file
    if False:
        # file exist
        # print("file "+file_path+ " exist!")
        df = pd.read_csv(file_path, index_col=False)
    else:
        response = requests.get(base_url, params=params, timeout=100)
        if response.status_code == 200:
            data = response.json()
            if "Time Series (Daily)" in data:
                df = pd.DataFrame(data["Time Series (Daily)"]).T  # Transpose to get dates as rows
                # Get today's date in the format YYYY-MM-DD
                # today_date = datetime.today().strftime('%Y-%m-%d')
                # print(today_date)
                
                # need to adjust the stck headers to make sure there is a 'date' col in front
                # COMMENT THIS OOUT for reading from a .csv file 
                # if df.index.name == None:
                df.index.name = 'date'

                ###########################################################################
                # the inut is in descending order, let's first make sure they are ascending
                #
                if params['ascend']:
                    df.sort_values(by='date', ascending=True, inplace=True)
                    # print('>now the df should be in ascending order')
                    # print(df.head)
                # persist the gathered file into a CSV file with file name = <symbol>.csv
                # make sure the index header is set to date
                df.to_csv(symbol+'.csv', index=True) # we want to save the date index
            else:
                print("Error fetching data:", data.get("Note", "Unknown Error"))
                return None
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None

    # drop all the unused col
    df.drop(['7. dividend amount', '8. split coefficient'], axis='columns', inplace=True)

    headers_list = df.columns.tolist()
    # print('Num of col= ', len(df.columns))
    # print(headers_list)

    df.rename(columns={'1. open': 'open'}, inplace=True)
    df.rename(columns={'2. high': 'high'}, inplace=True)
    df.rename(columns={'3. low': 'low'}, inplace=True)
    df.rename(columns={'4. close': 'close'}, inplace=True)
    df.rename(columns={'5. adjusted close': 'adjusted close'}, inplace=True)
    df.rename(columns={'6. volume': 'volume'}, inplace=True)
    
    # headers_list = df.columns.tolist()
    # print('>Num of col after rename= ', len(df.columns))
    # print(headers_list)
    return df

################################################
# Function to fetch FRED data
def get_FRED_Data( df: DataFrame, feed_name: str, start_date_timestamp: str, param_name: str):
    # Do stuff
    data_source = 'fred'
    new_df = pdr.DataReader(feed_name, data_source, start_date_timestamp)
    
    # Convert the index into a column
    new_df = new_df.reset_index()             # this will get the date index into its own column
    new_df.rename(columns={'DATE': 'date'}, inplace=True)    # date col heading has the wrong case
    # new_df['date'] = new_df['date'].dt.strftime('%Y-%m-%d')
    df = merge_feed_data_frame(df, new_df)