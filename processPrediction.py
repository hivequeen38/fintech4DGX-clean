import os
from typing import List
import pandas as pd
from pandas import DataFrame

print("All libraries loaded for "+ __file__)
#########################################
# 1. First see if there is a file outstanding
# if so load it into df
#

def process_prediction_results(symbol: str, date_str: str, close: float, results: List[str], num_of_days: int):
    ''' symbol is start of the file name, results is an array of predictions from oldest to newest'''
    file_path = symbol+ "_" + str(num_of_days) + "d_predictions.csv"
    df: DataFrame

    if os.path.isfile(file_path):
        # file exist
        df = pd.read_csv(file_path)
        # df['date'] = pd.to_datetime(df['date'])  # Convert to datetime if it's not already
    else:
        # columns = ['date', 'close', 'p-1', 'p-2', 'p-3', 'p-4', 'p-5', 'p-6', 'p-7', 'p-8', 'p-9', 'p-10', 'p-11', 'p-12', 'p-13', 'p-14', 'p-15']
        columns = [f'p-{i}' for i in range(1, num_of_days + 1)]
        columns.insert(0, 'date')
        columns.insert(1,'close')
        df = DataFrame(columns= columns)

    # df['date'] = pd.to_datetime(df['date'])  # Convert to datetime if it's not already
    df.set_index('date', inplace=True)
    # Handle duplicate dates if necessary
    df = df[~df.index.duplicated(keep='last')]

    # the read dataframe should have a first col called 'prediciton date', 'today close', then followed by p-1, p-2... to p-5
    #
    results.insert(0,close)

    # Create a DataFrame for the new row
    new_row_df = DataFrame([results], columns=df.columns)
    new_row_df.index = [date_str]

    # see if there already is a row with this date, if so, will delete it
    # Convert the new date to datetime
    # new_date = pd.to_datetime(date_str)
    # formatted_date = new_date.strftime('%Y-%m-%d')

    # Update if exists, append if not
    # Assuming new_row_df is a DataFrame with the same columns as df and only one row
    new_row_series = new_row_df.iloc[0]
    df.loc[date_str] = new_row_series

    print('New prediciton results:')
    print(df.head)

    # now store this back to disk
    df.to_csv(file_path, index=True) # we want to save the date index
    
def process_prediction_results_test(symbol: str, date_str: str, close: float, results: List[str], num_of_days: int):
    ''' symbol is start of the file name, results is an array of predictions from oldest to newest'''
    file_path = symbol+ "_" + str(num_of_days) + "d_predictions_test.csv"
    df: DataFrame

    if os.path.isfile(file_path):
        # file exist
        df = pd.read_csv(file_path)
        # df['date'] = pd.to_datetime(df['date'])  # Convert to datetime if it's not already
    else:
        columns = []
        columns.insert(0, 'date')
        columns.insert(1,'close')
        columns.extend([f'p-{i}' for i in range(1, num_of_days + 1)])
        # columns.insert(16, 'comment')

        df = DataFrame(columns= columns)

    # df['date'] = pd.to_datetime(df['date'])  #!!! Convert to datetime if it's not already
    # df.set_index('date', inplace=True)
    # Handle duplicate dates if necessary
    # Modified to keep all dups
    # df = df[~df.index.duplicated(keep='last')]

    results.insert(0,close)
    results.insert(0,date_str)

    # Create a DataFrame for the new row
    new_row_df = DataFrame([results], columns=df.columns)
    # new_row_df.index = [date_str]

    # see if there already is a row with this date, if so, will delete it
    # Convert the new date to datetime
    # new_date = pd.to_datetime(date_str)
    # formatted_date = new_date.strftime('%Y-%m-%d')

    # just append new row at the bottom
    df = pd.concat([df, new_row_df])

    # new_row_series = new_row_df.iloc[0]
    # df.loc[date_str] = new_row_series

    print('New prediciton results:')
    print(df.head)

    # now store this back to disk
    df.to_csv(file_path, index=False) # we want to save the date index

    