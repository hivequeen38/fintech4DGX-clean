import numpy as np
import pandas as pd
from pandas import DataFrame
# import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging


# This file contains the code to message the data
# Input is a panda dataframe that is thegathered output from fetchBulkData routine
#
def fill_data(df: DataFrame)-> DataFrame:
    ''' this will return a dataframe that is ready for training'''
 
    ####################################################
    # check for missing data and fill them
    #
    # Check for missing values in each column
    missing_data = df.isnull().sum()
    print("Missing values in each column:\n", missing_data)

    # to ensure all data are float in there, need to catch and fix
    # Apply pd.to_numeric to all columns except 'date'
    df[df.columns.difference(['date'])] = df[df.columns.difference(['date'])].apply(pd.to_numeric, errors='coerce')

    # Replace inf/-inf with NaN so ffill/bfill can handle them
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Final safety: fill any remaining NaN (e.g. all-NaN columns with 0% data coverage)
    # with 0 so the scaler never receives NaN. Zero-variance columns are harmless to training.
    remaining_nan = df.isnull().sum().sum()
    if remaining_nan > 0:
        all_nan_cols = [c for c in df.columns if df[c].isna().all()]
        if all_nan_cols:
            print(f"WARNING: {len(all_nan_cols)} all-NaN columns filled with 0: {all_nan_cols}")
        df.fillna(0, inplace=True)

    logging.debug("\nDataFrame after handling missing values:\n"+ str(df.head))
    # df.to_csv('TMP_etl.csv', index=False) # we want to save the date index
    return df
