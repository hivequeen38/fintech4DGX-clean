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

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    logging.debug("\nDataFrame after handling missing values:\n"+ str(df.head))
    # df.to_csv('TMP_etl.csv', index=False) # we want to save the date index
    return df
