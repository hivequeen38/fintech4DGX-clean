
import pandas as pd
from pandas import DataFrame
import numpy as np
import os
import sys  
import logging      
import time
from datetime import datetime
import pytz
import trendConfig
import trendAnalysisBlackBox
import trendAnalysisFromTodayNew
import shutil
import multiZoneAnalysisNew

print("All libraries loaded for "+ __file__)

# fetch closing price & date from one of the intermediate result files
def fetchDateAndClosing_OLD(param: dict[str]):
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

    last_date = param['end_date']

    mask = df['date'] == last_date
    last_cp_vol = 0

    if mask.any():
        last_close = df.loc[mask, 'adjusted close'].values[0]
        if 'cp_sentiment_ratio' in param['selected_columns']:
            last_cp_vol = df.loc[mask, 'cp_sentiment_ratio'].values[0]
    else:
        print(f"Error: No data found for date {last_date}")
        sys.exit(1)
    return last_date, last_close, last_cp_vol

def fetchDateAndClosing(param: dict[str, any]):
    symbol = param['symbol']
    file_path = symbol + '_TMP.csv'
    last_date = param['end_date']
    
    if os.path.isfile(file_path):
        # file exist
        df = pd.read_csv(file_path)
    else:
        print('>>> ERROR, file '+ file_path + ' Not found!')
        return None, None, None, None  # Return four None values to match return signature
    
    # Instead of using the last row, filter by the specific date we want
    mask = df['date'] == last_date
    last_cp_vol = 0  # Default value if cp_sentiment_ratio isn't found

    if mask.any():
        last_close = df.loc[mask, 'adjusted close'].values[0]
        if 'cp_sentiment_ratio' in param['selected_columns'] and 'cp_sentiment_ratio' in df.columns:
            last_cp_vol = df.loc[mask, 'cp_sentiment_ratio'].values[0]

        if 'options_volume_ratio' in param['selected_columns'] and 'options_volume_ratio' in df.columns:
            last_options_vol = df.loc[mask, 'options_volume_ratio'].values[0]
            last_options_vol_pct = round(last_options_vol * 100, 3)
        else:
            last_options_vol = None
        return last_date, last_close, last_cp_vol, last_options_vol
    else:
        print(f"Error: No data found for date {last_date}, falling back to most recent available date")
        # Fall back to the most recent available date in the TMP file
        df_sorted = df.sort_values('date', ascending=False)
        last_row = df_sorted.iloc[0]
        fallback_date = last_row['date']
        last_close = last_row['adjusted close']
        last_cp_vol = 0
        if 'cp_sentiment_ratio' in param['selected_columns'] and 'cp_sentiment_ratio' in df.columns:
            last_cp_vol = last_row['cp_sentiment_ratio']
        if 'options_volume_ratio' in param['selected_columns'] and 'options_volume_ratio' in df.columns:
            last_options_vol = last_row['options_volume_ratio']
        else:
            last_options_vol = None
        print(f"Falling back to date: {fallback_date}")
        return fallback_date, last_close, last_cp_vol, last_options_vol

# process result files
def processDeltaFromTodayResults( symbol: str, incr_df: DataFrame, dateStr: str, closingPrice: float, comment: str, last_cp_vol: float, param: dict[str], last_vol_ratio=None):
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
        # columns.insert(16, 'comment')
        columns.append('comment')   # this will add to the end
        df = DataFrame(columns= columns)
        df = df.reset_index(drop=True)

    df.loc[len(df)] = np.nan  # This appends a row filled with NaNs at the end

    # Step 2: Set values for specific columns in the new row
    df.loc[len(df)-1, 'date'] = dateStr  
    df.loc[len(df)-1, 'close'] = closingPrice 
    input_col = ['p1', 'p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15']
    for target_item in input_col:
        df.loc[len(df)-1, target_item] = incr_df.iloc[0][target_item] 
    
    # check if cp_sentiment_ratio is used, if so append comment    
    features = param['selected_columns']
    if 'cp_sentiment_ratio' in features:
        last_cp_vol_str = f"{float(last_cp_vol):.2f}"
        comment = comment + ' C/PS= ' + last_cp_vol_str

    # check if options_volume_ratio is used, if so append comment    
    if 'options_volume_ratio' in features:
        last_vol_ratio_str = f"{float(last_vol_ratio):.2f}"
        comment = comment + ' O/PS= ' + last_vol_ratio_str + '%'

    df.loc[len(df)-1, 'comment'] = comment 

    # now store this back to disk
    df.to_csv(file_path, index=False, header=True) # we want to save the date index

    print(">>> Final result delta from today")
    print(df)


#########################
# PROCESS END DATE
def QA_processEndDate ( param, target_date: str, load_cache: bool = True, comment: str = ""):
    param['end_date'] = target_date
    eastern = pytz.timezone('US/Eastern')
    run_date = datetime.now(eastern).strftime('%Y-%m-%d %H:%M')
    main(param, end_date = target_date, run_date= run_date, input_comment=comment + target_date, load_cache = load_cache)

def QA_mz_processEndDate ( param, target_date: str, load_cache: bool = True, comment: str = ""):
    param['end_date'] = target_date
    eastern = pytz.timezone('US/Eastern')
    run_date = datetime.now(eastern).strftime('%Y-%m-%d %H:%M')
    mz_main(param, run_date= run_date, input_comment=comment + target_date, load_cache = load_cache)

def process_first_5_days(param, incr_df, turn_random_on: bool, comment: str, use_timesplit: bool=False):
    target_size_elements = [1,2,3,4,5]
    for target_item in target_size_elements:
        param["target_size"] = target_item
        param["comment"]=comment + str(target_item)
        # trendAnalysisBlackBox.analyze_trend(trendConfig.config, param, target_item, incr_df, turn_random_on, True)
        trendAnalysisFromTodayNew.analyze_trend(trendConfig.config, param, target_item, incr_df, turn_random_on, True, use_timesplit)
    
def process_last_10_days(param, incr_df, turn_random_on: bool, comment: str, use_timesplit: bool=False):
    target_size_elements = [6,7,8,9,10,11,12,13,14,15]
        # for first 5 days at 3%, then rest of the 15 days at 5%

    for target_item in target_size_elements:
        param["target_size"] = target_item
        param["comment"]=comment + str(target_item)
        # trendAnalysisBlackBox.analyze_trend(trendConfig.config, param, target_item, incr_df, turn_random_on, True)
        trendAnalysisFromTodayNew.analyze_trend(trendConfig.config, param, target_item, incr_df, turn_random_on, True, use_timesplit)
    
def inference_first_5_days(param, incr_df, turn_random_on: bool, comment: str, use_timesplit: bool=False):
    target_size_elements = [1,2,3,4,5]
    for target_item in target_size_elements:
        param["target_size"] = target_item
        param["comment"]=comment + str(target_item)
        # trendAnalysisBlackBox.analyze_trend(trendConfig.config, param, target_item, incr_df, turn_random_on, True)
        trendAnalysisFromTodayNew.make_inference(trendConfig.config, param, target_item, incr_df, turn_random_on, True, use_timesplit)
    
def inference_last_10_days(param, incr_df, turn_random_on: bool, comment: str, use_timesplit: bool=False):
    target_size_elements = [6,7,8,9,10,11,12,13,14,15]
        # for first 5 days at 3%, then rest of the 15 days at 5%

    for target_item in target_size_elements:
        param["target_size"] = target_item
        param["comment"]=comment + str(target_item)
        # trendAnalysisBlackBox.analyze_trend(trendConfig.config, param, target_item, incr_df, turn_random_on, True)
        trendAnalysisFromTodayNew.make_inference(trendConfig.config, param, target_item, incr_df, turn_random_on, True, use_timesplit)

#####################################
# Individual Train calls
#
def individual_train(param: dict[str], incr_df: pd.DataFrame, turn_random_on: bool,  comment: str, use_timesplit: bool):

    # for first 5 days at 3%, then rest of the 15 days at 5%
    param["threshold"] = 0.03
    process_first_5_days(param, incr_df, turn_random_on, comment, use_timesplit)

    param["threshold"] = 0.05
    process_last_10_days(param, incr_df, turn_random_on, comment, use_timesplit)


    # at this point there are 15x individual result files that's been updated. 
    # grab the current date and the closing price from the first one 
    # NVDA_1d_predictions_test.csv
    #
    dateStr, closing_price, last_cp_vol = fetchDateAndClosing(param)
    processDeltaFromTodayResults(param["symbol"], incr_df, dateStr, closing_price, comment, last_cp_vol, param)
    return

#####################################   
# Main Multi-zone call for training
#
def mz_main(param: dict[str], run_date=None, input_comment=None, load_cache=True):
    logging.basicConfig(level=logging.INFO)

    if (run_date is None):
        eastern = pytz.timezone('US/Eastern')
    run_date = datetime.now(eastern).strftime('%Y-%m-%d %H:%M')

    if (input_comment is None):
        input_comment = ''

    # load today's data
    # trendAnalysisBlackBox.load_data_to_cache(trendConfig.config, param)
    if load_cache == True:
        trendAnalysisFromTodayNew.load_data_to_cache(trendConfig.config, param)
    
    comment = 'Train (Fixed)' + ' RD=' + str(run_date) + ' ' + input_comment
    # for MZ, there is opnly one call 
    multiZoneAnalysisNew.analyze_trend(trendConfig.config, param, comment, turn_random_on= False, use_cached_data=True)

    comment = 'Train (Random)' + ' RD=' + str(run_date) + ' ' + input_comment
    # for MZ, there is opnly one call 
    multiZoneAnalysisNew.analyze_trend(trendConfig.config, param, comment, turn_random_on= True, use_cached_data=True)

#########################################
# MAIN LOOP enters here for training
def main(param: dict[str], end_date: str=None, run_date=None, input_comment=None, load_cache=True):
    logging.basicConfig(level=logging.INFO)

    if (run_date is None):
        eastern = pytz.timezone('US/Eastern')
    run_date = datetime.now(eastern).strftime('%Y-%m-%d %H:%M')

    if (end_date is None):
        end_date = datetime.now().strftime('%Y-%m-%d')
    param['end_date'] = end_date

    if (input_comment is None):
        input_comment = '(' + param["model_name"] + ')'

    # load today's data
    # trendAnalysisBlackBox.load_data_to_cache(trendConfig.config, param)
    if load_cache == True:
        trendAnalysisFromTodayNew.load_data_to_cache(trendConfig.config, param)
    
    #  DO FIXED SEED
    #
    input_col = ['date', 'close', 'p1', 'p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15', 'comment']
    incr_df = pd.DataFrame(columns=input_col)
    incr_df = incr_df.reset_index(drop=True)
    
    # for first 5 days at 3%, then rest of the 15 days at 5%
    comment = 'Training (Fixed)' + ' RD=' + str(run_date) + ' ' + input_comment

    use_time_split = param.get('use_time_split', False)

    param["threshold"] = 0.03
    process_first_5_days(param, incr_df, False, comment, use_time_split)

    param["threshold"] = 0.05
    process_last_10_days(param, incr_df, False, comment, use_time_split)


    # at this point there are 15x individual result files that's been updated. 
    # grab the current date and the closing price from the first one 
    # NVDA_1d_predictions_test.csv
    #

    # dateStr, closing_price, last_cp_vol, last_vol_ratio = fetchDateAndClosing(param)
    result = fetchDateAndClosing(param)
    print(f"fetchDateAndClosing returned: {result}")
    print(f"Number of values: {len(result)}")
    dateStr, closing_price, last_cp_vol, last_vol_ratio = result  # This will still error but you'll see the values
    processDeltaFromTodayResults(param["symbol"], incr_df, dateStr, closing_price, comment, last_cp_vol, param, last_vol_ratio)

    # RANDOM SEED
    #
    # input_col = ['date', 'close', 'p1', 'p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15']
    # incr_df = pd.DataFrame(columns=input_col)
    # incr_df = incr_df.reset_index(drop=True)
    
    # # for first 5 days at 3%, then rest of the 15 days at 5%

    # comment = 'Training (Random)'  + ' RD=' + str(run_date) + ' ' + input_comment

    # param["threshold"] = 0.03
    # process_first_5_days(param, incr_df, True, comment)

    # param["threshold"] = 0.05
    # process_last_10_days(param, incr_df, True, comment)


    # # at this point there are 15x individual result files that's been updated. 
    # # grab the current date and the closing price from the first one 
    # # NVDA_1d_predictions_test.csv
    # #
 
    # dateStr, closing_pric, last_cp_vol = fetchDateAndClosing(param)
    # processDeltaFromTodayResults(param["symbol"], incr_df, dateStr, closing_price, comment, last_cp_vol, param)

    # now rename the tmp file by a date tagged tmp file so it can be recreated
    oldFileName = param["symbol"] + '_TMP.csv'
    newFileName = param['symbol'] + '_' + dateStr + '_TMP.csv'

    subfolder = param["symbol"] + "_data"
    os.makedirs(subfolder, exist_ok=True)
    filepath = os.path.join(subfolder, newFileName)

    try:
        shutil.copyfile(oldFileName, filepath)
        print(f"File copied from {oldFileName} to {filepath}")
    except FileNotFoundError:
        print(f"The file {oldFileName} does not exist.")
    except PermissionError:
        print("You do not have permission to access or modify this file.")
    except IsADirectoryError:
        print(f"One of the paths refers to a directory, not a file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # os.rename(oldFileName, newFileName)

#########################################
# MAIN LOOP enters here for training
def inference(param: dict[str], run_date=None, input_comment=None, load_cache=True):
    logging.basicConfig(level=logging.INFO)

    if (run_date is None):
        eastern = pytz.timezone('US/Eastern')
    run_date = datetime.now(eastern).strftime('%Y-%m-%d %H:%M')

    if (input_comment is None):
        input_comment = ''

    # load today's data
    # trendAnalysisBlackBox.load_data_to_cache(trendConfig.config, param)
    if load_cache == True:
        trendAnalysisFromTodayNew.load_data_to_cache(trendConfig.config, param)
    
    #  DO FIXED SEED
    #
    input_col = ['date', 'close', 'p1', 'p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15', 'comment']
    incr_df = pd.DataFrame(columns=input_col)
    incr_df = incr_df.reset_index(drop=True)
    
    # for first 5 days at 3%, then rest of the 15 days at 5%
    comment = 'Refactored make_prediction_test(Fixed)' + ' Rundate= ' + str(run_date) + ' ' + input_comment
    param["threshold"] = 0.03
    inference_first_5_days(param, incr_df, False, comment)

    param["threshold"] = 0.05
    inference_last_10_days(param, incr_df, False, comment)


    # at this point there are 15x individual result files that's been updated. 
    # grab the current date and the closing price from the first one 
    # NVDA_1d_predictions_test.csv
    #

    dateStr, closing_price, last_cp_vol = fetchDateAndClosing(param)
    processDeltaFromTodayResults(param["symbol"], incr_df, dateStr, closing_price, comment, last_cp_vol, param)

    # # RANDOM SEED
    # #
    # input_col = ['date', 'close', 'p1', 'p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15']
    # incr_df = pd.DataFrame(columns=input_col)
    # incr_df = incr_df.reset_index(drop=True)
    
    # target_size_elements = [1,2,3,4,5]
    # # for first 5 days at 3%, then rest of the 15 days at 5%

    # comment = 'Refactored make_prediction_test (Random)'  + ' Rundate= ' + str(run_date) + ' ' + input_comment

    # param["threshold"] = 0.03
    # inference_first_5_days(param, incr_df, True, comment)

    # param["threshold"] = 0.05
    # inference_last_10_days(param, incr_df, True, comment)


    # # at this point there are 15x individual result files that's been updated. 
    # # grab the current date and the closing price from the first one 
    # # NVDA_1d_predictions_test.csv
    # #
 
    # dateStr, closing_price, last_cp_vol = fetchDateAndClosing(param)
    # processDeltaFromTodayResults(param["symbol"], incr_df, dateStr, closing_price, comment, last_cp_vol, param)

    # #  DO FIXED SEED with timesplit
    # #
    # input_col = ['date', 'close', 'p1', 'p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15', 'comment']
    # incr_df = pd.DataFrame(columns=input_col)
    # incr_df = incr_df.reset_index(drop=True)
    
    # # for first 5 days at 3%, then rest of the 15 days at 5%
    # comment = 'time split (Fixed)' + ' Rundate= ' + str(run_date) + ' ' + input_comment
    # param["threshold"] = 0.03
    # inference_first_5_days(param, incr_df, False, comment, True)

    # param["threshold"] = 0.05
    # inference_last_10_days(param, incr_df, False, comment, True)


    # # at this point there are 15x individual result files that's been updated. 
    # # grab the current date and the closing price from the first one 
    # # NVDA_1d_predictions_test.csv
    # #

    # dateStr, closing_price, last_cp_vol = fetchDateAndClosing(param)
    # processDeltaFromTodayResults(param["symbol"], incr_df, dateStr, closing_price, comment, last_cp_vol, param)

