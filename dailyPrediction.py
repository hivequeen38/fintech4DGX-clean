import pandas as pd
from pandas import DataFrame
import json
import logging
import trendAnalysis
import trendConfig

print("All libraries loaded for "+ __file__)

########################################################################################################
# this module will
# 1. assume there is another module will loop and call into here when it's time to do daily training
# 2. Fetch from NVDA_trend.csv to get all the output from various runs
# 3. fetch the one with best avergae of Down F1
# 4. load up the config for that run and then do training wth latest data
# 5. check result, and calculate predicted vs actual
# 6. Send alert when there is a change from previous day
#

def dailyPredictionProcess( stock_list: [str], target_size: int, threshold_size: float):
    '''Input contains a list of stocks'''
    for symbol in stock_list:
        print('==> Now processing daily prediciton for symbol= '+ symbol)
        file_path = symbol+ '_trend.csv'
        df = pd.read_csv(file_path)

        filtered_df = df[(df['target_size']== target_size) & (df['threshold']==threshold_size)]
        # Find the index of the row with the maximum value in the specified column
    
        if not filtered_df.empty:  # Check if the filtered DataFrame is not empty
            # note the space at the end of the column name
            max_index = filtered_df['Test F1 for class 2: '].idxmax()
            print("Row with the highest 'Test F1 for class 2: ' under specified conditions:", max_index)
        else:
            print("No rows found matching the specified conditions")
            return

        max_index = filtered_df['Test F1 for class 2: '].idxmax()

        # Retrieve the row with the maximum value
        row_with_max_value = filtered_df.loc[max_index]

        print(row_with_max_value)
        runtime_str = row_with_max_value['run_time']    # runtime is the key to the data structure used to run it
        print('==> (' + str(target_size) + '/' + str(threshold_size) + ') the runtime with the best F1 for DOWN is: ' + runtime_str)

        # now open the <symbol>_trend.json and fetch the entry
        file_path = symbol+ '_trend.json'

        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # # Find the matching entry
        # matching_entry = next((entry for entry in data if entry[0] == runtime_str), None)

        # if matching_entry is not None:
        #     print("Found entry:", matching_entry)
        # else:
        #     print("No entry found with the specified 'run_time'.")
        #     exit
        
        # data_dict = {entry[0]: entry[1:] for entry in data}
        data_dict = {entry[0]: entry[1] for entry in data if isinstance(entry[1], dict)}


        # if runtime_str in data_dict:
        #     entry = data_dict[runtime_str]
        #     print("Found entry:", entry)
        # else:
        #     print("No entry found for date", runtime_str)

        if runtime_str in data_dict:
            entry = data_dict[runtime_str]
            if isinstance(entry, dict):
                print("Found entry:", entry)
                param = entry  # Assuming you need to use this dictionary as 'param'
            else:
                print("Retrieved entry is not a dictionary")
        else:
            print("No entry found for date", runtime_str)

        param = entry

        # Call and use the entry and do another training/prediction cycle, this will be the official one for prediction
        param['comment'] = 'Official Run with target=' + str(target_size) + ' & threshold='+ str(threshold_size)
        trendAnalysis.analyze_trend(trendConfig.config, param)







