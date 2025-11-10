import pandas as pd
import numpy as np
from pandas import DataFrame
import datetime
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def filter_out_comments(df, output_file):
    """
    Filter out rows containing a specific phrase in the 'comment' column
    and save the remaining rows to a new file.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame with 'comment' column
    output_file (str): Path to save the filtered DataFrame
    
    Returns:
    pandas.DataFrame: Filtered DataFrame
    """
    # Create mask for rows that don't contain the phrase
    mask = ~df['comment'].str.contains('now processing end_date=', na=False)
    
    # Apply mask to get filtered DataFrame
    filtered_df = df[mask]
    
    # Save to file
    filtered_df.to_csv(output_file, index=False)
    
    # Print summary of operation
    rows_removed = len(df) - len(filtered_df)
    print(f"Removed {rows_removed} rows containing the phrase")
    print(f"Saved {len(filtered_df)} remaining rows to {output_file}")
    
    return filtered_df

import pandas as pd

def filter_keep_comments(df, output_file, keep_matches=True):
    """
    Filter DataFrame rows based on whether they contain a specific phrase in the 'comment' column.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame with 'comment' column
    output_file (str): Path to save the filtered DataFrame
    keep_matches (bool): If True, keep rows containing the phrase. If False, remove them.
    
    Returns:
    pandas.DataFrame: Filtered DataFrame
    """
    # Create mask for rows containing the phrase
    mask = df['comment'].str.contains('QA AAII', na=False)
    
    # If we want to remove matches instead of keeping them, invert the mask
    if not keep_matches:
        mask = ~mask
    
    # Apply mask to get filtered DataFrame
    filtered_df = df[mask]
    
    # Save to file
    filtered_df.to_csv(output_file, index=False)
    
    # Print summary of operation
    action = "Kept" if keep_matches else "Removed"
    rows_affected = len(df) - len(filtered_df)
    print(f"{action} {len(filtered_df)} rows containing the phrase")
    print(f"Saved to {output_file}")
    
    return filtered_df

##########################################################
# Function to filter rows based on a specific date
#
def filter_by_date(df, input_date):
    # Convert input_date to a datetime object and extract the date part
    input_date = pd.to_datetime(input_date).date()

    # Ensure df['date_only'] contains only date components
    df['date_only'] = pd.to_datetime(df['date'], format='mixed').dt.date

    # Return a new DataFrame with rows matching the input date
    return df[df['date_only'] == input_date]


###################################################################################################
# Function to find the row 16 entries before the input date !!! Not used
# def find_previous_16th_row(df, input_date):
#     # Ensure the input_date is in datetime format
#     input_date = pd.to_datetime(input_date)
    
#     # Check if the input_date is present in the 'date' column
#     if input_date not in df['date'].values:
#         raise ValueError(f"The date {input_date} is not found in the DataFrame.")
    
#     # Find the index of the row with the input date
#     input_index = df[df['date'] == input_date].index[0]
    
#     # Calculate the index of the row 16 entries before
#     target_index = input_index - 16
    
#     # Check if target_index is valid
#     if target_index < 0:
#         raise ValueError("Not enough previous entries to count back 15 rows.")
    
#     # Return the entire row at the target index
#     return df.iloc[[target_index]]  # Use double brackets to keep it as DataFrame


def print_results(actual: list, pred: list):   
    # Now calculate the accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(actual, pred, average=None)

   # Define class labels (if you have specific names for classes, replace these)
    class_labels = ['__', 'UP', 'DN']

    # Iterate over each class and print the results
    for i, label in enumerate(class_labels):
        # Check if the index exists in the precision array and handle cases where the index is out of bounds
        if i < len(precision):
            prec = 'Undefined' if np.isnan(precision[i]) else precision[i]
        else:
            prec = 'Undefined'

        if i < len(recall):
            rec = 'Undefined' if np.isnan(recall[i]) else recall[i]
        else:
            rec = 'Undefined'

        if i < len(f1):
            f1_score = 'Undefined' if np.isnan(f1[i]) else f1[i]
        else:
            f1_score = 'Undefined'
        
        print(f"{label}: Precision = {prec}, Recall = {rec}, F1-Score = {f1_score}")

def process_single_prediction(prediction_df: DataFrame, master_df: DataFrame, num_of_days: int, file_index: int, cum_result_df: DataFrame):
    '''
    The prediction_df should only have one entry
    '''
    comment = prediction_df.iloc[0]['comment']
    prediction_date = prediction_df.iloc[0]['date_only']

    # keep just the p1...p15 columns
    input_df = prediction_df.drop(columns=['date', 'close', 'comment', 'date_only'])
    # transpose the results into a col
    # Transpose the row into a column
    transposed_df = input_df.T.reset_index()
 
    # Rename the columns for clarity
    transposed_df.columns = ['P_offset', 'Pred']

    # This should have 15 entries, we want to add something up front so we have 16 entries
    # Create an empty row (with NaN values for each column)
    empty_row = pd.DataFrame([[None] * len(transposed_df.columns)], columns=transposed_df.columns)

    # Add the empty row to the top of the DataFrame
    df = pd.concat([empty_row, transposed_df], ignore_index=True)

    # now we need to take the master df (which has everything, and generate a smaller df where)
    # 1. There are only 16 entries, starting with the prediction date, and then 15 days after that (a total of 16 rows)
    #
    # Ensure that 'date' column is in datetime format
    master_df['date'] = pd.to_datetime(master_df['date'])

    # make sure these are all the right type
    prediction_date = pd.to_datetime(prediction_date)
    master_df['date'] = pd.to_datetime(master_df['date'])

    # Find the index of the row where 'date' matches the prediction_date
    prediction_date_index = master_df[master_df['date'] == prediction_date].index[0]

    # Select the row with the matching date and the next 15 rows
    result_df = master_df.iloc[prediction_date_index:prediction_date_index + num_of_days+ 1]

    # Just keep date, and adjusted close
    selected_columns = ['date', 'adjusted close']
    result_df = result_df[selected_columns]   # only keep those slected features defined in the param, this includes label
    result_df.reset_index(drop=True, inplace=True)

    # now concatinate the result_df with the transposed 
    # now we want to add this col to the prediction_df
    # Concatenate column-wise (axis=1)
    result_df = pd.concat([result_df, df], axis=1)

    # now the col is 'date', 'closing', 'P-offset', 'Pred'
    # now populate a new col bassed on the reference closing price
    # 
    reference_price = result_df.loc[0]['adjusted close']
    # Add a new column 'ref_close' and populate from row 1 onwards with the reference_price
    result_df['ref_close'] = None  # Initialize the column with None or NaN
    result_df.loc[1:, 'ref_close'] = reference_price  # Populate from row 1 onwards

    # now calculate the gain
    result_df['gain'] = None
    result_df.loc[1:, 'gain'] = (result_df.loc[1:, 'adjusted close'] - result_df.loc[1:, 'ref_close']) / result_df.loc[1:, 'ref_close']
    # Ensure the 'gain' column is in numeric format
    result_df.loc[1:, 'gain'] = pd.to_numeric(result_df.loc[1:, 'gain'], errors='coerce')

    # Now round to 2 decimal places
    # def round_to_3_sig_figs(x):
    #     if x == 0:
    #         return 0
    #     return round(x, -int(np.floor(np.log10(abs(x)))) + 2)

    def round_to_3_sig_figs(x):
        # Check for NaN or zero values first
        if pd.isna(x) or x == 0:
            return x
        return round(x, -int(np.floor(np.log10(abs(x)))) + 2)
    
    # result_df.loc[1:, 'gain'] = result_df.loc[1:, 'gain'].apply(round_to_3_sig_figs)
    # Then modify the apply call to handle NaN values:
    result_df.loc[1:, 'gain'] = result_df.loc[1:, 'gain'].apply(lambda x: round_to_3_sig_figs(x))

    # now clculate the actual UP/DN/__
    # Create a new column 'actual' based on the conditions
    result_df['actual'] = None
    result_df.loc[1:5,'actual'] = np.where(result_df.loc[1:5,'gain'] >= 0.03, 1, np.where(result_df.loc[1:5,'gain'] <= -0.03, 2, 0))
    result_df.loc[6:,'actual'] = np.where(result_df.loc[6:,'gain'] >= 0.05, 1, np.where(result_df.loc[6:,'gain'] <= -0.05, 2, 0))

    # Create a dictionary for mapping
    conversion_dict = {'__': 0, 'UP': 1, 'DN': 2}

    # Create the new column starting from index 1
    result_df.loc[1:, 'pred_val'] = df.loc[1:, 'Pred'].map(conversion_dict)

    # Now calculate the accuracy
    pred = result_df['pred_val'].dropna().to_list()
    actual = result_df['actual'].dropna().to_list()

    print_results(pred, actual)

    # Now add the preserved comment to a comment field
    result_df['comment'] = None
    result_df.loc[0, 'comment'] = comment

    new_order = ['date', 'P_offset', 'ref_close', 'adjusted close', 'gain', 'Pred', 'actual', 'pred_val', 'comment']
    result_df = result_df[new_order]

    # Save the file
    file_path = 'verify_prediction_result_'+ prediction_date.strftime('%Y-%m-%d') + '_'+ str(file_index) + '.csv'
    result_df.to_csv(file_path, index=False)

    # now append the result_df to the cum_df
    cum_result_df = pd.concat([cum_result_df, result_df], ignore_index=True)

    return cum_result_df

##############################
# verify predictions
# (the main entrance)
#
def main_verify_predictions(symbol: str, num_of_days: int, last_date: str, cum_result_df: DataFrame, master_df: DataFrame):
    '''
    The input_df is the prediction result df (The full thing)
    master_df contains all the tradeing dates and closeing prices
    the dates in the prediction results is in Y:M:D H:M
    '''    
    # read the input_df from results
    file_path = symbol+ "_" + str(num_of_days) + "d_from_today_predictions.csv"
    input_df = pd.read_csv(file_path)
    print('>prediction data loaded from disk in ' + file_path)

    # filter out QA stuff
    input_df = filter_keep_comments(input_df, 'QA_PLTR_Reference.csv')


    # based on the date time, find the index of the matching date from the master_df
    # Find the index where 'date' matches the input 'last_date'
    #
    matching_index = master_df.index[master_df['date'] == last_date]  # Assume no dupe date
    # matching_index = master_df.index[master_df['date'].astype(str) == str(last_date)]

    prediction_date_index = matching_index - num_of_days

    # now go back 16 days to get the prediciton date
    prediction_date = master_df.loc[prediction_date_index, 'date']

    # Since it's a Series, use .iloc[0] to extract the value
    prediction_date = prediction_date.iloc[0]

    # now find all the matching predictions made on that date
    predictions_made_df = filter_by_date(input_df, prediction_date)

    # cum_result_df might be empty (for the first call). 
    # in that case initialize
    if cum_result_df is None:
        cum_result_df = pd.DataFrame()

    # Iterate through rows and process each one
    for index, row in predictions_made_df.iterrows():
        row_df = pd.DataFrame([row]) 
        cum_result_df = process_single_prediction( row_df, master_df, num_of_days, index, cum_result_df )
    
    return cum_result_df

#### Main debug flow starts here
# because we use last day as the reference, we now have to use the first day then then index in 15 offset to get the end date
symbol = 'PLTR'
start_date_str = "2024-03-26"
end_date_str = "2025-03-05"

 # Read in the master_df from the TMP.csv file
file_path = symbol + '_TMP.csv'  
master_df = pd.read_csv(file_path)
print('>master data loaded from cache in ' + file_path)

# master_date list 
master_df['date'] = pd.to_datetime(master_df['date'])
start_date = pd.to_datetime(start_date_str)
end_date = pd.to_datetime(end_date_str)

master_date_df = master_df[(master_df['date'] >= start_date) & (master_df['date'] <= end_date)]
master_date_df['date'] = master_date_df['date'].dt.strftime('%Y-%m-%d')
master_date_list = master_date_df['date'].tolist()  # the master date list has to be strings

master_df['date'] = master_df['date'].dt.strftime('%Y-%m-%d')

cum_result_df = pd.DataFrame()
date_offset = 15

for index, date in enumerate(master_date_list):
    matching_index = master_df.index[master_df['date'] == date]  # Assume no dupe date

    end_date_index = matching_index + date_offset

    # now go back 16 days to get the prediciton date
    end_date = master_df.loc[end_date_index, 'date'].values[0]
    cum_result_df = main_verify_predictions(symbol, date_offset, end_date, cum_result_df, master_df)
    # cum_result_df = main_verify_predictions('NVDA', 15, '2024-03-28', cum_result_df)
    # cum_result_df = main_verify_predictions('NVDA', 15, '2024-04-01', cum_result_df)

# now print the results of the cum_df
# Now calculate the accuracy
pred = cum_result_df['pred_val'].dropna().to_list()
actual = cum_result_df['actual'].dropna().to_list()

print('>>> The cumulative result Stats: <<<')
print_results(pred, actual)

print('\n')

# Now process the (Old Process) based on comment

# Filter the DataFrame
# filtered_df = df[df['comment'].str.contains('(Old Process)', na=False, regex=False)]

# store away cum_result_df
file_path = 'cum_verify_prediction_result_'+ start_date_str + '_'+ end_date_str + '.csv'
cum_result_df.to_csv(file_path, index=False)