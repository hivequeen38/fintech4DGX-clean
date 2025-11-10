import pandas as pd
from datetime import datetime
import mainDeltafromToday
import daily_main
import NVDA_param
import PLTR_param

######################################################
# General PLan
# 1. have a start and end date
# 2. grab the TMP.file so all the master dates are in
# 4. iterate each day in the file from start to end
# 5. For each date, set it as the end date, and then run trendMainDealtaFromToday
# 6. When all done stich all the results together into a single DF, then get F1
#
symbol = 'PLTR'
start_date_str = "2021-09-30"
# start_end_date_str = "2024-03-26"
start_end_date_str = "2025-03-03"
end_end_date_str = "2025-03-05"

# 2. grab the TMP.file so all the master dates are in
file_path = symbol+ '_TMP.csv'
df = pd.read_csv(file_path)

# now find all the rows in the df that is from start date to end date
df['date'] = pd.to_datetime(df['date'])     # convert date field to datetime so we can do comparison

# Convert 'start_date' and 'end_date' to datetime format
start_date = pd.to_datetime(start_end_date_str)
end_date = pd.to_datetime(end_end_date_str)

# Filter the DataFrame based on the date range
filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
filtered_df['date'] = filtered_df['date'].dt.strftime('%Y-%m-%d')     # change it back to str

# now just keep the date col
filtered_date = filtered_df['date'].dropna().unique()

# Convert to a list if needed
dates_list = filtered_date.tolist()

load_cache=True

param = PLTR_param.AAII_reference
# param = daily_main.NVDA_ref_param   

# 4. iterate each day in the file from start to end 
for target_date in dates_list:
    print('>>>QA Now processing date= ' + target_date + ' as end date')
    # mainDeltafromToday.QA_mz_processEndDate(param, target_date, load_cache, comment = '(QA) now end_date= ')
    mainDeltafromToday.QA_processEndDate(param, target_date, load_cache, comment = '(QA AAII_Ref) now end_date= ')
    load_cache=False
