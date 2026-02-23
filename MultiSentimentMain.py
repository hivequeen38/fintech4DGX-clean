import pandas as pd
from pandas import DataFrame
# from alpha_vantage.timeseries import TimeSeries
# import alpha_vantage.fundamentaldata as av_fund
import json
from google.cloud import storage
from google.oauth2 import service_account
# import vonage
# from sinch import Client
from datetime import datetime, time, timedelta
import time as time_module
import os
import requests
from dataclasses import dataclass
from collections import OrderedDict
import fetchMultiSentiment
import googleCloudUtil
import LRUCache
from bs4 import BeautifulSoup
import get_daily_results
import get_osillator
import get_historical_html
# import mainDeltafromToday  # Commented out for sentiment-only mode
import NVDA_param
import SMCI_param
import PLTR_param
import APP_param
import META_param
import MSTR_param
import INOD_param
import QQQ_param
import TSM_param
import CRDO_param
import ANET_param
import ALAB_param

# import inferenceOnly

print("> All libraries loaded")

# define class for results to be posted
@dataclass
class ReportToPost:
    symbol: str
    content: str


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
    "sentiment": {
        "last_timestamp": "20240129T1331"
    }, 
    "eodhd": {

    }
}

stocks_to_watch = {
    "NVDA": {
        "relevance_cutoff": 0.33,
        "last_timestamp": "20240228T1357"
    },
    "META": {
        "relevance_cutoff": 0.5,
        "last_timestamp": "20240227T1450"
    },
    "NOW": {
        "relevance_cutoff": 0.5,
        "last_timestamp": "20240226T1518"
    },
    "MSFT": {
        "relevance_cutoff": 0.5,
        "last_timestamp": "20240228T1254"
    },
    "MSTR": {
        "relevance_cutoff": 0.25,
        "last_timestamp": "20240227T0628"
    },
    "GOOG": {
        "relevance_cutoff": 0.5,
        "last_timestamp": "20240228T1350"
    },
    "AMZN": {
        "relevance_cutoff": 0.5,
        "last_timestamp": "20240226T1415"
    },
    "ADBE": {
        "relevance_cutoff": 0.5,
        "last_timestamp": "20240226T2245"
    },
    "QCOM": {
        "relevance_cutoff": 0.5,
        "last_timestamp": "20240228T0956"
    },
    "CRYPTO:BTC": {
         "relevance_cutoff": 0.50,
        "last_timestamp": "20240129T1331"
    }
}

####################################
def saveToFile(timeStampStr: str, body: str)-> str:
    file_path = 'Sentiment_'+ timeStampStr + '.html'
 
    # Write the HTML table to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(body)
    return file_path

####################################
def seconds_until(target_hour, target_minute=0, target_second=0):
    # Get the current datetime
    now = datetime.now()

    # Create a datetime object for the target time today
    target_time = datetime.combine(now.date(), time(target_hour, target_minute, target_second))

    # If the target time is already past, set it for the next day
    if now >= target_time:
        target_time += timedelta(days=1)

    # Calculate the number of seconds until the target time
    return (target_time - now).total_seconds()

####################################
# This will default to 60 min, but then will skip the rest hours
#
def in_black_out_period()-> bool:
    # Get the current local time
    now = datetime.now().time()
    nowstr = now.strftime("%H%M")
    # print('==> Check if blackout time. Now= '+ nowstr)

    # Define the start and end times
    start_time = time(23, 0)  # 11 PM
    end_time = time(4, 0)    # 4 AM
    in_period: bool = False

    # Check if the current time is within the range
    if start_time <= now or now < end_time:
        print("==> Current time is within the blckout range.")
        in_period = True
    return in_period

####################################
def send_sms_message(public_url: str):
    client = vonage.Client(key="94c89edb", secret="FQfg2yQlVja1zz43")
    sms = vonage.Sms(client)

    file_content = 'Sentiment report: '+ ' ' + public_url+ ' '

    # now send to Dowsk's number
    responseData = sms.send_message(
        {
            "from": "17742250231",
            "to": "16508239634",
            "text": file_content,
        }
    )

    if responseData["messages"][0]["status"] == "0":
        print("Message sent successfully to Dowsk.")
    else:
        print(f"Message failed with error: {responseData['messages'][0]['error-text']}")

####################################
def send_sms_via_sinch(public_url: str):

    sinch_client = Client(
        key_id="8ce12697-164b-46b1-ad14-5071fa2ff759",
        key_secret="fqdy3VwRyZdX5U961eOjgfCNM1",
        project_id="11433ed2-f50e-4f69-94b1-811aa25b6ec3"
    )

    file_content = 'Sentiment report: '+ ' ' + public_url+ ' '
    send_batch_response = sinch_client.sms.batches.send(
        body=file_content,
        to=["16508239634"],
        from_="12085797231",
        delivery_report="none"
    )

    # print('SMS send response via Sinch: ')
    # print(send_batch_response)

####################################
def is_market_open()-> bool:
    '''returns open, closed, or market not found'''
    api_key = config["alpha_vantage"]["key"]
    base_url = config["alpha_vantage"]["url"]
    params = {
                "apikey": api_key,
                "function": 'MARKET_STATUS'
            }
    status_str: str = None
    response = requests.get(base_url, params=params, timeout=100)
    if response.status_code == 200:
        data = response.json()
        
        for market in data['markets']:
            if market['region'] == 'United States' and market['market_type'] == 'Equity':
                status_str = market['current_status']
                break
        if status_str == None:
            print("Market not found")
    else:
        print(f"Error {response.status_code}: {response.text}")
    
    if status_str == 'open':
        return True
    else:
        return False


def handle_flexible_master_sentiment(new_url: str):

    # Step 1: Read the CSV file into a DataFrame
    file_path = 'master_urls.csv'  # Replace with your actual file path
    df = pd.read_csv(file_path, header=None, names=['URL'])

  # Create a new DataFrame with the new URL
    new_df = pd.DataFrame([new_url], columns=['URL'])

    # Concatenate the new DataFrame with the existing one, placing the new URL at the top
    df = pd.concat([new_df, df], ignore_index=True)

    # Step 4: Keep only the first 95 entries
    df = df.head(95)

    # Step 4.5: df table back to a file
    df.to_csv(file_path, index=False, header=False)
    
    # Create hyperlinks in the 'URL' column
    df['URL'] = df.apply(lambda row: f'<a href="{row["URL"]}">{row["URL"]}</a>', axis=1)

    # Step 5: Generate a HTML table with a larger font size using Styler
    styled_df = df.style.set_table_styles([
        {'selector': 'table', 'props': [('font-size', '16px')]}
    ])

    # Use to_html() to render the styled DataFrame to HTML with escape=False to prevent escaping of HTML tags
    html_table = styled_df.to_html(escape=False)

    # Step 6: (Optional) Save the HTML table to a file
    with open('urls_table.html', 'w') as file:
        file.write(html_table)

    return(html_table)

from datetime import datetime, time, timedelta
import pytz

def time_until_230pm():
    # Define the target time in Pacific Time
    pacific_tz = pytz.timezone('America/Los_Angeles')  # This handles PST/PDT automatically
    
    # Get current time in UTC and convert to Pacific Time
    now_utc = datetime.now(pytz.UTC)
    now_pacific = now_utc.astimezone(pacific_tz)
    
    # Create today's 3:40 PM target in Pacific Time
    today_target_naive = datetime.combine(now_pacific.date(), time(17, 30))
    today_target = pacific_tz.localize(today_target_naive)
    
    # Determine if we should target today or tomorrow
    if now_pacific < today_target:
        target_time = today_target
    else:
        # If it's already past 2:40 PM Pacific, set target to 2:40 PM Pacific tomorrow
        tomorrow = now_pacific.date() + timedelta(days=1)
        tomorrow_target_naive = datetime.combine(tomorrow, time(14, 40))
        target_time = pacific_tz.localize(tomorrow_target_naive)

    # Calculate time difference in the same timezone
    time_difference = target_time - now_pacific
    seconds_remaining = time_difference.total_seconds()

    if seconds_remaining <= 2 * 3600:
        return seconds_remaining  # Return the actual time difference in seconds
    else:
        return 2 * 3600  # Return 2 hours in seconds

def time_until_230pm_OLD():
    now = datetime.now()
    today_target = datetime.combine(now.date(), time(14, 40))  # 2:30 PM today
    # today_target = datetime.combine(now.date(), time(12, 20))  # 2:15 PM HI time today 

    if now < today_target:
        target_time = today_target
    else:
        # If it's already past 2:30 PM, set target to 2:30 PM tomorrow
        target_time = datetime.combine(now.date() + timedelta(days=1), time(14, 40))
        # target_time = datetime.combine(now.date() + timedelta(days=1), time(12, 20))

    time_difference = target_time - now
    seconds_remaining = time_difference.total_seconds()

    if seconds_remaining <= 2 * 3600:
        return seconds_remaining  # Return the actual time difference in seconds
    else:
        return 2 * 3600  # Return 2 hours in seconds

# Add this function to your existing code
def get_bitcoin_price():
    """
    Function to fetch the current Bitcoin price.
    """
    try:
        response = requests.get('https://api.coindesk.com/v1/bpi/currentprice.json')
        data = response.json()
        bitcoin_price = data['bpi']['USD']['rate_float']
        print(f"Bitcoin price at {datetime.now()}: ${bitcoin_price}")
        
        # Optional: Log to file or perform additional actions with the price
        with open('bitcoin_prices.log', 'a') as log_file:
            log_file.write(f"{datetime.now()},{bitcoin_price}\n")
            
        return bitcoin_price
    except Exception as e:
        print(f"Error fetching Bitcoin price: {e}")
        return None

# Add this function to check if it's 1 PM
def is_bitcoin_check_time():
    """
    Check if current time is in the Bitcoin price check window.
    Returns True if time is between 12:59 PM and 2:00 PM.
    """
    now = datetime.now().time()
    start_time = time(12, 59)  # 12:59 PM
    end_time = time(14, 0)     # 2:00 PM
    
    return start_time <= now <= end_time

# Modify your sleep_time calculation to ensure we check often enough to not miss 1 PM
def calculate_sleep_time():
    # Get the minimum of time until 2:30 PM and time until 1 PM
    time_to_bitcoin_check = seconds_until(13, 0, 0)  # 1 PM
    market_time = time_until_230pm()
    
    # Don't sleep more than 2 hours to ensure we don't miss the 1 PM check
    return min(market_time, time_to_bitcoin_check, 2 * 3600)


##################################################################################################
# MAIN FLOW
# the forever while loop that will wake up periodically, and fetch data from the last timestamp
#
anySymbolChanged: bool

googleCloudUtil.cleanup_old_report()

# define the article cache 
article_cache = LRUCache.LRUCache(500)

#define variable used to track when market closes
market_last_open: bool = is_market_open()
need_to_run_prediction: bool = False
bitcoin_price_checked_today = False

# check google cloud to see if a master sentiment file is present,
# if not create it, 
# if yes, fetch it
# then make sure the local file handle is setup
#
# Path to your service account key file
key_path = "sentiment-412417-27b17b73abd5.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

# Initialize a client
client = storage.Client()

# Define your bucket name and file name
bucket_name = 'sentiment-report'
file_name = "master_sentiment.html"

# Get the bucket
bucket = client.bucket(bucket_name)

# Check if the file exists
blob = bucket.blob(file_name)

while True:
    # Read latest stocks to watch and updated timestamp if they exist``
    #
    # Check if the file exists
    file_path = 'stocks_to_watch.json'
    if os.path.exists(file_path):
        # File exists, you can proceed with reading the file
        with open(file_path, 'r') as file:
            # Perform file read operations here
            stocks_to_watch = json.load(file)
    else:
        # File does not exist
        print(f"The file {file_path} does not exist. use coded default instead")
        
    anySymbolChanged = False       # this is set to true if ANY of the stock report has changed
    file_path: str
    result_df = DataFrame()

    # Bitcoin price check at 1 PM
    if is_bitcoin_check_time() and not bitcoin_price_checked_today:
        print("It's Bitcoin check time - checking Bitcoin price...")
        bitcoin_price = get_bitcoin_price()
        bitcoin_price_checked_today = True

        if bitcoin_price is not None:
            price_message = f"Current Bitcoin price: ${bitcoin_price}"
    
    # firs thing in the loop is to check if market open
    # 1. call to see if market open
    # 2. if market_last_open == True and mow it returns false, then mareket just closed
    # 3. Set market_last_open to false
    # 4. Process end of day prediciton stuff
    #
    current_market_status = is_market_open()
    print('==>Market last open ='+ str(market_last_open))
    print('==>Current Market status ='+ str(current_market_status))


    if (current_market_status is False and market_last_open is True):
        # market just closed, do end of market processing
        need_to_run_prediction = True
        print('==> Market Just closed')

    market_last_open = current_market_status
        
    if (need_to_run_prediction is True):
        # now that we know we need to run prediction, check if it's at least 2:30
        now = datetime.now().time()
        nowstr = now.strftime("%H%M")

        # check if the current time is at least 2:30 in the afternoon, if so do some inference
        # Convert the nowstr to an integer
        now_int = int(nowstr)

        # Define the target time (2:30 PM)
        target_time = time(14, 40)  # 2:40 PM
        # target_time = time(12, 20)  # 12:20 PM for Hawaii
        target_time_str = target_time.strftime("%H%M")
        target_time_int = int(target_time_str)

        # Check if nowstr is the same or greater than 2:30 PM
        if now_int >= target_time_int:
            print("The current time is the same or greater than 2:30 PM, need to process new training & do daily prediction")
            # now do some daily stuff

            # get today's date string ['NVDA', 'PLTR', 'APP', 'ANET', 'CRDO', 'ALAB' ]
            today_date_str = datetime.now().strftime("%Y-%m-%d")

            # Training code commented out for sentiment-only mode
            # mainDeltafromToday.main(CRDO_param.reference, end_date = today_date_str)
            # mainDeltafromToday.main(ANET_param.reference, end_date = today_date_str)
            # mainDeltafromToday.main(ALAB_param.reference, end_date = today_date_str)
            # mainDeltafromToday.main(APP_param.reference, end_date = today_date_str)
            # mainDeltafromToday.main(NVDA_param.reference, end_date = today_date_str)
            # mainDeltafromToday.main(PLTR_param.reference, end_date = today_date_str)
            # mainDeltafromToday.main(INOD_param.reference, end_date = today_date_str)
            # get_historical_html.upload_all_results(today_date_str)

            # mainDeltafromToday.main(CRDO_param.AAII_option_vol_ratio, end_date = today_date_str)
            # mainDeltafromToday.main(ANET_param.AAII_option_vol_ratio, end_date = today_date_str)
            # mainDeltafromToday.main(ALAB_param.AAII_option_vol_ratio, end_date = today_date_str)
            # mainDeltafromToday.main(APP_param.AAII_option_vol_ratio, end_date = today_date_str)
            # mainDeltafromToday.main(NVDA_param.AAII_option_vol_ratio, end_date = today_date_str)
            # mainDeltafromToday.main(PLTR_param.AAII_option_vol_ratio, end_date = today_date_str)
            # mainDeltafromToday.main(INOD_param.AAII_option_vol_ratio, end_date = today_date_str)
            # get_historical_html.upload_all_results(today_date_str)

            print("✓ Sentiment gathering complete (training skipped in sentiment-only mode)")
            
            need_to_run_prediction = False
            bitcoin_price_checked_today = False
        else:
            print("Need to run daily prediction at 2:30 PM & not there yet")
    
    if not in_black_out_period():
        ##########################################################
        # Now loop through each stock symbol & fetch the results
        # 
        for symbol, details in stocks_to_watch.items():
            relevance_cutoff = details["relevance_cutoff"]
            last_timestamp = details["last_timestamp"]

            try:    # add try catch block because of content causing error
                current_df, new_timestamp = fetchMultiSentiment.fetch_all_data(config, symbol, relevance_cutoff, last_timestamp, article_cache)
            except Exception as e:  # catch all exceptions
                print(f"⚠️  Error getting sentiment for {symbol}: {e}")
                print(f"   Skipping {symbol} and continuing...")
                continue  # skip this symbol and move to next
            else:
                # print("Routine X executed successfully")

                if new_timestamp != last_timestamp:
                    # the file have changed
                    print('Timestamp have changed for '+ symbol + ': Old = ' + last_timestamp + ', new= ' + new_timestamp)

                    stocks_to_watch[symbol]["last_timestamp"]= new_timestamp

                    if result_df.empty:
                        result_df = current_df
                    else:
                        result_df = pd.concat([result_df, current_df], ignore_index=True)
                # else:
                    # print('The sentiment file have not changed for sumbol= '+symbol + ' or there are all filtered out')


        ###
        # after all the symbols has been processed, if there is any result, then convert to html then upload
        #
        if not result_df.empty:
            file_path: str = ""
            
            # Convert to HTML, but don't include the URL column
            html_table = result_df.drop('URL', axis=1).to_html(escape=False)

            # create the filepath
            # use the timestamp for NOW instead of any reports for file path
            #
            now = datetime.now()
            nowTimestampStr = now.strftime("%Y%m%dT%H%M")
            file_path = saveToFile(nowTimestampStr, html_table)

            # Path to your service account key file
            # key_path = "sentiment-412417-27b17b73abd5.json"

            # Create a service account credential
            credentials = service_account.Credentials.from_service_account_file(
                key_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

            # Create a client
            client = storage.Client(credentials=credentials, project=credentials.project_id)

            # Define your bucket name and file details
            # bucket_name = 'sentiment-report'
            destination_blob_name = file_path
            source_file_name = file_path

            # Get the bucket
            bucket = client.bucket(bucket_name)

            # Create a blob and upload the file
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_file_name)

            # Get the public URL
            public_url = blob.public_url

            print(f"The file is uploaded and publicly accessible at: {public_url}")

            ################
            # Now deal with the master sentiment file
            # add a new URL to the table
            # new_row = f"<tr><td><a href='{public_url}'>{public_url}</a></td></tr>"
            # new_row = f"<tr><td style='font-size: 20px;'><a href='{public_url}'>{public_url}</a></td></tr>"

            # Insert the new row at the top of the table in the BeautifulSoup object
            # table.insert(1, BeautifulSoup(new_row, 'html.parser'))

            master_file = handle_flexible_master_sentiment(public_url)


            # Save the updated HTML back to the file and upload to GCS
            # with open(file_name, 'w') as f:
            #     # f.write(str(soup))
            #     f.write(new_master_html)

            blob = bucket.blob(file_name)
            # blob.upload_from_filename(file_name)
            # print(f"Updated {file_name} uploaded to bucket {bucket_name}.")

            # blob.upload_from_filename(master_filepath)
            blob.upload_from_string(master_file, content_type='text/html')

            print(f"Updated master_file uploaded to bucket {bucket_name}.")
            # Upload the content directly from the variable
            
            # Now we can delete the local file
            os.remove(file_path)

            ###
            # Use Nextmo to send messager via Vonnage
            #
            #send_sms_message(public_url)
            # send_sms_via_sinch(public_url)
            

            # If we are here then at least one timestamp has changed, dump it into file
            #
            # Writing JSON data
            with open('stocks_to_watch.json', 'w') as file:
                json.dump(stocks_to_watch, file, indent=4)

    # figure out how long to sleep
    # sleep_time = time_until_230pm()
    sleep_time = calculate_sleep_time()
    print('Sleep for: ' + str(sleep_time) + ' seconds')
    time_module.sleep(sleep_time)
    ###
    # end of while loop