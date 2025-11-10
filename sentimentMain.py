import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.preprocessing import StandardScaler

from alpha_vantage.timeseries import TimeSeries
import alpha_vantage.fundamentaldata as av_fund
import json
import numpy as np
# from datetime import datetime
import fetchSentiment
from google.cloud import storage
from google.oauth2 import service_account
from twilio.rest import Client
import vonage
import time

print("All libraries loaded")

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

####
# the forever while loop that will wake up periodically, and fetch data from the last timestamp
#
while True:
    hasChanged: bool;

    file_path: str
    hasChanged, file_path = fetchSentiment.fetch_all_data(config)

    ###
    # upload a file to bucket if it has changed
    #
    if hasChanged:
    # Path to your service account key file
        key_path = "sentiment-412417-27b17b73abd5.json"

        # Create a service account credential
        credentials = service_account.Credentials.from_service_account_file(
            key_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        # Create a client
        client = storage.Client(credentials=credentials, project=credentials.project_id)

        # Define your bucket name and file details
        bucket_name = 'sentiment-report'
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

        ###
        # now use Twilo to message phones
        # Twilo phone number: 18885721641
        #


        # # Message to send
        # file_content = 'NVDA Sentiment report: '+ 'file_path' + ' ' + public_url

        # # Send the message
        # message = client.messages.create(
        #     to="16508236296",
        #     from_="18885721641",
        #     body=file_content)

        # print('message.sid = ' + message.sid)

        ###
        # Use Nextmo to send messager via Vonnage
        #
        client = vonage.Client(key="94c89edb", secret="FQfg2yQlVja1zz43")
        sms = vonage.Sms(client)

        file_content = 'NVDA Sentiment report: '+ ' ' + public_url+ ' '

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

    # about to go to sleep for 60 min
    time.sleep(60*60)
    ###
    # end of while loop