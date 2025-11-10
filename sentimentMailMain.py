import fetchSentiment
from google.cloud import storage
from google.oauth2 import service_account
import time
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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

stocks_to_watch = {
    'NVDA', 'AMD'
}

# setup dictionary of stock symbol to last_timestamp lookup
#
stock_last_timestamp_dict = {}
now = datetime.now()
now_timestamp = now.strftime("%Y%m%dT%H%M%S")

for symbol in stocks_to_watch:
    print('setting up stock= '+ symbol)
    stock_last_timestamp_dict.update({symbol: now_timestamp})

# DEBUG print
print(stock_last_timestamp_dict)

####
# the forever while loop that will wake up periodically, and fetch data from the last timestamp
#

# Setup email 
#
sender_email = "milton.soong@gmail.com"
receiver_email = "linda.dowsk.liu@gmail.com, msoongtest@gmail.com"
password = "dzal yryd wabi ugnh"  # Be cautious with your password
subject = "New NVDA sentiment Report"
body = "This is the body of the email."

while True:
    hasChanged: bool;

    file_path: str
    hasChanged, file_path = fetchSentiment.fetch_all_data(config)

    ###
    # upload a file to bucket if it has changed
    #
    if hasChanged:
    # # Path to your service account key file
    #     key_path = "sentiment-412417-27b17b73abd5.json"

    #     # Create a service account credential
    #     credentials = service_account.Credentials.from_service_account_file(
    #         key_path,
    #         scopes=["https://www.googleapis.com/auth/cloud-platform"],
    #     )

    #     # Create a client
    #     client = storage.Client(credentials=credentials, project=credentials.project_id)

    #     # Define your bucket name and file details
    #     bucket_name = 'sentiment-report'
    #     destination_blob_name = file_path
    #     source_file_name = file_path

    #     # Get the bucket
    #     bucket = client.bucket(bucket_name)

    #     # Create a blob and upload the file
    #     blob = bucket.blob(destination_blob_name)
    #     blob.upload_from_filename(source_file_name)

    #     # Get the public URL
    #     public_url = blob.public_url

    #     print(f"The file is uploaded and publicly accessible at: {public_url}")

        ###
        # Now get the content out to subscribers
        #
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = 'NVDA Sentiment Report: '+file_path

        # Open the file and read its content into 'body'
        with open(file_path, 'r', encoding='utf-8') as file:
            body = file.read()

        # Add body to email
        message.attach(MIMEText(body, "html"))

        # SMTP server configuration
        smtp_server = "smtp.gmail.com"
        port = 587  # For starttls

        # Start a secure SMTP connection
        server = smtplib.SMTP(smtp_server, port)
        server.starttls()  # Secure the connection

        try:
            server.login(sender_email, password)
            text = message.as_string()
            server.sendmail(sender_email, receiver_email, text)
            print("Email sent successfully to " + receiver_email)
        except Exception as e:
            print(f"Error: {e}")

        finally:
            server.quit()

    # about to go to sleep for 60 min
    time.sleep(60*60)
    ###
    # end of while loop