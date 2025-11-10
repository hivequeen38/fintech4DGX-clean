import requests
import pandas as pd
from pandas import DataFrame
import datetime
import yaml
import pandas_datareader as pdr
import os
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from datetime import datetime, timedelta
import pytz
from typing import Tuple

# import alpha_vantage.fundamentaldata as av_fund

################################################
# return all the merged data into a single DF
#
def fetch_all_data(config) -> Tuple[bool, str]:
    """Function get ALL the relevant strings."""
   
    base_url = config["alpha_vantage"]["url"]
    api_key = config["alpha_vantage"]["key"]
    df= pd.DataFrame()

    function = "NEWS_SENTIMENT"
    tickers = "NVDA"
    topic = "technology"
    time_from = config["sentiment"]["last_timestamp"]
    sort = "LATEST"
    limit= "50"

    #####
    # fetch the sentiment data
    #
    params = {
                "apikey": api_key,
                "function": function,
                "tickers": tickers,
                "topic": topic,
                "sort": sort,
                "limit": limit,
                "time_from": time_from
            }
    print('DEBUG: about to get sentiment from timestamp: '+ time_from)
    response = requests.get(base_url, params=params, timeout=100)
    if response.status_code == 200:
        data_feed = response.json()
        # Prepare a list to hold extracted data
        extracted_data = []

        # Iterate through each article
        for article in data_feed['feed']:
            # Extract title, URL, and summary
            title = article['title']
            url = article['url']
            summary = article['summary']
            time_published = article['time_published']

            # Find NVDA data
            nvda_data = next((item for item in article['ticker_sentiment'] if item['ticker'] == 'NVDA'), None)

            if nvda_data:
                relevance_score = nvda_data['relevance_score']
                sentiment_score = nvda_data['ticker_sentiment_score']
                sentiment_label = nvda_data['ticker_sentiment_label']
            else:
                # Default values if NVDA data is not found
                relevance_score = None
                sentiment_score = None
                sentiment_label = None

            # Append to the list
            extracted_data.append([time_published, relevance_score, title, sentiment_label, summary, sentiment_score, url])

        # Create DataFrame
        df = pd.DataFrame(extracted_data, columns=['Time', 'NVDA Relevance Score', 'Title', 'NVDA Sentiment Label', 'Summary', 'NVDA Sentiment Score', 'URL'])

        # Convert NVDA Relevance Score to numeric, as it might be a string
        df['NVDA Relevance Score'] = pd.to_numeric(df['NVDA Relevance Score'], errors='coerce')

        # Filter out entries with relevance score < 0.33
        filtered_df = df[df['NVDA Relevance Score'] >= 0.25]

        print(filtered_df)

        # filtered_df.to_csv('NVDA_Sentiment.csv'); 

        # Create hyperlinks in the 'Title' column
        df = filtered_df
        df['Title'] = df.apply(lambda row: f'<a href="{row["URL"]}">{row["Title"]}</a>', axis=1)

        # Resetting the index and dropping it
        df.reset_index(drop=True, inplace=True)

        # Convert to HTML, but don't include the URL column
        html_table = df.drop('URL', axis=1).to_html(escape=False)

        # Now html_table contains the HTML representation of the DataFrame with clickable titles
        print(html_table)
        # Specify the file path and name
        latest_time: str = df['Time'][0]

        # Convert the string to a datetime object
        timestamp = datetime.strptime(latest_time, "%Y%m%dT%H%M%S")
        
        # add one to go to the next second
        timestamp = timestamp + timedelta(seconds=1)

        # Format the timestamp as a string in 'YYYYMMDDTHHMM' format
        formatted_timestamp = timestamp.strftime("%Y%m%dT%H%M")

        hasChanged: bool = False
        file_path: str = ""

        if (formatted_timestamp) != config["sentiment"]["last_timestamp"]:
            # the file have changed

            config["sentiment"]["last_timestamp"] = formatted_timestamp

            print(formatted_timestamp)
            file_path = 'NVDA_'+ formatted_timestamp + '.html'
 
            # Write the HTML table to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(html_table)
            hasChanged= True
            # DEBUG see of the config has changed
            print('The config last time stamp is now: '+ config["sentiment"]["last_timestamp"])
        else:
            print('The sentiment file have not changed')
        return hasChanged, file_path
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

    #########################################################################