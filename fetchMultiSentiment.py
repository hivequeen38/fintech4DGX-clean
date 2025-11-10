import requests
import pandas as pd
from pandas import DataFrame
import datetime
from datetime import datetime, timedelta
from typing import Tuple
import LRUCache

# import alpha_vantage.fundamentaldata as av_fund

def addArticlesToCache (df: DataFrame, articleCache: LRUCache):
    # List to hold indices of rows to be dropped
    rows_to_drop = []

    # Iterate through DataFrame rows
    num_dupe_found = 0
    for index, row in df.iterrows():
        # Add URL to cache and check if it's a cache hit
        if articleCache.add(row['Title'], None):  # Assuming the value is not important in this case
            # Mark row for deletion if it's a cache hit
            # we have a cache hit for URL
            # print('==> DUPLICATE ARTICLE found: '+row['Title'])
            num_dupe_found += 1    
            rows_to_drop.append(index)

    # Drop rows from DataFrame
    df.drop(rows_to_drop, inplace=True)
    if num_dupe_found > 0:
        print('==> # DUPLICATE ARTICLE found: '+ str(num_dupe_found))


################################################
# return all the merged data into a single DF
#
def fetch_all_data(config, symbol, relevance_cutoff, timestampStr, article_cache: LRUCache) -> Tuple[str, str]:
    """Function get ALL the relevant strings. DF could be real or empty, str could be real or empty"""

    base_url = config["alpha_vantage"]["url"]
    api_key = config["alpha_vantage"]["key"]
    df= pd.DataFrame()

    function = "NEWS_SENTIMENT"
    tickers = symbol
    topic = "technology"
    time_from = timestampStr
    sort = "LATEST"
    limit= "100"

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
    # print('DEBUG: about to get sentiment: ' + tickers+ ' from timestamp: '+ time_from)
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

            # Find data for symbol
            symbol_data = next((item for item in article['ticker_sentiment'] if item['ticker'] == tickers), None)

            if symbol_data:
                relevance_score = symbol_data['relevance_score']
                sentiment_score = symbol_data['ticker_sentiment_score']
                sentiment_label = symbol_data['ticker_sentiment_label']
            else:
                # Default values if symbol data is not found
                relevance_score = None
                sentiment_score = None
                sentiment_label = None

            # Append to the list
            extracted_data.append([tickers, time_published, relevance_score, title, sentiment_label, summary, sentiment_score, url])

        # Create DataFrame
        df = pd.DataFrame(extracted_data, columns=['symbol','Time','Relevance', 'Title', 'Sentiment', 'Summary','Sentiment Score', 'URL'])

        # DEBUG
        # print(">DF for sumbol:"+ tickers)
        # print(df.head)

        # Convert Relevance Score to numeric, as it might be a string
        df['Relevance'] = pd.to_numeric(df['Relevance'], errors='coerce')

        # Filter out entries with relevance score < relevance_cutoff
        filtered_df = df[df['Relevance'] >= relevance_cutoff]

        # now filter out anything from motley fool
        filtered_df = filtered_df[~filtered_df['URL'].str.startswith('https://www.fool.com/')]
        df = filtered_df

        addArticlesToCache(df, article_cache)

        formatted_timestamp: str = timestampStr

        # the filtering might have resulted in an empty set, in which case should still update timestamp but move on
        # df might get emptied by adding to cache becasue of duplicate article check
        #
        if not df.empty:

            # Create hyperlinks in the 'Title' column
            df['Title'] = df.apply(lambda row: f'<a href="{row["URL"]}">{row["Title"]}</a>', axis=1)

            # Resetting the index and dropping it
            df.reset_index(drop=True, inplace=True)

            # Now html_table contains the HTML representation of the DataFrame with clickable titles
            # print(html_table)
            # Specify the file path and name
            latest_time: str = df['Time'][0]

            # Convert the string to a datetime object
            timestamp = datetime.strptime(latest_time, "%Y%m%dT%H%M%S")
            
            # add one to go to the next min
            timestamp = timestamp + timedelta(minutes=1)

            # Format the timestamp as a string in 'YYYYMMDDTHHMM' format
            formatted_timestamp = timestamp.strftime("%Y%m%dT%H%M")
            
        return df, formatted_timestamp
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

    #########################################################################