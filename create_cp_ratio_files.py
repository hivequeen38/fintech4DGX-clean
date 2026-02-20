#!/usr/bin/env python3
"""
Create missing CP ratio files for stocks that need options_volume_ratio data
This script fetches historical options data from Alpha Vantage and saves to CSV files
"""

import pandas as pd
import fetchBulkData
import trendConfig

# Stocks to process
stocks = ['NVDA', 'CRDO', 'PLTR', 'APP', 'INOD']

api_key = trendConfig.alpha_vantage_API_KEY

for symbol in stocks:
    print(f"\n{'='*60}")
    print(f"Processing {symbol}...")
    print(f"{'='*60}\n")

    try:
        # Read the TMP file to get all dates
        master_df = pd.read_csv(f'{symbol}_TMP.csv')

        # Extract dates
        dates_df = master_df[['date']].copy()

        # Convert to datetime then to string format for API
        dates_df['date'] = pd.to_datetime(dates_df['date'])
        dates_df['date'] = dates_df['date'].dt.strftime('%Y-%m-%d')

        print(f"Found {len(dates_df)} dates to process for {symbol}")
        print(f"Date range: {dates_df['date'].min()} to {dates_df['date'].max()}")

        # Call the function to fetch and save CP ratios with sentiments
        # This will create {symbol}-cp_ratios_sentiment_w_volume.csv
        result_df = fetchBulkData.get_historical_cp_ratios_with_sentiments_new(
            symbol, dates_df, api_key
        )

        print(f"\n✓ Successfully created {symbol}-cp_ratios_sentiment_w_volume.csv")
        print(f"  Rows: {len(result_df)}")

    except Exception as e:
        print(f"\n✗ Error processing {symbol}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*60}")
print("CP ratio file creation complete!")
print(f"{'='*60}\n")
