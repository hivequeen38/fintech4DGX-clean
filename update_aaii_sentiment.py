#!/usr/bin/env python3
"""
Automatically fetch and update AAII Sentiment Survey data
Run this script weekly (Thursday nights) to update sentiment.csv with the latest AAII data

AAII publishes their Investor Sentiment Survey every Thursday after market close.
This survey shows the percentage of individual investors who are bullish, bearish,
and neutral on the stock market for the next six months.
"""

import pandas as pd
import requests
from datetime import datetime
import os
import sys
from bs4 import BeautifulSoup
import re

# AAII data sources (will try multiple sources)
AAII_URLS = [
    'https://www.aaii.com/files/surveys/sentiment.csv',
    'https://www.aaii.com/sentimentsurvey',  # Main sentiment survey page
]
LOCAL_FILE = 'sentiment.csv'
BACKUP_FILE = 'sentiment_backup.csv'


def scrape_aaii_webpage():
    """Scrape AAII sentiment data from their webpage"""
    print("Attempting to scrape AAII sentiment survey webpage...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }

    try:
        response = requests.get('https://www.aaii.com/sentimentsurvey', headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to find the latest sentiment data in the page
        # This is a best-effort approach and may need adjustment based on AAII's page structure
        print("⚠️  Web scraping from AAII webpage is not fully implemented yet.")
        print("   The AAII website structure needs to be analyzed to extract the data.")
        return None

    except Exception as e:
        print(f"✗ Error scraping AAII webpage: {e}")
        return None


def fetch_aaii_data_csv():
    """Fetch the latest AAII sentiment data from their CSV file"""
    print("Fetching AAII sentiment data from CSV...")
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/csv,text/plain,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.aaii.com/',
        }
        response = requests.get(AAII_URLS[0], headers=headers, timeout=30)
        response.raise_for_status()

        # Save to temporary file to parse
        temp_file = 'sentiment_temp.csv'
        with open(temp_file, 'w') as f:
            f.write(response.text)

        # Read the CSV
        df = pd.read_csv(temp_file)

        # Clean up temp file
        os.remove(temp_file)

        print(f"✓ Successfully fetched {len(df)} records from AAII CSV")
        return df

    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching CSV from AAII: {e}")
        return None
    except Exception as e:
        print(f"✗ Error processing AAII CSV: {e}")
        return None


def manual_input_sentiment():
    """Allow manual input of sentiment data"""
    print()
    print("="*60)
    print("Manual Sentiment Data Entry")
    print("="*60)
    print()
    print("Please enter the latest AAII sentiment survey data.")
    print("You can find this at: https://www.aaii.com/sentimentsurvey")
    print()

    try:
        date_str = input("Enter date (YYYY-MM-DD or MM-DD-YY): ").strip()
        bullish = float(input("Enter Bullish %: ").strip())
        bearish = float(input("Enter Bearish %: ").strip())
        spread = bullish - bearish

        # Create a dataframe with this single record
        df = pd.DataFrame({
            'date': [date_str],
            'Bullish': [bullish],
            'Bearish': [bearish],
            'Spread': [spread]
        })

        print()
        print(f"✓ Entered: {date_str} - Bullish={bullish}%, Bearish={bearish}%, Spread={spread:.1f}%")
        return df

    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        return None
    except Exception as e:
        print(f"\n✗ Error in manual input: {e}")
        return None


def fetch_aaii_data():
    """Fetch the latest AAII sentiment data - tries multiple methods"""
    # Try CSV first
    df = fetch_aaii_data_csv()
    if df is not None:
        return df

    # Try web scraping
    df = scrape_aaii_webpage()
    if df is not None:
        return df

    # If all automated methods fail, offer manual input
    print()
    print("⚠️  Automated data fetching failed.")
    print()
    response = input("Would you like to enter the data manually? (Y/N): ").strip().upper()
    if response == 'Y':
        return manual_input_sentiment()

    return None


def load_local_sentiment():
    """Load the local sentiment.csv file"""
    if not os.path.exists(LOCAL_FILE):
        print(f"✗ Local file {LOCAL_FILE} not found!")
        return None

    try:
        df = pd.read_csv(LOCAL_FILE)
        print(f"✓ Loaded {len(df)} records from local file")
        return df
    except Exception as e:
        print(f"✗ Error reading local file: {e}")
        return None


def parse_date(date_str):
    """Parse date from various formats"""
    # Try different date formats
    formats = ['%m-%d-%y', '%Y-%m-%d', '%m/%d/%y', '%m/%d/%Y']

    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue

    # Last resort: let pandas figure it out
    try:
        return pd.to_datetime(date_str)
    except:
        return None


def normalize_dataframe(df):
    """Normalize the dataframe format"""
    # Ensure column names are correct
    df.columns = df.columns.str.strip()

    # Convert date column
    if 'date' not in df.columns and 'Date' in df.columns:
        df.rename(columns={'Date': 'date'}, inplace=True)

    # Parse dates
    df['date'] = df['date'].apply(parse_date)

    # Drop rows with invalid dates
    df = df.dropna(subset=['date'])

    # Sort by date
    df = df.sort_values('date')

    # Ensure required columns exist
    required_cols = ['date', 'Bullish', 'Bearish', 'Spread']
    for col in required_cols:
        if col not in df.columns:
            print(f"⚠️  Warning: Column '{col}' not found in dataframe")

    return df


def merge_sentiment_data(local_df, remote_df):
    """Merge local and remote data, keeping only new records"""
    # Normalize both dataframes
    local_df = normalize_dataframe(local_df.copy())
    remote_df = normalize_dataframe(remote_df.copy())

    # Find the latest date in local file
    latest_local_date = local_df['date'].max()
    print(f"Latest date in local file: {latest_local_date.strftime('%Y-%m-%d')}")

    # Filter remote data for new records only
    new_records = remote_df[remote_df['date'] > latest_local_date]

    if len(new_records) == 0:
        print("✓ No new records found. Local file is up to date.")
        return local_df, 0

    print(f"Found {len(new_records)} new record(s):")
    for idx, row in new_records.iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d')}: Bullish={row['Bullish']:.1f}%, Bearish={row['Bearish']:.1f}%, Spread={row['Spread']:.1f}")

    # Concatenate the dataframes
    merged_df = pd.concat([local_df, new_records], ignore_index=True)

    # Sort by date
    merged_df = merged_df.sort_values('date')

    # Remove duplicates (keeping the last occurrence)
    merged_df = merged_df.drop_duplicates(subset=['date'], keep='last')

    return merged_df, len(new_records)


def save_sentiment_data(df, backup=True):
    """Save the sentiment data to file"""
    # Create backup if requested
    if backup and os.path.exists(LOCAL_FILE):
        try:
            import shutil
            shutil.copy2(LOCAL_FILE, BACKUP_FILE)
            print(f"✓ Created backup: {BACKUP_FILE}")
        except Exception as e:
            print(f"⚠️  Warning: Could not create backup: {e}")

    # Format dates consistently
    df_to_save = df.copy()
    df_to_save['date'] = pd.to_datetime(df_to_save['date']).dt.strftime('%Y-%m-%d')

    # Save to CSV
    try:
        df_to_save.to_csv(LOCAL_FILE, index=False)
        print(f"✓ Saved {len(df_to_save)} records to {LOCAL_FILE}")
        return True
    except Exception as e:
        print(f"✗ Error saving file: {e}")
        return False


def main():
    """Main function"""
    print("="*60)
    print("AAII Sentiment Survey Data Updater")
    print("="*60)
    print()

    # Load local data
    local_df = load_local_sentiment()
    if local_df is None:
        print("Aborting: Cannot proceed without local file")
        sys.exit(1)

    # Fetch remote data
    remote_df = fetch_aaii_data()
    if remote_df is None:
        print("Aborting: Cannot fetch data from AAII")
        sys.exit(1)

    print()

    # Merge the data
    merged_df, new_count = merge_sentiment_data(local_df, remote_df)

    print()

    # Save if there are new records
    if new_count > 0:
        if save_sentiment_data(merged_df, backup=True):
            print()
            print(f"✅ Successfully updated {LOCAL_FILE} with {new_count} new record(s)")
        else:
            print()
            print("❌ Failed to save updated data")
            sys.exit(1)
    else:
        print()
        print("✅ No updates needed - your sentiment.csv is already up to date!")

    print()
    print("="*60)


if __name__ == "__main__":
    main()
