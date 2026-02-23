#!/usr/bin/env python3
"""
Add weekly AAII Sentiment Survey data to sentiment.csv

Simple script to manually add the latest AAII sentiment data every Thursday night.
Visit https://www.aaii.com/sentimentsurvey to get the latest numbers.

The AAII Investor Sentiment Survey is published every Thursday after market close.
"""

import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import shutil

LOCAL_FILE = 'sentiment.csv'
BACKUP_FILE = 'sentiment_backup.csv'


def get_latest_thursday():
    """Get the date of the most recent Thursday"""
    today = datetime.now()
    # Thursday is weekday 3 (Monday=0)
    days_since_thursday = (today.weekday() - 3) % 7
    if days_since_thursday == 0 and today.hour < 16:
        # If it's Thursday but before 4pm, use last week's Thursday
        days_since_thursday = 7
    last_thursday = today - timedelta(days=days_since_thursday)
    return last_thursday.strftime('%Y-%m-%d')


def load_sentiment_file():
    """Load the sentiment.csv file"""
    if not os.path.exists(LOCAL_FILE):
        print(f"✗ Error: {LOCAL_FILE} not found!")
        return None

    try:
        df = pd.read_csv(LOCAL_FILE)
        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date')
        return df
    except Exception as e:
        print(f"✗ Error reading {LOCAL_FILE}: {e}")
        return None


def create_backup():
    """Create a backup of the sentiment file"""
    if os.path.exists(LOCAL_FILE):
        try:
            shutil.copy2(LOCAL_FILE, BACKUP_FILE)
            print(f"✓ Created backup: {BACKUP_FILE}")
            return True
        except Exception as e:
            print(f"⚠️  Warning: Could not create backup: {e}")
            return False
    return True


def add_sentiment_entry(df, date_str, bullish, bearish, neutral=None):
    """Add a new sentiment entry to the dataframe"""
    try:
        # Parse the date
        entry_date = pd.to_datetime(date_str)

        # Calculate spread
        spread = bullish - bearish

        # Check if entry already exists
        if entry_date in df['date'].values:
            print(f"\n⚠️  Warning: An entry for {entry_date.strftime('%Y-%m-%d')} already exists!")
            existing = df[df['date'] == entry_date].iloc[0]
            print(f"   Existing: Bullish={existing['Bullish']:.1f}%, Bearish={existing['Bearish']:.1f}%, Spread={existing['Spread']:.1f}")
            print(f"   New:      Bullish={bullish:.1f}%, Bearish={bearish:.1f}%, Spread={spread:.1f}")
            response = input("\n   Overwrite existing entry? (Y/N): ").strip().upper()
            if response != 'Y':
                print("   Cancelled - keeping existing entry")
                return df, False

            # Remove existing entry
            df = df[df['date'] != entry_date]

        # Create new row
        new_row = pd.DataFrame({
            'date': [entry_date],
            'Bullish': [bullish],
            'Bearish': [bearish],
            'Spread': [spread]
        })

        # Append and sort
        df = pd.concat([df, new_row], ignore_index=True)
        df = df.sort_values('date')

        print(f"\n✓ Added: {entry_date.strftime('%Y-%m-%d')} - Bullish={bullish:.1f}%, Bearish={bearish:.1f}%, Spread={spread:.1f}")

        return df, True

    except Exception as e:
        print(f"\n✗ Error adding entry: {e}")
        return df, False


def save_sentiment_file(df):
    """Save the updated sentiment file"""
    try:
        # Format dates as YYYY-MM-DD
        df_to_save = df.copy()
        df_to_save['date'] = df_to_save['date'].dt.strftime('%Y-%m-%d')

        # Save to CSV
        df_to_save.to_csv(LOCAL_FILE, index=False)
        print(f"\n✓ Saved {len(df_to_save)} records to {LOCAL_FILE}")
        return True
    except Exception as e:
        print(f"\n✗ Error saving file: {e}")
        return False


def main():
    """Main function"""
    print("="*60)
    print("AAII Sentiment Survey - Weekly Update")
    print("="*60)
    print()

    # Load existing data
    df = load_sentiment_file()
    if df is None:
        sys.exit(1)

    # Show current status
    latest_date = df['date'].max()
    latest_entry = df[df['date'] == latest_date].iloc[0]
    print(f"Current latest entry: {latest_date.strftime('%Y-%m-%d')}")
    print(f"  Bullish: {latest_entry['Bullish']:.2f}%")
    print(f"  Bearish: {latest_entry['Bearish']:.2f}%")
    print(f"  Spread:  {latest_entry['Spread']:.2f}")
    print()

    # Suggest this week's Thursday
    suggested_date = get_latest_thursday()
    print(f"Suggested date for this week: {suggested_date}")
    print()
    print("Get the latest data from: https://www.aaii.com/sentimentsurvey")
    print()

    # Get user input
    try:
        print("Enter new AAII sentiment data:")
        print("-" * 40)

        # Date input with default
        date_input = input(f"Date [{suggested_date}]: ").strip()
        date_str = date_input if date_input else suggested_date

        # Bullish percentage
        bullish_str = input("Bullish %: ").strip()
        bullish = float(bullish_str)

        # Bearish percentage
        bearish_str = input("Bearish %: ").strip()
        bearish = float(bearish_str)

        # Optional: Neutral percentage (for validation)
        neutral_str = input("Neutral % (optional): ").strip()
        if neutral_str:
            neutral = float(neutral_str)
            total = bullish + bearish + neutral
            if abs(total - 100.0) > 0.5:
                print(f"\n⚠️  Warning: Bullish + Bearish + Neutral = {total:.1f}% (should be ~100%)")
                response = input("Continue anyway? (Y/N): ").strip().upper()
                if response != 'Y':
                    print("Cancelled")
                    sys.exit(0)

        print()

        # Create backup
        create_backup()

        # Add the entry
        df, success = add_sentiment_entry(df, date_str, bullish, bearish)

        if success:
            # Save the file
            if save_sentiment_file(df):
                print()
                print("="*60)
                print("✅ Successfully updated sentiment.csv")
                print("="*60)
            else:
                print()
                print("❌ Failed to save the updated file")
                print(f"Your backup is safe at: {BACKUP_FILE}")
                sys.exit(1)
        else:
            print("\nNo changes made to sentiment.csv")

    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(0)
    except ValueError as e:
        print(f"\n✗ Invalid input: {e}")
        print("Please enter numeric values for percentages")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
