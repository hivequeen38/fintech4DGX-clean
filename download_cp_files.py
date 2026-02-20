#!/usr/bin/env python3
"""
Download CP ratio files from Google Cloud Storage
Run this on the machine that NEEDS the files
"""

import os
from google.cloud import storage
from google.oauth2 import service_account

# Service account credentials
key_path = "sentiment-412417-27b17b73abd5.json"
credentials = service_account.Credentials.from_service_account_file(
    key_path,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

# Initialize the client with credentials
client = storage.Client(credentials=credentials, project=credentials.project_id)

# Your bucket name (same as used in get_historical_html.py)
bucket_name = 'ml-prediction-results'
bucket = client.bucket(bucket_name)

# List all CP ratio files in the bucket
print("Checking Google Cloud Storage for CP ratio files...")
blobs = list(bucket.list_blobs(prefix='cp_ratio_files/'))

if not blobs:
    print("No CP ratio files found in Google Cloud Storage!")
    print("Make sure you've uploaded them first from the other machine")
else:
    print(f"\nFound {len(blobs)} files to download:")
    for blob in blobs:
        print(f"  - {blob.name}")

    print("\nDownloading files...")
    for blob in blobs:
        # Extract just the filename (remove the cp_ratio_files/ prefix)
        filename = blob.name.split('/')[-1]

        if filename:  # Skip if it's just the folder
            blob.download_to_filename(filename)
            print(f"✓ Downloaded {filename}")

    print("\n✅ All files downloaded successfully!")

    # Verify the files
    print("\nVerifying downloaded files:")
    import glob
    local_files = glob.glob('*-cp_ratios_sentiment_w_volume.csv')
    for f in local_files:
        size = os.path.getsize(f)
        print(f"  {f}: {size:,} bytes")
