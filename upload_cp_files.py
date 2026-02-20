#!/usr/bin/env python3
"""
Upload CP ratio files to Google Cloud Storage
Run this on the machine that HAS the files
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

# Find all CP ratio files
import glob
cp_files = glob.glob('*-cp_ratios_sentiment_w_volume.csv')

if not cp_files:
    print("No CP ratio files found in current directory!")
    print("Make sure you're in the directory with the files")
else:
    print(f"Found {len(cp_files)} files to upload:")
    for f in cp_files:
        print(f"  - {f}")

    print("\nUploading to Google Cloud Storage...")
    for filename in cp_files:
        blob_name = f'cp_ratio_files/{filename}'
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(filename)
        print(f"✓ Uploaded {filename} -> gs://{bucket_name}/{blob_name}")

    print("\n✅ All files uploaded successfully!")
