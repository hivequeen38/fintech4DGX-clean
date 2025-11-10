from google.cloud import storage
from datetime import datetime, timedelta
import os

# Set up your Google Cloud credentials
# Path to your service account key file
def cleanup_old_report():
    key_path = "sentiment-412417-27b17b73abd5.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

    # Initialize a client
    storage_client = storage.Client()

    # Define your bucket name
    bucket_name = 'sentiment-report'

    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)

    # Calculate the date 10 days ago
    ten_days_ago = datetime.utcnow() - timedelta(days=7)

    # List all blobs in the bucket
    blobs = bucket.list_blobs()

    # Loop through each blob
    for blob in blobs:
        # Convert blob time to a datetime object
        blob_time = blob.time_created.replace(tzinfo=None)
        
        # Check if the blob is older than 10 days
        if blob_time < ten_days_ago:
            # Delete the blob
            print(f"Deleting {blob.name}...")
            blob.delete()
    print("Deletion complete.")

# a main entry to call directly to cleanup stuff
cleanup_old_report()
