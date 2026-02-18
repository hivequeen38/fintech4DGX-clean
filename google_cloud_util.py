from google.cloud import storage
from google.oauth2 import service_account

key_path = "sentiment-412417-27b17b73abd5.json"


def upload_file_to_bucket(bucket_name: str, file_path: str)-> str:
    # Create a service account credential
    credentials = service_account.Credentials.from_service_account_file(
        key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    # Create a client
    client = storage.Client(credentials=credentials, project=credentials.project_id)

    # Define your bucket name and file details
    # bucket_name = 'sentiment-report'
    destination_blob_name = file_path
    source_file_name = file_path

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Create a blob and upload the file
    blob = bucket.blob(destination_blob_name)
    blob.cache_control = 'no-cache, max-age=0'
    blob.upload_from_filename(source_file_name)
    blob.patch()  # Apply the cache_control metadata update

    # Get the public URL
    public_url = blob.public_url

    print(f"The file is uploaded and publicly accessible at: {public_url}")
    return public_url

