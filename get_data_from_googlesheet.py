import pandas as pd
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os.path
import pickle

def get_google_sheet_data(spreadsheet_id, range_name):
    """
    Fetches data from a Google Sheet.
    
    Args:
        spreadsheet_id: The ID of the spreadsheet (found in the URL).
        range_name: The range to retrieve (e.g., 'Sheet1!A1:E10').
    
    Returns:
        DataFrame containing the sheet data.
    """
    # Define the scopes
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    
    creds = None
    # The file token.pickle stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If credentials don't exist or are invalid, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret_993570327918-efrut96fggfsbl9cnuspqlheg1439o24.apps.googleusercontent.com.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    # Build the service
    service = build('sheets', 'v4', credentials=creds)
    
    # Call the Sheets API
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id,
                                range=range_name).execute()
    values = result.get('values', [])
    
    if not values:
        print('No data found.')
        return pd.DataFrame()
    
    # Fix for mismatched columns
    headers = values[0]
    data_rows = values[1:]
    
    # Create a list to store processed rows
    processed_rows = []
    
    # Process each data row to match the header length
    for row in data_rows:
        # Extend shorter rows with None values to match header length
        if len(row) < len(headers):
            row = row + [None] * (len(headers) - len(row))
        # Truncate longer rows to match header length
        elif len(row) > len(headers):
            row = row[:len(headers)]
        processed_rows.append(row)
    
    # Create DataFrame with properly aligned columns
    df = pd.DataFrame(processed_rows, columns=headers)
    
    # Convert numeric columns to proper data types
    for col in df.columns:
        # Try to convert to numeric, ignore errors for non-numeric columns
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
    return df

# Example usage
if __name__ == "__main__":
    # Replace with your Google Sheet ID and range
    # The spreadsheet ID is the part of the URL after /d/ and before /edit
    # For example: https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit
    # The spreadsheet ID would be: 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms
    spreadsheet_id = "17nM1b4K6V1al_z-YK1bq38cfzZZRvhfXJNQtuEVugxU"
    range_name = "Sheet1!A1:DM2"  # Adjust this according to your needs
    
    # Get the data
    df = get_google_sheet_data(spreadsheet_id, range_name)
    
    # Now you can work with the data as you would with any pandas DataFrame
    print(df.head())
    df.to_csv('debug_fetched_googlesheet.csv', index=False)

     # Ensure the 'Spread' column is numeric before calculating mean
    if 'Spread' in df.columns:
        df['Spread'] = pd.to_numeric(df['Spread'], errors='coerce')
        print("sentiment spread =", df['Spread'].mean())
    
    print("cp sentiment ratio =" , df['cp_sentiment_ratio'].mean())
    
    # Continue with your training or inference...