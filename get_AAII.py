import imaplib
import email
import email.header
import re
from datetime import datetime

def get_aaii_sentiment():
    """
    Connects to IMAP, finds the most recent email from noreply@tm.openai.com
    with subject starting with "AAII sentiment", extracts Bull and Bear percentages.
    
    Returns a tuple of (bull_value, bear_value) or (None, None) if not found.
    """
    # Configuration
    IMAP_SERVER = 'imap.gmail.com'
    EMAIL_ACCOUNT = 'milton.soong@gmail.com'
    EMAIL_PASSWORD = 'erovqdtudvngxkbr'  # or normal password if no 2FA
    
    # Email search criteria
    EXPECTED_SENDER = "noreply@tm.openai.com"
    EXPECTED_SUBJECT_KEYWORDS = ["AAII"]  # More flexible, just look for AAII
    
    # Two regex patterns - one for bull-first, one for bear-first format
    BULL_FIRST_PATTERN = r"(?:bull(?:ish)?\s+|aaii\s+bull(?:ish)?\s+)(\d+(?:\.\d+)?)\s+(?:bear(?:ish)?\s+|aaii\s+bear(?:ish)?\s+)(\d+(?:\.\d+)?)"
    BEAR_FIRST_PATTERN = r"(?:bear(?:ish)?\s+|aaii\s+bear(?:ish)?\s+)(\d+(?:\.\d+)?)\s+(?:bull(?:ish)?\s+|aaii\s+bull(?:ish)?\s+)(\d+(?:\.\d+)?)"
    
    # 1. Connect to IMAP
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
    mail.select("INBOX")
    
    # 2. Search for today's email with AAII in the subject
    today_str = datetime.now().strftime("%d-%b-%Y")
    search_criteria = f'(FROM "{EXPECTED_SENDER}" SUBJECT "AAII" SENTON "{today_str}")'
    status, email_ids = mail.search(None, search_criteria)
    
    if status != "OK" or not email_ids or email_ids[0] == b'':
        print(f"No matching emails found for today ({today_str}).")
        
        # Try searching without the date constraint as fallback, but still with AAII in subject
        fallback_criteria = f'(FROM "{EXPECTED_SENDER}" SUBJECT "AAII")'
        status, email_ids = mail.search(None, fallback_criteria)
        
        if status != "OK" or not email_ids or email_ids[0] == b'':
            print("No matching emails found at all.")
            mail.logout()
            return None, None
        print("Found matching emails without date constraint, using the most recent one.")
    
    # 3. Get the latest matching email
    id_list = email_ids[0].split()
    if not id_list:
        print("No matching emails found with AAII in subject.")
        mail.logout()
        return None, None
        
    latest_email_id = id_list[-1]  # Get the most recent email
    
    # 4. Fetch the email content
    status, msg_data = mail.fetch(latest_email_id, "(RFC822)")
    if status != "OK":
        print("Failed to fetch the email content.")
        mail.logout()
        return None, None
    
    # 5. Parse the email
    raw_email = msg_data[0][1]
    msg = email.message_from_bytes(raw_email)
    
    # Get email date for reference
    email_date = msg.get("Date")
    print(f"Processing email from: {email_date}")
    
    # 6. Extract the subject and body
    raw_subject = msg.get("Subject", "")
    print(f"Raw email subject: {raw_subject}")
    
    # Decode the subject properly in case it's encoded
    subject = email.header.decode_header(raw_subject)
    subject = ' '.join([str(text, charset or 'utf-8') if isinstance(text, bytes) else text for text, charset in subject])
    print(f"Decoded email subject: {subject}")
    
    # Check if this is indeed an AAII-related email - more flexible now
    if not any(keyword in subject for keyword in EXPECTED_SUBJECT_KEYWORDS):
        print(f"Email subject doesn't contain expected keywords: {subject}")
        mail.logout()
        return None, None
    
    # 7. Extract the email body
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition") or "")
            
            # Skip attachments
            if "attachment" in content_disposition:
                continue
                
            # Get text content
            if content_type == "text/plain" or content_type == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    body += payload.decode(part.get_charset() or 'utf-8', errors='replace')
    else:
        # Not multipart - just get the payload
        payload = msg.get_payload(decode=True)
        if payload:
            body += payload.decode(msg.get_charset() or 'utf-8', errors='replace')
    
    # 8. Look for sentiment values in the subject first using bull-first pattern
    subject_match = re.search(BULL_FIRST_PATTERN, subject, re.IGNORECASE)
    if subject_match:
        bull_value = float(subject_match.group(1))
        bear_value = float(subject_match.group(2))
        print(f"Found bull-first in subject - Bull: {bull_value}%, Bear: {bear_value}%")
        mail.logout()
        return bull_value, bear_value
    
    # 9. If not found, try bear-first pattern in subject
    subject_match = re.search(BEAR_FIRST_PATTERN, subject, re.IGNORECASE)
    if subject_match:
        bear_value = float(subject_match.group(1))
        bull_value = float(subject_match.group(2))
        print(f"Found bear-first in subject - Bull: {bull_value}%, Bear: {bear_value}%")
        mail.logout()
        return bull_value, bear_value
    
    # 10. Look for sentiment values in the body using bull-first pattern
    body_match = re.search(BULL_FIRST_PATTERN, body, re.IGNORECASE)
    if body_match:
        bull_value = float(body_match.group(1))
        bear_value = float(body_match.group(2))
        print(f"Found bull-first in body - Bull: {bull_value}%, Bear: {bear_value}%")
        mail.logout()
        return bull_value, bear_value
    
    # 11. If not found, try bear-first pattern in body
    body_match = re.search(BEAR_FIRST_PATTERN, body, re.IGNORECASE)
    if body_match:
        bear_value = float(body_match.group(1))
        bull_value = float(body_match.group(2))
        print(f"Found bear-first in body - Bull: {bull_value}%, Bear: {bear_value}%")
        mail.logout()
        return bull_value, bear_value
    
    # 10. If not found, print some debugging info and return None
    print("Could not find AAII sentiment values in the email.")
    print(f"Subject for reference: {subject}")
    # Print first 100 chars of body for debugging
    print(f"Body excerpt: {body[:100]}...")
    mail.logout()
    return None, None

def update_sentiment_csv(bull_value, bear_value):
    """
    Updates the sentiment.csv file with the latest bull/bear values.
    
    Args:
        bull_value (float): The bull percentage value
        bear_value (float): The bear percentage value
    
    Returns:
        bool: True if successful, False otherwise
    """
    import csv
    import os
    from datetime import datetime
    
    # Calculate spread
    spread_value = bull_value - bear_value
    
    # Get today's date in YYYY-MM-DD format
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    # Path to the CSV file
    csv_path = "sentiment.csv"
    
    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    try:
        # If file doesn't exist, create it with headers
        if not file_exists:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['date', 'bullish', 'bearish', 'spread'])
        
        # Append the new data
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([today_date, bull_value, bear_value, spread_value])
        
        print(f"Successfully updated {csv_path} with new data:")
        print(f"Date: {today_date}, Bull: {bull_value}%, Bear: {bear_value}%, Spread: {spread_value}%")
        
        return True
    
    except Exception as e:
        print(f"Error updating CSV file: {e}")
        return False

if __name__ == "__main__":
    bull, bear = get_aaii_sentiment()
    if bull is not None and bear is not None:
        print(f"AAII Sentiment: Bull {bull}%, Bear {bear}%")
        print(f"Spread: {bull - bear}%")
        
        # Update the CSV file
        update_sentiment_csv(bull, bear)
    else:
        print("Failed to retrieve AAII sentiment values.")