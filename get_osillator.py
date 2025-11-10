import imaplib
import email
import re
import io
from datetime import datetime

# You need PyPDF2 installed: pip install PyPDF2
import PyPDF2

# --------------- CONFIGURATION -----------------
IMAP_SERVER = 'imap.gmail.com'
EMAIL_ACCOUNT = 'milton.soong@gmail.com'
EMAIL_PASSWORD = 'erovqdtudvngxkbr'  # or normal password if no 2FA

# Example subject & sender you expect for the oscillator email
EXPECTED_SENDER = "milton.soong@gmail.com"

# Regex to find your oscillator reading
# For instance, look for a plus/minus number with optional decimal digits
# OSCILLATOR_PATTERN = r"([+-]?\d+(\.\d+)?)"
OSCILLATOR_PATTERN = r"S&P Short Range Oscillator.*?:\s*([+-]?\d+(?:\.\d+)?)%"

# --------------- END CONFIG --------------------




def get_latest_pdf_email_oscillator(input_date_str: str):
    """
    Connects to IMAP, finds the most recent email from EXPECTED_SENDER & SUBJECT 
    dated 'today', downloads the PDF attachment, parses out the oscillator value.
    
    Returns the float oscillator value or None if not found.
    """

    # 1. Connect to IMAP
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
    mail.select("INBOX")

    # 2. Search for today's email by subject/sender
    # IMAP search format:
    #   (FROM "somebody@domain.com" SUBJECT "subject text" ON "DD-MMM-YYYY")
    # Adjust or remove ON if needed.
    status, email_ids = mail.search(None, "(SUBJECT \"Short Range Oscillator\")")
    if status != "OK" or not email_ids or email_ids[0] == b'':
        print("No matching emails found for today.")
        mail.logout()
        return None

    # 3. Get the latest matching email (by ID)
    id_list = email_ids[0].split()
    latest_email_id = id_list[-1]

    # 4. Fetch the raw email
    status, msg_data = mail.fetch(latest_email_id, "(RFC822)")
    if status != "OK":
        print("Failed to fetch the email content.")
        mail.logout()
        return None

    raw_email = msg_data[0][1]
    msg = email.message_from_bytes(raw_email)

    # 5. Locate the PDF attachment & parse
    oscillator_value = None

    if msg.is_multipart():
        # Walk through the parts of the email to find the PDF
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition") or "")
            
            # Check if this part is a PDF attachment
            if content_type == "application/pdf" or "pdf" in content_disposition.lower():
                # Get the raw bytes of the attachment
                pdf_data = part.get_payload(decode=True)
                
                # Parse the PDF to extract text
                extracted_text = extract_text_from_pdf_bytes(pdf_data)
                
                # Use a regex to find the oscillator value in the text
                # match = re.search(OSCILLATOR_PATTERN, extracted_text)
                # The date is appearing between "Oscillator" and the value, so we need to account for that
                # pattern = r"S&P Short Range Oscillator\s+(?:[\w\s,]+?):\s*([+-]?\d+(?:\.\d+)?)%"
                # match = re.search(pattern, extracted_text, flags=re.S)
                # if match:
                #     oscillator_value = float(match.group(1))
                #     print("Found oscillator value in PDF:", oscillator_value)
                pattern = r"S&P Short Range Oscillator\s+((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}):\s*([+-]?\d+(?:\.\d+)?)%"
                match = re.search(pattern, extracted_text, flags=re.S)

                if match:
                    date_str = match.group(1)  # e.g. "December 19, 2024"
                    oscillator_value = float(match.group(2))
                    
                    # Convert date to YYYY-MM-DD format
                    date_obj = datetime.strptime(date_str, "%B %d, %Y")
                    formatted_date = date_obj.strftime("%Y-%m-%d")
                    
                    print("Date:", formatted_date)
                    print("Found oscillator value in PDF:", oscillator_value)
                    break
    else:
        # If not multipart, the email might have no attachment or a different structure.
        print("Email is not multipart or no PDF found.")
    
    mail.logout()
    return oscillator_value, formatted_date

def get_specific_date_pdf_email_oscillator(input_date_str: str):
    """
    Connects to IMAP, searches emails with oscillator PDFs, and finds the oscillator value 
    for the specific input date (YYYY-MM-DD format).
    
    Returns tuple of (oscillator_value, formatted_date) or (None, None) if not found.
    """
    # Convert input_date_str to the format used in PDF for matching
    try:
        date_obj = datetime.strptime(input_date_str, "%Y-%m-%d")
        target_date_str = date_obj.strftime("%B %d, %Y")
    except ValueError:
        print(f"Invalid date format for {input_date_str}. Use YYYY-MM-DD format.")
        return None, None

    # 1. Connect to IMAP
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
    mail.select("INBOX")

    # 2. Search for emails by subject
    status, email_ids = mail.search(None, "(SUBJECT \"Short Range Oscillator\")")
    if status != "OK" or not email_ids or email_ids[0] == b'':
        print("No matching emails found.")
        mail.logout()
        return None, None

    # Get all email IDs and process them until we find the matching date
    id_list = email_ids[0].split()
    
    # Process emails from newest to oldest
    for email_id in reversed(id_list):
        status, msg_data = mail.fetch(email_id, "(RFC822)")
        if status != "OK":
            continue

        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition") or "")
                
                if content_type == "application/pdf" or "pdf" in content_disposition.lower():
                    pdf_data = part.get_payload(decode=True)
                    extracted_text = extract_text_from_pdf_bytes(pdf_data)
                    
                    # Create pattern using the target date
                    pattern = f"S&P Short Range Oscillator\\s+{target_date_str}:\\s*([+-]?\\d+(?:\\.\\d+)?)%"
                    match = re.search(pattern, extracted_text, flags=re.S)

                    if match:
                        oscillator_value = float(match.group(1))
                        print(f"Found oscillator value for {input_date_str}: {oscillator_value}")
                        mail.logout()
                        return oscillator_value, input_date_str

    print(f"No oscillator value found for date: {input_date_str}")
    mail.logout()
    return None, None

def extract_text_from_pdf_bytes(pdf_bytes):
    """
    Uses PyPDF2 to extract text from in-memory PDF data.
    Returns the extracted text as a single string.
    """
    pdf_text = []
    with io.BytesIO(pdf_bytes) as pdf_file_obj:
        reader = PyPDF2.PdfReader(pdf_file_obj)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pdf_text.append(page_text)
    return "\n".join(pdf_text)

def get_oscillator_old():
    today_str = datetime.now().strftime("%d-%b-%Y")
    osc_value, dateStr = get_latest_pdf_email_oscillator(today_str)
    if osc_value is not None:
        print(f"Today's oscillator value: {osc_value}")
        print(f"Today's date: {dateStr}")
    else:
        print("Could not retrieve the oscillator value from today's email.")
    return osc_value, dateStr

def get_oscillator(input_date_str: str = None):
    if input_date_str is not None:
        oscillator_value, found_date = get_specific_date_pdf_email_oscillator(input_date_str)
    else :
        today_str = datetime.now().strftime("%d-%b-%Y")
        oscillator_value, found_date = get_latest_pdf_email_oscillator(today_str)

    if oscillator_value is not None:
        print(f"Oscillator value for {found_date}: {oscillator_value}")
    else:
        print("No matching oscillator value found")
    return oscillator_value

if __name__ == "__main__":
    get_oscillator()
    
