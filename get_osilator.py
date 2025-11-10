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
EMAIL_PASSWORD = 'erov qdtu dvng xkbr'  # or normal password if no 2FA

# Example subject & sender you expect for the oscillator email
EXPECTED_SUBJECT = "S&P Short Range Oscillator"
EXPECTED_SENDER = "milton.soong@gmail.com"

# Regex to find your oscillator reading
# For instance, look for a plus/minus number with optional decimal digits
OSCILLATOR_PATTERN = r"([+-]?\d+(\.\d+)?)"

# --------------- END CONFIG --------------------




def get_latest_pdf_email_oscillator():
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
    search_criteria = f'(FROM "{EXPECTED_SENDER}" SUBJECT "{EXPECTED_SUBJECT}" ON "{today_str}")'
    status, email_ids = mail.search(None, search_criteria)
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
                match = re.search(OSCILLATOR_PATTERN, extracted_text)
                if match:
                    oscillator_value = float(match.group(1))
                    print("Found oscillator value in PDF:", oscillator_value)
                    break
    else:
        # If not multipart, the email might have no attachment or a different structure.
        print("Email is not multipart or no PDF found.")
    
    mail.logout()
    return oscillator_value


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


if __name__ == "__main__":
    osc_value = get_latest_pdf_email_oscillator()
    if osc_value is not None:
        print(f"Today's oscillator value: {osc_value}")
    else:
        print("Could not retrieve the oscillator value from today's email.")

