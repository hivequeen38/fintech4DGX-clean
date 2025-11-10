from datetime import datetime
import csv
import imaplib
import email
import re
import io
import PyPDF2
from typing import List, Tuple
import get_osillator

def process_prediction_files(stock_symbols: List[str], date_str: str, oscillator_value: float = None) -> str:
    """
    Process prediction files for given stock symbols and consolidate them into a single CSV.
    Now includes oscillator value at the bottom of the report.
    
    Args:
        stock_symbols (List[str]): List of stock symbols to process
        date_str (str): Date string to filter predictions (format: YYYY-MM-DD)
        oscillator_value (float, optional): Oscillator value to append to report
    
    Returns:
        str: Name of the output consolidated file
    """
    
    # Store the consolidated output
    consolidated_rows = []
    
    for symbol in stock_symbols:
        filename = f"{symbol}_15d_from_today_predictions.csv"
        
        try:
            # Read and process the file
            with open(filename, 'r') as file:
                if consolidated_rows:  # Only add blank line if not the first symbol
                    consolidated_rows.append([])  # Empty row for carriage return
                consolidated_rows.append([f"<{symbol}>"])  # This will be the separator row
                
                # Read and filter rows
                for line in file:
                    # Split the line by comma
                    row = line.strip().split(',')
                    
                    # Check if this row is for today
                    if row[0] == date_str:
                        consolidated_rows.append(row)
                        
        except FileNotFoundError:
            print(f"Warning: File not found for symbol {symbol}")
            continue
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    # Add oscillator value at the bottom if available
    if oscillator_value is not None:
        consolidated_rows.append([])  # Empty row for spacing
        consolidated_rows.append([f"Today's Oscillator = {oscillator_value}"])
    
    # Write consolidated data to output file
    output_filename = f'consolidated_predictions_{date_str}.csv'
    with open(output_filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(consolidated_rows)
    
    return output_filename

def process_all_predictions(date_str: str):
    # Define your stock symbols here
    symbols = ["NVDA", "PLTR", "APP", "INOD", "META", "MSTR"]  # Replace with your symbols
    
    try:
        # First get the oscillator value
        oscillator_value = get_osillator.get_latest_pdf_email_oscillator(date_str)
        
        # Then create the consolidated file with the oscillator value
        output_file = process_prediction_files(symbols, date_str, oscillator_value)
        print(f"Successfully created consolidated file: {output_file}")
        print(f"Included oscillator value: {oscillator_value}%")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Keep your existing get_latest_pdf_email_oscillator and extract_text_from_pdf_bytes functions...

# Usage
if __name__ == "__main__":
    today = datetime.now().strftime('%Y-%m-%d')
    # today = '2025-02-13'
    process_all_predictions(today)