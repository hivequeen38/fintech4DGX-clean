from datetime import datetime
import csv

def process_prediction_files(stock_symbols, date_str: str):
   
    
    # Store the consolidated output
    consolidated_rows = []
    
    for symbol in stock_symbols:
        filename = f"{symbol}_15d_from_today_predictions.csv"
        
        try:
            # Read and process the file
            with open(filename, 'r') as file:
                if consolidated_rows:  # Only add blank line if not the first symbol
                    consolidated_rows.append([])  # Empty row for carriage return# Add stock symbol as a separator
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
        output_file = process_prediction_files(symbols, date_str)
        print(f"Successfully created consolidated file: {output_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Usage
if __name__ == "__main__":
    process_all_predictions()
    