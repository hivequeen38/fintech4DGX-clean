from datetime import datetime
import csv
from typing import List, Dict
import os
import get_osillator

def create_html_table(data: List[Dict], date_str: str, oscillator_value: float = None) -> str:
    """
    Creates an HTML table from the provided data.
    
    Args:
        data (List[Dict]): List of dictionaries containing the CSV data
        date_str (str): The date string for the report
        oscillator_value (float, optional): Oscillator value to append
    
    Returns:
        str: HTML content as a string
    """
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Predictions Report - {date}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                text-align: center;
                margin-bottom: 20px;
            }}
            .stock-symbol {{
                font-weight: bold;
                color: #2c5282;
                font-size: 1.2em;
                padding: 15px 0;
                background-color: #ebf8ff;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th {{
                background-color: #4a5568;
                color: white;
                padding: 12px;
                text-align: center;
                font-size: 0.9em;
            }}
            td {{
                padding: 8px;
                text-align: center;
                border: 1px solid #e2e8f0;
            }}
            .up {{
                color: #38a169;
                font-weight: bold;
            }}
            .down {{
                color: #e53e3e;
                font-weight: bold;
            }}
            .neutral {{
                color: #718096;
            }}
            .oscillator {{
                margin-top: 20px;
                text-align: right;
                font-weight: bold;
                font-size: 1.1em;
                color: #2d3748;
            }}
            .comment {{
                font-size: 0.9em;
                color: #718096;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Stock Predictions Report - {date}</h1>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Close</th>
                        {prediction_headers}
                        <th>Comment</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
            {oscillator_section}
        </div>
    </body>
    </html>
    """
    
    # Create prediction column headers (p1 through p15)
    prediction_headers = ''.join([f'<th>P{i}</th>' for i in range(1, 16)])
    
    # Process table rows
    table_rows = []
    current_symbol = None
    
    for row in data:
        # Check if this is a symbol marker
        if len(row) == 1 and row[0].startswith('<') and row[0].endswith('>'):
            current_symbol = row[0][1:-1]  # Remove < >
            table_rows.append(f'<tr><td colspan="18" class="stock-symbol">{current_symbol}</td></tr>')
            continue
            
        # Skip empty rows
        if not row or all(cell == '' for cell in row):
            continue
            
        # Process data row
        cells = []
        for i, value in enumerate(row):
            if i < 2:  # Date and Close
                cells.append(f'<td>{value}</td>')
            elif i < 17:  # Predictions p1-p15
                class_name = ''
                if value == 'UP':
                    class_name = 'up'
                elif value == 'DN':
                    class_name = 'down'
                else:
                    class_name = 'neutral'
                cells.append(f'<td class="{class_name}">{value}</td>')
            else:  # Comment
                cells.append(f'<td class="comment" style="text-align: left; padding-left: 10px;">{value}</td>')
        
        table_rows.append(f'<tr>{"".join(cells)}</tr>')
    
    # Create oscillator section if value exists
    oscillator_section = ''
    if oscillator_value is not None:
        oscillator_section = f'<div class="oscillator">Today\'s Oscillator = {oscillator_value}%</div>'
    
    # Generate final HTML
    html_content = html_template.format(
        date=date_str,
        prediction_headers=prediction_headers,
        table_rows='\n'.join(table_rows),
        oscillator_section=oscillator_section
    )
    
    return html_content

def csv_to_html(input_filename: str, date_str: str, oscillator_value: float = None) -> str:
    """
    Converts a CSV file to HTML format.
    
    Args:
        input_filename (str): Path to the input CSV file
        date_str (str): Date string for filtering records
        oscillator_value (float, optional): Oscillator value to append
        
    Returns:
        str: Path to the generated HTML file
    """
    # Read CSV data
    data = []
    with open(input_filename, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Skip header row
        for row in reader:
            data.append(row)
    
    # Generate HTML content
    html_content = create_html_table(data, date_str, oscillator_value)
    
    # Write HTML file
    output_filename = input_filename.replace('.csv', '.html')
    with open(output_filename, 'w', encoding='utf-8') as file:
        file.write(html_content)
    
    return output_filename

def process_prediction_files(stock_symbols: List[str], date_str: str, oscillator_value: float = None) -> str:
    """
    Process prediction files and create both CSV and HTML outputs.
    """
    # First create the CSV file as before
    consolidated_rows = []
    
    for symbol in stock_symbols:
        filename = f"{symbol}_15d_from_today_predictions.csv"
        
        try:
            with open(filename, 'r') as file:
                if consolidated_rows:
                    consolidated_rows.append([])
                consolidated_rows.append([f"<{symbol}>"])
                
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    if row['date'] == date_str:
                        output_row = [
                            row['date'],
                            row['close'],
                            *[row[f'p{i}'] for i in range(1, 16)],
                            row['comment']
                        ]
                        consolidated_rows.append(output_row)
                        
        except FileNotFoundError:
            print(f"Warning: File not found for symbol {symbol}")
            continue
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    # Add oscillator value
    if oscillator_value is not None:
        consolidated_rows.append([])
        consolidated_rows.append([f"Today's Oscillator = {oscillator_value}%"])
    
    # Write CSV file
    csv_filename = f'consolidated_predictions_{date_str}.csv'
    with open(csv_filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(consolidated_rows)
    
    # Convert to HTML
    html_filename = csv_to_html(csv_filename, date_str, oscillator_value)
    
    return html_filename

def process_all_predictions(date_str: str):
    """
    Main function to process all predictions and generate reports.
    """
    symbols = ["NVDA", "PLTR", "APP", "INOD", "META", "MSTR", "QQQ"]
    
    try:
        # Get oscillator value first
        oscillator_value, osc_date = get_osillator.get_latest_pdf_email_oscillator(date_str)
        
        # Process files and create HTML
        output_file = process_prediction_files(symbols, date_str, oscillator_value)
        print(f"Successfully created HTML report: {output_file}")
        print(f"Included oscillator value: {oscillator_value}%")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # today = datetime.now().strftime('%Y-%m-%d')
    today = '2025-03-21'
    process_all_predictions(today)