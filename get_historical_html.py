from datetime import datetime
import pandas as pd
from typing import Optional
from bs4 import BeautifulSoup
import os
from datetime import datetime
import google_cloud_util
import get_osillator

INITIAL_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Stock Trend Predictions</title>
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            table-layout: auto;
        }
        .stock-column {
            width: 1%;
            white-space: nowrap;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        h1 {
            text-align: center;
            color: #333;
        }
    </style>
</head>
<body>
    <h1>Stock Trend Predictions</h1>
    <table>
        <thead>
            <tr>
                <th class="stock-column">Stock</th>
                <th>Time</th>
                <th>Link</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>NVDA</td>
                <td></td>
                <td><a href=""></a></td>
            </tr>
            <tr>
                <td>PLTR</td>
                <td></td>
                <td><a href=""></a></td>
            </tr>
              <tr>
                <td>SMCI</td>
                <td></td>
                <td><a href=""></a></td>
            </tr>
            <tr>
                <td>APP</td>
                <td></td>
                <td><a href=""></a></td>
            </tr>
            <tr>
                <td>INDO</td>
                <td></td>
                <td><a href=""></a></td>
            </tr>
            <tr>
                <td>META</td>
                <td></td>
                <td><a href=""></a></td>
            </tr>
            <tr>
                <td>MSTR</td>
                <td></td>
                <td><a href=""></a></td>
            </tr>
             <tr>
                <td>ANET</td>
                <td></td>
                <td><a href=""></a></td>
            </tr>
        </tbody>
    </table>
</body>
</html>"""

def create_historical_html_table_OLD(df: pd.DataFrame, symbol: str, oscillator_value: Optional[float] = None) -> str:
    """
    Creates an HTML table from historical DataFrame for a single stock.
    Includes visual separation between different dates.
    
    Args:
        df (pd.DataFrame): DataFrame containing historical data
        symbol (str): Stock symbol (e.g., 'NVDA')
        oscillator_value (float, optional): Current oscillator value
    
    Returns:
        str: HTML content as a string
    """
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{symbol} Historical Predictions</title>
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
            .date-separator {{
                border-top: 3px solid #2d3748;
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
                text-align: left;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{symbol} Historical Predictions</h1>
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
    
    # Sort DataFrame by date (descending) and then by comment
    df_sorted = df.sort_values(['date', 'comment'], ascending=[False, True])
    
    # Keep track of the previous date to detect changes
    previous_date = None
    
    for _, row in df_sorted.iterrows():
        cells = []
        current_date = row["date"]
        
        # Check if this is a new date
        is_new_date = previous_date is not None and current_date != previous_date
        
        # Add date and close
        cells.append(f'<td>{current_date}</td>')
        cells.append(f'<td>{row["close"]}</td>')
        
        # Add predictions (p1-p15)
        for i in range(1, 16):
            value = row[f'p{i}']
            class_name = ''
            if value == 'UP':
                class_name = 'up'
            elif value == 'DN':
                class_name = 'down'
            else:
                class_name = 'neutral'
            cells.append(f'<td class="{class_name}">{value}</td>')
        
        # Add comment with left alignment
        cells.append(f'<td class="comment" style="text-align: left; padding-left: 10px;">{row["comment"]}</td>')
        
        # Add the date-separator class if this is a new date
        separator_class = ' class="date-separator"' if is_new_date else ''
        table_rows.append(f'<tr{separator_class}>{"".join(cells)}</tr>')
        
        # Update previous date
        previous_date = current_date
    
    # Create oscillator section if value exists
    oscillator_section = ''
    if oscillator_value is not None:
        oscillator_section = f'<div class="oscillator">Today\'s Oscillator = {oscillator_value}%</div>'
    
    # Generate final HTML
    html_content = html_template.format(
        symbol=symbol,
        prediction_headers=prediction_headers,
        table_rows='\n'.join(table_rows),
        oscillator_section=oscillator_section
    )
    
    return html_content

def create_historical_html_table(df: pd.DataFrame, symbol: str, oscillator_value: Optional[float] = None) -> str:
    """
    Creates an HTML table from historical DataFrame for a single stock.
    Includes visual separation between different dates.
    Highlights rows containing "(ref)" in the comment section with a black background.
    
    Args:
        df (pd.DataFrame): DataFrame containing historical data
        symbol (str): Stock symbol (e.g., 'NVDA')
        oscillator_value (float, optional): Current oscillator value
    
    Returns:
        str: HTML content as a string
    """
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{symbol} Historical Predictions</title>
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
            .date-separator {{
                border-top: 3px solid #2d3748;
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
            .reference-row {{
                background-color: #000000;
                color: #ffffff;
            }}
            .reference-row .up {{
                color: #38a169;
                font-weight: bold;
            }}
            .reference-row .down {{
                color: #e53e3e;
                font-weight: bold;
            }}
            .reference-row .neutral {{
                color: #cccccc;
            }}
            .reference-row .comment {{
                color: #ffffff;
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
                text-align: left;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{symbol} Historical Predictions</h1>
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
    
    # Sort DataFrame by date (descending) and then by comment
    df_sorted = df.sort_values(['date', 'comment'], ascending=[False, True])
    
    # Keep track of the previous date to detect changes
    previous_date = None
    
    for _, row in df_sorted.iterrows():
        cells = []
        current_date = row["date"]
        
        # Check if this is a new date
        is_new_date = previous_date is not None and current_date != previous_date
        
        # Check if this row contains "(ref)" in the comment
        is_reference = "(ref)" in str(row["comment"])
        
        # Add date and close
        cells.append(f'<td>{current_date}</td>')
        cells.append(f'<td>{row["close"]}</td>')
        
        # Add predictions (p1-p15)
        for i in range(1, 16):
            value = row[f'p{i}']
            class_name = ''
            if value == 'UP':
                class_name = 'up'
            elif value == 'DN':
                class_name = 'down'
            else:
                class_name = 'neutral'
            cells.append(f'<td class="{class_name}">{value}</td>')
        
        # Add comment with left alignment
        cells.append(f'<td class="comment" style="text-align: left; padding-left: 10px;">{row["comment"]}</td>')
        
        # Determine row classes
        row_classes = []
        if is_new_date:
            row_classes.append("date-separator")
        if is_reference:
            row_classes.append("reference-row")
        
        # Add the class attribute if there are any classes
        class_attr = ''
        if row_classes:
            class_attr = f' class="{" ".join(row_classes)}"'
        
        table_rows.append(f'<tr{class_attr}>{"".join(cells)}</tr>')
        
        # Update previous date
        previous_date = current_date
    
    # Create oscillator section if value exists
    oscillator_section = ''
    if oscillator_value is not None:
        oscillator_section = f'<div class="oscillator">Today\'s Oscillator = {oscillator_value}%</div>'
    
    # Generate final HTML
    html_content = html_template.format(
        symbol=symbol,
        prediction_headers=prediction_headers,
        table_rows='\n'.join(table_rows),
        oscillator_section=oscillator_section
    )
    
    return html_content

def generate_historical_html(df: pd.DataFrame, symbol: str, oscillator_value: Optional[float] = None) -> str:
    """
    Generate HTML file from historical DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing historical data
        symbol (str): Stock symbol (e.g., 'NVDA')
        oscillator_value (float, optional): Current oscillator value
        
    Returns:
        str: Path to the generated HTML file
    """
    # Generate HTML content
    html_content = create_historical_html_table(df, symbol, oscillator_value)
    
    # Write HTML file
    output_filename = f'{symbol}_result.html'
    with open(output_filename, 'w', encoding='utf-8') as file:
        file.write(html_content)
    
    return output_filename

def get_recent_entries(df, count):
    # Create a copy to avoid modifying the original
    temp_df = df.copy()
    
    # Convert to datetime using 'mixed' format to handle different date formats
    temp_df['date'] = pd.to_datetime(temp_df['date'], format='mixed').dt.normalize()
    
    # Get unique dates sorted in descending order
    unique_dates = temp_df['date'].sort_values(ascending=False).unique()
    
    # Select the most recent 'count' dates
    selected_dates = unique_dates[:count]
    
    # Filter the original dataframe using the same mixed format conversion
    mask = pd.to_datetime(df['date'], format='mixed').dt.normalize().isin(selected_dates)
    result_df = df[mask].copy()
    
    # Sort by date descending to show most recent first
    result_df = result_df.sort_values('date', ascending=False)
    
    return result_df

def process_all_predictions(oscillator_value):
    '''
    this will take the latest predictions from ALL stocks, 
    1. upload each file to S3
    2. updates the overall html file with link to each file and a time stamp

    '''
    stocks = ['NVDA', 'PLTR', 'APP', 'INOD', 'META', 'MSTR', 'ANET']
    for stock in stocks:
        df = pd.read_csv(f'{stock}_15d_from_today_predictions.csv')
        recent_df = get_recent_entries(df, 16)
        output_file = generate_historical_html(
            df=recent_df,
            symbol=stock,
            oscillator_value=oscillator_value  # Optional
        )
        # upload_to_s3(output_file, stock)
        print(f"Successfully created HTML report: {output_file}")
    
    # now update the overall html file
    update_stock_table(main_html_path="stock_trends.html")

    # now upload stuff to S3

# Example usage:
# if __name__ == "__main__":
#     # Load your data into a DataFrame
#     df = pd.read_csv('PLTR_15d_from_today_predictions.csv')
#     recent_df = get_recent_entries(df, 16)

#     # Generate the HTML file
#     output_file = generate_historical_html(
#         df=recent_df,
#         symbol='PLTR',
#         oscillator_value=2.5  # Optional
#     )
#     df = pd.read_csv('APP_15d_from_today_predictions.csv')
#     recent_df = get_recent_entries(df, 16)

#     # Generate the HTML file
#     output_file = generate_historical_html(
#         df=recent_df,
#         symbol='APP',
#         oscillator_value=2.5  # Optional
#     )
#     df = pd.read_csv('INOD_15d_from_today_predictions.csv')
#     recent_df = get_recent_entries(df, 16)

#     # Generate the HTML file
#     output_file = generate_historical_html(
#         df=recent_df,
#         symbol='INOD',
#         oscillator_value=2.5  # Optional
#     )
#     df = pd.read_csv('META_15d_from_today_predictions.csv')
#     recent_df = get_recent_entries(df, 16)

#     # Generate the HTML file
#     output_file = generate_historical_html(
#         df=recent_df,
#         symbol='META',
#         oscillator_value=2.5  # Optional
#     )
#     df = pd.read_csv('MSTR_15d_from_today_predictions.csv')
#     recent_df = get_recent_entries(df, 16)

#     # Generate the HTML file
#     output_file = generate_historical_html(
#         df=recent_df,
#         symbol='MSTR',
#         oscillator_value=2.5  # Optional
#     )

def update_stock_table(main_html_path="stock_trends.html"):
    """
    Update the main HTML table with latest stock results.
    Uses the file modification timestamp for each stock's result file.
    If the file doesn't exist or is empty, initializes it with the template.
    
    Args:
        main_html_path (str): Path to the main HTML file
    """
    print(f"Checking file: {main_html_path}")
    
    # Check if file exists and has content
    if not os.path.exists(main_html_path) or os.path.getsize(main_html_path) == 0:
        print("File is empty or doesn't exist. Creating new file with template...")
        # Initialize the file with the template
        with open(main_html_path, 'w', encoding='utf-8') as f:
            f.write(INITIAL_HTML)
        print("Template written to file.")
    
    # Verify the file has content
    if os.path.getsize(main_html_path) == 0:
        print("Error: File is still empty after initialization!")
        return
    
    print("Reading HTML file...")
    # Read the HTML file
    with open(main_html_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"File content length: {len(content)}")
        soup = BeautifulSoup(content, 'html.parser')
    
    # Get all rows from the table body
    rows = soup.find('tbody').find_all('tr')
    print(f"Found {len(rows)} rows in table")
    
    # Process each stock
    for row in rows:
        stock = row.find('td').text.strip()
        result_file = f"{stock}_result.html"
        print(f"Processing stock: {stock}")
        
        # Skip if results file doesn't exist
        if not os.path.exists(result_file):
            print(f"No results file found for {stock}")
            continue
        
        # Get the timestamp of the CSV file, not the results file
        csv_file = f"{stock}_15d_from_today_predictions.csv"
        
        if os.path.exists(csv_file):
            file_timestamp = datetime.fromtimestamp(os.path.getmtime(csv_file))
            timestamp_str = file_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            # Update with the CSV file's timestamp - ensure string is properly set
            time_cell = row.find_all('td')[1]
            time_cell.clear()  # Clear existing content
            time_cell.append(timestamp_str)  # Add new content
            print(f"Updated {stock} timestamp to {timestamp_str} from CSV file")
        else:
            print(f"Warning: CSV file {csv_file} not found, using result file timestamp instead")
            result_timestamp = datetime.fromtimestamp(os.path.getmtime(result_file))
            timestamp_str = result_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            time_cell = row.find_all('td')[1]
            time_cell.clear()  # Clear existing content
            time_cell.append(timestamp_str)  # Add new content
            print(f"Updated {stock} timestamp to {timestamp_str} from result file")
        
        # Update link
        link_cell = row.find_all('td')[2]
        link = link_cell.find('a')
        if not link:
            link = soup.new_tag('a')
            link_cell.append(link)
        link['href'] = result_file
        link.string = f"View {stock} Results"
    
    print("Saving updated HTML...")
    # Save the updated HTML - use soup's string representation directly instead of prettify()
    with open(main_html_path, 'w', encoding='utf-8') as f:
        f.write(str(soup))
    print("Update complete")
    
    # Verify the file was updated correctly by reading it back
    with open(main_html_path, 'r', encoding='utf-8') as f:
        verify_content = f.read()
        verify_soup = BeautifulSoup(verify_content, 'html.parser')
        verify_rows = verify_soup.find('tbody').find_all('tr')
        for row in verify_rows:
            stock = row.find('td').text.strip()
            timestamp = row.find_all('td')[1].text.strip()
            print(f"Verification - {stock}: {timestamp}")
    return main_html_path


def upload_all_results(input_date_str: str = None):

    oscillator_input = get_osillator.get_oscillator(input_date_str)
    # create a list of files for each sstock to be uploaded (they should exist already
    stocks = ['NVDA', 'PLTR', 'APP', 'ANET', 'CRDO', 'ALAB', 'TSLA' ]

    for stock in stocks:
        df = pd.read_csv(stock + '_15d_from_today_predictions.csv')
        recent_df = get_recent_entries(df, 16)

        # filter out all (QA) entries
        # Create mask for rows that don't contain the phrase
        # delete checking for closeing bracket) 
        mask = ~recent_df['comment'].str.contains('QA', na=False)
    
        # Apply mask to get filtered DataFrame
        filtered_df = recent_df[mask]
        
        # Generate the HTML file
        output_file = generate_historical_html(
            df=filtered_df,
            symbol=stock,
            oscillator_value=oscillator_input  # Optional
        )
        # file_path = f'{stock}_result.html'
        google_cloud_util.upload_file_to_bucket('ml-prediction-results', output_file)

    main_html_path = update_stock_table(main_html_path="stock_trends.html")
    main_file_url = google_cloud_util.upload_file_to_bucket('ml-prediction-results', main_html_path)

if __name__ == "__main__":
    upload_all_results('2025-02-21')
