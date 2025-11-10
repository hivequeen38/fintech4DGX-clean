import pandas as pd
import re

def process_sentiment_data(filename):
    # Read the file as a simple text file first
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Parse the header and get column names
    header = lines[0].strip()
    # Find where actual column names are
    column_names = re.split(r'\s+', header.strip())
    
    # Get date, bullish, neutral, bearish columns
    if len(column_names) >= 4:
        date_col, bull_col, neutral_col, bear_col = column_names[:4]
    else:
        # If the column names are combined, use default names
        date_col = "date"
        bull_col = "Bullish" 
        neutral_col = "Neutral"
        bear_col = "Bearish"
    
    # Process data rows
    data = []
    for line in lines[1:]:  # Skip the header
        if not line.strip():  # Skip empty lines
            continue
            
        # Split the first part (date) and the rest
        parts = line.strip().split('\t')
        if len(parts) < 4:
            # If the tabs aren't properly separating, try to extract values
            # using regex to find patterns like 36.0% or 36.0
            row_text = line.strip()
            # Extract date (assuming it's at the beginning)
            date_match = re.search(r'(\d+-\d+-\d+)', row_text)
            if date_match:
                date = date_match.group(1)
                # Remove date from text for easier processing
                row_text = row_text[date_match.end():].strip()
            else:
                date = ""
                
            # Extract percentage values
            percentages = re.findall(r'(\d+\.\d+%?|\d+%?|#N/A)', row_text)
            if len(percentages) >= 3:
                bullish = percentages[0].rstrip('%')
                neutral = percentages[1].rstrip('%')
                bearish = percentages[2].rstrip('%')
                row = [date, bullish, neutral, bearish]
                data.append(row)
        else:
            # Clean each part by removing % signs
            date = parts[0].strip()
            bullish = parts[1].strip().rstrip('%')
            neutral = parts[2].strip().rstrip('%')
            bearish = parts[3].strip().rstrip('%')
            row = [date, bullish, neutral, bearish]
            data.append(row)
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=[date_col, bull_col, neutral_col, bear_col])
    
    # Convert percentage columns to float, handling special values
    for col in [bull_col, neutral_col, bear_col]:
        # Replace special values like #N/A with NaN
        df[col] = df[col].replace(['#N/A', 'N/A', 'nan', 'NaN', ''], pd.NA)
        # Convert to float, errors='coerce' will convert problematic values to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove the Neutral column
    df = df.drop(neutral_col, axis=1)
    
    # Add a Spread column (handle NaN values gracefully)
    df['Spread'] = df[bull_col].sub(df[bear_col])
    
    # Write to a new CSV file
    df.to_csv(filename, index=False)
    
    print(f"Successfully processed {filename}")
    print(f"Removed '{neutral_col}' column and added 'Spread' column")
    print(f"Final columns: {', '.join(df.columns)}")
    print(f"Saved as standard comma-delimited CSV")

if __name__ == "__main__":
    process_sentiment_data("sentiment.csv")