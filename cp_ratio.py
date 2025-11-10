import pandas as pd
import json
import time
import os
import trendConfig

def calculate_cp_ratio_by_date(df):
    # Group by date and type
    daily_grouped = df.groupby(['date', 'type']).agg({
        'volume': 'sum',
        'open_interest': 'sum'
    }).reset_index()
    
    # Pivot the data to get calls and puts side by side
    daily_ratios = daily_grouped.pivot_table(
        index='date',
        columns='type',
        values=['volume', 'open_interest']
    ).fillna(0)
    
    # Calculate daily ratios
    daily_ratios['cp_volume_ratio'] = daily_ratios[('volume', 'call')] / \
                                     daily_ratios[('volume', 'put')].replace(0, float('inf'))
    daily_ratios['cp_oi_ratio'] = daily_ratios[('open_interest', 'call')] / \
                                 daily_ratios[('open_interest', 'put')].replace(0, float('inf'))
    
    return daily_ratios.sort_index()

# Example usage:
# Assuming your data is in the variable 'options_data'

import requests
import pandas as pd

def get_options_data(symbol, date, api_key):
    """
    Fetch options data from Alpha Vantage API
    
    Parameters:
    symbol (str): Stock symbol (e.g., 'IBM')
    date (str): Date in YYYY-MM-DD format
    api_key (str): Alpha Vantage API key
    """
    
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "HISTORICAL_OPTIONS",
        "symbol": symbol,
        "date": date,
        "apikey": api_key
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json()
        
        # Check if we got valid data
        if "data" not in data:
            print(f"Error: No data returned. Response: {data}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        
        # Convert numeric columns
        numeric_columns = ['strike', 'last', 'mark', 'bid', 'bid_size', 
                         'ask', 'ask_size', 'volume', 'open_interest',
                         'implied_volatility', 'delta', 'gamma', 'theta',
                         'vega', 'rho']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Convert date columns
        df['date'] = pd.to_datetime(df['date'])
        df['expiration'] = pd.to_datetime(df['expiration'])
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def get_historical_cp_ratios(symbol, dates_df, api_key):
    """
    Get C/P ratios for dates provided in DataFrame
    """
    
    results = []
    
    for date in dates_df['date']:
        date_str = date.strftime('%Y-%m-%d')
        try:
            print(f"Fetching data for {date_str}")
            
            response = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "HISTORICAL_OPTIONS",
                    "symbol": symbol,
                    "date": date_str,
                    "apikey": api_key
                }
            )
            
            data = response.json()
            
            if "data" in data and data["data"]:
                df = pd.DataFrame(data["data"])
                
                # Convert volume and open_interest to numeric
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce')
                
                daily_totals = df.groupby('type').agg({
                    'volume': 'sum',
                    'open_interest': 'sum'
                }).reset_index()
                
                call_data = daily_totals[daily_totals['type'] == 'call'].iloc[0]
                put_data = daily_totals[daily_totals['type'] == 'put'].iloc[0]
                
                cp_volume_ratio = call_data['volume'] / put_data['volume'] if put_data['volume'] != 0 else float('inf')
                cp_oi_ratio = call_data['open_interest'] / put_data['open_interest'] if put_data['open_interest'] != 0 else float('inf')
                
                results.append({
                    'date': date,
                    'call_volume': float(call_data['volume']),
                    'put_volume': float(put_data['volume']),
                    'call_oi': float(call_data['open_interest']),
                    'put_oi': float(put_data['open_interest']),
                    'cp_volume_ratio': float(cp_volume_ratio),
                    'cp_oi_ratio': float(cp_oi_ratio)
                })
            
            # Sleep to respect API rate limits
            time.sleep(12)
            
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.set_index('date', inplace=True)
        results_df.sort_index(inplace=True)
    
    return results_df

def get_historical_cp_ratios_with_sentiments_VERYOLD(symbol, dates_df, api_key):
    """
    Get C/P ratios for dates provided in DataFrame
    """
    
    results = []
    
    for date in dates_df['date']:
        date_str = date.strftime('%Y-%m-%d')
        try:
            print(f"Fetching data for {date_str}")
            
            response = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "HISTORICAL_OPTIONS",
                    "symbol": symbol,
                    "date": date_str,
                    "apikey": api_key
                }
            )
            
            data = response.json()
            
            if "data" in data and data["data"]:
                df = pd.DataFrame(data["data"])
                
                # Convert to numeric
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce')
                df['bid'] = pd.to_numeric(df['bid'], errors='coerce')
                df['ask'] = pd.to_numeric(df['ask'], errors='coerce')
                df['last'] = pd.to_numeric(df['last'], errors='coerce')
                
                # Compute midpoint and apply naive sentiment rule
                df['mid'] = (df['bid'] + df['ask']) / 2
                
                def naive_sentiment(row):
                    # If no volume, no actual trades -> "no trade"
                    if row['volume'] == 0 or pd.isna(row['last']):
                        return "no trade"
                    
                    # If midpoint is missing or zero, cannot compare properly
                    if pd.isna(row['mid']):
                        return "unknown"
                    
                    # Compare last to midpoint
                    if row['last'] >= row['mid']:
                        # near the ask => a buy
                        if row['type'] == 'call':
                            return "bullish"
                        else:  # row['type'] == 'put'
                            return "bearish"
                    else:
                        # near the bid => a sell
                        if row['type'] == 'call':
                            return "bearish"
                        else:
                            return "bullish"
                
                df['sentiment'] = df.apply(naive_sentiment, axis=1)
                
                # Now df has a 'sentiment' column for each option row
                # -----------------------------------------------------------------
                # Next, do your existing grouping to get total volume, open interest, etc.
                # -----------------------------------------------------------------
                
                daily_totals = df.groupby('type').agg({
                    'volume': 'sum',
                    'open_interest': 'sum'
                }).reset_index()
                
                # Safely extract call/put rows
                call_row = daily_totals[daily_totals['type'] == 'call']
                put_row = daily_totals[daily_totals['type'] == 'put']
                
                # Handle missing call or put data
                if not call_row.empty:
                    call_vol = call_row.iloc[0]['volume']
                    call_oi = call_row.iloc[0]['open_interest']
                else:
                    call_vol, call_oi = 0, 0
                
                if not put_row.empty:
                    put_vol = put_row.iloc[0]['volume']
                    put_oi = put_row.iloc[0]['open_interest']
                else:
                    put_vol, put_oi = 0, 0
                
                # Ratios
                cp_volume_ratio = call_vol / put_vol if put_vol != 0 else float('inf')
                cp_oi_ratio = call_oi / put_oi if put_oi != 0 else float('inf')
                
                # OPTIONAL: You can also aggregate naive sentiment across all rows
                # For example, sum the volumes where sentiment == 'bullish' vs. 'bearish'
                bullish_volume = df.loc[df['sentiment'] == 'bullish', 'volume'].sum()
                bearish_volume = df.loc[df['sentiment'] == 'bearish', 'volume'].sum()
                
                # You could decide an overall "daily sentiment" if you like
                if bullish_volume > bearish_volume:
                    day_sentiment = "bullish"
                elif bearish_volume > bullish_volume:
                    day_sentiment = "bearish"
                else:
                    day_sentiment = "neutral"
                
                # Finally, append your results
                results.append({
                    'date': date,  # Or whatever your date variable is
                    'call_volume': float(call_vol),
                    'put_volume': float(put_vol),
                    'call_oi': float(call_oi),
                    'put_oi': float(put_oi),
                    'cp_volume_ratio': float(cp_volume_ratio),
                    'cp_oi_ratio': float(cp_oi_ratio),
                    'daily_sentiment': day_sentiment,
                    # If you want the entire DataFrame's sentiment, you could store it
                    # or some other breakdown of bullish/bearish volume
                    'bullish_volume': float(bullish_volume),
                    'bearish_volume': float(bearish_volume),
                })
                
            # Sleep for rate limit
            time.sleep(12)
            
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.set_index('date', inplace=True)
        results_df.sort_index(inplace=True)
    
    return results_df

def get_historical_cp_ratios_with_sentiments_OLD(symbol, dates_df, api_key):
    """
    Get C/P ratios for dates provided in DataFrame
    """
    
    results = []
    
    for date_str in dates_df['date']:
        
        try:
            print(f"Fetching data for {date_str}")
            
            response = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "HISTORICAL_OPTIONS",
                    "symbol": symbol,
                    "date": date_str,
                    "apikey": api_key
                }
            )
            
            data = response.json()
            
            if "data" in data and data["data"]:
                df = pd.DataFrame(data["data"])
                
                # Convert to numeric
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce')
                df['bid'] = pd.to_numeric(df['bid'], errors='coerce')
                df['ask'] = pd.to_numeric(df['ask'], errors='coerce')
                df['last'] = pd.to_numeric(df['last'], errors='coerce')
                
                # Compute midpoint and apply naive sentiment rule
                df['mid'] = (df['bid'] + df['ask']) / 2
                
                def naive_sentiment(row):
                    # If no volume, no actual trades -> "no trade"
                    if row['volume'] == 0 or pd.isna(row['last']):
                        return "no trade"
                    
                    # If midpoint is missing or zero, cannot compare properly
                    if pd.isna(row['mid']):
                        return "unknown"
                    
                    # Compare last to midpoint
                    if row['last'] >= row['mid']:
                        # near the ask => a buy
                        if row['type'] == 'call':
                            return "bullish"
                        else:  # row['type'] == 'put'
                            return "bearish"
                    else:
                        # near the bid => a sell
                        if row['type'] == 'call':
                            return "bearish"
                        else:
                            return "bullish"
                
                df['sentiment'] = df.apply(naive_sentiment, axis=1)
                
                # Now df has a 'sentiment' column for each option row
                # -----------------------------------------------------------------
                # Next, do your existing grouping to get total volume, open interest, etc.
                # -----------------------------------------------------------------
                
                daily_totals = df.groupby('type').agg({
                    'volume': 'sum',
                    'open_interest': 'sum'
                }).reset_index()
                
                # Safely extract call/put rows
                call_row = daily_totals[daily_totals['type'] == 'call']
                put_row = daily_totals[daily_totals['type'] == 'put']
                
                # Handle missing call or put data
                if not call_row.empty:
                    call_vol = call_row.iloc[0]['volume']
                    call_oi = call_row.iloc[0]['open_interest']
                else:
                    call_vol, call_oi = 0, 0
                
                if not put_row.empty:
                    put_vol = put_row.iloc[0]['volume']
                    put_oi = put_row.iloc[0]['open_interest']
                else:
                    put_vol, put_oi = 0, 0
                
                # Ratios
                cp_volume_ratio = call_vol / put_vol if put_vol != 0 else float('inf')
                cp_oi_ratio = call_oi / put_oi if put_oi != 0 else float('inf')
                
                # OPTIONAL: You can also aggregate naive sentiment across all rows
                # For example, sum the volumes where sentiment == 'bullish' vs. 'bearish'
                bullish_volume = df.loc[df['sentiment'] == 'bullish', 'volume'].sum()
                bearish_volume = df.loc[df['sentiment'] == 'bearish', 'volume'].sum()
                
                # You could decide an overall "daily sentiment" if you like
                if bullish_volume > bearish_volume:
                    day_sentiment = "bullish"
                elif bearish_volume > bullish_volume:
                    day_sentiment = "bearish"
                else:
                    day_sentiment = "neutral"
                
                # Finally, append your results
                results.append({
                    'date': date_str,  # Or whatever your date variable is
                    'call_volume': float(call_vol),
                    'put_volume': float(put_vol),
                    'call_oi': float(call_oi),
                    'put_oi': float(put_oi),
                    'cp_volume_ratio': float(cp_volume_ratio),
                    'cp_oi_ratio': float(cp_oi_ratio),
                    'daily_sentiment': day_sentiment,
                    # If you want the entire DataFrame's sentiment, you could store it
                    # or some other breakdown of bullish/bearish volume
                    'bullish_volume': float(bullish_volume),
                    'bearish_volume': float(bearish_volume),
                })
                
            # Sleep for rate limit
            time.sleep(12)
            
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            continue

        # at this point save the intermediate results
        file_path = symbol+ '-' + "cp_ratios_sentiment_w_volume.csv"
        cp_ratios_df = pd.DataFrame(results)
        cp_ratios_df.to_csv(file_path)
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.set_index('date', inplace=True)
        results_df.sort_index(inplace=True)
    
    return results_df


def get_historical_cp_ratios_with_sentiments_new(symbol, dates_df, api_key):
    """
    Get C/P ratios for dates provided in DataFrame with improved handling of edge cases
    Continues from where it left off if a saved file exists
    """
    file_path = f"{symbol}-cp_ratios_sentiment_w_volume.csv"
    
    # First create an empty DataFrame with the correct structure and types
    results_df = pd.DataFrame(columns=[
        'date', 'call_volume', 'put_volume', 'call_oi', 'put_oi',
        'cp_volume_ratio', 'cp_oi_ratio', 'daily_sentiment',
        'bullish_volume', 'bearish_volume'
    ])
    
    # Set proper dtypes upfront
    results_df = results_df.astype({
        'date': 'object',
        'call_volume': 'float64',
        'put_volume': 'float64',
        'call_oi': 'float64',
        'put_oi': 'float64',
        'cp_volume_ratio': 'float64',
        'cp_oi_ratio': 'float64',
        'daily_sentiment': 'object',
        'bullish_volume': 'float64',
        'bearish_volume': 'float64'
    })
    
    # Check if we already have saved results
    if os.path.exists(file_path):
        # Load existing results
        results_df = pd.read_csv(file_path)
        
        # Find dates that haven't been processed yet
        processed_dates = set(results_df['date'])
        all_dates = set(dates_df['date'])
        remaining_dates = sorted(list(all_dates - processed_dates))
        
        print(f"Found existing file with {len(results_df)} entries")
        print(f"Continuing with {len(remaining_dates)} remaining dates")
        
        # Update dates_df to only include remaining dates
        dates_df = pd.DataFrame({'date': remaining_dates})
    else:
        print(f"Starting fresh - no existing file found")
    
    for date_str in dates_df['date']:
        try:
            print(f"Fetching data for {date_str}")
            
            response = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "HISTORICAL_OPTIONS",
                    "symbol": symbol,
                    "date": date_str,
                    "apikey": api_key
                }
            )
            
            data = response.json()
            
            # Initialize row with zeros and default values
            row = {
                'date': date_str,
                'call_volume': 0.0,
                'put_volume': 0.0,
                'call_oi': 0.0,
                'put_oi': 0.0,
                'cp_volume_ratio': 0.0,
                'cp_oi_ratio': 0.0,
                'daily_sentiment': 'no_trades',
                'bullish_volume': 0.0,
                'bearish_volume': 0.0
            }
            
            if "data" in data and data["data"]:
                df = pd.DataFrame(data["data"])
                
                if not df.empty:
                    # Convert numeric columns
                    for col in ['volume', 'open_interest', 'bid', 'ask', 'last']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    
                    # Calculate midpoint
                    df['mid'] = ((df['bid'] + df['ask']) / 2).fillna(df['last'])
                    
                    # Calculate sentiment
                    df['sentiment'] = df.apply(lambda x: 
                        'no_trade' if x['volume'] == 0 or pd.isna(x['last']) else
                        'bullish' if (x['last'] >= x['mid'] and x['type'] == 'call') or 
                                    (x['last'] < x['mid'] and x['type'] == 'put') else
                        'bearish', axis=1)
                    
                    # Calculate totals by type
                    type_totals = df.groupby('type').agg({
                        'volume': 'sum',
                        'open_interest': 'sum'
                    }).fillna(0)
                    
                    # Extract values safely
                    row['call_volume'] = float(type_totals.loc['call', 'volume']) if 'call' in type_totals.index else 0.0
                    row['put_volume'] = float(type_totals.loc['put', 'volume']) if 'put' in type_totals.index else 0.0
                    row['call_oi'] = float(type_totals.loc['call', 'open_interest']) if 'call' in type_totals.index else 0.0
                    row['put_oi'] = float(type_totals.loc['put', 'open_interest']) if 'put' in type_totals.index else 0.0
                    
                    # Calculate ratios
                    if row['put_volume'] > 0:
                        row['cp_volume_ratio'] = row['call_volume'] / row['put_volume']
                    if row['put_oi'] > 0:
                        row['cp_oi_ratio'] = row['call_oi'] / row['put_oi']
                    
                    # Calculate sentiment volumes
                    row['bullish_volume'] = float(df[df['sentiment'] == 'bullish']['volume'].sum())
                    row['bearish_volume'] = float(df[df['sentiment'] == 'bearish']['volume'].sum())
                    
                    # Determine overall sentiment
                    if row['bullish_volume'] == 0 and row['bearish_volume'] == 0:
                        row['daily_sentiment'] = 'no_trades'
                    elif row['bullish_volume'] > row['bearish_volume']:
                        row['daily_sentiment'] = 'bullish'
                    elif row['bearish_volume'] > row['bullish_volume']:
                        row['daily_sentiment'] = 'bearish'
                    else:
                        row['daily_sentiment'] = 'neutral'
            
            # Append row to results DataFrame
            results_df.loc[len(results_df)] = row
            
            # Save intermediate results after each successful fetch
            results_df.to_csv(file_path, index=False)
            print(f"Saved intermediate results: {len(results_df)} total entries")
            
            time.sleep(12)
            
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            results_df.loc[len(results_df)] = row
            continue
    
    # Force fill any remaining NaN values
    float_cols = ['call_volume', 'put_volume', 'call_oi', 'put_oi', 
                  'cp_volume_ratio', 'cp_oi_ratio', 'bullish_volume', 'bearish_volume']
    
    results_df[float_cols] = results_df[float_cols].fillna(0.0)
    results_df['daily_sentiment'] = results_df['daily_sentiment'].fillna('no_trades')
    
    # Set index and sort
    if not results_df.empty:
        results_df.set_index('date', inplace=True)
        results_df.sort_index(inplace=True)
        results_df.reset_index(inplace=True)
    
    # Final check to ensure no NaN values
    results_df = results_df.fillna({
        'call_volume': 0.0,
        'put_volume': 0.0,
        'call_oi': 0.0,
        'put_oi': 0.0,
        'cp_volume_ratio': 0.0,
        'cp_oi_ratio': 0.0,
        'daily_sentiment': 'no_trades',
        'bullish_volume': 0.0,
        'bearish_volume': 0.0
    })
    
    # only keep the 4 columns (Note: you had 'call_volumne' with typo, I fixed it)
    columns = ['date', 'call_volume', 'put_volume', 'cp_volume_ratio', 'cp_oi_ratio', 'bullish_volume', 'bearish_volume']
    results_df = results_df[columns]

    print(results_df.to_string())  # This will show all values
    print(results_df.dtypes)  # This will show the data types of each column
    return results_df

def get_historical_cp_ratios_with_sentiments(symbol, dates_df, api_key):
    """
    Get C/P ratios for dates provided in DataFrame, continuing from where we left off
    """
    
    # Check if we already have saved results
    file_path = f"{symbol}-cp_ratios_sentiment_w_volume.csv"
    
    if os.path.exists(file_path):
        # Load existing results
        existing_results = pd.read_csv(file_path)
        
        # Convert date column to datetime for proper comparison
        existing_results['date'] = pd.to_datetime(existing_results['date'])
        dates_df['date'] = pd.to_datetime(dates_df['date'])
        
        # Find dates that haven't been processed yet
        processed_dates = set(existing_results['date'].dt.strftime('%Y-%m-%d'))
        all_dates = set(dates_df['date'].dt.strftime('%Y-%m-%d'))
        remaining_dates = all_dates - processed_dates
        
        # Filter dates_df to only include remaining dates
        dates_df = dates_df[dates_df['date'].dt.strftime('%Y-%m-%d').isin(remaining_dates)]
        
        # Convert existing results back to list format to continue appending
        results = existing_results.to_dict('records')
        
        print(f"Found existing file with {len(existing_results)} entries")
        print(f"Continuing with {len(remaining_dates)} remaining dates")
    else:
        results = []
        print(f"Starting fresh - no existing file found")
    
    # Convert dates back to string format for API calls
    dates_df['date'] = dates_df['date'].dt.strftime('%Y-%m-%d')
    
    for date_str in dates_df['date']:
        try:
            print(f"Fetching data for {date_str}")
            
            # [... rest of your existing code for fetching and processing data ...]
            
            response = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "HISTORICAL_OPTIONS",
                    "symbol": symbol,
                    "date": date_str,
                    "apikey": api_key
                }
            )
            
            # [... rest of your processing code ...]
            
            # Append to results
            results.append({
                'date': date_str,
                'call_volume': float(call_vol),
                'put_volume': float(put_vol),
                'call_oi': float(call_oi),
                'put_oi': float(put_oi),
                'cp_volume_ratio': float(cp_volume_ratio),
                'cp_oi_ratio': float(cp_oi_ratio),
                'daily_sentiment': day_sentiment,
                'bullish_volume': float(bullish_volume),
                'bearish_volume': float(bearish_volume),
            })
            
            # Save intermediate results after each successful fetch
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(file_path, index=False)
            print(f"Saved intermediate results: {len(results)} total entries")
            
            # Sleep for rate limit
            time.sleep(12)
            
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            continue
    
    # Create final DataFrame
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.set_index('date', inplace=True)
        results_df.sort_index(inplace=True)
    
    return results_df

# Example usage:
# Assuming your DataFrame is called 'your_dates_df' and has a 'date' column
api_key = "H896AQT2GYE4ZO8Z"
symbol = "TSLA"

master_df = pd.read_csv(symbol+'_TMP.csv')
columns=['date']
your_dates_df = master_df[['date']].copy()  # Create a proper copy
your_dates_df.loc[:, 'date'] = pd.to_datetime(your_dates_df['date'])

your_dates_df['date'] = pd.to_datetime(your_dates_df['date'])
# change date to string
your_dates_df['date'] = your_dates_df['date'].dt.strftime('%Y-%m-%d')

cp_ratios_df = get_historical_cp_ratios_with_sentiments_new(symbol, your_dates_df, api_key)
print(cp_ratios_df)

file_path = symbol+ '-' + "cp_ratios_sentiment_w_volume.csv"
cp_ratios_df.to_csv(file_path)
