import pandas as pd
from pandas import DataFrame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
import fetchBulkData
import fetchBulkDataCached
import etl
import logging
import os
from tqdm import tqdm
import time

    
def download_data(config, param, use_global_cache=False):
    max_retries = 12
    retry_count = 0
    sleep_time = 300
    
    while retry_count < max_retries:
        try:
            if use_global_cache:
                df = fetchBulkDataCached.fetch_all_data(config, param)
            else:
                df = fetchBulkData.fetch_all_data(config, param)
            # print(df.head)
            break
            
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                # If we've reached the maximum retries, propagate the error
                raise Exception(f"Failed to fetch all data after {max_retries} attempts: {str(e)}")
            
            print(f"Attempt {retry_count} failed: {str(e)}. Retrying in {sleep_time/60} minutes...")
            time.sleep(sleep_time)  # Sleep for the specified time (default 5 minutes)
    # Use the bulk fetch code to get all the needed stuff in one call
    
    
    symbol = param['symbol']
    # df.to_csv(symbol + '_PRE_TMP'+'.csv', index=False) # we want to save the date index
    
    #########################################################################
    # add a normalize volume
    # No date filtering
    #
    df['volume_norm'] = df['volume'] / df['volume'].rolling(window=20).mean()

    # there might be holes before the window of 20 so needs to backfill


    # add volitility
    df['daily_return'] = df['adjusted close'].pct_change()

    if param.get('volatility_window') is not None:
        window_size = param['volatility_window']
    else:
        window_size = 13
    df['volatility'] = df['daily_return'].rolling(window=window_size).std()

    # add VWAP
    # Assuming 'high', 'low', and 'close' columns exist in your DataFrame
    df['typical_price'] = (df['high'] + df['low'] + df['adjusted close']) / 3
    df['tp_volume'] = df['typical_price'] * df['volume']
    df['VWAP'] = df['tp_volume'].cumsum() / df['volume'].cumsum()

    df = etl.fill_data(df)

    #############################
    # Add volume oscillator
    #
    # Setting periods for fast and slow moving averages
    fast_period = 3
    slow_period = 5

    # Calculate both EMAs as a separate DataFrame
    ema_df = pd.DataFrame({
        'Fast_EMA': df['volume'].ewm(span=fast_period, adjust=False).mean(),
        'Slow_EMA': df['volume'].ewm(span=slow_period, adjust=False).mean()
    })

      # Calculate the Volume Oscillator
    ema_df['Volume_Oscillator'] = (ema_df['Fast_EMA'] - ema_df['Slow_EMA']) / ema_df['Slow_EMA'] *100
    # Concatenate with original DataFrame
    ema_df['volume_volatility'] = df['volume'].pct_change().rolling(15).std()

    df = pd.concat([df, ema_df], axis=1)

    # Print the result
    # print(df[['volume', 'Fast_EMA', 'Slow_EMA', 'Volume_Oscillator']])
    df.drop(columns=['Fast_EMA', 'Slow_EMA'], inplace=True)

    ####################################
    # Calculate our own SPY Oscillator
    #
    # Calculate the short-term EMA (22-day EMA by default)
    short_window = 22
    df['short_ema'] = df['SPY_close'].ewm(span=short_window, adjust=False).mean()
    
    long_window = 50
    # Calculate the long-term EMA (50-day EMA by default)
    df['long_ema'] = df['SPY_close'].ewm(span=long_window, adjust=False).mean()
    
    # Calculate the oscillator: (long-term EMA - short-term EMA) / short-term EMA * 100
    df['calc_spy_oscillator'] = ((df['long_ema'] - df['short_ema']) / df['short_ema']) * 100
    df.drop(columns=['long_ema', 'short_ema'], inplace=True)

    ####################################
    # Calculate our own SP500 Oscillator
    #
    # Calculate the short-term EMA (22-day EMA by default)
    short_window = 22
    df['short_ema'] = df['SPY_close'].ewm(span=short_window, adjust=False).mean()
    
    long_window = 50
    # Calculate the long-term EMA (50-day EMA by default)
    df['long_ema'] = df['SPY_close'].ewm(span=long_window, adjust=False).mean()
    
    # Calculate the oscillator: (long-term EMA - short-term EMA) / short-term EMA * 100
    df['calc_SP500_oscillator'] = ((df['long_ema'] - df['short_ema']) / df['short_ema']) * 100
    df.drop(columns=['long_ema', 'short_ema'], inplace=True)

    num_data_points = len(df)
    df = df.reset_index()
    display_date_range = "from " + str(df['date'].iloc[0]) + " to " + str(df['date'].iloc[num_data_points-1])
    logging.info("Number data points: " + str(num_data_points) + " and display date range= " + str(display_date_range))

    df.sort_values(by='date', inplace=True) # probably superfluous but does not hurt

    #################################################################
    # Add days of week and month 
    #
    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    #################################################################
    # Add lag features 
    #
    df['price_lag_1'] = df['adjusted close'].shift(1)
    df['price_lag_5'] = df['adjusted close'].shift(5)
    df['price_lag_15'] = df['adjusted close'].shift(15)
    
    df['price_change_1'] = df['adjusted close'].pct_change()
    df['price_change_5'] = df['adjusted close'].pct_change(periods=5)
    df['price_change_15'] = df['adjusted close'].pct_change(periods=15)

    #################################################################
    # caluculate all the relative features
    #
    df['relative_volume'] = df['volume'] / df['volume'].rolling(window=30).mean()
    df['relative_close'] = df['adjusted close'] / df['adjusted close'].rolling(window=30).mean()
    df['relative_high'] = df['high'] / df['high'].rolling(window=30).mean()
    df['relative_low'] = df['low'] / df['low'].rolling(window=30).mean()


    #################################################################
    # calculate last close to next high/low features (NOT FOR ML)
    #
    df['l_close_1_next_high'] = df['high'] / df['adjusted close'].shift(-1)
    df['l_close_1_next_low'] = df['low'] / df['adjusted close'].shift(-1)
    df['rolling_next_high'] = df['l_close_1_next_high'].rolling(window=20).mean()
    df['rolling_next_low'] = df['l_close_1_next_low'].rolling(window=20).mean()
    df['next_pred_high'] = df['rolling_next_high'] * df['adjusted close']
    df['next_pred_low'] = df['rolling_next_low'] * df['adjusted close']


    #################################################################
    # calculate all the meta labelling fields
    #
    # vs SPY and QQQ
    df['rs_sp500'] = df['adjusted close'] / df['SPY_close']
    df['rs_nasdaq'] = df['adjusted close'] / df['qqq_close']
    
    # 2. Relative trends (like your SOX implementation)
    df['rs_sp500_trend'] = df['rs_sp500'].pct_change(15)
    df['rs_nasdaq_trend'] = df['rs_nasdaq'].pct_change(15)

    if symbol == 'NVDA' or symbol == 'SMCI' or symbol == 'CRDO' or symbol == 'TSM' or symbol == 'ANET' or symbol == 'ALAB' or symbol == 'TSLA':
        # 1. Relative Strength Features
        # vs AMD/INtc and Broadcom
        df['rs_amd'] = df['adjusted close'] / df['AMD_close']
        df['rs_intc'] = df['adjusted close'] / df['INTC_close']
        df['rs_avgo'] = df['adjusted close'] / df['AVGO_close']
        # df['rs_sox'] = df['adjusted close'] / df['SOX']
        df['rs_smh'] = df['adjusted close'] / df['SMH_close']
        df['rs_amd_trend'] = df['rs_amd'].pct_change(5)
        df['rs_intc_trend'] = df['rs_intc'].pct_change(5)
        df['rs_avgo_trend'] = df['rs_avgo'].pct_change(5)
        # Add multiple trend windows for different signals
        # df['rs_sox_trend_short'] = df['rs_sox'].pct_change(5)   # 1 week
        # df['rs_sox_trend_med'] = df['rs_sox'].pct_change(15)    # 3 weeks
        # df['rs_sox_trend_long'] = df['rs_sox'].pct_change(30)   # 6 weeks
        # df['rs_sox_volatility'] = df['rs_sox_trend_med'].rolling(15).std()
        df['rs_smh_trend'] = df['rs_smh'].pct_change(15)
        df['tsm_price_change_1'] = df['TSMC_close'].pct_change()

    if symbol == 'PLTR':
        df['rs_ita'] = df['adjusted close'] / df['ITA_close']
        df['rs_igv'] = df['adjusted close'] / df['IGV_close']
        df['rs_ita_trend'] = df['rs_ita'].pct_change(5)
        df['rs_igv_trend'] = df['rs_igv'].pct_change(5)
    
    if symbol == 'APP':
        df['rs_gamr'] = df['adjusted close'] / df['GAMR_close']
        df['rs_socl'] = df['adjusted close'] / df['SOCL_close']
        df['rs_gamr_trend'] = df['rs_gamr'].pct_change(5)
        df['rs_socl_trend'] = df['rs_socl'].pct_change(5)
        df['rs_u'] = df['adjusted close'] / df['U_close']
        df['rs_ttwo'] = df['adjusted close'] / df['TTWO_close']
        df['rs_apps'] = df['adjusted close'] / df['APPS_close']
        df['rs_u_trend'] = df['rs_u'].pct_change(5)
        df['rs_ttwo_trend'] = df['rs_ttwo'].pct_change(5)
        df['rs_apps_trend'] = df['rs_apps'].pct_change(5)

    if symbol == 'META':
        df['rs_socl'] = df['adjusted close'] / df['SOCL_close']
        df['rs_socl_trend'] = df['rs_socl'].pct_change(5)
        df['rs_xlc'] = df['adjusted close'] / df['XLC_close']
        df['rs_xlc_trend'] = df['rs_xlc'].pct_change(5)
        

    # merge all the features together
    df = etl.fill_data(df)
    
 
    return df, num_data_points, display_date_range

def calculate_regime(df: DataFrame)-> DataFrame:
    '''
    Calculate the regime of the stock
    just add a column to the dataframe
    do not do other deletion or reindexing
    '''

    # the daily_return field should already be in the df
    # 2. Calculate rolling volatility over a 15-day window
    df['rolling_vol'] = df['daily_return'].rolling(window=15).std()

    # 1) Choose a threshold. Common ways include:
    #    - the median of historical volatility
    #    - a particular percentile (e.g., 70th, 80th)
    #    - or a fixed value (less common unless you have domain insight)

    threshold = df['rolling_vol'].median()
    # or, for the 75th percentile:
    # threshold = df['rolling_vol'].quantile(0.75)

    # 2) Label each row based on whether rolling_vol exceeds threshold
    df['vol_regime'] = np.where(df['rolling_vol'] > threshold, 'HIGH_VOL', 'LOW_VOL')

    # Now df['vol_regime'] contains "HIGH_VOL" or "LOW_VOL" labels for each row.
    return df

class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        """Save model when validation loss decreases."""
        self.best_model = model.state_dict().copy()

class LossIncreaseStopping:
    def __init__(self, threshold=0.01, consecutive_checks=2, verbose=True):
        """
        Stop training when loss starts increasing beyond a threshold.
        
        Args:
            threshold (float): How much relative increase in loss is acceptable
                             (e.g., 0.01 means 1% increase)
            consecutive_checks (int): Number of consecutive increases needed before stopping
            verbose (bool): If True, prints information about loss increases
        """
        self.threshold = threshold
        self.consecutive_checks = consecutive_checks
        self.verbose = verbose
        self.previous_loss = None
        self.increase_count = 0
        self.early_stop = False
        self.best_loss = float('inf')
        self.best_model = None
        
    def __call__(self, current_loss, model):
        # Always save if it's the best loss seen
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_model = model.state_dict().copy()
        
        if self.previous_loss is not None:
            # Calculate relative increase
            relative_increase = (current_loss - self.previous_loss) / self.previous_loss
            
            if relative_increase > self.threshold:
                self.increase_count += 1
                if self.verbose:
                    print(f'Loss increased by {relative_increase*100:.2f}% ',
                          f'({self.increase_count}/{self.consecutive_checks})')
                
                if self.increase_count >= self.consecutive_checks:
                    self.early_stop = True
                    if self.verbose:
                        print(f'Stopping: Loss increased for {self.consecutive_checks}',
                              'consecutive epochs')
            else:
                self.increase_count = 0
                
        self.previous_loss = current_loss

def train_with_early_stopping(model, train_loader, val_loader, num_epochs, optimizer, 
                            criterion, device, scheduler=None, l1_lambda=0):
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=7, min_delta=1e-4)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            num_classes = outputs.size(-1)
            outputs = outputs.view(-1, num_classes)
            labels = labels.view(-1)
            
            loss = criterion(outputs, labels)
            
            # L1 regularization if specified
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for name, p in model.named_parameters() 
                            if 'weight' in name)
                loss = loss + l1_lambda * l1_norm
                
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                
                outputs = model(inputs)
                num_classes = outputs.size(-1)
                outputs = outputs.view(-1, num_classes)
                labels = labels.view(-1)
                
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Step scheduler if provided
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        
        # Print epoch summary
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # Early stopping check
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            # Load best model
            model.load_state_dict(early_stopping.best_model)
            break
    
    return model

class TrendBasedStopping:
    def __init__(self, window_size=10, threshold=0.05, verbose=True):
        """
        Stop training when the trend of loss starts increasing.
        
        Args:
            window_size (int): Number of epochs to consider for trend
            threshold (float): How much relative increase in trend is acceptable
            verbose (bool): If True, prints information about loss trends
        """
        self.window_size = window_size
        self.threshold = threshold
        self.verbose = verbose
        self.losses = []
        self.early_stop = False
        self.best_loss = float('inf')
        self.best_model = None
        
    def calculate_trend(self):
        """Calculate the average rate of change over the window"""
        if len(self.losses) < self.window_size:
            return -float('inf')  # Negative trend (improving) if not enough data
        
        recent_losses = self.losses[-self.window_size:]
        # Calculate average rate of change
        changes = [(recent_losses[i] - recent_losses[i-1]) / recent_losses[i-1] 
                  for i in range(1, len(recent_losses))]
        return sum(changes) / len(changes)
    
    def __call__(self, current_loss, model):
        self.losses.append(current_loss)
        
        # Save best model
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_model = model.state_dict().copy()
        
        # Calculate trend if we have enough data
        if len(self.losses) >= self.window_size:
            trend = self.calculate_trend()
            
            # if self.verbose:
            #     print(f'Current trend: {trend*100:.2f}% per epoch')
            
            # Stop if trend is positive (increasing) beyond threshold
            if trend > self.threshold:
                self.early_stop = True
                if self.verbose:
                    print(f'Stopping: Loss trend increased by {trend*100:.2f}% per epoch')
                    print(f'Best loss achieved: {self.best_loss:.4f}')
                return True
        
        return False

def train_with_trend_based_stopping(model, train_loader, val_loader, num_epochs, optimizer, 
                            criterion, device, scheduler=None, l1_lambda=0):
    # Initialize early stopping
    # stopping = analysisUtil.LossIncreaseStopping(
    #     threshold=0.01,  # Stop if loss increases by more than 1%
    #     consecutive_checks=2  # Need 2 consecutive increases to stop
    # )
    stopping = TrendBasedStopping(
        window_size=10,  # Look at trends over 5 epochs
        threshold=0.05  # Stop if trend shows 1% increase per epoch
    )
        
    for epoch in range(num_epochs):
        # Training phase
        total_loss = 0
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            num_classes = outputs.size(-1)
            outputs = outputs.view(-1, num_classes)
            labels = labels.view(-1)
            
            loss = criterion(outputs, labels)
            
            # L1 regularization if specified
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for name, p in model.named_parameters() 
                            if 'weight' in name)
                loss = loss + l1_lambda * l1_norm
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
    
        # Check if we should stop
        stopping(avg_loss, model)
        
        if stopping.early_stop:
            print(f"Stopping at epoch {epoch+1} due to increasing trend")
            # Load the best model we saw
            model.load_state_dict(stopping.best_model)
            break
        
        # Print epoch summary
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return model

def train_with_trend_based_stopping_OLDNEW(model, train_loader, val_loader, num_epochs, optimizer, 
                            criterion, device, scheduler=None, l1_lambda=0):
      # Initialize early stopping
    # stopping = analysisUtil.LossIncreaseStopping(
    #     threshold=0.01,  # Stop if loss increases by more than 1%
    #     consecutive_checks=2  # Need 2 consecutive increases to stop
    # )
    stopping = TrendBasedStopping(
        window_size=10,  # Look at trends over 5 epochs
        threshold=0.05  # Stop if trend shows 1% increase per epoch
    )
    
    # Create a progress bar for all epochs
    epoch_bar = tqdm(range(num_epochs), desc='Training')

    for epoch in epoch_bar:
        # epoch_desc = f'Epoch {epoch+1}/{num_epochs}'

        # Training phase
        total_loss = 0
        model.train()

        # Create progress bar for the training batches with a dynamic description
        # train_bar = tqdm(train_loader, desc=epoch_desc, leave=False)
    
        for inputs, labels in train_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            num_classes = outputs.size(-1)
            outputs = outputs.view(-1, num_classes)
            labels = labels.view(-1)
            
            loss = criterion(outputs, labels)
            
            # L1 regularization if specified
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for name, p in model.named_parameters() 
                            if 'weight' in name)
                loss = loss + l1_lambda * l1_norm
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            # Update progress bar with current loss
            # train_bar.set_postfix(loss=f'{loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
    
        # Check if we should stop
        stopping(avg_loss, model)
        
        if stopping.early_stop:
            print(f"Stopping at epoch {epoch+1} due to increasing trend")
            # Load the best model we saw
            model.load_state_dict(stopping.best_model)
            break

        # Only update the description every 10 epochs
        if (epoch + 1) % 10 == 0:
            epoch_bar.set_postfix(loss=f'{avg_loss:.4f}')
            # Force the progress bar to update
            epoch_bar.refresh()
    
    return model

def train_with_trend_based_stopping_NEW(model, train_loader, val_loader, num_epochs, optimizer, 
                            criterion, device, scheduler=None, l1_lambda=0):
    # Initialize early stopping
    stopping = TrendBasedStopping(
        window_size=10,  # Look at trends over 10 epochs
        threshold=0.05   # Stop if trend shows 5% increase per epoch
    )
    
    # Instead of updating on every epoch, we'll manually control the progress bar
    print(f"Training for {num_epochs} epochs")
    pbar = tqdm(total=num_epochs, desc='Training')
    last_update = 0

    for epoch in range(num_epochs):
        # Training phase
        total_loss = 0
        model.train()
    
        for inputs, labels in train_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            num_classes = outputs.size(-1)
            outputs = outputs.view(-1, num_classes)
            labels = labels.view(-1)
            
            loss = criterion(outputs, labels)
            
            # L1 regularization if specified
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for name, p in model.named_parameters() 
                            if 'weight' in name)
                loss = loss + l1_lambda * l1_norm
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
    
        # Check if we should stop
        stopping(avg_loss, model)
        
        if stopping.early_stop:
            # Update progress bar one last time
            pbar.update(epoch + 1 - last_update)
            pbar.set_postfix(loss=f'{avg_loss:.4f}', status="Early stopping")
            pbar.close()
            print(f"Stopping at epoch {epoch+1} due to increasing trend")
            # Load the best model we saw
            model.load_state_dict(stopping.best_model)
            break

        # Only update the progress bar every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Update by the number of epochs since last update
            pbar.update(epoch + 1 - last_update)
            pbar.set_postfix(loss=f'{avg_loss:.4f}')
            last_update = epoch + 1
    
    # Make sure to close the progress bar at the end
    if not stopping.early_stop:
        pbar.update(num_epochs - last_update)
        pbar.close()
    
    return model

def build_ai_sector_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a DataFrame with AI sector index information.
    Returns: the df that has a new ai_index column added
    """

    # Define AI basket tickers that are already columns in your df
    ai_basket = ['MSFT', 'META','GOOGL', 'CRM']

    # Create equal-weighted AI index as a new column
    df['ai_index'] = df[ai_basket].mean(axis=1)

    # Or if you want custom weights:
    weights = {
        'NVDA': 0.3,
        'AMD': 0.2,
        'MSFT': 0.2,
        'GOOGL': 0.15,
        'CRM': 0.15
    }
    df['ai_index'] = sum(df[ticker] * weight for ticker, weight in weights.items())

def fetchDateAndClosing(param: dict[str]):
    symbol = param['symbol']
    file_path = symbol + '_TMP.csv'
    if os.path.isfile(file_path):
        # file exist
        df = pd.read_csv(file_path)
    else:
        print('>>> ERROR, file '+ file_path + ' Not found!')
        return None, None
    
    # Get the last row of the DataFrame
    last_row = df.iloc[-1]

    # Fetch the values of 'date' and 'close' columns
    # last_date = last_row['date']
    # last_close = last_row['close']

    # use the date from param 'last_date'
    # !!! Debug this tomorrow
    last_date = param['end_date']
    last_close = df.loc[df['date'] == last_date, 'adjusted close'].values[0]
    return last_date, last_close

def processDeltaFromTodayResults( symbol: str, incr_df: DataFrame, dateStr: str, closingPrice: float, comment: str):
    file_path = symbol+ "_" + "15d_from_today_predictions.csv"
    num_of_days = 15
    df: DataFrame

    if os.path.isfile(file_path):
        # file exist
        df = pd.read_csv(file_path,index_col=False)
        # df['date'] = pd.to_datetime(df['date'])  # Convert to datetime if it's not already
    else:
        columns = []
        columns.insert(0, 'date')
        columns.insert(1,'close')
        columns.extend([f'p{i}' for i in range(1, num_of_days + 1)])
        columns.insert(16, 'comment')
        df = DataFrame(columns= columns)
        df = df.reset_index(drop=True)

    df.loc[len(df)] = np.nan  # This appends a row filled with NaNs at the end

    # Step 2: Set values for specific columns in the new row
    df.loc[len(df)-1, 'date'] = dateStr  
    df.loc[len(df)-1, 'close'] = closingPrice 
    input_col = ['p1', 'p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15']
    for target_item in input_col:
        df.loc[len(df)-1, target_item] = incr_df.iloc[0][target_item] 

    df.loc[len(df)-1, 'comment'] = comment 

    # now store this back to disk
    df.to_csv(file_path, index=False, header=True) # we want to save the date index

    print(">>> Final result delta from today")
    print(df)

