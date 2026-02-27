import sys
import pandas as pd
from pandas import DataFrame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from joblib import dump, load

from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import json
import time
from datetime import datetime
from datetime import timedelta
import pytz
import os
import math
import fetchBulkData
import etl
import processPrediction
import analysisUtil

print("All libraries loaded for "+ __file__)


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced multi-class classification.

    gamma=2.0: down-weights easy correct predictions (neutral class), forcing
    the model to focus training effort on hard minority examples (UP/DN).
    label_smoothing: prevents overconfidence on the dominant neutral class by
    softening the target distribution.
    """
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        ce_loss = F.cross_entropy(
            input, target,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class StockTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads, num_layers, dropout_rate):
        super(StockTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)

         # Initialize weights (per advise to cut down randomness)
        self.init_weights()
    
    def init_weights(self):
        # Initialize Linear Layer with Xavier Uniform
        nn.init.xavier_uniform_(self.fc.weight)
        # Set biases to zero
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.transformer_encoder(x)
        # Additional layers as necessary, e.g., a fully connected layer
        x = self.fc(x)
        return x

# new class to handle regularization
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads, num_layers, dropout_rate, embedded_dim):
        super(TransformerModel, self).__init__()
        
        # Set the embedding dimension (can be a parameter or fixed)
        self.embedding_dim = embedded_dim  # You can make this a parameter if desired
        
        # Input projection layer to match transformer's expected input dimension
        self.input_projection = nn.Linear(input_dim, self.embedding_dim)
        
        # Positional encoding with dropout
        self.positional_encoding = PositionalEncoding(self.embedding_dim, dropout_rate)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, 
            nhead=num_heads, 
            dropout=dropout_rate, 
            batch_first=True  # Set to True if input shape is [batch_size, seq_length, embedding_dim]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc = nn.Linear(self.embedding_dim, num_classes)
    
    def forward(self, src):
    # src: [batch_size, input_dim]
        # print("Input x shape:", src.shape)  # Should be [batch_size, seq_length, embedding_dim]
        if src.dim() == 2:
            src = src.unsqueeze(1)  # Now src has shape [batch_size, 1, input_dim]

        x = self.input_projection(src)  # [batch_size, seq_length, embedding_dim]
        x = self.positional_encoding(x)  # [batch_size, seq_length, embedding_dim]
        # print("After positional encoding x shape:", src.shape)  # Should be unchanged

        x = self.transformer_encoder(x)  # [batch_size, seq_length, embedding_dim]
        # print("After transformer encoder output shape:", x.shape)  # Should be [batch_size, seq_length, embedding_dim]

        # x = x.mean(dim=1)  # [batch_size, embedding_dim]
        # print("After mean pooling output shape:", x.shape)  # Should be [batch_size, embedding_dim]

        x = self.dropout(x)
        x = self.fc(x)  # [batch_size, num_classes]
        # x = self.activation(x)    # remove activation for multi class
        # print("Final output shape:", x.shape)  # Should be [batch_size, num_classes]
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout_rate, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        position = torch.arange(0, max_len).unsqueeze(1).float()  # Shape: [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )  # Shape: [embedding_dim/2]

        pe = torch.zeros(max_len, embedding_dim)  # Shape: [max_len, embedding_dim]
        pe[:, 0::2] = torch.sin(position * div_term)  # Shape: [max_len, embedding_dim/2]
        pe[:, 1::2] = torch.cos(position * div_term)  # Shape: [max_len, embedding_dim/2]

        pe = pe.unsqueeze(0)  # Shape: [1, max_len, embedding_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_length, embedding_dim]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def build_model(param: dict, input_dim: int, num_classes: int = 3) -> nn.Module:
    """
    Model factory: instantiate a model based on param['model_type'].
    Defaults to 'transformer' if the key is absent (backward compatible).

    To add a new architecture:
      1. Define a new nn.Module class above this function
      2. Add an elif branch here
      3. Add 'model_type': '<name>' to the param dict for that stock/config
    """
    model_type = param.get('model_type', 'transformer')

    if model_type == 'transformer':
        return TransformerModel(
            input_dim=input_dim,
            num_classes=num_classes,
            num_heads=param['headcount'],
            num_layers=param['num_layers'],
            dropout_rate=param['dropout_rate'],
            embedded_dim=param['embedded_dim'],
        )
    elif model_type == 'multi_horizon_transformer':
        return MultiHorizonTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            num_heads=param['headcount'],
            num_layers=param['num_layers'],
            dropout_rate=param['dropout_rate'],
            embedded_dim=param['embedded_dim'],
            num_horizons=param.get('num_horizons', 15),
        )
    # elif model_type == 'lstm':
    #     return LSTMModel(...)
    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Supported types: ['transformer', 'multi_horizon_transformer']"
        )


CLASS_LABELS = {0: '__', 1: 'UP', 2: 'DN'}  # 0=flat, 1=up, 2=down — DO NOT CHANGE


class MultiHorizonHead(nn.Module):
    """Architecture-agnostic multi-horizon classification head.

    Input:  (batch, d_model)         — pooled encoder representation.
    Output: (batch, num_horizons, 3) — logits for all horizons at once.

    Reusable across encoders (Transformer, LSTM, etc.) via composition.
    """
    def __init__(self, d_model: int, num_horizons: int = 15, num_classes: int = 3):
        super().__init__()
        self.num_horizons = num_horizons
        self.num_classes = num_classes
        self.fc = nn.Linear(d_model, num_horizons * num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, d_model)
        return self.fc(z).view(-1, self.num_horizons, self.num_classes)


class MultiHorizonTransformer(nn.Module):
    """Transformer encoder + multi-horizon head.

    Predicts all 15 horizons in one forward pass.
    Uses last-token pooling: z = encoder_output[:, -1, :].
    Class encoding: 0=flat, 1=UP, 2=DOWN (CLASS_LABELS).
    Output shape: (batch, num_horizons, num_classes).
    """
    def __init__(self, input_dim: int, num_classes: int, num_heads: int,
                 num_layers: int, dropout_rate: float, embedded_dim: int,
                 num_horizons: int = 15):
        super().__init__()
        self.embedding_dim = embedded_dim
        self.input_projection = nn.Linear(input_dim, embedded_dim)
        self.positional_encoding = PositionalEncoding(embedded_dim, dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedded_dim,
            nhead=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.head = MultiHorizonHead(embedded_dim, num_horizons=num_horizons, num_classes=num_classes)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: (batch, D) or (batch, L, D)
        if src.dim() == 2:
            src = src.unsqueeze(1)          # (batch, 1, D)
        x = self.input_projection(src)      # (batch, L, d_model)
        x = self.positional_encoding(x)     # (batch, L, d_model)
        x = self.transformer_encoder(x)     # (batch, L, d_model)
        x = self.dropout(x)
        z = x[:, -1, :]                     # last-token pooling: (batch, d_model)
        return self.head(z)                 # (batch, num_horizons, num_classes)


def build_multi_horizon_labels(df: DataFrame, num_horizons: int = 15):
    """Generate forward-return class labels for all horizons simultaneously.

    Thresholds:
        h=1..5:   ±3%  (short-term)
        h=6..15:  ±5%  (medium-term)

    Class encoding (matches CLASS_LABELS):
        0 = flat/neutral
        1 = UP  (return >= thr)
        2 = DOWN (return <= -thr)

    Returns:
        labels:   np.ndarray shape (n_usable, num_horizons), dtype int64
        n_usable: int — number of usable rows (len(df) - num_horizons)
    """
    closes = df['adjusted close'].values.astype(np.float64)
    n = len(closes)
    n_usable = n - num_horizons
    if n_usable <= 0:
        raise ValueError(f"Not enough data: {n} rows, need at least {num_horizons + 1}")
    labels = np.zeros((n_usable, num_horizons), dtype=np.int64)
    for h in range(1, num_horizons + 1):
        thr = 0.03 if h <= 5 else 0.05
        future_closes = closes[h:h + n_usable]
        current_closes = closes[:n_usable]
        ret = (future_closes - current_closes) / current_closes
        labels[:, h - 1] = np.where(ret >= thr, 1, np.where(ret <= -thr, 2, 0))
    return labels, n_usable


def DEPRECATED_download_data(config, param):

    # Use the bulk fetch code to get all the needed stuff in one call
    df = fetchBulkData.fetch_all_data(config, param)
    # print(df.head)
    df = etl.fill_data(df)      # added on 10/15/24 
 
    # print('>DF is cleaned up after ETL')
    # print(df.head)
    
    #########################################################################
    # add a normalize volume
    # No date filtering
    #
    df['volume_norm'] = df['volume'] / df['volume'].rolling(window=20).mean()

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

    df['volume_volatility'] = df['volume'].pct_change().rolling(15).std()

    # df = etl.fill_data(df)

    #############################
    # Add volume oscillator
    #
    # Setting periods for fast and slow moving averages
    fast_period = 3
    slow_period = 5

    # Calculate the fast and slow EMA of volume
    df['Fast_EMA'] = df['volume'].ewm(span=fast_period, adjust=False).mean()
    df['Slow_EMA'] = df['volume'].ewm(span=slow_period, adjust=False).mean()

    # Calculate the Volume Oscillator
    df['Volume_Oscillator'] = df['Fast_EMA'] - df['Slow_EMA'] / df['Slow_EMA'] *100

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
    df['short_ema'] = df['SP500'].ewm(span=short_window, adjust=False).mean()
    
    long_window = 50
    # Calculate the long-term EMA (50-day EMA by default)
    df['long_ema'] = df['SP500'].ewm(span=long_window, adjust=False).mean()
    
    # Calculate the oscillator: (long-term EMA - short-term EMA) / short-term EMA * 100
    df['calc_SP500_oscillator'] = ((df['long_ema'] - df['short_ema']) / df['short_ema']) * 100
    df.drop(columns=['long_ema', 'short_ema'], inplace=True)

    num_data_points = len(df)
    df = df.reset_index()
    display_date_range = "from " + str(df['date'].iloc[0]) + " to " + str(df['date'].iloc[num_data_points-1])
    print("Number data points: " + str(num_data_points) + " and display date range= " + str(display_date_range))

    df.sort_values(by='date', inplace=True)

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
    # calculate all the meta labelling fields
    #
    # 1. Relative Strength Features
    
    # vs AMD/INtc and Broadcom
    df['rs_amd'] = df['adjusted close'] / df['AMD_close']
    df['rs_intc'] = df['adjusted close'] / df['INTC_close']
    df['rs_avgo'] = df['adjusted close'] / df['AVGO_close']
    df['rs_sox'] = df['adjusted close'] / df['SOX']
    df['rs_smh'] = df['adjusted close'] / df['SMH_close']
    df['rs_amd_trend'] = df['rs_amd'].pct_change(15)
    df['rs_intc_trend'] = df['rs_intc'].pct_change(15)
    df['rs_avgo_trend'] = df['rs_avgo'].pct_change(15)
    
    # Add multiple trend windows for different signals
    df['rs_sox_trend_short'] = df['rs_sox'].pct_change(5)   # 1 week
    df['rs_sox_trend_med'] = df['rs_sox'].pct_change(15)    # 3 weeks
    df['rs_sox_trend_long'] = df['rs_sox'].pct_change(30)   # 6 weeks
    df['rs_sox_volatility'] = df['rs_sox_trend_med'].rolling(15).std()
    df['rs_smh_trend'] = df['rs_smh'].pct_change(15)
    df['tsm_price_change_1'] = df['TSMC_close'].pct_change()

    # vs SPY and QQQ
    df['rs_sp500'] = df['adjusted close'] / df['SPY_close']
    df['rs_nasdaq'] = df['adjusted close'] / df['qqq_close']
    
    # 2. Relative trends (like your SOX implementation)
    df['rs_sp500_trend'] = df['rs_sp500'].pct_change(15)
    df['rs_nasdaq_trend'] = df['rs_nasdaq'].pct_change(15)

    # 3. Idiosyncratic return: stock alpha vs market / sector
    #    ret_Nd_rel_X = stock N-day return minus reference N-day return.
    #    Decomposes the stock's move into market/sector drift vs stock-specific alpha.
    #    rs_smh_trend / rs_sp500_trend are 15-day only; these add 5d and 10d windows.
    df['price_change_10'] = df['adjusted close'].pct_change(10)
    df['ret_5d_rel_SPY']  = df['price_change_5']  - df['SPY_close'].pct_change(5)
    df['ret_10d_rel_SPY'] = df['price_change_10'] - df['SPY_close'].pct_change(10)

    if 'SMH_close' in df.columns and df['SMH_close'].notna().sum() > 0:
        df['ret_5d_rel_SMH']  = df['price_change_5']  - df['SMH_close'].pct_change(5)
        df['ret_10d_rel_SMH'] = df['price_change_10'] - df['SMH_close'].pct_change(10)
    else:
        df['ret_5d_rel_SMH']  = np.nan
        df['ret_10d_rel_SMH'] = np.nan

    # merge all the features together
    print("shape of feature data in a df: "+ str(df.shape))
    return df, num_data_points, display_date_range

#################################################################################
# calculate label
def calculate_label(df: DataFrame, param: dict[str]):

    # Define your threshold for significant change (e.g., 2%)
    threshold = param['threshold']

    df['adjusted close'] = df['adjusted close'].astype(float)
    # Calculate the percentage change after [target_size] days
    price_change = (df['adjusted close'].shift(-param['target_size']) - df['adjusted close']) / df['adjusted close']

    # Label based on the defined threshold
    # df['label'] = 0  # Default label for minor changes
    df.loc[:, 'label'] = 0  # This sets all rows in the 'label' column to 0

    df['label'] = np.where(price_change >= threshold, '1', df['label'])   # 1 for significant increase
    df['label'] = np.where(price_change <= -threshold, '2', df['label']) # 2 for significant decrease


    # Drop the last n rows which will have NaN labels (future price unavailable)
    rows_before = len(df)
    df.dropna(subset=['label'], inplace=True)
    rows_dropped = rows_before - len(df)
    expected_drops = param['target_size']
    if rows_dropped != expected_drops:
        print(f"WARNING: dropna removed {rows_dropped} rows but expected {expected_drops} "
              f"(target_size={expected_drops}). Possible NaN in feature columns — check for data gaps.")
    else:
        print(f"dropna: removed {rows_dropped} trailing rows with NaN labels (expected). OK.")

    # Convert label column to integer
    df['label'] = df['label'].astype(int)

    #########################################################################
    symbol = param['symbol']
    df.to_csv(symbol + '_TMP'+'.csv', index=False) # we want to save the date index

    # print(df.head())

def multi_class_accuracy(preds, y):
    """
    Returns accuracy per batch for multi-class classification.
    preds: Model predictions
    y: Ground truth labels
    """
    # Get the class with the highest probability
    predicted_classes = preds.argmax(dim=1)
    
    # Compare with true labels
    correct = (predicted_classes == y).float()  # Convert into float for division
    acc = correct.sum() / len(correct)
    return acc

# integrate this into the validation loop:
def validate(model, val_loader, criterion, device):
    # from latest suggestion 
    model.eval()  # Put the model in evaluation mode
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            # Move inputs and labels to the appropriate device
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            predictions = model(inputs)

            # Get num_classes from predictions
            num_classes = predictions.size(-1)  # Should be 3 in your case

            # Reshape predictions and labels
            predictions_flat = predictions.view(-1, num_classes)  # Shape: [batch_size * seq_length, num_classes]
            labels_flat = labels.view(-1)  # Shape: [batch_size * seq_length]
            loss = criterion(predictions_flat, labels_flat)
            total_loss += loss.item() * inputs.size(0)  # Multiply by batch size

            # Use your multi_class_accuracy function
            acc = multi_class_accuracy(predictions_flat, labels_flat)
            batch_size = labels.size(0)
            total_correct += acc.item() * batch_size  # Multiply by batch size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
    return avg_loss, avg_accuracy

def make_prediciton( model, raw_df: DataFrame, param: dict[str], currentDateTime, symbol):
    # i only want to keep the last 20 row of the df
    # RAW DF HAS ALL TEH DATA not date limited
    # print(raw_df.head)
    # raw_df.drop('label', axis=1, inplace=True)

    selected_columns = param['selected_columns']

    missing = [c for c in selected_columns if c not in raw_df.columns]
    if missing:
        raise ValueError(f"[{param['symbol']}] selected_columns has features not in dataframe: {missing}")

    raw_df = raw_df[selected_columns]   # only keep those slected features defined in the param, this includes label

    # for inference only use the last stretch of the most recent data, from today going back target size plus batch size
    last_df = raw_df.tail(param['batch_size'] + param['target_size']).copy()
    last_df.loc[:, 'label'] = 0
    last_df = last_df[selected_columns]
    # last_df = last_df.drop('label', axis=1)     # got to lose the label col

    # Load the scaler from disk
    scaler_filename = param['symbol'] + '_' + param['model_name'] + '_scaler.joblib'
    if not os.path.exists(scaler_filename):
        raise FileNotFoundError(
            f"Scaler not found for [{param['symbol']} / {param['model_name']}]: {scaler_filename}. "
            f"Run training first to generate this file.")
    scaler = load(scaler_filename)

    # Assuming new_data_df is your new incoming data for inference
    features_array = scaler.transform(last_df.drop(['label'], axis=1))

    # Convert to PyTorch Tensor and add batch dimension
    input_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)  # Shape: [1, sequence_length, input_dim]

    model.eval()  # Set to evaluation mode

    with torch.no_grad():
        # Generate predictions
        prediction = model(input_tensor)  # Expected output shape: [1, sequence_length, num_classes]
        print("Prediction shape:", prediction.shape)

        # Apply softmax and argmax over the correct dimensions
        probabilities = torch.softmax(prediction, dim=2)  # Class dimension at index 2
        predicted_class = torch.argmax(probabilities, dim=2).squeeze(0)  # Shape: [sequence_length]

    # Extract the last [target_size] predictions
    last_target_predictions = predicted_class[-(param['target_size']):]

    # Convert prediction to a human-readable form
    class_labels = {0: "__", 1: "UP", 2: "DN"}
    last_target_labels = [class_labels[pred.item()] for pred in last_target_predictions]

    print("Last" + str(param['target_size']) + " predicted labels:", last_target_labels)
    # The last prediction corresponds to the latest data


    date_str = currentDateTime.strftime('%Y-%m-%d')
    close = raw_df['adjusted close'].iloc[-1]

    processPrediction.process_prediction_results(symbol, date_str, close, last_target_labels, param['target_size'])
    return
def next_weekday(dt):
    # If the current day is Friday (4), add 3 days to get to Monday
    # If the current day is Saturday (5), add 2 days to get to Monday
    # Otherwise, just add 1 day
    if dt.weekday() == 4:  # Friday
        return dt + timedelta(days=3)
    elif dt.weekday() == 5:  # Saturday
        return dt + timedelta(days=2)
    else:
        return dt + timedelta(days=1)
    
def make_prediciton_test_suspect( model, raw_df: DataFrame, param: dict[str], currentDateTime, symbol, incr_df: DataFrame):
    # i only want to keep the last 20 row of the df
    # RAW DF HAS ALL TEH DATA not date limited
    # print(raw_df.head)


    if param.get('end_date') is not None:
        end_date_str = param['end_date']
    else:
        end_date_str = currentDateTime.strftime('%Y-%m-%d')

    raw_df.drop('label', axis=1)

    selected_columns = param['selected_columns']

    # raw_df = raw_df[selected_columns]   # only keep those slected features defined in the param, this includes label

    # for inference only use the last stretch of the most recent data, from today going back target size plus batch size

    # Number of additional entries to include after 'end_date', this is the simulated "new data" for the unit test
    N = 1
    df = raw_df.copy()

    # Convert 'Date' column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Convert 'end_date' to datetime
    end_date = pd.to_datetime(end_date_str)

    # Get the day of the week
    day_of_week = end_date.strftime('%A')  # Full weekday name (e.g., 'Monday')
    print('==> End Date: '+ end_date_str + ' is a ' + str(day_of_week))

    # Find the index of the 'end_date' entry or the next one
    end_index = df[df['date'] >= end_date].index[0]

    # Create a new DataFrame from the start to 'end_date' + N entries
    # new_df = df.iloc[:end_index + N + 1]  # DEBUG the potential off by one close price problem
    new_df = df.iloc[:end_index + N]
    print(new_df)


    last_df = new_df.tail(param['batch_size'] + param['target_size']).copy()
    last_df.loc[:, 'label'] = 0
    last_df = last_df[selected_columns]

    # Load the scaler from disk
    scaler_filename = param['symbol'] + '_' + param['model_name'] + '_scaler.joblib'
    if not os.path.exists(scaler_filename):
        raise FileNotFoundError(
            f"Scaler not found for [{param['symbol']} / {param['model_name']}]: {scaler_filename}. "
            f"Run training first to generate this file.")
    scaler = load(scaler_filename)

    # Convert DataFrame to NumPy array
    # features_array = last_df.to_numpy()

    # Assuming new_data_df is your new incoming data for inference
    features_array = scaler.transform(last_df.drop(['label'], axis=1))

     # Convert to PyTorch Tensor and add batch dimension
    input_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)  # Shape: [1, sequence_length, input_dim]

    model.eval()  # Set to evaluation mode

    with torch.no_grad():
        # Generate predictions
        prediction = model(input_tensor)  # Expected output shape: [1, sequence_length, num_classes]
        print("Prediction shape:", prediction.shape)

        # Apply softmax and argmax over the correct dimensions
        probabilities = torch.softmax(prediction, dim=2)  # Class dimension at index 2
        predicted_class = torch.argmax(probabilities, dim=2).squeeze(0)  # Shape: [sequence_length]


    # Extract the last [target_size] predictions
    last_target_predictions = predicted_class[-(param['target_size']):]

    # Convert prediction to a human-readable form
    class_labels = {0: "__", 1: "UP", 2: "DN"}
    last_target_labels = [class_labels[pred.item()] for pred in last_target_predictions]

    print("Last " + str(param['target_size']) + " predicted labels:", last_target_labels)
    # The last prediction corresponds to the latest data

    next_date = next_weekday(end_date)
    print("Next weekday dateshould be:", next_date)

    close = new_df['adjusted close'].iloc[-1]
    print("Next weekday Close:", close)

    # date_str = next_date.strftime('%Y-%m-%d %H:%M')     #make sure to inclide hours and min as well
    date_str = currentDateTime.strftime('%Y-%m-%d %H:%M')     #make sure to inclide hours and min as well NOTE this is the run time, not necessarily the end time

    # Now add last col data to the incremental df that was passed in.
    # print(type(incr_df))  # This should print <class 'pandas.core.frame.DataFrame'>
    col_name = "p"+ str(param['target_size'])
    result = last_target_labels[-1]
    print('>>> Result for day '+ col_name + ' ='+ result )
    incr_df[col_name] = [result]

    processPrediction.process_prediction_results_test(symbol, date_str, close, last_target_labels, param['target_size'])
    return

def make_prediciton_test( model, raw_df: DataFrame, param: dict[str], currentDateTime, symbol, incr_df: DataFrame):
    # i only want to keep the last 20 row of the df
    # RAW DF HAS ALL TEH DATA not date limited
    selected_columns = param['selected_columns']

    missing = [c for c in selected_columns if c not in raw_df.columns]
    if missing:
        raise ValueError(f"[{param['symbol']}] selected_columns has features not in dataframe: {missing}")

    # Step 1: Select columns first (including 'label')
    new_df = raw_df[selected_columns].copy()
    
    # Step 2: Handle date filtering (assuming you want to keep this logic)
    
    # Step 3: Select the last batch_size + target_size rows
    last_df = new_df.tail(param['batch_size'] + param['target_size']).copy()
    
    # Step 4: Replace label with dummy values (if necessary for prediction)
    last_df['label'] = 0  # or remove this line if you don't need dummy labels
    
    # Step 5: Prepare features for scaling (consistent with training)
    features_to_scale = last_df.drop(['label'], axis=1)
    
    # Step 6: Load and apply scaler
    scaler_filename = param['symbol'] + '_' + param['model_name'] + '_scaler.joblib'
    if not os.path.exists(scaler_filename):
        raise FileNotFoundError(
            f"Scaler not found for [{param['symbol']} / {param['model_name']}]: {scaler_filename}. "
            f"Run training first to generate this file.")
    scaler = load(scaler_filename)
    features_array = scaler.transform(features_to_scale)
    
    # Continue with the rest of your prediction logic...
    input_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)

    print(f"Feature array shape: {features_array.shape}")
    print(f"Input tensor shape: {input_tensor.shape}")

    model.eval()  # Set to evaluation mode

    # If you're using a GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("Using device:", device)

    model = model.to(device)

    with torch.no_grad():
        # Generate predictions
        prediction = model(input_tensor)  # Expected output shape: [1, sequence_length, num_classes]
        print("Prediction shape:", prediction.shape)

        # Apply softmax and argmax over the correct dimensions
        probabilities = torch.softmax(prediction, dim=2)  # Class dimension at index 2
        predicted_class = torch.argmax(probabilities, dim=2).squeeze(0)  # Shape: [sequence_length]

    # Extract the last [target_size] predictions
    last_target_predictions = predicted_class[-(param['target_size']):]

    # Convert prediction to a human-readable form
    class_labels = {0: "__", 1: "UP", 2: "DN"}
    last_target_labels = [class_labels[pred.item()] for pred in last_target_predictions]

    print("Last" + str(param['target_size']) + " predicted labels:", last_target_labels)
    # The last prediction corresponds to the latest data


    # date_str = currentDateTime.strftime('%Y-%m-%d')
    date_str = param['end_date']
    close = raw_df['adjusted close'].iloc[-1]

     # Now add last col data to the incremental df that was passed in.
    # print(type(incr_df))  # This should print <class 'pandas.core.frame.DataFrame'>
    col_name = "p"+ str(param['target_size'])
    result = last_target_labels[-1]
    print('>>> Result for day '+ col_name + ' ='+ result )
    incr_df[col_name] = [result]

    processPrediction.process_prediction_results_test(symbol, date_str, close, last_target_labels, param['target_size'])



#######################################################################
# in case there are specific feature value to override for the last day
def feature_value_override(df: DataFrame, param: dict[str]):
    if param.get('current_unemploy') is not None:
        latest_unemploy_data = param['current_unemploy']
        df.loc[df.index[-1], 'unemploy'] = latest_unemploy_data
    return df

#################################################################################
# calculate class weight (not normalized)
def calculate_class_weight(total_samples_df: DataFrame, num_classes: int):
    total_sample_count = len(total_samples_df)
    class_weights = []
    for i in range(num_classes):
        sample_count = len(total_samples_df[total_samples_df['label'] == i])
        if sample_count == 0:
            print(f"Class {i}: 0 samples — absent from fold, weight set to 0.0")
            print("-" * 40)
            class_weights.append(0.0)
            continue
        sample_weight = total_sample_count / (num_classes * sample_count)
        class_weights.append(sample_weight)
        # Print distribution and weight for each class
        print(f"Class {i}: {sample_count} samples (weight: {sample_weight:.3f})")
        print("-" * 40)

    # WI = total_sample_count / (num_classes * sample_count)
    return class_weights

def time_based_split(df, n_splits=3, val_size=0.15):
    # Ensure the dataframe is sorted by date
    # df = df.sort_values('date')   # do we need dates here? Already been striped
    
    # Create TimeSeriesSplit object
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Initialize lists to store our train, validation, and test sets
    train_sets = []
    val_sets = []
    test_sets = []
    
    # Split the data
    for train_index, test_index in tscv.split(df):
        # Split the train_index further into train and validation
        train_val_split = int(len(train_index) * (1 - val_size))
        
        train_sets.append(df.iloc[train_index[:train_val_split]])
        val_sets.append(df.iloc[train_index[train_val_split:]])
        test_sets.append(df.iloc[test_index])
    
    return train_sets, val_sets, test_sets

#################################################################################
### Main code starts here
def analyze_trend( config: dict[str, str], param: dict[str], current_day_offset: str, incr_df: DataFrame, turn_random_on: bool, use_cached_data: bool, use_time_split: bool=False):

    symbol = param['symbol']

    if turn_random_on:
        random_seed = random.randint(0, 2**32 - 1)
    else:
        random_seed = 42

    random.seed(random_seed)  # Python's built-in random lib
    np.random.seed(random_seed)  # Numpy lib
    torch.manual_seed(random_seed)  # PyTorch

    # If you are using CUDA
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if (use_cached_data):
        symbol = param['symbol']
        file_path = symbol + '_TMP.csv'  
        df = pd.read_csv(file_path)
        print('> data loaded from cache')
    else:
        df = load_data_to_cache(config, param)

    df = feature_value_override(df, param)  

    # Filtering operation by date
    start_date = param['start_date']
    df = df[df['date'] >= start_date]

    # delete the stuff beyong end date if that is specified
    if param.get('end_date') is not None:
        end_date = param['end_date']
        df = df[df['date'] <= end_date]

    # need to save copy that has the end date chopped off
    raw_df = df.copy()      # preserve the original, there the end of it contains the data needed for prediction this is not truncated with end date
  

    # from here onward df only contains what is between start and end
    selected_columns = param['selected_columns']
    df = df[selected_columns]   # only keep those slected features defined in the param, this includes label

    # Split the Data: Divide your data into training, validation, and test sets.
    if use_time_split:
        train_sets, val_sets, test_sets = time_based_split(df, n_splits=3, val_size=param['validation_size'])
        train_df = pd.concat(train_sets + val_sets + test_sets[:-1])
        val_df = val_sets[-1]
        test_df = test_sets[-1]
    else:
        shuffle_splits = param.get('shuffle_splits', False)
        rs = random_seed if shuffle_splits else None
        train_df, test_df = train_test_split(df, test_size=param['test_size'], shuffle=shuffle_splits, random_state=rs)
        train_df, val_df = train_test_split(train_df, test_size=param['validation_size'], shuffle=shuffle_splits, random_state=rs)
        

    # print("Training Set Statistics:")
    # print(train_df.describe())

    # print("\nValidation Set Statistics:")
    # print(val_df.describe())

    # print("\nTest Set Statistics:")
    # print(test_df.describe())

    # Getting the labels
    train_labels = train_df['label'].values

    # new way to use sample count to calculate class weight
    num_labels = train_df['label'].nunique()
    if isinstance(num_labels, pd.Series):
        num_labels = num_labels.iloc[0]  # or num_labels.values[0]
    class_weights = calculate_class_weight(train_df, num_labels)
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}  # dynamic: safe if a class is absent from the fold
    class_weights_tensor = torch.tensor(list(class_weights_dict.values()), dtype=torch.float)

    # Feature Scaling: Normalize or standardize your features. Transformers typically require input data to be scaled.
    scaler_type = param['scaler_type']
    
    if scaler_type == 'Robust':
        scaler = RobustScaler()
    elif scaler_type == 'MinMax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    train_features = scaler.fit_transform(train_df.drop(['label'], axis=1))
    val_features = scaler.transform(val_df.drop(['label'], axis=1))
    test_features = scaler.transform(test_df.drop(['label'], axis=1))

    # Save the scaler to disk
    # this needs to be symbol and model specific
    scaler_filename = param['symbol'] + '_' + param['model_name'] + '_scaler.joblib'
    dump(scaler, scaler_filename)

    # Format Data for PyTorch: Convert your data into PyTorch tensors and create DataLoader instances.
    train_data = TensorDataset(torch.FloatTensor(train_features), torch.FloatTensor(train_df['label'].values))
    val_data = TensorDataset(torch.FloatTensor(val_features), torch.FloatTensor(val_df['label'].values))
    test_data = TensorDataset(torch.FloatTensor(test_features), torch.FloatTensor(test_df['label'].values))

    # Creating DataLoaders
    batch_size = param['batch_size']
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        if len(split_df) < batch_size:
            print(f'WARNING: {param["symbol"]} {param["model_name"]} — {split_name} split has {len(split_df)} rows < batch_size {batch_size}')
    train_loader = DataLoader(train_data, shuffle=param['shuffle'], batch_size=batch_size, drop_last=False)
    val_loader   = DataLoader(val_data,   shuffle=False,            batch_size=batch_size, drop_last=False)
    test_loader  = DataLoader(test_data,  shuffle=False,            batch_size=batch_size, drop_last=False)

    # Define Your Transformer Model: Use PyTorch’s Transformer model or define your own.
    # Num heads in this chatGPT example = 2, in the datascitribe article it is 4, can test to find out
    # Data Sci also have number of features to be much higher (64)
    #
    # For a three-class classification problem, the output layer of your model 
    # should have three units (one for each class) and typically use a softmax 
    # activation function, which generalizes the binary sigmoid function to multiple classes.
    feature_count=train_features.shape[1]
    head_count = param['headcount']
    embedded_dim = param['embedded_dim']
    if embedded_dim % head_count != 0:
        print(f'ERROR: embedded_dim ({embedded_dim}) must be divisible by headcount ({head_count})')
        sys.exit()

    num_classes=3

    ############################
    # NEW REGULARIZATION CODE
    # Instantiate the model
    model = build_model(param, input_dim=feature_count, num_classes=num_classes)
    
    # Check if CUDA (GPU support) is available and use it, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move class weights to device before creating criterion to avoid device mismatch
    class_weights_tensor = class_weights_tensor.to(device)

    # Define loss function
    if num_classes == 1:
        criterion = nn.BCELoss()
    else:
        criterion = FocalLoss(weight=class_weights_tensor, gamma=2.0, label_smoothing=0.1)

    # Define optimizer with L2 regularization (weight decay)
    optimizer = optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param["l2_weight_decay"]  # L2 regularization
    )

    # Move your model to the device
    model.to(device)

    # Training Loop:
    num_epochs = param['num_epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # L1 regularization NEW
    l1_lambda = param["l1_lambda"]
    
     # # Step 1: Record the start time
    start_time = time.time()
    
    model = analysisUtil.train_with_trend_based_stopping(model, train_loader, val_loader, num_epochs, optimizer, 
                            criterion, device, scheduler, l1_lambda)

    end_time = time.time()

    # Step 4: Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f">>>Time elapsed for training: {elapsed_time} seconds")
    
    ###
    # training complete, save the model
    # model name should be model_<symbol>_<random>_<timesplit>_<horizon>.pth
    #
    if turn_random_on:
        random_str = 'random'
    else:
        random_str = 'fixed'
    
    if use_time_split:
        time_split_str = 'timesplit'
    else:
        time_split_str = 'noTimesplit'

    
    eastern = pytz.timezone('US/Eastern')
    currentDateTime = datetime.now(eastern)
    date = currentDateTime.strftime("%Y-%m-%d %H:%M:%S")

    # save model to folder /model
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    model_name = os.path.join(model_dir, 'model_'+ param['symbol'] + '_' + param['model_name'] + '_' + random_str + '_' + time_split_str + '_' + str(current_day_offset) +'.pth')
    torch.save(model, model_name)
   
    print(f"Model saved as {model_name}")

    # Validate The model
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print(f'Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_acc*100:.2f}%')

            
    # Step 4: Evaluation
    # After training, evaluate your model's performance on the test set using appropriate metrics (accuracy, F1-score, etc.).
    # Confusion Matrix, Precision, Recall, and F1-Score:
    # For these metrics, you can use functions from sklearn.metrics. First, ensure that you've computed the predictions for your validation or test set.


    # Assuming y_preds and y_trues are the lists of predictions and true labels
    y_preds = np.array([])
    y_trues = np.array([])

    for inputs, labels in val_loader:
        # Move inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass and get predictions
        # Make sure to move your inputs and labels to the same device as your model
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=-1)

        # Flatten the arrays
        preds_flat = preds.cpu().numpy().flatten()    # Shape: [batch_size * seq_length]
        labels_flat = labels.cpu().numpy().flatten()  # Shape: [batch_size * seq_length]

        y_preds = np.concatenate((y_preds, preds_flat))
        y_trues = np.concatenate((y_trues, labels_flat))


    # Precision, Recall, F1-Score
    precision, recall, f1, _ = precision_recall_fscore_support(y_trues, y_preds, average=None)

    # You might want to average these for a summary metric
    average_precision, average_recall, average_f1, _ = precision_recall_fscore_support(y_trues, y_preds, average='macro')

    print('====> Validation set performance <====')
    print('Precision for class 0: '+str(precision[0]))
    print('Precision for class 1: '+str(precision[1]))
    print('Precision for class 2: '+str(precision[2]))
    print('Recall for class 0: '+ str(recall[0]))
    print('Recall for class 1: '+ str(recall[1]))
    print('Recall for class 2: '+ str(recall[2]))
    print('F1 for class 0: '+ str(f1[0]))
    print('F1 for class 1: '+ str(f1[1]))
    print('F1 for class 2: '+ str(f1[2]))
    
    print('==>Avergae Precision: '+ str(average_precision))
    print('==>Avergae Recall: '+ str(average_recall))
    print('==>Avergae F1: '+ str(average_f1))

    #############################################################################
    # TEST
    # next step is to run the test data
    # Assuming 'test_loader' is your DataLoader for the test dataset
    # and 'model' is your trained model
    print('======> Start test <======')


    # Assuming y_preds and y_trues are the lists of predictions and true labels
    test_preds = np.array([])
    test_trues = np.array([])

    for inputs, labels in test_loader:
        # Forward pass and get predictions
        # Make sure to move your inputs and labels to the same device as your model
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        # Convention: apply softmax before argmax (consistent with inference and ROC paths).
        # Softmax does not change the argmax result but ensures the probability convention
        # is uniform across all paths, so confidence scores and future calibration work correctly.
        preds = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)

        test_preds_flat = preds.cpu().numpy().flatten()    # Shape: [batch_size * seq_length]
        test_labels_flat = labels.cpu().numpy().flatten()  # Shape: [batch_size * seq_length]

        test_preds = np.concatenate((test_preds, test_preds_flat))
        test_trues = np.concatenate((test_trues, test_labels_flat))

    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_trues, test_preds, average=None)

    print('====> Test set performance <====')
    print('Precision for class 0: '+ str(test_precision[0]))
    print('Precision for class 1: '+ str(test_precision[1]))
    print('Precision for class 2: '+ str(test_precision[2]))
    print('Recall for class 0: '+ str(test_recall[0]))
    print('Recall for class 1: '+ str(test_recall[1]))
    print('Recall for class 2: '+ str(test_recall[2]))
    print('F1 for class 0: '+ str(test_f1[0]))
    print('F1 for class 1: '+ str(test_f1[1]))
    print('F1 for class 2: '+ str(test_f1[2]))


    # Step 6: Visualization
    # Plot training and validation loss, accuracy, or other relevant metrics over epochs to assess model performance.

    # For visualizing the confusion matrix, you can use libraries like matplotlib or seaborn.

    # Plotting Confusion Matrix
    # Assuming y_true and y_pred are your true and predicted labels
    # cm = confusion_matrix(y_trues, y_preds)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.show()
    # Save the figure
    
    # symbol = param['symbol']

    confusion_file_name = symbol+ '_' + str(date) + '_' + 'confusion_matrix.png'
    # plt.savefig(confusion_file_name)  # You can change the filename and extension as needed


    #############
    # now do ROC curve stuff
    # also implement ROC-AUC
    # Assuming y_true is your true labels and y_probs are the predicted probabilities for each class

    softmax = torch.nn.Softmax(dim=2)
    y_probs = []
    y_trues = []

    for inputs, labels in val_loader:
        # Move inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        # probabilities = softmax(outputs).detach()  # Detach the tensor from the graph
        probabilities = softmax(outputs)

        # Reshape probabilities and labels to 2D arrays
        batch_size, seq_length, num_classes = probabilities.shape
        probabilities_flat = probabilities.view(-1, num_classes)  # Shape: [batch_size * seq_length, num_classes]
        labels_flat = labels.view(-1)

        # y_probs.append(probabilities.cpu().numpy())
        # Convert to CPU and NumPy arrays
        # y_probs.append(probabilities_flat.cpu().numpy())
        y_probs.append(probabilities_flat.detach().cpu().numpy())
        y_trues.append(labels_flat.detach().cpu().numpy())

    y_probs = np.concatenate(y_probs, axis=0)  # Shape: [total_samples, num_classes]
    y_trues = np.concatenate(y_trues, axis=0)  # Shape: [total_samples]

    # print('==> y_probs.shape: '+ str(y_probs.shape))
    print('==> y_probs.shape:', y_probs.shape)
    print('==> y_trues.shape:', y_trues.shape)


    y_true_binarized = label_binarize(y_trues, classes=[0, 1, 2])
    n_classes = y_true_binarized.shape[1]

    # If y_pred is categorical labels, convert it to probabilities (e.g., using a softmax output from your model)

    # Assuming y_true_binarized and y_probs are already defined as shown previously
    fpr = dict()
    tpr = dict()
    roc_auc_scores = dict()  # Rename this variable to avoid conflict
    # auc_scores = []  # This can remain the same since it's not causing a conflict

    for i in range(n_classes):
        # Extract true labels and predicted probabilities for class i
        y_true_class = y_true_binarized[:, i]     # Shape: [total_samples]
        y_prob_class = y_probs[:, i]              # Shape: [total_samples]
        
        # Compute ROC curve
        fpr[i], tpr[i], _ = roc_curve(y_true_class, y_prob_class)
        
        # Compute AUC score
        auc_score = roc_auc_score(y_true_class, y_prob_class)
        roc_auc_scores[i] = auc_score
        print(f"AUC for class {i}: {auc_score:.2f}")

    ###################################################
    # Now save the results into a ever running log
    #
    nan_in_val  = any(np.isnan(v) for v in f1)
    nan_in_test = any(np.isnan(v) for v in test_f1)
    if nan_in_val or nan_in_test:
        print(f"WARNING: NaN metrics detected — one or more classes absent from val/test set.")
        print(f"  val  F1={f1}  test F1={test_f1}")
        print(f"  Result NOT saved to {symbol}_trend.jsonl to avoid corrupt history.")
    else:
        result_dictionary = {
            'Validation Loss': val_loss,
            'Validation Accuracy': val_acc,
            'Avergae Precision': str(average_precision),
            'Avergae Recall': str(average_recall),
            'Avergae F1': str(average_f1),
            'F1 for Class 0 [no change]': str(f1[0]),
            'F1 for Class 1 [Up]':str(f1[1]),
            'F1 for Class 2 [Down]':str(f1[2]),
            'Test F1 for class 0: ': float(test_f1[0]),
            'Test F1 for class 1: ': float(test_f1[1]),
            'Test F1 for class 2: ': float(test_f1[2]),
        }

        result_dictionary.update(roc_auc_scores)    # add the AUC results into results

        # Creating a row
        # tiume stamp, stock symbol, the input param, and then output
        row = [date, param, result_dictionary]

        # Append one record per line to the JSONL file (no full rewrite needed)
        jsonl_file_path = symbol + '_trend.jsonl'
        with open(jsonl_file_path, 'a') as file:
            file.write(json.dumps(row) + '\n')

    # Read full JSONL history to rebuild the CSV (read-only, no rewrite)
    jsonl_file_path = symbol + '_trend.jsonl'
    with open(jsonl_file_path, 'r') as file:
        data = [json.loads(line) for line in file if line.strip()]

    # Process each record in JSON data
    processed_data = []
    for record in data:
        # Flatten the structure
        run_time, parameters, results = record
        flat_record = {'run_time': run_time}
        flat_record.update(parameters)
        flat_record.update(results)
        processed_data.append(flat_record)

    # Create DataFrame
    df = pd.DataFrame(processed_data)

    # now convert the json file into a csv file
    csv_file_path = symbol+ '_trend.csv'
    df.to_csv(csv_file_path, index=True)

    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc_scores[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC')
    plt.legend(loc="lower right")

    ROC_file_name = symbol+ '_' + str(date) + '_' + 'ROC.png'
    # plt.savefig(ROC_file_name)  # You can change the filename and extension as needed

    # plt.show()

    ############################################################
    # Now make prediction
    # 
    make_prediciton_test(model, raw_df, param, currentDateTime, symbol, incr_df)

def analyze_trend_multi_horizon(
    config: dict, param: dict, current_day_offset: str,
    incr_df: DataFrame, turn_random_on: bool, use_cached_data: bool,
    save_dir: str = None,
):
    """Train a MultiHorizonTransformer that predicts all 15 horizons in one pass.

    Key differences from analyze_trend():
    - shuffle_splits=True is a hard error (look-ahead bias on multi-horizon labels)
    - Labels: (N, 15) forward-return classes, not single-horizon label column
    - Optimizer: AdamW with gradient clipping (max_norm=1.0)
    - Stopping: EarlyStopping(patience=15) on val loss (not TrendBasedStopping)
    - Loss: FocalLoss per horizon, normalized by num_horizons
    - Checkpoint: state_dict format {state_dict, config, calib_temp, train_date}
    - Scaler saved as {SYM}_{model_name}_mh_scaler.joblib

    Args:
        save_dir: Override directory for model + scaler files (for test isolation).
                  If None, uses 'model/' for model and CWD for scaler (production default).
    """
    symbol = param['symbol']

    if param.get('shuffle_splits', False):
        raise ValueError(
            f"[{symbol}] shuffle_splits=True is not allowed for model_type='multi_horizon_transformer'. "
            "It produces misleadingly inflated metrics. Set shuffle_splits=False."
        )

    # --- Seed setup ---
    if turn_random_on:
        random_seed = random.randint(0, 2**32 - 1)
    else:
        random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- Load data ---
    if use_cached_data:
        file_path = symbol + '_TMP.csv'
        df = pd.read_csv(file_path)
        print('> data loaded from cache')
    else:
        df = load_data_to_cache(config, param)

    df = feature_value_override(df, param)
    start_date = param['start_date']
    df = df[df['date'] >= start_date].reset_index(drop=True)
    if param.get('end_date') is not None:
        df = df[df['date'] <= param['end_date']].reset_index(drop=True)

    # --- Build multi-horizon label matrix (N_usable, 15) ---
    NUM_HORIZONS = param.get('num_horizons', 15)
    labels_mh, n_usable = build_multi_horizon_labels(df, num_horizons=NUM_HORIZONS)

    # --- Select feature columns (exclude single-horizon 'label') ---
    feature_cols = [c for c in param['selected_columns'] if c != 'label']
    feature_cols = [c for c in feature_cols if c in df.columns]
    features = df[feature_cols].iloc[:n_usable].values  # (n_usable, D)

    # --- Chronological split (no shuffle) ---
    n_test = int(n_usable * param['test_size'])
    n_val  = int((n_usable - n_test) * param['validation_size'])
    n_train = n_usable - n_test - n_val

    train_features = features[:n_train]
    val_features   = features[n_train:n_train + n_val]
    test_features  = features[n_train + n_val:]
    train_labels   = labels_mh[:n_train]
    val_labels     = labels_mh[n_train:n_train + n_val]
    test_labels    = labels_mh[n_train + n_val:]

    print(f"[{symbol} MH] train={n_train}, val={n_val}, test={n_test}, features={features.shape[1]}")

    # --- Scaler (fit on train only) ---
    scaler_type = param.get('scaler_type', 'MinMax')
    if scaler_type == 'Robust':
        scaler = RobustScaler()
    elif scaler_type == 'MinMax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    train_features = scaler.fit_transform(train_features)
    val_features   = scaler.transform(val_features)
    test_features  = scaler.transform(test_features)

    scaler_filename = f"{symbol}_{param['model_name']}_mh_scaler.joblib"
    if save_dir:
        scaler_path = os.path.join(save_dir, scaler_filename)
    else:
        scaler_path = scaler_filename
    dump(scaler, scaler_path)

    # --- DataLoaders ---
    batch_size = param['batch_size']
    train_data = TensorDataset(torch.FloatTensor(train_features), torch.LongTensor(train_labels))
    val_data   = TensorDataset(torch.FloatTensor(val_features),   torch.LongTensor(val_labels))
    test_data  = TensorDataset(torch.FloatTensor(test_features),  torch.LongTensor(test_labels))

    for split_name, split_size in [('train', n_train), ('val', n_val), ('test', n_test)]:
        if split_size < batch_size:
            print(f'WARNING: [{symbol}] {split_name} split has {split_size} rows < batch_size {batch_size}')

    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=False)
    val_loader   = DataLoader(val_data,   shuffle=False, batch_size=batch_size, drop_last=False)
    test_loader  = DataLoader(test_data,  shuffle=False, batch_size=batch_size, drop_last=False)

    # --- Model ---
    feature_count = train_features.shape[1]
    head_count    = param['headcount']
    embedded_dim  = param['embedded_dim']
    if embedded_dim % head_count != 0:
        print(f'ERROR: embedded_dim ({embedded_dim}) must be divisible by headcount ({head_count})')
        sys.exit()

    model = build_model(param, input_dim=feature_count, num_classes=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # --- Per-horizon FocalLoss instances with class weights ---
    focal_losses = []
    for h in range(NUM_HORIZONS):
        h_df = pd.DataFrame({'label': train_labels[:, h]})
        cw = calculate_class_weight(h_df, 3)
        cw_tensor = torch.tensor(list({i: w for i, w in enumerate(cw)}.values()),
                                  dtype=torch.float).to(device)
        focal_losses.append(FocalLoss(weight=cw_tensor, gamma=2.0, label_smoothing=0.1))

    # --- AdamW optimizer + cosine scheduler ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['l2_weight_decay'],
    )
    num_epochs = param['num_epochs']
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    l1_lambda  = param.get('l1_lambda', 0)

    early_stopping = analysisUtil.EarlyStopping(patience=15, min_delta=1e-4)

    use_amp   = (device.type == 'cuda')
    amp_scaler = torch.amp.GradScaler(enabled=use_amp)

    start_time = time.time()

    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()
        total_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)  # (batch, 15)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                outputs = model(inputs)  # (batch, 15, 3)
                horizon_loss = sum(
                    focal_losses[h](outputs[:, h, :], labels[:, h])
                    for h in range(NUM_HORIZONS)
                ) / NUM_HORIZONS
                if l1_lambda > 0:
                    l1_norm = sum(p.abs().sum() for name, p in model.named_parameters()
                                  if 'weight' in name)
                    horizon_loss = horizon_loss + l1_lambda * l1_norm

            amp_scaler.scale(horizon_loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            total_train_loss += horizon_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        scheduler.step()

        # --- Validation phase ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    outputs = model(inputs)
                    val_horizon_loss = sum(
                        focal_losses[h](outputs[:, h, :], labels[:, h])
                        for h in range(NUM_HORIZONS)
                    ) / NUM_HORIZONS
                total_val_loss += val_horizon_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}')

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            model.load_state_dict(early_stopping.best_model)
            break

    elapsed = time.time() - start_time
    print(f">>>Time elapsed for multi-horizon training: {elapsed:.1f}s")

    # --- Save checkpoint (state_dict format) ---
    if turn_random_on:
        random_str = 'random'
    else:
        random_str = 'fixed'

    eastern = pytz.timezone('US/Eastern')
    currentDateTime = datetime.now(eastern)
    train_date = currentDateTime.strftime("%Y-%m-%d %H:%M:%S")

    model_dir = save_dir if save_dir else 'model'
    os.makedirs(model_dir, exist_ok=True)
    model_filename = f"model_{symbol}_{param['model_name']}_mh_{random_str}_noTimesplit.pth"
    model_path = os.path.join(model_dir, model_filename)

    assert not os.path.abspath(model_path).startswith(os.path.abspath('/workspace/model/')) or save_dir is None, \
        "Test must not write to production model dir"

    torch.save({
        'state_dict': model.state_dict(),
        'config': param,
        'calib_temp': 1.0,
        'train_date': train_date,
    }, model_path)
    print(f"Model saved as {model_path}")

    # --- Test set evaluation ---
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs  = inputs.to(device, dtype=torch.float32)
            outputs = model(inputs)                                   # (batch, 15, 3)
            preds   = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)  # (batch, 15)
            all_preds.append(preds.cpu().numpy())
    test_preds = np.concatenate(all_preds, axis=0)  # (N_test, 15)

    print('====> Multi-horizon TEST set performance <====')
    bucket_f1 = {'h1_5': [], 'h6_15': []}
    for h in range(NUM_HORIZONS):
        p, r, f1_h, _ = precision_recall_fscore_support(
            test_labels[:, h], test_preds[:, h], average=None, labels=[0, 1, 2], zero_division=0
        )
        macro = np.mean(f1_h)
        bucket = 'h1_5' if h < 5 else 'h6_15'
        bucket_f1[bucket].append(macro)
        print(f'  h={h+1:2d}: F1[flat={f1_h[0]:.3f}  UP={f1_h[1]:.3f}  DN={f1_h[2]:.3f}]  macro={macro:.3f}')

    for bname, vals in bucket_f1.items():
        print(f'  Bucket {bname}: mean macro-F1 = {np.mean(vals):.3f}')

    return model, test_preds, test_labels, model_path, scaler_path


def load_data_to_cache(config: dict[str, str], param: dict[str]):
    df, num_data_points, display_date_range = analysisUtil.download_data(config, param)    
    calculate_label(df, param)  # Call subroutine to calicalte the label 
    symbol = param['symbol']

    # only keep data between start and end date
    # Filtering operation by date
    start_date = param['start_date']
    df = df[df['date'] >= start_date]

    # delete the stuff beyong end date if that is specified
    # do not do this as it breaks QA 
    # if param.get('end_date') is not None:
    #     end_date = param['end_date']
    #     df = df[df['date'] <= end_date]
        
    df.to_csv(symbol+ '_TMP'+'.csv', index=False) # we want to save the date index

    last_row = df.iloc[-1]  # Get the last row of the DataFrame
    data_value = last_row['date']  # Access the 'date' column in the last row
    print('>> Last day= ' + str(data_value))  # Print the value
    return df

def make_inference( config: dict[str, str], param: dict[str], current_day_offset: str, incr_df: DataFrame, turn_random_on: bool, use_cached_data: bool, use_time_split: bool=False):
    
    if turn_random_on:
        random_seed = random.randint(0, 2**32 - 1)
    else:
        random_seed = 42

    random.seed(random_seed)  # Python's built-in random lib
    np.random.seed(random_seed)  # Numpy lib
    torch.manual_seed(random_seed)  # PyTorch

    # If you are using CUDA
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ###
    # training complete, save the model
    # model name should be model_<symbol>_<random>_<timesplit>_<horizon>.pth
    #
    if turn_random_on:
        random_str = 'random'
    else:
        random_str = 'fixed'
    
    if use_time_split:
        time_split_str = 'timesplit'
    else:
        time_split_str = 'noTimesplit'

    # load it from the model subdirectory
    model_name = 'model/model_'+ param['symbol'] + '_' + param['model_name'] + '_' + random_str + '_' + time_split_str + '_' + str(current_day_offset) +'.pth'
    print('> loading model name: ' + model_name + '...')
    model = torch.load(model_name)

    ############################################################
    # Now make prediction
    #
    eastern = pytz.timezone('US/Eastern')
    currentDateTime = datetime.now(eastern)
    if (use_cached_data):
        symbol = param['symbol']
        file_path = symbol + '_TMP.csv'  
        df = pd.read_csv(file_path)
        print('> data loaded from cache')
    else:
        df = load_data_to_cache(config, param)

    df = feature_value_override(df, param)
    raw_df = df.copy()      # preserve the original, there the end of it contains the data needed for prediction this is not truncated with end date

    make_prediciton_test(model, raw_df, param, currentDateTime, param['symbol'], incr_df)

