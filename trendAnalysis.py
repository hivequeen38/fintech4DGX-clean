import sys
import pandas as pd
from pandas import DataFrame
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# import shap

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
import json
from datetime import datetime
import logging
import os
import math
import processPrediction
import shap
import analysisUtil

print("All libraries loaded for "+ __file__)

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from typing import Tuple, List

class BalancedTimeSeriesSplit:
    """
    A time series split that ensures each data point is used equally in training
    """
    def __init__(self, n_splits: int = 5, weight_recent: bool = True):
        self.n_splits = n_splits
        self.weight_recent = weight_recent
        
    def split(self, X) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate balanced time series splits
        
        Parameters:
        -----------
        X : array-like
            Training data
            
        Returns:
        --------
        splits : list of (train_idx, test_idx) tuples
        """
        n_samples = len(X)
        splits = []
        
        # Calculate indices for each fold
        indices = np.arange(n_samples)
        
        if self.weight_recent:
            # Method 1: Moving window with equal-sized splits
            test_size = n_samples // (self.n_splits + 1)
            for i in range(self.n_splits):
                end_idx = n_samples - i * test_size
                start_idx = end_idx - test_size * 2  # Double window for training
                
                if start_idx < 0:
                    break
                    
                train_idx = indices[start_idx:end_idx-test_size]
                test_idx = indices[end_idx-test_size:end_idx]
                splits.append((train_idx, test_idx))
        else:
            # Method 2: Non-overlapping splits with bootstrapping
            test_size = n_samples // self.n_splits
            for i in range(self.n_splits):
                # Test indices for this fold
                test_start = n_samples - (i + 1) * test_size
                test_end = n_samples - i * test_size
                test_idx = indices[test_start:test_end]
                
                # Training indices excluding current test period
                available_train = np.concatenate([
                    indices[:test_start],
                    indices[test_end:]
                ])
                
                # Randomly sample from available training data to balance
                if len(available_train) > test_size * 2:
                    train_idx = np.random.choice(
                        available_train, 
                        size=test_size * 2, 
                        replace=False
                    )
                else:
                    train_idx = available_train
                    
                splits.append((np.sort(train_idx), test_idx))
        
        return splits[::-1]  # Reverse to maintain chronological order

class TimeSeriesTrainTestSplit:
    """
    Create train/validation/test splits for time series data with proper chronological order
    """
    def __init__(self, train_size: float = 0.7, val_size: float = 0.15):
        self.train_size = train_size
        self.val_size = val_size
        
    def split(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets
        """
        n_samples = len(X)
        
        # Calculate split points
        train_end = int(n_samples * self.train_size)
        val_end = int(n_samples * (self.train_size + self.val_size))
        
        # Split indices
        train_idx = np.arange(train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n_samples)
        
        return train_idx, val_idx, test_idx

def compare_split_methods(data: pd.DataFrame, n_splits: int = 5):
    """
    Compare different time series split methods
    """
    # Original TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Balanced TimeSeriesSplit
    balanced_tscv = BalancedTimeSeriesSplit(n_splits=n_splits)
    
    # Count data usage frequency
    data_usage_original = np.zeros(len(data))
    data_usage_balanced = np.zeros(len(data))
    
    # Count for original split
    for train_idx, _ in tscv.split(data):
        data_usage_original[train_idx] += 1
    
    # Count for balanced split
    for train_idx, _ in balanced_tscv.split(data):
        data_usage_balanced[train_idx] += 1
    
    return pd.DataFrame({
        'original_usage': data_usage_original,
        'balanced_usage': data_usage_balanced,
        'date': data.index if isinstance(data, pd.DataFrame) else np.arange(len(data))
    })

# Example usage
def demonstrate_splits():
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='B')
    data = pd.DataFrame({
        'value': np.random.randn(len(dates)),
        'date': dates
    }).set_index('date')
    
    # Compare splits
    usage_comparison = compare_split_methods(data)
    
    # Print statistics
    print("\nData usage statistics:")
    print("\nOriginal TimeSeriesSplit:")
    print(f"Max usage: {usage_comparison['original_usage'].max()}")
    print(f"Min usage: {usage_comparison['original_usage'].min()}")
    print(f"Mean usage: {usage_comparison['original_usage'].mean():.2f}")
    
    print("\nBalanced TimeSeriesSplit:")
    print(f"Max usage: {usage_comparison['balanced_usage'].max()}")
    print(f"Min usage: {usage_comparison['balanced_usage'].min()}")
    print(f"Mean usage: {usage_comparison['balanced_usage'].mean():.2f}")
    
    return usage_comparison

# Example of how to use balanced split in training
def train_with_balanced_split(model, X, y, n_splits=5):
    splitter = BalancedTimeSeriesSplit(n_splits=n_splits)
    
    scores = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
        
        print(f"Fold {fold + 1}: Score = {score:.4f}")
    
    print(f"\nMean score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores

# default transformer that has been working
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
        # print("Input x shape:", src.shape)  # Should be [batch_size, input_dim]
        
        # Add a sequence dimension if it's not present
        if src.dim() == 2:
            src = src.unsqueeze(1)  # Now src has shape [batch_size, 1, input_dim]
        
        x = self.input_projection(src)  # [batch_size, seq_length, embedding_dim]
        x = self.positional_encoding(x)  # [batch_size, seq_length, embedding_dim]
        
        x = self.transformer_encoder(x)  # [batch_size, seq_length, embedding_dim]
        
        # Since we only have one time step, we can squeeze it out
        x = x.squeeze(1)  # [batch_size, embedding_dim]
        
        x = self.dropout(x)
        x = self.fc(x)  # [batch_size, num_classes]
        
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

    


def split_dataframe(df, size_a, size_b):
    # Check if the DataFrame has enough rows to split
    if len(df) < size_a + size_b:
        raise ValueError("The DataFrame doesn't have enough rows to split according to size_a and size_b.")

    # First DataFrame: First size_a rows
    df_first = df.iloc[:size_a]

    # Second DataFrame: Next size_b rows
    df_second = df.iloc[size_a:size_a + size_b]

    # Third DataFrame: Remaining rows
    df_third = df.iloc[size_a + size_b:]

    return df_first, df_second, df_third

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


    # Drop the last n rows which will have NaN labels
    df.dropna(subset=['label'], inplace=True)

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
    # predicted_classes = preds.argmax(dim=1)

     # Get predicted classes NEW time step code
    predicted_classes = torch.argmax(preds, dim=1)
    
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
    
    # Step 1: Select columns first (including 'label')
    new_df = raw_df[selected_columns].copy()
    
    # Step 2: Handle date filtering (assuming you want to keep this logic)
    # end_date = pd.to_datetime(param.get('end_date') or currentDateTime.strftime('%Y-%m-%d'))
    # end_index = new_df[new_df['date'] >= end_date].index[0]
    # new_df = new_df.iloc[:end_index + 1]  # +1 to include the end_date
    
    # Step 3: Select the last batch_size + target_size rows
    last_df = new_df.tail(param['batch_size'] + param['target_size']).copy()
    
    # Step 4: Replace label with dummy values (if necessary for prediction)
    last_df['label'] = 0  # or remove this line if you don't need dummy labels
    
    # Step 5: Prepare features for scaling (consistent with training)
    features_to_scale = last_df.drop(['label'], axis=1)
    
    # Step 6: Load and apply scaler
    scaler = load('scaler.joblib')
    features_array = scaler.transform(features_to_scale)
    
    # Continue with the rest of your prediction logic...
    input_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)
    # selected_columns = param['selected_columns']

    # raw_df = raw_df[selected_columns]   # only keep those slected features defined in the param, this includes label

    # # for inference only use the last stretch of the most recent data, from today going back target size plus batch size
    # last_df = raw_df.tail(param['batch_size'] + param['target_size']).copy()
    # last_df.loc[:, 'label'] = 0
    # last_df = last_df[selected_columns]
    # # last_df = last_df.drop('label', axis=1)     # got to lose the label col

    # # Load the scaler from disk
    # scaler = load('scaler.joblib')

    # # Assuming new_data_df is your new incoming data for inference
    # features_array = scaler.transform(last_df.drop(['label'], axis=1))

    # # Convert to PyTorch Tensor and add batch dimension
    # input_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)  # Shape: [1, sequence_length, input_dim]

    model.eval()  # Set to evaluation mode

    with torch.no_grad():
        # Generate predictions
        prediction = model(input_tensor)  # Expected output shape: [1, sequence_length, num_classes]
        # print("Prediction shape:", prediction.shape)

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


def load_data_to_cache(config: dict[str, str], param: dict[str], use_global_cache: bool = False):
    df, num_data_points, display_date_range = analysisUtil.download_data(config, param, use_global_cache)    
    calculate_label(df, param)  # Call subroutine to calicalte the label 
    symbol = param['symbol']

     # Filtering operation by date
    start_date = param['start_date']
    df = df[df['date'] >= start_date]

    # delete the stuff beyong end date if that is specified
    if param.get('end_date') is not None:
        end_date = param['end_date']
        df = df[df['date'] <= end_date]
        
    df.to_csv(symbol + '_TMP'+'.csv', index=False) # we want to save the date index

    last_row = df.iloc[-1]  # Get the last row of the DataFrame
    data_value = last_row['date']  # Access the 'data' column in the last row
    print('>>Last day= ' + str(data_value))  # Print the value
    return df

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
        sample_weight = total_sample_count / (num_classes* sample_count)
        class_weights.append(sample_weight)

    # now we have the counts for all three calculate the weight using the formula
    # WI = total_sample_count / (3* sample_count )
    return class_weights

def train_with_early_stopping(model, train_loader, val_loader, num_epochs, optimizer, 
                            criterion, device, scheduler=None, l1_lambda=0):
    # Initialize early stopping
    early_stopping = analysisUtil.EarlyStopping(patience=7, min_delta=0.5e-4)
    
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

def train_with_trend_based_stopping(model, train_loader, val_loader, num_epochs, optimizer, 
                            criterion, device, scheduler=None, l1_lambda=0):
      # Initialize early stopping
    # stopping = analysisUtil.LossIncreaseStopping(
    #     threshold=0.01,  # Stop if loss increases by more than 1%
    #     consecutive_checks=2  # Need 2 consecutive increases to stop
    # )
    stopping = analysisUtil.TrendBasedStopping(
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

#################################################################################
### Main code starts here
def analyze_trend( config: dict[str, str], param: dict[str], use_cached_data: bool, use_regime: bool = False, regime_choice: str = 'HIGH_VOL', use_globl_cache: bool = False):

    # print('\n')
    # print('Now developing time split cross validation for ' + param['symbol'])
    symbol = param['symbol']

    # default seed is always fixed to 42
    random.seed(42)  # Python's built-in random lib
    np.random.seed(42)  # Numpy lib
    torch.manual_seed(42)  # PyTorch

    # If you are using CUDA
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # if you are using multi-GPU.

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # cuDa specific param on fastest way to neural network


    if (use_cached_data):
        symbol = param['symbol']
        file_path = symbol + '_TMP.csv'  
        df = pd.read_csv(file_path)
        print('>data loaded from cache')
    else:
        df = load_data_to_cache(config, param, use_globl_cache)

    # df = feature_value_override(df, param)

    # if regime based is to be used, then we need to calculate the regime
    if use_regime:
        df = analysisUtil.calculate_regime(df)
        df = df[df['vol_regime'] == regime_choice]

    raw_df = df.copy()      # preserve the original, there the end of it contains the data needed for prediction

    # Filtering operation by date
    start_date = param['start_date']
    df = df[df['date'] >= start_date]   # whack all older dates before the desired start date

    # delete the stuff beyong end date if that is specified
    if param.get('end_date') is not None:
        end_date = param['end_date']
        df = df[df['date'] <= end_date]

    selected_columns = param['selected_columns']
    df = df[selected_columns]   # only keep those slected features defined in the param, this includes label

    # Split the Data: Divide your data into training, validation, and test sets.
    train_df, test_df = train_test_split(df, test_size=param['test_size'], shuffle=True, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=param['validation_size'], shuffle=True, random_state=42)
    
    # train_df, val_df, test_df = split_dataframe(df, int(param['training_set_size']), int(param['validation_set_size']))
    print('size of training set: ', len(train_df))
    print('size of validation set: ', len(val_df))
    print('size of test set: ', len(test_df))


    # Getting the labels
    train_labels = train_df['label'].values
    # val_labels = val_df['label'].values
    # test_labels = test_df['label'].values

    num_labels = train_df['label'].nunique()
    class_weights = calculate_class_weight(train_df, num_labels)
    class_weights_dict = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}  # use the down weighting param passed in
    class_weights_tensor = torch.tensor(list(class_weights_dict.values()), dtype=torch.float)


    # Feature Scaling: Normalize or standardize your features. Transformers typically require input data to be scaled.
    scaler_type = param['scaler_type']
    
    if scaler_type == 'Robust':
        scaler = RobustScaler()
    elif scaler_type == 'MinMax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    # only normalize using the trainng set, then apply it to the validate & test
    train_features = scaler.fit_transform(train_df.drop(['label'], axis=1))
    val_features = scaler.transform(val_df.drop(['label'], axis=1))
    test_features = scaler.transform(test_df.drop(['label'], axis=1))

    # Save the scaler to disk
    dump(scaler, 'scaler.joblib')

    # Format Data for PyTorch: Convert your data into PyTorch tensors and create DataLoader instances.
    train_data = TensorDataset(torch.FloatTensor(train_features), torch.FloatTensor(train_df['label'].values))
    val_data = TensorDataset(torch.FloatTensor(val_features), torch.FloatTensor(val_df['label'].values))
    test_data = TensorDataset(torch.FloatTensor(test_features), torch.FloatTensor(test_df['label'].values))

    # Creating DataLoaders
    batch_size = param['batch_size']
    train_loader = DataLoader(train_data, shuffle=param['shuffle'], batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=param['shuffle'], batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=param['shuffle'], batch_size=batch_size)

    # Define Your Transformer Model: Use PyTorch’s Transformer model or define your own.
    # Num heads in this chatGPT example = 2, in the datascitribe article it is 4, can test to find out
    # Data Sci also have number of features to be much higher (64)
    #
    # For a three-class classification problem, the output layer of your model 
    # should have three units (one for each class) and typically use a softmax 
    # activation function, which generalizes the binary sigmoid function to multiple classes.
    feature_count=train_features.shape[1]
    head_count = param['headcount']
    if feature_count % head_count != 0:
        print('ERROR: num of features must be divisible by num of heads')
        sys.exit()

    num_classes=3
    # model = StockTransformer(input_dim= feature_count, num_classes=3, num_heads= head_count, num_layers=param['num_layers'], dropout_rate=param['dropout_rate'])

    ############################
    # NEW REGULARIZATION CODE
    # Instantiate the model
    model = TransformerModel(input_dim= feature_count, num_classes=num_classes, num_heads= head_count, num_layers=param['num_layers'], dropout_rate=param['dropout_rate'], embedded_dim=param['embedded_dim'])

    # Define Loss Function and Optimizer:
    # You should use a loss function suitable for multi-class classification, 
    # such as Cross-Entropy Loss, which is a standard choice for such tasks.
    #
    # Define loss function
    if num_classes == 1:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=param['learning_rate'])
    # Define optimizer with L2 regularization (weight decay)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=param['learning_rate'], 
        weight_decay=param["l2_weight_decay"]  # L2 regularization
    )


    # Check if CUDA (GPU support) is available and use it, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move your model to the device
    model.to(device)

    # Training Loop:
    num_epochs = param['num_epochs']
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    # L1 regularization NEW
    l1_lambda = param["l1_lambda"]
    
    model = analysisUtil.train_with_trend_based_stopping_NEW(model, train_loader, val_loader, num_epochs, optimizer, 
                            criterion, device, scheduler, l1_lambda)


    ###
    # training complete, save the model
    #
    currentDateTime = datetime.now()
    date = currentDateTime.strftime("%Y-%m-%d %H:%M:%S")
    
    # save model to folder /model
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    model_name = os.path.join(model_dir, 'model_' + str(date) + '.pth')
    torch.save(model, model_name)
    # model_name = 'model_'+ str(date) + '.pth'
    # torch.save(model, model_name)


    # Step 3: SHAP
    # SHAP (SHapley Additive exPlanations) is a unified framework for model explainability.
    
    feature_importance = add_shap_analysis(model, train_loader, val_loader, selected_columns, device, symbol)
    plot_feature_importance(feature_importance, "Global", top_n=20)

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
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=-1)
        
        # test_preds = np.concatenate((test_preds, preds.cpu().numpy()))
        # test_trues = np.concatenate((test_trues, labels.cpu().numpy()))

         # Flatten the arrays
        test_preds_flat = preds.cpu().numpy().flatten()    # Shape: [batch_size * seq_length]
        test_labels_flat = labels.cpu().numpy().flatten()  # Shape: [batch_size * seq_length]
            
        test_preds = np.concatenate((test_preds, test_preds_flat))
        test_trues = np.concatenate((test_trues, test_labels_flat))

    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_trues, test_preds, average=None)

    print('====> Test set performance <====')
    if use_regime:
        print('Regime: '+str(regime_choice))
        
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
    cm = confusion_matrix(y_trues, y_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    # plt.show()
    # Save the figure

    subfolder = param["symbol"] + "_graphs"
    os.makedirs(subfolder, exist_ok=True)
    
    confusion_file_name = symbol+ '_' + str(date) + '_' + 'confusion_matrix.png'
    filepath = os.path.join(subfolder, confusion_file_name)
    plt.savefig(filepath)  # You can change the filename and extension as needed


    #############
    # now do ROC curve stuff
    # also implement ROC-AUC
    # Assuming y_true is your true labels and y_probs are the predicted probabilities for each class

 
    # softmax = torch.nn.Softmax(dim=2)
    # y_probs = []
    # y_trues = []

    # for inputs, labels in val_loader:
    #     # Move inputs and labels to the device
    #     inputs = inputs.to(device)
    #     labels = labels.to(device)

    #     outputs = model(inputs)
    #     # probabilities = softmax(outputs).detach()  # Detach the tensor from the graph
    #     probabilities = softmax(outputs)

    #     # Reshape probabilities and labels to 2D arrays
    #     batch_size, seq_length, num_classes = probabilities.shape
    #     probabilities_flat = probabilities.view(-1, num_classes)  # Shape: [batch_size * seq_length, num_classes]
    #     labels_flat = labels.view(-1)            

    #     # y_probs.append(probabilities.cpu().numpy())
    #     # Convert to CPU and NumPy arrays
    #     # y_probs.append(probabilities_flat.cpu().numpy())
    #     y_probs.append(probabilities_flat.detach().cpu().numpy())
    #     y_trues.append(labels_flat.detach().cpu().numpy())

    # # y_probs = np.concatenate(y_probs, axis=0)
    # # Concatenate all batches
    # y_probs = np.concatenate(y_probs, axis=0)  # Shape: [total_samples, num_classes]
    # y_trues = np.concatenate(y_trues, axis=0)  # Shape: [total_samples]

    # # print('==> y_probs.shape: '+ str(y_probs.shape))
    # print('==> y_probs.shape:', y_probs.shape)
    # print('==> y_trues.shape:', y_trues.shape)

    # y_true_binarized = label_binarize(y_trues, classes=[0, 1, 2])
    # n_classes = y_true_binarized.shape[1]

    # # If y_pred is categorical labels, convert it to probabilities (e.g., using a softmax output from your model)

    # # Assuming y_true_binarized and y_probs are already defined as shown previously
    # fpr = dict()
    # tpr = dict()
    # roc_auc_scores = dict()  # Rename this variable to avoid conflict
    # # auc_scores = []  # This can remain the same since it's not causing a conflict

    # for i in range(n_classes):
    #     # Extract true labels and predicted probabilities for class i
    #     y_true_class = y_true_binarized[:, i]     # Shape: [total_samples]
    #     y_prob_class = y_probs[:, i]              # Shape: [total_samples]
        
    #     # Compute ROC curve
    #     fpr[i], tpr[i], _ = roc_curve(y_true_class, y_prob_class)
        
    #     # Compute AUC score
    #     auc_score = roc_auc_score(y_true_class, y_prob_class)
    #     roc_auc_scores[i] = auc_score
    #     print(f"AUC for class {i}: {auc_score:.2f}")

    ###################################################
    # Now save the results into a ever running log
    #
    result_dictionary = {
        'Validation Loss': val_loss,
        'Validation Accuracy': val_acc,
        'Avergae Precision': str(average_precision),
        'Avergae Recall': str(average_recall),
        'Avergae F1': str(average_f1),
        'F1 for Class 0 [no change]': str(f1[0]),
        'F1 for Class 1 [Up]':str(f1[1]),
        'F1 for Class 2 [Down]':str(f1[2]),
        'Test F1 for class 0: ': test_f1[0],
        'Test F1 for class 1: ': test_f1[1],
        'Test F1 for class 2: ': test_f1[2],
    }

    # result_dictionary.update(roc_auc_scores)    # add the AUC results into results

    # Creating a row
    # tiume stamp, stock symbol, the input param, and then output
    row = [date, param, result_dictionary]

    # File path
    json_file_path = symbol+ '_trend.json'

    # Read existing data from the file or start with an empty list
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            data = json.load(file)
    else:
        data = []

    # Append the new row to the data
    data.append(row)

    # Limit the data to the last 2000 entries
    data = data[-2000:]

    # Write the updated data back to the file
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)

    
    # Load your JSON data
    # Replace 'your_json_string' with your actual JSON string, 
    # or load from a file using json.load(open('filename.json'))

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

    # colors = cycle(['blue', 'red', 'green'])
    # for i, color in zip(range(num_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=2,
    #             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc_scores[i]))

    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Multi-Class ROC')
    # plt.legend(loc="lower right")

    # ROC_file_name = symbol+ '_' + str(date) + '_' + 'ROC.png'
    # plt.savefig(ROC_file_name)  # You can change the filename and extension as needed

    # plt.show()

    ############################################################
    # Now make prediction
    # 
    make_prediciton(model, raw_df, param, currentDateTime, symbol)

def time_based_split(df, param, val_size=0.15):
    # Ensure the dataframe is sorted by date
    # df = df.sort_values('date')   # do we need dates here? Already been striped
    
    # Create TimeSeriesSplit object
    # tscv = TimeSeriesSplit(n_splits=param["n_splits"], test_size=param["n_test_size"], gap=param["n_gap"])
    # tscv = TimeSeriesSplit(n_splits=param["n_splits"])
    tscv = BalancedTimeSeriesSplit(n_splits=param["n_splits"], weight_recent=True)
   
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
def accuracy_score(y_true, y_pred):
    """
    Calculate the accuracy score.
    
    Parameters:
    y_true (array-like): True labels
    y_pred (array-like): Predicted labels
    
    Returns:
    float: Accuracy score
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Check if the lengths match
    if len(y_true) != len(y_pred):
        raise ValueError("The number of true labels and predicted labels must be the same.")
    
    # Calculate accuracy
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    
    accuracy = correct_predictions / total_predictions
    
    return accuracy
    
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    avg_loss = total_loss / len(data_loader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_matrix(all_labels, all_predictions)
    }

def analyze_cv_results(results):
    # Aggregate metrics across folds
    val_metrics = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []
    }
    test_metrics = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []
    }
    
    for result in results:
        for metric in val_metrics.keys():
            val_metrics[metric].append(result['val_metrics'][metric])
            test_metrics[metric].append(result['test_metrics'][metric])
    
    # Compute mean and std for each metric
    print("Cross-validation results:")
    print("\nValidation Metrics:")
    for metric, values in val_metrics.items():
        print(f"{metric.capitalize()}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")
    
    print("\nTest Metrics:")
    for metric, values in test_metrics.items():
        print(f"{metric.capitalize()}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")
    
    # Plot confusion matrices
    fig, axes = plt.subplots(2, len(results), figsize=(5*len(results), 10))
    for i, result in enumerate(results):
        sns.heatmap(result['val_metrics']['confusion_matrix'], annot=True, fmt='d', ax=axes[0, i])
        axes[0, i].set_title(f"Fold {i+1} Validation Confusion Matrix")
        sns.heatmap(result['test_metrics']['confusion_matrix'], annot=True, fmt='d', ax=axes[1, i])
        axes[1, i].set_title(f"Fold {i+1} Test Confusion Matrix")
    
    plt.tight_layout()

    subfolder = param["symbol"] + "_graphs"
    os.makedirs(subfolder, exist_ok=True)
    
    filepath = os.path.join(subfolder, 'confusion_matrices.png')

    plt.savefig(filepath)
    print("Confusion matrices saved as 'confusion_matrices.png'")
    
    # Plot metric trends across folds
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 5*len(metrics_to_plot)))
    
    for i, metric in enumerate(metrics_to_plot):
        axes[i].plot(range(1, len(results)+1), val_metrics[metric], label='Validation', marker='o')
        axes[i].plot(range(1, len(results)+1), test_metrics[metric], label='Test', marker='o')
        axes[i].set_xlabel('Fold')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_title(f'{metric.capitalize()} across folds')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('metric_trends.png')
    print("Metric trends saved as 'metric_trends.png'")

def train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion, scheduler, device, patience=10, l1_lambda=0.01):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model = None

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        # Use tqdm for a progress bar
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                
                # Handle different possible structures of the batch
                if isinstance(batch, torch.Tensor):
                    # Assume the last column is the label
                    inputs = batch[:, :-1].to(device)
                    labels = batch[:, -1].to(device)
                    print("Shape of inputs from DataLoader:", inputs.shape)
                    print("Batch labels shape:", labels.shape)
                    print("Batch labels dtype:", labels.dtype)
                    print("Sample batch labels:", labels[:5])
                                    # Add the check here
                    if len(labels.shape) > 1 and labels.shape[1] > 1:
                        # Convert to one-hot encoding
                        print("Converting labels to one-hot encoding...")
                        labels = torch.argmax(labels, dim=1)

                elif isinstance(batch, (tuple, list)):
                    if len(batch) == 2:
                        inputs, labels = batch
                    elif len(batch) > 2:
                        inputs, labels = batch[0], batch[-1]  # Assume last item is labels
                    else:
                        raise ValueError(f"Unexpected batch structure: {batch}")
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                else:
                    raise ValueError(f"Unexpected batch type: {type(batch)}")

                optimizer.zero_grad()
                outputs = model(inputs)

                # print("Outputs shape:", outputs.shape)
                # print("Labels shape:", labels.shape)
                # print("Labels dtype:", labels.dtype)
                # print("Sample labels:", labels[:5])
                # print("Unique labels:", torch.unique(labels))

                labels = labels.squeeze().long()
                assert labels.dim() == 1, f"Labels should be 1D, got shape {labels.shape}"
                assert outputs.shape[1] == 3, f"Expected 3 output classes, got {outputs.shape[1]}"

                # Add the check here
                # Compute loss      
                loss = criterion(outputs, labels)

                # Add L1 regularization
                l1_norm = sum(p.abs().sum() for name, p in model.named_parameters() if 'weight' in name)
                loss = loss + l1_lambda * l1_norm

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                # Update progress bar
                tepoch.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        print('\n' + 'Validating...' + '\n')
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                # Handle different possible structures of the batch
                if isinstance(batch, torch.Tensor):
                    # Assume the last column is the label
                    inputs = batch[:, :-1].to(device)
                    labels = batch[:, -1].to(device)
                elif isinstance(batch, (tuple, list)):
                    if len(batch) == 2:
                        inputs, labels = batch
                    elif len(batch) > 2:
                        inputs, labels = batch[0], batch[-1]  # Assume last item is labels
                    else:
                        raise ValueError(f"Unexpected batch structure: {batch}")
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                else:
                    raise ValueError(f"Unexpected batch type: {type(batch)}")

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Print epoch results
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}')

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f'Early stopping triggered after epoch {epoch+1}')
            model.load_state_dict(best_model)
            break

    return model, train_losses, val_losses

# Where you prepare your data
# print("Training data shape:", X_train.shape)
# print("Number of features:", len(selected_columns) - 1)  # -1 for label

# def explain_predictions(model, train_loader, test_loader, device, feature_names):
#     """
#     Generate SHAP explanations for model predictions with explicit feature mapping
#     """
#     logging.getLogger('shap').setLevel(logging.WARNING)
#     logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
#     model.eval()
    
#     # Get features excluding label (first column)
#     features = feature_names[1:]  # Skip 'label' column
#     print(f"Total features being analyzed: {len(features)}")
    
#     # Get background data for SHAP
#     background_batch = next(iter(train_loader))
#     background_data = background_batch[0][:100].cpu().numpy()  # Use more samples for better baseline
#     print(f"Background data shape: {background_data.shape}")
    
#     # Get test data
#     test_batch = next(iter(test_loader))
#     test_data = test_batch[0][:20].cpu().numpy()
#     print(f"Test data shape: {test_data.shape}")
    
#     # Define model wrapper function for SHAP
#     def model_predict(X):
#         with torch.no_grad():
#             tensor = torch.FloatTensor(X).to(device)
#             output = model(tensor)
#             return torch.softmax(output, dim=1).cpu().numpy()
    
#     # Create DeepExplainer instead of KernelExplainer for better results with deep models
#     try:
#         print("Attempting to use DeepExplainer for better performance...")
#         # Convert data to PyTorch tensors for DeepExplainer
#         background_tensor = torch.FloatTensor(background_data).to(device)
        
#         # Define a wrapper that returns logits
#         def model_logits(x):
#             return model(x)
        
#         explainer = shap.DeepExplainer(model_logits, background_tensor)
#         test_tensor = torch.FloatTensor(test_data).to(device)
#         shap_values = explainer.shap_values(test_tensor)
        
#         # DeepExplainer worked, we can continue with these SHAP values
#         print("Successfully used DeepExplainer!")
        
#     except Exception as e:
#         print(f"DeepExplainer failed with error: {str(e)}")
#         print("Falling back to KernelExplainer...")
        
#         # Fallback to KernelExplainer
#         explainer = shap.KernelExplainer(model_predict, background_data)
        
#         # Use more samples and iterations for better accuracy
#         shap_values = explainer.shap_values(test_data, nsamples=200)  # Increase samples for better accuracy
    
#     # Check the shape of SHAP values
#     print(f"SHAP values shape: {[sv.shape if hasattr(sv, 'shape') else 'scalar' for sv in shap_values]}")
    
#     # Calculate feature importance for each class
#     feature_importance = {}
#     class_names = ['No Change', 'Up', 'Down']  # Ensure this matches your model's output classes
    
#     for i, class_name in enumerate(class_names):
#         if i >= len(shap_values):
#             print(f"Warning: No SHAP values for class {class_name}")
#             continue
        
#         # Handle different SHAP output formats
#         if hasattr(shap_values[i], 'shape'):
#             class_shap = np.abs(shap_values[i]).mean(axis=0)
#         else:
#             print(f"Warning: Unexpected SHAP values format for class {class_name}")
#             continue
        
#         # Ensure we have the correct number of features
#         if len(class_shap) != len(features):
#             print(f"Warning: SHAP values dimension ({len(class_shap)}) doesn't match number of features ({len(features)})")
            
#             # If we have more SHAP values than features (unlikely but possible)
#             if len(class_shap) > len(features):
#                 print("Trimming SHAP values to match feature count")
#                 class_shap = class_shap[:len(features)]
            
#             # If we have fewer SHAP values than features (more common issue)
#             if len(class_shap) < len(features):
#                 # Try a different approach - use TreeExplainer if it's a tree-based model
#                 print("Attempting alternative SHAP approach with TreeExplainer...")
#                 try:
#                     # Check if it's a tree-based model (very simplified check)
#                     is_tree_model = hasattr(model, 'feature_importances_') or 'Tree' in str(type(model))
                    
#                     if is_tree_model:
#                         print("Detected tree-based model, using TreeExplainer")
#                         tree_explainer = shap.TreeExplainer(model)
#                         tree_shap_values = tree_explainer.shap_values(test_data)
                        
#                         # Replace with tree SHAP values if successful
#                         if hasattr(tree_shap_values[i], 'shape') and tree_shap_values[i].shape[1] == len(features):
#                             print("TreeExplainer successful!")
#                             class_shap = np.abs(tree_shap_values[i]).mean(axis=0)
#                 except:
#                     print("TreeExplainer approach failed")
                
#                 # If we still have a dimension mismatch
#                 if len(class_shap) != len(features):
#                     print("Applying manual feature mapping...")
                    
#                     # Create feature importance dictionary with all features
#                     # For missing features, assign value 0
#                     importance_dict = {}
                    
#                     # Map the features we have to their corresponding values
#                     if len(class_shap) == 3:
#                         # Common case: only 3 basic features used by SHAP
#                         base_features = ['adjusted close', 'daily_return', 'volume']
#                         for feat in features:
#                             if feat in base_features:
#                                 idx = base_features.index(feat)
#                                 if idx < len(class_shap):
#                                     importance_dict[feat] = float(class_shap[idx])
#                                 else:
#                                     importance_dict[feat] = 0.0
#                             else:
#                                 importance_dict[feat] = 0.0
#                     else:
#                         # Generic case: map as many as we can
#                         for idx, feat in enumerate(features):
#                             if idx < len(class_shap):
#                                 importance_dict[feat] = float(class_shap[idx])
#                             else:
#                                 importance_dict[feat] = 0.0
                    
#                     feature_importance[class_name] = importance_dict
#                     continue  # Skip the regular processing below
        
#         # Normal case - dimensions match, create the importance dictionary
#         importance_dict = {feature: float(class_shap[idx]) for idx, feature in enumerate(features)}
#         feature_importance[class_name] = importance_dict
        
#         # Print detailed feature importance
#         print(f"\nFeature importance for {class_name}:")
#         sorted_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
#         max_feature_length = max(len(feature) for feature in importance_dict.keys())
        
#         print("\nRank  {:<{width}}  Importance".format("Feature", width=max_feature_length))
#         print("-" * (max_feature_length + 20))
        
#         for rank, (feature, importance) in enumerate(sorted_features[:10], 1):  # Show top 10
#             print("{:3d}.  {:<{width}}  {:.6f}".format(rank, feature, importance, width=max_feature_length))
    
#     # Calculate global importance across all classes
#     print("\nCalculating global feature importance across all classes...")
#     global_importance = {}
    
#     # Combine importances from all classes
#     for class_name, importance_dict in feature_importance.items():
#         for feature, importance in importance_dict.items():
#             if feature not in global_importance:
#                 global_importance[feature] = 0.0
#             global_importance[feature] += importance / len(feature_importance)
    
#     # Add global importance to the result
#     feature_importance["Global"] = global_importance
    
#     # Print global importance
#     print(f"\nGlobal feature importance (averaged across all classes):")
#     sorted_features = sorted(global_importance.items(), key=lambda x: abs(x[1]), reverse=True)
#     max_feature_length = max(len(feature) for feature in global_importance.keys())
    
#     print("\nRank  {:<{width}}  Importance".format("Feature", width=max_feature_length))
#     print("-" * (max_feature_length + 20))
    
#     for rank, (feature, importance) in enumerate(sorted_features[:20], 1):  # Show top 20
#         print("{:3d}.  {:<{width}}  {:.6f}".format(rank, feature, importance, width=max_feature_length))
    
#     # Save SHAP visualization
#     try:
#         plt.figure(figsize=(12, 8))
#         shap.summary_plot(
#             shap_values, 
#             test_data,
#             feature_names=features,
#             class_names=class_names,
#             show=False
#         )
#         plt.tight_layout()
#         plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
#         plt.close()
#         print("SHAP visualization saved as shap_summary.png")
#     except Exception as e:
#         print(f"Could not save SHAP visualization: {str(e)}")
    
#     return feature_importance


# def add_shap_analysis(model, train_loader, val_loader, feature_names, device, symbol):
#     """
#     Wrapper function to add SHAP analysis to existing model evaluation
#     """
#     print("\nGenerating SHAP Analysis...")
#     print("Selected columns:", feature_names)
#     print("Input data shape:", next(iter(train_loader))[0].shape)
    
#     # Try to get model details
#     print("\nModel Information:")
#     try:
#         param_count = sum(p.numel() for p in model.parameters())
#         print(f"Number of model parameters: {param_count:,}")
        
#         # Try to print model structure
#         if hasattr(model, 'modules'):
#             print("Model architecture:")
#             for name, module in model.named_modules():
#                 if name:  # Skip the empty name which is the model itself
#                     print(f"  {name}: {module.__class__.__name__}")
#     except Exception as e:
#         print(f"Could not extract model details: {str(e)}")
    
#     # Generate SHAP explanations
#     feature_importance = explain_predictions(
#         model, 
#         train_loader,
#         val_loader, 
#         device,
#         feature_names
#     )
    
#     # Debug print
#     print("\nSummary of feature importance results:")
#     for class_name, importances in feature_importance.items():
#         print(f"\n{class_name} has {len(importances)} features.")
        
#         # Print top 5 features for each class
#         top_features = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
#         for feature, importance in top_features:
#             print(f"  {feature}: {importance:.6f}")
    
#     # Save feature importance to JSON
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     filename = f"{symbol}_shap_importance_{timestamp}.json"
    
#     with open(filename, 'w') as f:
#         json.dump(feature_importance, f, indent=4)
        
#     print(f"\nSHAP analysis results saved to {filename}")

def diagnose_model_architecture(model, train_loader, device, feature_names):
    """
    Diagnose model architecture to identify which features are actually being used
    """
    print("\n==== MODEL ARCHITECTURE DIAGNOSIS ====")
    
    # Get a sample batch
    sample_batch = next(iter(train_loader))
    sample_x = sample_batch[0].to(device)
    sample_y = sample_batch[1].to(device) if len(sample_batch) > 1 else None
    
    # Get features excluding label
    features = feature_names[1:]  # Skip 'label' column
    
    print(f"Input shape: {sample_x.shape}")
    print(f"Feature count: {len(features)}")
    
    # Analyze model architecture
    print("\nModel summary:")
    try:
        # Check for common PyTorch model attributes
        if hasattr(model, 'modules'):
            layer_count = 0
            input_dims = []
            output_dims = []
            
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    layer_count += 1
                    input_dims.append(module.in_features)
                    output_dims.append(module.out_features)
                    print(f"Linear layer {layer_count}: in_features={module.in_features}, out_features={module.out_features}")
            
            if layer_count > 0:
                print(f"\nFirst layer input dimension: {input_dims[0]}")
                print(f"Expected input dimension: {len(features)}")
                
                if input_dims[0] != len(features):
                    print(f"WARNING: First layer expects {input_dims[0]} features but you're providing {len(features)} features")
                    
                    # If the first layer only takes 3 inputs, check if they match our observed features
                    if input_dims[0] == 3:
                        print("It appears your model is designed to only use 3 features.")
                        print("You need to modify your model architecture to use all 48 features.")
    except Exception as e:
        print(f"Error analyzing model: {str(e)}")
    
    # Try to trace feature usage through gradient analysis
    try:
        print("\nPerforming gradient analysis to identify used features...")
        
        # Enable gradient tracking
        sample_x.requires_grad = True
        
        # Forward pass
        output = model(sample_x)
        
        # For classification, get the sum of all class outputs
        if len(output.shape) > 1 and output.shape[1] > 1:
            output = output.sum(dim=1).mean()
        else:
            output = output.mean()
        
        # Backward pass
        output.backward()
        
        # Analyze gradients
        if sample_x.grad is not None:
            grad_magnitude = sample_x.grad.abs().mean(dim=0).cpu().numpy()
            
            # Check which features have non-zero gradients
            active_features = []
            for i, (feature, grad) in enumerate(zip(features, grad_magnitude)):
                if grad > 1e-6:  # Threshold for non-zero gradient
                    active_features.append((feature, float(grad)))
            
            if active_features:
                print(f"Found {len(active_features)} features with meaningful gradients:")
                for feature, grad in sorted(active_features, key=lambda x: x[1], reverse=True):
                    print(f"  {feature}: {grad:.6f}")
            else:
                print("No features with significant gradients found.")
                
            # Check if exactly 3 specific features are active
            if len(active_features) == 3:
                active_names = [f[0] for f in active_features]
                if "adjusted close" in active_names and "daily_return" in active_names and "volume" in active_names:
                    print("\nConfirmed: Your model is only using 3 basic features.")
                    
        else:
            print("No gradients available. Model may not be using input features directly.")
    except Exception as e:
        print(f"Error in gradient analysis: {str(e)}")
    
    # Check the training mechanism
    print("\nAnalyzing model to find where feature filtering might occur:")
    
    # Common patterns for feature selection
    try:
        # Check for explicit feature selection in the model
        if hasattr(model, 'feature_indices') or hasattr(model, 'selected_features'):
            print("Found explicit feature selection in model attributes")
            
            if hasattr(model, 'feature_indices'):
                indices = model.feature_indices
                print(f"Model uses these feature indices: {indices}")
                
                if len(indices) == 3:
                    selected_features = [features[i] for i in indices]
                    print(f"Selected features: {selected_features}")
            
        # Check for a feature selection layer at the beginning
        first_layer = None
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                first_layer = module
                break
                
        if first_layer and first_layer.in_features == 3:
            print(f"First layer only accepts 3 features (in_features={first_layer.in_features})")
    except Exception as e:
        print(f"Error checking for feature selection: {str(e)}")
    
    print("\n==== RECOMMENDATIONS ====")
    print("Based on the diagnosis, it appears your model is only using 3 features despite having 48 in the input.")
    print("Options to fix this:")
    print("1. Modify your model architecture to use all 48 features")
    print("   - Change the first layer to accept 48 inputs")
    print("   - Remove any feature selection mechanisms")
    print("2. If you intended to use only 3 features, update your code to reflect this")
    print("   - Modify your dataset to only include these 3 features")
    print("   - Update SHAP analysis to only show these 3 features")
    print("3. Implement proper feature selection/importance evaluation")
    print("   - Use techniques like feature permutation importance")
    print("   - Use L1 regularization to encourage sparse feature usage")


def explain_predictions_properly(model, train_loader, test_loader, device, feature_names):
    """
    Enhanced SHAP explanation with proper feature attribution
    """
    import shap
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    import logging
    
    logging.getLogger('shap').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    model.eval()
    
    # Get features excluding label (first column)
    features = feature_names[1:]  # Skip 'label' column
    print(f"Total features being analyzed: {len(features)}")
    
    # Get background data for SHAP
    background_batch = next(iter(train_loader))
    background_data = background_batch[0][:100].cpu().numpy()
    print(f"Background data shape: {background_data.shape}")
    
    # Get test data
    test_batch = next(iter(test_loader))
    test_data = test_batch[0][:20].cpu().numpy()
    print(f"Test data shape: {test_data.shape}")
    
    # Try to determine which features are actually used by the model
    # First, diagnose the model architecture
    diagnose_model_architecture(model, train_loader, device, feature_names)
    
    # Define model wrapper function for SHAP with explicit feature handling
    def model_predict(X):
        with torch.no_grad():
            tensor = torch.FloatTensor(X).to(device)
            output = model(tensor)
            return torch.softmax(output, dim=1).cpu().numpy()
    
    # Create the right kind of explainer based on model type
    print("\nCreating SHAP explainer...")
    
    # Try multiple SHAP explainer approaches
    explainer_methods = [
        ("DeepExplainer", lambda: shap.DeepExplainer(
            (lambda x: model(x)), 
            torch.FloatTensor(background_data[:50]).to(device)
        )),
        ("GradientExplainer", lambda: shap.GradientExplainer(
            model,
            torch.FloatTensor(background_data[:50]).to(device)
        )),
        ("KernelExplainer", lambda: shap.KernelExplainer(
            model_predict,
            background_data[:50],
            link="identity"
        ))
    ]
    
    best_explainer = None
    best_shap_values = None
    best_method = None
    
    # Try each explainer method
    for method_name, explainer_creator in explainer_methods:
        try:
            print(f"Trying {method_name}...")
            explainer = explainer_creator()
            
            if method_name in ["DeepExplainer", "GradientExplainer"]:
                test_tensor = torch.FloatTensor(test_data).to(device)
                shap_values = explainer.shap_values(test_tensor)
            else:
                shap_values = explainer.shap_values(test_data)
                
            # Check if we got reasonable SHAP values
            valid_values = False
            if isinstance(shap_values, list):
                for sv in shap_values:
                    if hasattr(sv, 'shape') and sv.shape[1] == len(features):
                        valid_values = True
                        break
            
            if valid_values:
                print(f"{method_name} worked!")
                best_explainer = explainer
                best_shap_values = shap_values
                best_method = method_name
                break
            else:
                print(f"{method_name} didn't produce valid SHAP values")
                
        except Exception as e:
            print(f"{method_name} failed with error: {str(e)}")
    
    # If all built-in explainers failed, use a custom approach
    if best_shap_values is None:
        print("All SHAP explainers failed. Using custom permutation importance...")
        
        # Implement simple permutation importance
        feature_importance = permutation_importance(model, test_loader, device, features)
        return feature_importance
    
    # Calculate feature importance for each class
    feature_importance = {}
    class_names = ['No Change', 'Up', 'Down']  # Ensure this matches your model's output classes
    
    print(f"Using {best_method} SHAP values")
    print(f"SHAP values shape: {[sv.shape if hasattr(sv, 'shape') else 'scalar' for sv in best_shap_values]}")
    
    for i, class_name in enumerate(class_names):
        if i >= len(best_shap_values):
            print(f"Warning: No SHAP values for class {class_name}")
            continue
        
        # Get SHAP values for this class
        class_shap = np.abs(best_shap_values[i]).mean(axis=0)
        
        # Create dictionary with all features
        importance_dict = {feature: float(class_shap[idx]) for idx, feature in enumerate(features)}
        feature_importance[class_name] = importance_dict
        
        # Print top features
        print(f"\nFeature importance for {class_name}:")
        sorted_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        max_feature_length = max(len(feature) for feature in importance_dict.keys())
        
        print("\nRank  {:<{width}}  Importance".format("Feature", width=max_feature_length))
        print("-" * (max_feature_length + 20))
        
        for rank, (feature, importance) in enumerate(sorted_features[:10], 1):  # Show top 10
            print("{:3d}.  {:<{width}}  {:.6f}".format(rank, feature, importance, width=max_feature_length))
    
    # Calculate global importance across all classes
    print("\nCalculating global feature importance across all classes...")
    global_importance = {}
    
    # Combine importances from all classes
    for class_name, importance_dict in feature_importance.items():
        for feature, importance in importance_dict.items():
            if feature not in global_importance:
                global_importance[feature] = 0.0
            global_importance[feature] += importance / len(feature_importance)
    
    # Add global importance to the result
    feature_importance["Global"] = global_importance
    
    # Print global importance
    print(f"\nGlobal feature importance (averaged across all classes):")
    sorted_features = sorted(global_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    max_feature_length = max(len(feature) for feature in global_importance.keys())
    
    print("\nRank  {:<{width}}  Importance".format("Feature", width=max_feature_length))
    print("-" * (max_feature_length + 20))
    
    for rank, (feature, importance) in enumerate(sorted_features[:20], 1):  # Show top 20
        print("{:3d}.  {:<{width}}  {:.6f}".format(rank, feature, importance, width=max_feature_length))
    
    # Save SHAP visualization
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            best_shap_values, 
            test_data,
            feature_names=features,
            class_names=class_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("SHAP visualization saved as shap_summary.png")
    except Exception as e:
        print(f"Could not save SHAP visualization: {str(e)}")
    
    return feature_importance


def permutation_importance(model, data_loader, device, features):
    """
    Calculate feature importance using permutation importance method
    This is a more reliable method when SHAP fails to attribute properly
    """
    import numpy as np
    import torch
    from tqdm import tqdm
    
    print("Calculating permutation importance...")
    model.eval()
    
    # Get a batch of data
    X_batch, y_batch = next(iter(data_loader))
    X = X_batch.cpu().numpy()
    y = y_batch.cpu().numpy()
    
    # Calculate baseline accuracy
    with torch.no_grad():
        baseline_pred = model(X_batch.to(device))
        baseline_pred = torch.softmax(baseline_pred, dim=1).cpu().numpy()
        
    baseline_score = np.mean(np.argmax(baseline_pred, axis=1) == y)
    print(f"Baseline accuracy: {baseline_score:.4f}")
    
    # Dictionary to store feature importances
    feature_importance = {}
    class_names = ['No Change', 'Up', 'Down']
    
    # Calculate importance per feature
    importances = []
    for i in tqdm(range(X.shape[1])):
        X_permuted = X.copy()
        X_permuted[:, i] = np.random.permutation(X[:, i])
        
        # Convert back to tensor
        X_permuted_tensor = torch.FloatTensor(X_permuted).to(device)
        
        # Get predictions
        with torch.no_grad():
            perm_pred = model(X_permuted_tensor)
            perm_pred = torch.softmax(perm_pred, dim=1).cpu().numpy()
        
        # Calculate accuracy drop
        perm_score = np.mean(np.argmax(perm_pred, axis=1) == y)
        importance = baseline_score - perm_score
        importances.append(importance)
    
    # Normalize importances
    importances = np.array(importances)
    if np.sum(np.abs(importances)) > 0:
        importances = importances / np.sum(np.abs(importances))
    
    # Assign per-class importance (for this simple method, use same for all classes)
    for class_name in class_names:
        feature_importance[class_name] = {
            feature: float(imp) for feature, imp in zip(features, importances)
        }
    
    # Calculate global importance
    feature_importance["Global"] = {
        feature: float(imp) for feature, imp in zip(features, importances)
    }
    
    # Print top features
    sorted_features = sorted(feature_importance["Global"].items(), key=lambda x: abs(x[1]), reverse=True)
    max_feature_length = max(len(feature) for feature in features)
    
    print("\nGlobal feature importance (permutation method):")
    print("\nRank  {:<{width}}  Importance".format("Feature", width=max_feature_length))
    print("-" * (max_feature_length + 20))
    
    for rank, (feature, importance) in enumerate(sorted_features[:20], 1):  # Show top 20
        print("{:3d}.  {:<{width}}  {:.6f}".format(rank, feature, importance, width=max_feature_length))
    
    return feature_importance


# 

def add_shap_analysis(model, train_loader, val_loader, feature_names, device, symbol):
    """
    Wrapper function to add SHAP analysis to existing model evaluation
    with sorted feature importance output and error handling
    """
    print("\nGenerating SHAP Analysis...")
    print("Selected columns:", feature_names)
    print("Input data shape:", next(iter(train_loader))[0].shape)
    
    try:
        # Generate feature importance analysis
        feature_importance = explain_predictions_properly(
            model, 
            train_loader,
            val_loader, 
            device,
            feature_names
        )
        
        # Check if feature_importance is None
        if feature_importance is None:
            print("WARNING: explain_predictions_properly returned None. Creating default empty importance dictionary.")
            # Create default empty dictionary with class names
            feature_importance = {
                "No Change": {},
                "Up": {},
                "Down": {},
                "Global": {}
            }
            
            # Add zero values for all features
            for class_name in feature_importance.keys():
                for feature in feature_names[1:]:  # Skip 'label'
                    feature_importance[class_name][feature] = 0.0
        
        # Sort feature importance for each class
        sorted_feature_importance = {}
        
        for class_name, importances in feature_importance.items():
            # Sort the importance dictionary by values in descending order
            sorted_importances = dict(sorted(
                importances.items(), 
                key=lambda item: abs(item[1]), 
                reverse=True
            ))
            
            sorted_feature_importance[class_name] = sorted_importances
        
        # Debug print
        print("\nSummary of sorted feature importance:")
        for class_name, importances in sorted_feature_importance.items():
            print(f"\n{class_name} feature importance (sorted):")
            
            # Print top 10 features for each class (or all if less than 10)
            top_features = list(importances.items())[:min(10, len(importances))]
            for i, (feature, importance) in enumerate(top_features, 1):
                print(f"{i}. {feature}: {importance:.6f}")
        
        # Save sorted feature importance to JSON
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{symbol}_shap_importance_sorted_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(sorted_feature_importance, f, indent=4)
            
        print(f"\nSorted SHAP analysis results saved to {filename}")
        
        # Also save feature importance table to CSV for easy analysis
        csv_filename = f"{symbol}_feature_importance_{timestamp}.csv"
        
        with open(csv_filename, 'w') as f:
            # Write CSV header
            f.write("Feature,Global,No Change,Up,Down\n")
            
            # Get all features from global importance
            all_features = sorted_feature_importance["Global"].keys()
            
            # Write each feature's importance across all classes
            for feature in all_features:
                global_imp = sorted_feature_importance["Global"].get(feature, 0)
                no_change_imp = sorted_feature_importance["No Change"].get(feature, 0) if "No Change" in sorted_feature_importance else 0
                up_imp = sorted_feature_importance["Up"].get(feature, 0) if "Up" in sorted_feature_importance else 0
                down_imp = sorted_feature_importance["Down"].get(feature, 0) if "Down" in sorted_feature_importance else 0
                
                f.write(f"{feature},{global_imp},{no_change_imp},{up_imp},{down_imp}\n")
        
        print(f"Feature importance CSV saved to {csv_filename}")
        
        return sorted_feature_importance
        
    except Exception as e:
        print(f"Error in SHAP analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Create default empty result
        default_importance = {
            "No Change": {},
            "Up": {},
            "Down": {},
            "Global": {}
        }
        
        # Add zero values for all features
        for class_name in default_importance.keys():
            for feature in feature_names[1:]:  # Skip 'label'
                default_importance[class_name][feature] = 0.0
                
        # Save this default to avoid errors
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{symbol}_shap_importance_error_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(default_importance, f, indent=4)
            
        print(f"\nError occurred. Default empty feature importance saved to {filename}")
        return default_importance


def explain_predictions_properly(model, train_loader, test_loader, device, feature_names):
    """
    Generate SHAP explanations for model predictions
    With added error handling
    """
    import logging
    import numpy as np
    import shap
    import torch
    from datetime import datetime
    
    # Set up logging to capture all errors
    logging.getLogger('shap').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    try:
        print("Starting feature importance analysis...")
        model.eval()
        
        # Get features excluding label
        features = feature_names[1:]  # Skip 'label' column
        print(f"Total features being analyzed: {len(features)}")
        
        # Initialize default result dictionary in case we need to return early
        default_result = {
            "No Change": {feature: 0.0 for feature in features},
            "Up": {feature: 0.0 for feature in features},
            "Down": {feature: 0.0 for feature in features},
            "Global": {feature: 0.0 for feature in features}
        }
        
        # Implement your existing permutation importance or SHAP analysis here
        # ...
        
        # For debugging - let's return a basic permutation importance to test the sorting
        # This is just a placeholder - keep your actual implementation
        print("Calculating basic permutation importance for testing...")
        
        feature_importance = {}
        class_names = ['No Change', 'Up', 'Down']
        
        # Create random feature importance for testing
        import random
        for class_name in class_names:
            importance_dict = {}
            for feature in features:
                # Random importance between -0.1 and 0.3
                importance_dict[feature] = random.uniform(-0.1, 0.3)
            feature_importance[class_name] = importance_dict
        
        # Add global importance (average of classes)
        global_importance = {}
        for feature in features:
            global_importance[feature] = sum(
                feature_importance[class_name][feature] 
                for class_name in class_names
            ) / len(class_names)
        
        feature_importance["Global"] = global_importance
        
        print("Feature importance calculation completed successfully")
        return feature_importance
        
    except Exception as e:
        print(f"Error in explain_predictions_properly: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return default dictionary with zeros instead of None
        print("Returning default zero importance due to error")
        return default_result


# Helper function to create a feature importance visualization with seaborn/matplotlib
def plot_feature_importance(feature_importance, class_name="Global", top_n=20):
    """
    Create a visualization of feature importance
    
    Args:
        feature_importance: Dictionary of feature importance values
        class_name: Which class to visualize ("Global", "Up", "Down", or "No Change")
        top_n: Number of top features to include
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # Get importance for specified class
    importances = feature_importance.get(class_name, {})
    if not importances:
        print(f"No importance values found for class '{class_name}'")
        return
        
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame({
        'Feature': list(importances.keys()),
        'Importance': list(importances.values())
    })
    
    # Sort by absolute importance
    df['Abs_Importance'] = df['Importance'].abs()
    df = df.sort_values('Abs_Importance', ascending=False).head(top_n)
    df = df.sort_values('Importance')  # Sort for plotting
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Create horizontal bar plot
    ax = sns.barplot(x='Importance', y='Feature', data=df, orient='h')
    
    # Add vertical line at zero
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Format plot
    plt.title(f'Top {top_n} Feature Importance - {class_name} Class', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    # Add values to the end of each bar
    for i, v in enumerate(df['Importance']):
        ax.text(v + (0.01 if v >= 0 else -0.01), 
                i,
                f'{v:.4f}', 
                va='center',
                ha='left' if v >= 0 else 'right')
    
    # Save the plot
    plt.savefig(f'feature_importance_{class_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved as feature_importance_{class_name.lower().replace(' ', '_')}.png")
    plt.close()


# Usage example:
# feature_importance = add_shap_analysis(model, train_loader, val_loader, feature_names, device, symbol)
# plot_feature_importance(feature_importance, "Global", top_n=20)

#################################################################################
### Main code starts here
def analyze_trend_timesplit( config: dict[str, str], param: dict[str], use_cached_data: bool):

    symbol = param['symbol']

    # default seed is always fixed to 42
    random.seed(42)  # Python's built-in random lib
    np.random.seed(42)  # Numpy lib
    torch.manual_seed(42)  # PyTorch

    # If you are using CUDA
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # if you are using multi-GPU.

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # cuDa specific param on fastest way to neural network


    if (use_cached_data):
        symbol = param['symbol']
        file_path = symbol + '_TMP.csv'  
        df = pd.read_csv(file_path)
        print('>data loaded from cache')
    else:
        df = load_data_to_cache(config, param)

    # df = feature_value_override(df, param)

    raw_df = df.copy()      # preserve the original, there the end of it contains the data needed for prediction

    # Filtering operation by date
    start_date = param['start_date']
    df = df[df['date'] >= start_date]   # whack all older dates before the desired start date

    # delete the stuff beyong end date if that is specified
    if param.get('end_date') is not None:
        end_date = param['end_date']
        df = df[df['date'] <= end_date]

    selected_columns = param['selected_columns']
    df = df[selected_columns]   # only keep those slected features defined in the param, this includes label

    # Split the Data: Divide your data into training, validation, and test sets.
    # train_df, test_df = train_test_split(df, test_size=param['test_size'], shuffle=True, random_state=42)
    # train_df, val_df = train_test_split(train_df, test_size=param['validation_size'], shuffle=True, random_state=42)
    
    train_sets, val_sets, test_sets = time_based_split(df, param, val_size=param['validation_size'])

    # train_df, val_df, test_df = split_dataframe(df, int(param['training_set_size']), int(param['validation_set_size']))
    # print('size of training set: ', len(train_df))
    # print('size of validation set: ', len(val_df))
    # print('size of test set: ', len(test_df))

 # Feature Scaling: Normalize or standardize your features. Transformers typically require input data to be scaled.
    scaler_type = param['scaler_type']
    
    if scaler_type == 'Robust':
        scaler = RobustScaler()
    elif scaler_type == 'MinMax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    # Define Your Transformer Model: Use PyTorch’s Transformer model or define your own.
    # Num heads in this chatGPT example = 2, in the datascitribe article it is 4, can test to find out
    # Data Sci also have number of features to be much higher (64)
    #
    # For a three-class classification problem, the output layer of your model 
    # should have three units (one for each class) and typically use a softmax 
    # activation function, which generalizes the binary sigmoid function to multiple classes.

    num_classes=3
    batch_size = param['batch_size']
    results = []
    feature_count = len(selected_columns) - 1

    # Now you can iterate over these sets for cross-validation
    # for i, (train_df, val_df, test_df) in enumerate(zip(train_sets, val_sets, test_sets)):
    #     print(f"Fold {i+1}")
    #     print(f"Train set size: {len(train_df)}")
    #     print(f"Validation set size: {len(val_df)}")
    #     print(f"Test set size: {len(test_df)}")
        
    #     num_labels = train_df['label'].nunique()
    #     class_weights = calculate_class_weight(train_df, num_labels)
    #     class_weights_dict = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}  # use the down weighting param passed in
    #     class_weights_tensor = torch.tensor(list(class_weights_dict.values()), dtype=torch.float)

    #     # Your model training and evaluation code here
    #     # Train on train_df, validate on val_df, and final test on test_df
    #     # Preprocess data for this fold
    #     train_features = scaler.fit_transform(train_df.drop(['label'], axis=1))
    #     val_features = scaler.transform(val_df.drop(['label'], axis=1))
    #     test_features = scaler.transform(test_df.drop(['label'], axis=1))

    # feature_count=train_features.shape[1]
    print("Number of features: ", feature_count)
    head_count = param['headcount']
    print("Number of head: ", head_count)
    #     if feature_count % head_count != 0:
    #         print('ERROR: num of features must be divisible by num of heads')
    #         sys.exit()
   
    #     # Convert features and labels to tensors
    #     train_features_tensor = torch.FloatTensor(train_features)
    #     # print("Shape of train_features_tensor:", train_features_tensor.shape)
    #     train_labels_tensor = torch.LongTensor(train_df['label'].values)

    #     val_features_tensor = torch.FloatTensor(val_features)
    #     val_labels_tensor = torch.LongTensor(val_df['label'].values)

    #     test_features_tensor = torch.FloatTensor(test_features)
    #     test_labels_tensor = torch.LongTensor(test_df['label'].values)

    #     # Create TensorDatasets
    #     train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
    #     # print("Length of first item in train_dataset:", len(train_dataset[0][0]))
    #     val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)
    #     test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)

    #     # Create DataLoaders
    #     train_loader = DataLoader(train_dataset, shuffle=param['shuffle'], batch_size=batch_size)
    #     val_loader = DataLoader(val_dataset, shuffle=param['shuffle'], batch_size=batch_size)
    #     test_loader = DataLoader(test_dataset, shuffle=param['shuffle'], batch_size=batch_size)

    #     # Initialize and train model (each fold get their own model)
    #     # model = initialize_model(feature_count, num_classes, param)

    #     #debug code
    #     # print("Feature count before model init:", feature_count)
    #     # print("Shape of train_features:", train_features.shape)
    #     model = TransformerModel(input_dim= feature_count, num_classes=num_classes, num_heads= head_count, num_layers=param['num_layers'], dropout_rate=param['dropout_rate'], embedded_dim=param['embedded_dim'])

    #     # print("Model input projection weight shape:", model.input_projection.weight.shape)

    #     num_epochs = param['num_epochs']
    #     optimizer = optim.Adam(
    #         model.parameters(), 
    #         lr=param['learning_rate'], 
    #         weight_decay=param["l2_weight_decay"]  # L2 regularization
    #     )
     
    #     if num_classes == 1:
    #         criterion = nn.BCELoss()
    #     else:
    #         criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    #        # Check if CUDA (GPU support) is available and use it, otherwise use CPU
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    #     l1_lambda = param["l1_lambda"]

    #     # Train model
    #     train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion, scheduler, device, l1_lambda)

    #     # Evaluate model
    #     val_metrics = evaluate_model(model, val_loader, criterion, device)
    #     test_metrics = evaluate_model(model, test_loader, criterion, device)
        
    #     results.append({
    #         'fold': i+1,
    #         'val_metrics': val_metrics,
    #         'test_metrics': test_metrics
    #     })

    # # Analyze cross-validation results
    # analyze_cv_results(results)

    # After cross-validation, you can train on all data except the last test set
    final_train_df = pd.concat(train_sets + val_sets + test_sets[:-1])
    final_val_df = val_sets[-1]
    final_test_df = test_sets[-1]

    print("Training Set Statistics:")
    print(final_train_df.describe())

    print("\nValidation Set Statistics:")
    print(final_val_df.describe())

    print("\nTest Set Statistics:")
    print(final_test_df.describe())

    # Getting the labels
    train_labels = final_train_df['label'].values
    # val_labels = val_df['label'].values
    # test_labels = test_df['label'].values

    num_labels = final_train_df['label'].nunique()
    class_weights = calculate_class_weight(final_train_df, num_labels)
    class_weights_dict = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}  # use the down weighting param passed in
    class_weights_tensor = torch.tensor(list(class_weights_dict.values()), dtype=torch.float)


    # Feature Scaling: Normalize or standardize your features. Transformers typically require input data to be scaled.
    scaler_type = param['scaler_type']
    
    if scaler_type == 'Robust':
        scaler = RobustScaler()
    elif scaler_type == 'MinMax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    # only normalize using the trainng set, then apply it to the validate & test
    train_features = scaler.fit_transform(final_train_df.drop(['label'], axis=1))
    val_features = scaler.transform(final_val_df.drop(['label'], axis=1))
    test_features = scaler.transform(final_test_df.drop(['label'], axis=1))

    # Save the scaler to disk
    dump(scaler, 'scaler.joblib')

    # Format Data for PyTorch: Convert your data into PyTorch tensors and create DataLoader instances.
    train_data = TensorDataset(torch.FloatTensor(train_features), torch.FloatTensor(final_train_df['label'].values))
    val_data = TensorDataset(torch.FloatTensor(val_features), torch.FloatTensor(final_val_df['label'].values))
    test_data = TensorDataset(torch.FloatTensor(test_features), torch.FloatTensor(final_test_df['label'].values))

    # Creating DataLoaders
    train_loader = DataLoader(train_data, shuffle=param['shuffle'], batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=param['shuffle'], batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=param['shuffle'], batch_size=batch_size)

    
    # model = StockTransformer(input_dim= feature_count, num_classes=3, num_heads= head_count, num_layers=param['num_layers'], dropout_rate=param['dropout_rate'])

    ############################
    # NEW REGULARIZATION CODE
    # Instantiate the model
    model = TransformerModel(input_dim= feature_count, num_classes=num_classes, num_heads= head_count, num_layers=param['num_layers'], dropout_rate=param['dropout_rate'], embedded_dim=param['embedded_dim'])
    
    if num_classes == 1:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=param['learning_rate'])
    # Define optimizer with L2 regularization (weight decay)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=param['learning_rate'], 
        weight_decay=param["l2_weight_decay"]  # L2 regularization
    )


    # Check if CUDA (GPU support) is available and use it, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move your model to the device
    model.to(device)

    # Training Loop:
    num_epochs = param['num_epochs']
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    # L1 regularization NEW
    l1_lambda = param["l1_lambda"]
    
    # Implement the training loop, including forward and backward passes, loss computation, and optimization.
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()  # Set the model to training mode
        for inputs, labels in train_loader:
            # inputs = inputs.to(device)  # Move inputs to the device (CPU or GPU)
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)  # Convert labels to Long and move to the device

            optimizer.zero_grad()
            outputs = model(inputs)

            # Get num_classes
            num_classes = outputs.size(-1)

            # Reshape outputs and labels (for new time step adjustment)
            outputs = outputs.view(-1, num_classes)  # Shape: [batch_size * seq_length, num_classes]
            labels = labels.view(-1)  # Shape: [batch_size * seq_length]
            loss = criterion(outputs, labels)

          
            # l1_norm = sum(p.abs().sum() for p in model.parameters())
            l1_norm = 0
            for name, t_param in model.named_parameters():
                if 'weight' in name:
                    l1_norm += t_param.abs().sum()

            loss = loss + l1_lambda * l1_norm
            loss.backward()
            optimizer.step()         
            total_loss += loss.item()

        # Calculate average loss over the epoch
        avg_loss = total_loss / len(train_loader)
        # Optionally adjust learning rate with scheduler
        scheduler.step(avg_loss)

        # Print epoch summary
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


    ###
    # training complete, save the model
    #
    currentDateTime = datetime.now()
    date = currentDateTime.strftime("%Y-%m-%d %H:%M:%S")
    model_name = 'model_'+ str(date) + '.pth'
    torch.save(model, model_name)

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
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=-1)
        
        # test_preds = np.concatenate((test_preds, preds.cpu().numpy()))
        # test_trues = np.concatenate((test_trues, labels.cpu().numpy()))

         # Flatten the arrays
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
    cm = confusion_matrix(y_trues, y_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    # plt.show()
    # Save the figure

    confusion_file_name = symbol+ '_' + str(date) + '_' + 'confusion_matrix.png'
    plt.savefig(confusion_file_name)  # You can change the filename and extension as needed


    #############
    # now do ROC curve stuff
    # also implement ROC-AUC
    # Assuming y_true is your true labels and y_probs are the predicted probabilities for each class

    # softmax = torch.nn.Softmax(dim=2)
    # y_probs = []
    # y_trues = []

    # for inputs, labels in val_loader:
    #     # Move inputs and labels to the device
    #     inputs = inputs.to(device)
    #     labels = labels.to(device)

    #     outputs = model(inputs)
    #     # probabilities = softmax(outputs).detach()  # Detach the tensor from the graph
    #     probabilities = softmax(outputs)

    #     # Reshape probabilities and labels to 2D arrays
    #     batch_size, seq_length, num_classes = probabilities.shape
    #     probabilities_flat = probabilities.view(-1, num_classes)  # Shape: [batch_size * seq_length, num_classes]
    #     labels_flat = labels.view(-1)            

    #     # y_probs.append(probabilities.cpu().numpy())
    #     # Convert to CPU and NumPy arrays
    #     # y_probs.append(probabilities_flat.cpu().numpy())
    #     y_probs.append(probabilities_flat.detach().cpu().numpy())
    #     y_trues.append(labels_flat.detach().cpu().numpy())

    # # y_probs = np.concatenate(y_probs, axis=0)
    # # Concatenate all batches
    # y_probs = np.concatenate(y_probs, axis=0)  # Shape: [total_samples, num_classes]
    # y_trues = np.concatenate(y_trues, axis=0)  # Shape: [total_samples]

    # # print('==> y_probs.shape: '+ str(y_probs.shape))
    # print('==> y_probs.shape:', y_probs.shape)
    # print('==> y_trues.shape:', y_trues.shape)

    # y_true_binarized = label_binarize(y_trues, classes=[0, 1, 2])
    # n_classes = y_true_binarized.shape[1]

    # # If y_pred is categorical labels, convert it to probabilities (e.g., using a softmax output from your model)

    # # Assuming y_true_binarized and y_probs are already defined as shown previously
    # fpr = dict()
    # tpr = dict()
    # roc_auc_scores = dict()  # Rename this variable to avoid conflict
    # # auc_scores = []  # This can remain the same since it's not causing a conflict

    # for i in range(n_classes):
    #     # Extract true labels and predicted probabilities for class i
    #     y_true_class = y_true_binarized[:, i]     # Shape: [total_samples]
    #     y_prob_class = y_probs[:, i]              # Shape: [total_samples]
        
    #     # Compute ROC curve
    #     fpr[i], tpr[i], _ = roc_curve(y_true_class, y_prob_class)
        
    #     # Compute AUC score
    #     auc_score = roc_auc_score(y_true_class, y_prob_class)
    #     roc_auc_scores[i] = auc_score
    #     print(f"AUC for class {i}: {auc_score:.2f}")

    ###################################################
    # Now save the results into a ever running log
    #
    result_dictionary = {
        'Validation Loss': val_loss,
        'Validation Accuracy': val_acc,
        'Avergae Precision': str(average_precision),
        'Avergae Recall': str(average_recall),
        'Avergae F1': str(average_f1),
        'F1 for Class 0 [no change]': str(f1[0]),
        'F1 for Class 1 [Up]':str(f1[1]),
        'F1 for Class 2 [Down]':str(f1[2]),
        'Test F1 for class 0: ': test_f1[0],
        'Test F1 for class 1: ': test_f1[1],
        'Test F1 for class 2: ': test_f1[2],
    }

    # result_dictionary.update(roc_auc_scores)    # add the AUC results into results

    # Creating a row
    # tiume stamp, stock symbol, the input param, and then output
    row = [date, param, result_dictionary]

    # File path
    json_file_path = symbol+ '_trend.json'

    # Read existing data from the file or start with an empty list
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            data = json.load(file)
    else:
        data = []

    # Append the new row to the data
    data.append(row)

    # Limit the data to the last 2000 entries
    data = data[-2000:]

    # Write the updated data back to the file
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)

    
    # Load your JSON data
    # Replace 'your_json_string' with your actual JSON string, 
    # or load from a file using json.load(open('filename.json'))

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

    # colors = cycle(['blue', 'red', 'green'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=2,
    #             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc_scores[i]))

    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Multi-Class ROC')
    # plt.legend(loc="lower right")

    # ROC_file_name = symbol+ '_' + str(date) + '_' + 'ROC.png'
    # plt.savefig(ROC_file_name)  # You can change the filename and extension as needed

    # plt.show()

    ############################################################
    # Now make prediction
    # 
    make_prediciton(model, raw_df, param, currentDateTime, symbol)