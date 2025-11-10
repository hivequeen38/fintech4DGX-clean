# PyTorch basics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# For optimization and learning rate scheduling
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# For numerical operations
import numpy as np

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# For metrics and evaluation
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

# For data manipulation
import pandas as pd

# For type hints (optional but recommended)
from typing import Dict, List, Tuple, Optional

# For saving/loading models
import os
import json
from datetime import datetime

# For progress tracking (optional but recommended)
from tqdm import tqdm

# For reproducibility
import random

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List


class StockDataset(Dataset):
    def __init__(self, 
                 data: np.ndarray,
                 targets: np.ndarray,
                 device: str = 'cuda'):
        """
        Args:
            data: Feature data of shape [n_samples, sequence_length, n_features]
            targets: Target data of shape [n_samples, n_horizons]
            device: Device to store the data on
        """
        self.data = torch.FloatTensor(data).to(device)
        self.targets = torch.LongTensor(targets).to(device)
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]

class DataProcessor:
    def __init__(self, sequence_length=20, n_horizons=15):
        self.sequence_length = sequence_length
        self.n_horizons = n_horizons
        self.scaler = StandardScaler()
        
    def create_sequences(self, df: pd.DataFrame, feature_columns: List[str]):
        # Check for infinite or NaN values
        if df[feature_columns].isna().any().any() or np.isinf(df[feature_columns]).any().any():
            print("Found NaN or infinite values in input features!")
            # Fill NaN with forward fill, then backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        scaled_data = self.scaler.fit_transform(df[feature_columns])
        
        # Check scaling results
        if np.isnan(scaled_data).any() or np.isinf(scaled_data).any():
            print("Found NaN or infinite values after scaling!")
            scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=1e6, neginf=-1e6)
        
        scaled_df = pd.DataFrame(scaled_data, columns=feature_columns, index=df.index)
        
        # Print statistics of scaled data
        print("Scaled data statistics:")
        print(scaled_df.describe())
        
        X, y = [], []
        
        for i in range(len(df) - self.sequence_length - self.n_horizons + 1):
            sequence = scaled_df.iloc[i:(i + self.sequence_length)][feature_columns].values
            
            # Check sequence values
            if np.isnan(sequence).any() or np.isinf(sequence).any():
                print(f"Found NaN or infinite values in sequence at index {i}")
                continue
                
            targets = []
            for h in range(self.n_horizons):
                target_idx = i + self.sequence_length + h
                if target_idx < len(df):
                    target = df.iloc[target_idx]['label']       # changed target to label
                    if not np.isnan(target) and not np.isinf(target):
                        targets.append(target)
                    else:
                        print(f"Found NaN or infinite target at index {target_idx}")
                        break
            
            if len(targets) == self.n_horizons:
                X.append(sequence)
                y.append(targets)
        
        X = np.array(X)
        y = np.array(y)
        
        # Final check of prepared data
        print("Final data shapes:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print("\nX statistics:")
        print(f"X min: {X.min()}, max: {X.max()}, mean: {X.mean()}")
        print("\ny statistics:")
        print(f"y min: {y.min()}, max: {y.max()}, mean: {y.mean()}")
        
        return X, y

# Usage example
def prepare_data(df: pd.DataFrame,
                test_size: float = 0.2,
                val_size: float = 0.1,
                random_state: int = 42) -> Tuple[StockDataset, StockDataset, StockDataset]:
    """
    Prepare data for training, validation, and testing.
    """
    # Define your feature columns
    feature_columns = [
      ##################################
        # company Stock fundamentals
        ##################################
        "adjusted close",
        "daily_return",     # this is good
        "volume",
        # "volume_norm",    # regular volume still have better F1
        'Volume_Oscillator',
        "volatility",
        "VWAP",           # Volume Weighted Average Price (10/5/24 VWAP > Volume)
        "high",           # high low is slightly underperforming
        "low",
        'volume_volatility',
        # "relative_volume",
        # "relative_close",
        # "relative_high",
        # "relative_low",

        ##################################
        # company fundamentals
        ##################################
        'EPS', 
        'estEPS',
        'surprisePercentage',
        'totalRevenue',
        'netIncome',
        
        ##################################
        # Stock Momentum 
        ##################################
        "MACD_Signal",
        "MACD",
        "MACD_Hist",
        "ATR",
        "RSI",
        # 'RSI_overbought',
        # 'RSI_oversold',
        # 'RSI_bullish_divergence',
        # 'RSI_bearish_divergence',
        # 'RSI_momentum_strength',
        'Real Upper Band',  # bollinger band (taken out as not that useful)
        'Real Middle Band',
        'Real Lower Band',

        ##################################
        # Monetary market
        ##################################
        "interest",
        "10year",
        "2year",
        "3month",
        'DTWEXBGS',         # DOLLAR INDEX (10/5/24 this not as good as using eur_close)
        'DFEDTARU',         # federal fund rate upper limit
        'BOGMBBM',          # Monetary Base; Reserve Balances
        # 'eur_close',
        'jpy_close',
        'twd_close',
        # 'USALOLITONOSTSAM', # Leading Indicators: Composite Leading Indicator: Normalised for United States (in 5/9/24, no data since Jan 24)

        ##################################
        # Other Indeces
        ##################################
        "SPY_close",
        "qqq_close",
        "VTWO_close",
        'SPY_stoch',        #the appromiate for SnP Oscillator using slowD (slightly better than SPY_close)
        'calc_spy_oscillator',  # self calculated SnP500 oscillator based on SPY
        'QQQ_stoch',      #the appromiate for NASDAQ Oscillator using slowD (very close but in combo not as good as using QQQ_close)
        'VTWO_stoch',       #the appromiate for Russell Oscillator using slowD (much better)
  
        ##################################
        # General Market fundamentals
        ##################################
        "VIXCLS",           # VIX from FRED
        # "VIXCLS_Y",         # VIX from YFinance
        "DCOILWTICO",       # WTI oil price
        # "unemploy",         # as of 5/9, still no satat for 5/1
        "FEDTARMDLR",       # Longer Run FOMC Summary of Economic Projections for the Fed Funds Rate, Median (probably not useful (in 5/9/24, last data point is 3/20/24)
        # "USEPUINDXD",       # Economic Policy Uncertainty Index for United States
        'UMCSENT',          # consumer sentiment (removed, slightly worse then base (no update since 3/1/24)
        # 'BSCICP03USM665S',  # sentiment OECD   (at 5/9/24, no data since Jan 24)
        'DGORDER',          # DURABLE GOODS ORDER
        'CORESTICKM159SFRBATL',     # Sticky Price Consumer Price Index less Food and Energy (no update in 5 mo, I think this is dead)
        # 'PCE',              # No data since 3/1
        'PCU33443344',      # Producer Price Index by Industry: Semiconductor and Other Electronic Component Manufacturing 
        'SAHMREALTIME',     # sahm unemployment (seems to work better than unemploy)
        'JTSJOL',           # non farm job opening
        'GDP',              # Add GDP
        'FINRA_debit',      # finra debit

        ##################################
        # other meta data
        #
        'day_of_week',
        'month',
        'price_lag_1',
        'price_lag_5',
        'price_lag_15',
        'price_change_1',
        'price_change_5',
        'price_change_15',

        #################################
        # Industry Specific
        #################################
        'rs_amd',
        'rs_intc',
        'rs_avgo',
        'rs_amd_trend',
        'rs_intc_trend',
        'rs_avgo_trend',
        'rs_sox_trend_short',
        'rs_sox_trend_med',
        'rs_sox_trend_long',
        'rs_sox_volatility',
        'rs_smh_trend',
        'TSMC_close',
        'tsm_price_change_1',
        'rs_sp500_trend',
        'rs_nasdaq_trend',
    ]   # !!! Needs refactor to pass that in
    
    # Initialize processor
    processor = DataProcessor(sequence_length=20, n_horizons=15)
    
    # Create sequences
    X, y = processor.create_sequences(df, feature_columns)
    
    # Split into train, validation, and test
    n_samples = len(X)
    test_samples = int(n_samples * test_size)
    val_samples = int(n_samples * val_size)
    
    # Create indices for splitting
    indices = np.arange(n_samples)
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    test_indices = indices[:test_samples]
    val_indices = indices[test_samples:test_samples + val_samples]
    train_indices = indices[test_samples + val_samples:]
    
    # Split data
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    # Create datasets
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = StockDataset(X_train, y_train, device=device)
    val_dataset = StockDataset(X_val, y_val, device=device)
    test_dataset = StockDataset(X_test, y_test, device=device)
    
    return train_dataset, val_dataset, test_dataset

# Example usage with your DataFrame
# if __name__ == "__main__":
#     # Load your data
#     df = pd.read_csv('your_data.csv')  # Replace with your data loading
    
#     # Prepare data
#     train_dataset, val_dataset, test_dataset = prepare_data(df)
    
#     # Get input dimension for model configuration
#     input_dim = train_dataset.data.shape[2]  # Number of features
    
#     # Create configuration
#     config = ModelConfig(
#         input_dim=input_dim,
#         hidden_dim=128,
#         num_horizons=15,
#         num_classes=3,  # flat=0, up=1, down=2
#         dropout=0.2,
#         learning_rate=0.001,
#         l1_lambda=1e-5,
#         l2_lambda=1e-4
#     )
    
#     # Create dataloaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config.batch_size,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True if config.device=='cuda' else False
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config.batch_size,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True if config.device=='cuda' else False
#     )
    
#     # Initialize and train model
#     model = MultiHorizonHybridClassifier(
#         input_dim=config.input_dim,
#         hidden_dim=config.hidden_dim,
#         num_horizons=config.num_horizons
#     ).to(config.device)
    
#     # Train model
#     history = train_with_regularization(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         num_epochs=config.num_epochs,
#         learning_rate=config.learning_rate,
#         l1_lambda=config.l1_lambda,
#         l2_lambda=config.l2_lambda
#     )


    
class ModelConfig:
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_horizons: int = 15,
                 num_classes: int = 3,
                 dropout: float = 0.2,
                 num_lstm_layers: int = 2,
                 learning_rate: float = 0.001,
                 l1_lambda: float = 1e-5,
                 l2_lambda: float = 1e-4,
                 batch_size: int = 32,
                 num_epochs: int = 100,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_horizons = num_horizons
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_lstm_layers = num_lstm_layers
        self.learning_rate = learning_rate
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = {k: v for k, v in self.__dict__.items()}
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Simplified example structure
class MultiHorizonTransformer(nn.Module):
    def __init__(self, input_dim, num_horizons=15):
        super().__init__()
        # Shared layers
        self.transformer = TransformerEncoder(...)
        self.shared_layers = nn.Sequential(...)
        
        # Separate output heads for each horizon
        self.horizon_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_horizons)
        ])
    
    def forward(self, x):
        # Shared feature extraction
        shared_features = self.transformer(x)
        shared_features = self.shared_layers(shared_features)
        
        # Get predictions for each horizon
        predictions = []
        for horizon_head in self.horizon_heads:
            horizon_pred = horizon_head(shared_features)
            predictions.append(horizon_pred)
            
        return torch.cat(predictions, dim=1)  # Shape: [batch_size, 15]
    
def multi_horizon_loss(predictions, targets):
    # Combined loss across all horizons
    losses = []
    for horizon in range(15):
        horizon_loss = mse_loss(predictions[:, horizon], targets[:, horizon])
        losses.append(horizon_loss)
    
    # You could weight different horizons differently
    weights = [1.0/h for h in range(1, 16)]  # More weight to shorter horizons
    total_loss = sum(w * l for w, l in zip(weights, losses))
    return total_loss

# class MultiHorizonClassifier(nn.Module):
#     def __init__(self, input_dim, num_horizons=15):
#         super().__init__()
#         self.transformer = TransformerEncoder(...)
#         self.shared_layers = nn.Sequential(...)
        
#         # Still output probabilities for each horizon
#         self.horizon_heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(hidden_dim, 1),
#                 nn.Sigmoid()  # For binary classification
#             ) for _ in range(num_horizons)
#         ])
        
#         # Store different thresholds for each horizon
#         # These could be learnable or fixed based on your needs
#         self.thresholds = {
#             'short': 0.6,    # e.g., horizons 1-5
#             'medium': 0.65,  # e.g., horizons 6-10
#             'long': 0.7      # e.g., horizons 11-15
#         }
    
#     def forward(self, x, apply_thresholds=True):
#         shared_features = self.transformer(x)
#         shared_features = self.shared_layers(shared_features)
        
#         # Get probability predictions
#         probs = []
#         for horizon_head in self.horizon_heads:
#             horizon_prob = horizon_head(shared_features)
#             probs.append(horizon_prob)
            
#         probs = torch.cat(probs, dim=1)  # Shape: [batch_size, 15]
        
#         if not apply_thresholds:
#             return probs
        
#         # Apply different thresholds for different horizons
#         predictions = []
#         for h in range(15):
#             if h < 5:
#                 threshold = self.thresholds['short']
#             elif h < 10:
#                 threshold = self.thresholds['medium']
#             else:
#                 threshold = self.thresholds['long']
                
#             pred = (probs[:, h] > threshold).float()
#             predictions.append(pred)
            
#         return torch.stack(predictions, dim=1)


def horizon_aware_loss(predictions, targets, horizons):
    losses = []
    
    for h in range(len(horizons)):
        if h < 5:  # Short-term predictions
            # Higher weights for up/down misclassifications
            # [flat, up, down] weights
            weights = torch.tensor([1.0, 1.2, 1.2])
        elif h < 10:
            weights = torch.tensor([1.0, 1.1, 1.1])
        else:
            weights = torch.tensor([1.0, 1.0, 1.0])
            
        criterion = nn.CrossEntropyLoss(weight=weights)
        horizon_loss = criterion(predictions[:, h], targets[:, h])
        
        # Decay weight for longer horizons
        horizon_weight = 1.0 / (1 + h * 0.1)
        losses.append(horizon_loss * horizon_weight)
    
    return sum(losses)
    
# Custom loss function that considers horizon-specific thresholds
# def horizon_aware_loss(predictions, targets, horizons):
#     losses = []
    
#     # You might want different loss weights or formulations
#     # based on the horizon
#     for h in range(15):
#         if h < 5:
#             # For short horizons: maybe higher penalty for false positives
#             horizon_loss = weighted_binary_cross_entropy(
#                 predictions[:, h],
#                 targets[:, h],
#                 pos_weight=torch.tensor(1.2)
#             )
#         elif h < 10:
#             # For medium horizons: balanced weights
#             horizon_loss = binary_cross_entropy(
#                 predictions[:, h],
#                 targets[:, h]
#             )
#         else:
#             # For long horizons: maybe higher penalty for false negatives
#             horizon_loss = weighted_binary_cross_entropy(
#                 predictions[:, h],
#                 targets[:, h],
#                 pos_weight=torch.tensor(0.8)
#             )
        
#         losses.append(horizon_loss)
    
#     return sum(losses)

class MultiHorizonHybridClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_horizons=15):
        super().__init__()
        self.num_horizons = num_horizons
        
        # Shared Feature Extractors
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        self.tcn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Horizon-specific attention
        self.horizon_attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=4)
            for _ in range(num_horizons)
        ])
        
        combined_dim = hidden_dim + 128  # LSTM + TCN features
        
        # Separate classifiers for each horizon
        self.horizon_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(combined_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 3),  # 3 classes: flat(0), up(1), down(2)
                nn.Softmax(dim=-1)
            ) for _ in range(num_horizons)
        ])
        
        # Different thresholds for different horizons
        # Adjusted for your class encoding
        self.horizon_thresholds = {
            'short': {'up': 0.3, 'down': 0.3},     # 1-5 days
            'medium': {'up': 0.5, 'down': 0.5},  # 6-10 days
            'long': {'up': 0.5, 'down': 0.5}       # 11-15 days
        }
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. LSTM pathway
        lstm_out, _ = self.lstm(x)
        
        # 2. TCN pathway
        x_tcn = x.transpose(1, 2)
        tcn_features = self.tcn(x_tcn)
        tcn_features = torch.mean(tcn_features, dim=2)  # ok this dim count is not related to the total feature count
        
        all_predictions = []
        
        for h in range(self.num_horizons):
            # Horizon-specific attention
            attn_out, _ = self.horizon_attention[h](
                lstm_out, lstm_out, lstm_out
            )
            
            lstm_features = attn_out[:, -1, :]
            combined = torch.cat([lstm_features, tcn_features], dim=1)
            horizon_preds = self.horizon_classifiers[h](combined)
            all_predictions.append(horizon_preds)
        
        return torch.stack(all_predictions, dim=1)  # [batch, horizons, 3]
    
    def predict_with_thresholds(self, x):
        probs = self.forward(x)
        predictions = []
        
        for h in range(self.num_horizons):
            if h < 5:
                thresholds = self.horizon_thresholds['short']
            elif h < 10:
                thresholds = self.horizon_thresholds['medium']
            else:
                thresholds = self.horizon_thresholds['long']
            
            # Apply thresholds - adjusted for your class encoding
            horizon_pred = torch.zeros_like(probs[:, h, 0])  # Default is flat (0)
            
            # Check up (1) and down (2) probabilities
            up_mask = probs[:, h, 1] > thresholds['up']
            down_mask = probs[:, h, 2] > thresholds['down']
            
            # Assign classes (flat=0, up=1, down=2)
            horizon_pred[up_mask] = 1    # Up class
            horizon_pred[down_mask] = 2  # Down class
            # Remains 0 (flat) when neither threshold is met
            
            predictions.append(horizon_pred)
            
        return torch.stack(predictions, dim=1)
    
    def get_l1_loss(self):
        """Calculate L1 regularization loss"""
        l1_loss = 0
        for name, param in self.named_parameters():
            # Typically apply L1 only to weights, not biases
            if 'weight' in name:
                l1_loss += torch.abs(param).sum()
        return l1_loss
    
    def get_l2_loss(self):
        """Calculate L2 regularization loss"""
        l2_loss = 0
        for name, param in self.named_parameters():
            # Typically apply L2 only to weights, not biases
            if 'weight' in name:
                l2_loss += torch.square(param).sum()
        return l2_loss

    def horizon_aware_loss(predictions, targets, horizons):
        losses = []
        
        for h in range(len(horizons)):
            if h < 5:  # Short-term predictions
                # Higher weights for up/down misclassifications
                # [flat, up, down] weights
                weights = torch.tensor([1.0, 1.2, 1.2])
            elif h < 10:
                weights = torch.tensor([1.0, 1.1, 1.1])
            else:
                weights = torch.tensor([1.0, 1.0, 1.0])
                
            criterion = nn.CrossEntropyLoss(weight=weights)
            horizon_loss = criterion(predictions[:, h], targets[:, h])
            
            # Decay weight for longer horizons
            horizon_weight = 1.0 / (1 + h * 0.1)
            losses.append(horizon_loss * horizon_weight)
        
        return sum(losses)

# Evaluation metrics
def compute_metrics(model, val_loader, horizons):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            predictions = model.predict_with_thresholds(batch_x)
            all_preds.append(predictions)
            all_targets.append(batch_y)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = {}
    for h in horizons:
        horizon_preds = all_preds[:, h]
        horizon_targets = all_targets[:, h]
        
        # Per-class metrics
        for class_idx, class_name in enumerate(['flat', 'up', 'down']):
            class_mask = horizon_targets == class_idx
            if class_mask.sum() > 0:
                accuracy = (horizon_preds[class_mask] == horizon_targets[class_mask]).float().mean()
                metrics[f'horizon_{h}_{class_name}_accuracy'] = accuracy.item()
    
    return metrics

def train_with_regularization(
    model, 
    train_loader, 
    val_loader, 
    num_epochs,
    learning_rate=0.001,
    l1_lambda=1e-5,
    l2_lambda=1e-4,
    validation_interval=1,
    grad_clip=1.0  # Add gradient clipping
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [],
        'val_loss': [],
        'metrics': [],
        'l1_loss': [],
        'l2_loss': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_l1_losses = []
        epoch_l2_losses = []
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Check for NaN in input
            if torch.isnan(batch_x).any() or torch.isnan(batch_y).any():
                print("NaN found in input data!")
                print("X stats:", torch.min(batch_x), torch.max(batch_x))
                print("Y stats:", torch.min(batch_y), torch.max(batch_y))
                continue
                
            # Forward pass
            predictions = model(batch_x)
            
            # Check for NaN in predictions
            if torch.isnan(predictions).any():
                print("NaN found in predictions!")
                continue
                
            # Calculate losses
            main_loss = horizon_aware_loss(predictions, batch_y, range(15))
            l1_loss = model.get_l1_loss()
            l2_loss = model.get_l2_loss()
            
            # Check if losses are NaN
            if torch.isnan(main_loss) or torch.isnan(l1_loss) or torch.isnan(l2_loss):
                print("NaN found in losses!")
                print(f"Main loss: {main_loss}")
                print(f"L1 loss: {l1_loss}")
                print(f"L2 loss: {l2_loss}")
                continue
            
            # Combine losses
            total_loss = main_loss + l1_lambda * l1_loss + l2_lambda * l2_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
            # Check for NaN gradients
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN gradient in {name}")
                    continue
            
            optimizer.step()
            
            # Store non-NaN losses
            if not torch.isnan(main_loss):
                epoch_losses.append(main_loss.item())
                epoch_l1_losses.append(l1_loss.item())
                epoch_l2_losses.append(l2_loss.item())
        
        # Calculate average losses for the epoch
        avg_loss = np.mean(epoch_losses)
        avg_l1_loss = np.mean(epoch_l1_losses)
        avg_l2_loss = np.mean(epoch_l2_losses)

        # Store in history
        history['train_loss'].append(avg_loss)
        history['l1_loss'].append(avg_l1_loss)
        history['l2_loss'].append(avg_l2_loss)
        
        # Print statistics for debugging
        if len(epoch_losses) > 0:
            avg_loss = np.mean(epoch_losses)
            avg_l1_loss = np.mean(epoch_l1_losses)
            avg_l2_loss = np.mean(epoch_l2_losses)

            # Store in history
            history['train_loss'].append(avg_loss)
            history['l1_loss'].append(avg_l1_loss)
            history['l2_loss'].append(avg_l2_loss)
                
            # these gets printed later on in validation phase
            # print(f"Epoch {epoch+1}/{num_epochs}")
            # print(f"Training Loss: {avg_loss:.4f}")
            # print(f"L1 Loss: {avg_l1_loss:.4f}")
            # print(f"L2 Loss: {avg_l2_loss:.4f}")
            
            # Print model parameter statistics
            # !!! comment out for now
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN found in parameter {name}")
                # print(f"{name}: min={param.min().item():.4f}, max={param.max().item():.4f}")

        # Validation phase
        if epoch % validation_interval == 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    val_predictions = model(batch_x)
                    val_loss = horizon_aware_loss(val_predictions, batch_y, range(15))
                    val_losses.append(val_loss.item())
            
            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Training Loss: {avg_loss:.4f}")
            print(f"L1 Loss: {avg_l1_loss:.4f}")
            print(f"L2 Loss: {avg_l2_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print("-" * 50)

             # Make sure to return the history
    return history

# Function to plot training history
def plot_training_history(history):
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot regularization losses
    plt.subplot(1, 2, 2)
    plt.plot(history['l1_loss'], label='L1 Loss')
    plt.plot(history['l2_loss'], label='L2 Loss')
    plt.title('Regularization Losses Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# # Usage example
# model = MultiHorizonHybridClassifier(input_dim=your_input_dim)

# Grid search for best regularization parameters (optional)
def find_best_regularization(
    model, 
    train_loader, 
    val_loader,
    l1_values=[1e-6, 1e-5, 1e-4],
    l2_values=[1e-5, 1e-4, 1e-3]
):
    best_val_loss = float('inf')
    best_params = None
    results = []
    
    for l1 in l1_values:
        for l2 in l2_values:
            print(f"Testing L1={l1}, L2={l2}")
            
            # Reset model weights
            model.apply(lambda m: m.reset_parameters() 
                       if hasattr(m, 'reset_parameters') else None)
            
            # Train with current regularization settings
            history = train_with_regularization(
                model,
                train_loader,
                val_loader,
                num_epochs=10,  # Quick training to find best params
                l1_lambda=l1,
                l2_lambda=l2
            )
            
            final_val_loss = min(history['val_loss'])
            results.append({
                'l1': l1,
                'l2': l2,
                'val_loss': final_val_loss
            })
            
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_params = {'l1': l1, 'l2': l2}
    
    return best_params, results

# # Find best regularization parameters
# best_params, reg_results = find_best_regularization(
#     model, train_loader, val_loader
# )

# # Train with best parameters
# history = train_with_regularization(
#     model,
#     train_loader,
#     val_loader,
#     num_epochs=100,
#     l1_lambda=best_params['l1'],
#     l2_lambda=best_params['l2']
# )

# # Plot results
# plot_training_history(history)

### another start
# Example usage with your DataFrame
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('NVDA_TMP.csv')  # Replace with your data loading
    
    # Prepare data
    train_dataset, val_dataset, test_dataset = prepare_data(df)
    
    # Get input dimension for model configuration
    input_dim = train_dataset.data.shape[2]  # Number of features
    
    # Create configuration
    config = ModelConfig(
        input_dim=input_dim,
        hidden_dim=128,
        num_horizons=15,
        num_classes=3,  # flat=0, up=1, down=2
        dropout=0.2,
        learning_rate=0.001,
        l1_lambda=1e-5,
        l2_lambda=1e-4
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if config.device=='cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if config.device=='cuda' else False
    )
    
    # Initialize and train model
    model = MultiHorizonHybridClassifier(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_horizons=config.num_horizons
    ).to(config.device)
    
    # Train model
    history = train_with_regularization(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        l1_lambda=config.l1_lambda,
        l2_lambda=config.l2_lambda
    )

# ### NOW START THE WHOLE THING
# # Set random seed for reproducibility
# set_seed(42)

# # Initialize configuration
# config = ModelConfig(
#     input_dim=your_input_dim,  # Replace with your actual input dimension
#     hidden_dim=128,
#     num_horizons=15,
#     num_classes=3,  # flat=0, up=1, down=2
#     dropout=0.2,
#     learning_rate=0.001,
#     l1_lambda=1e-5,
#     l2_lambda=1e-4
# )

# # Create datasets
# train_dataset = StockDataset(train_data, train_targets, device=config.device)
# val_dataset = StockDataset(val_data, val_targets, device=config.device)

# # Create dataloaders
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=config.batch_size,
#     shuffle=True,
#     num_workers=4,
#     pin_memory=True if config.device=='cuda' else False
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=config.batch_size,
#     shuffle=False,
#     num_workers=4,
#     pin_memory=True if config.device=='cuda' else False
# )

# # Initialize model
# model = MultiHorizonHybridClassifier(
#     input_dim=config.input_dim,
#     hidden_dim=config.hidden_dim,
#     num_horizons=config.num_horizons
# ).to(config.device)

# # Train model
# history = train_with_regularization(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     num_epochs=config.num_epochs,
#     learning_rate=config.learning_rate,
#     l1_lambda=config.l1_lambda,
#     l2_lambda=config.l2_lambda
# )

    if history is not None:
        plot_training_history(history)
    else:
        print("Training failed to return history")

    # Save model and config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/model_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), f"{save_dir}/model.pt")
    config.save(f"{save_dir}/config.json")



# Training loop example
# model = MultiHorizonHybridClassifier(input_dim=your_input_dim)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# for epoch in range(num_epochs):
#     model.train()
#     for batch_x, batch_y in train_loader:
#         optimizer.zero_grad()
#         predictions = model(batch_x)
#         loss = horizon_aware_loss(predictions, batch_y, range(15))
#         loss.backward()
#         optimizer.step()
    
#     # Validation
#     if epoch % validation_interval == 0:
#         metrics = compute_metrics(model, val_loader, range(15))
#         print(f"Epoch {epoch} metrics:", metrics)

# # Custom loss function for multi-horizon classification
# # def horizon_aware_loss(predictions, targets, horizons):
# #     losses = []
    
# #     # CrossEntropy for each horizon with different weights
# #     for h in range(len(horizons)):
# #         if h < 5:  # Short-term predictions
# #             # Maybe higher weights for misclassifying big moves
# #             weights = torch.tensor([1.2, 1.0, 1.2])
# #         elif h < 10:  # Medium-term
# #             weights = torch.tensor([1.1, 1.0, 1.1])
# #         else:  # Long-term
# #             weights = torch.tensor([1.0, 1.0, 1.0])
            
# #         criterion = nn.CrossEntropyLoss(weight=weights)
# #         horizon_loss = criterion(predictions[:, h], targets[:, h])
        
# #         # Optionally decay weight for longer horizons
# #         horizon_weight = 1.0 / (1 + h * 0.1)  # Decay factor
# #         losses.append(horizon_loss * horizon_weight)
    
# #     return sum(losses)

# # # Usage example
# # model = MultiHorizonHybridClassifier(input_dim=your_input_dim)
# # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # # Training loop
# # for epoch in range(num_epochs):
# #     for batch_x, batch_y in dataloader:
# #         optimizer.zero_grad()
# #         predictions = model(batch_x)
# #         loss = horizon_aware_loss(predictions, batch_y, range(15))
# #         loss.backward()
# #         optimizer.step()


# # # Usage
# # model = MultiHorizonClassifier(input_dim=your_dim)

# # # During training
# # probs = model(x, apply_thresholds=False)
# # loss = horizon_aware_loss(probs, targets, horizons)

# # # During inference
# # predictions = model(x, apply_thresholds=True)

def make_predictions(model, latest_data, data_processor, device='cuda'):
    """
    Make predictions for all 15 horizons using the latest data
    
    Args:
        model: Trained MultiHorizonHybridClassifier
        latest_data: DataFrame with the most recent sequence_length days of data
        data_processor: The same DataProcessor used for training
        device: 'cuda' or 'cpu'
    """
    model.eval()  # Set model to evaluation mode
    
    # Preprocess the latest data using the same processor
    feature_columns = [
        'close',
        'volume',
        'rsi',
        # Add all your features here (should match training features)
    ]
    
    # Scale the features using the same scaler
    scaled_data = data_processor.scaler.transform(latest_data[feature_columns])
    
    # Convert to tensor and add batch dimension
    x = torch.FloatTensor(scaled_data).unsqueeze(0).to(device)  # Shape: [1, sequence_length, n_features]
    
    with torch.no_grad():
        # Get predictions for all horizons
        predictions = model(x)
        # Apply thresholds to get class predictions
        class_predictions = model.predict_with_thresholds(x)
    
    # Convert predictions to numpy
    class_predictions = class_predictions.cpu().numpy()
    probabilities = predictions.cpu().numpy()
    
    # Create results DataFrame
    results = pd.DataFrame()
    results['horizon'] = range(1, 16)  # 1 to 15 days
    results['prediction'] = class_predictions[0]  # First batch
    
    # Add probabilities for each class
    results['prob_flat'] = probabilities[0, :, 0]
    results['prob_up'] = probabilities[0, :, 1]
    results['prob_down'] = probabilities[0, :, 2]
    
    # Add prediction labels
    results['prediction_label'] = results['prediction'].map({
        0: 'flat',
        1: 'up',
        2: 'down'
    })
    
    return results

# Usage example
def get_latest_predictions(model_path, config_path, latest_data):
    """
    Load model and make predictions
    """
    # Load model configuration
    config = ModelConfig.load(config_path)
    
    # Initialize model
    model = MultiHorizonHybridClassifier(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_horizons=config.num_horizons
    ).to(config.device)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path))
    
    # Initialize data processor
    data_processor = DataProcessor(
        sequence_length=20,  # Should match training
        n_horizons=15
    )
    
    # Make predictions
    predictions = make_predictions(
        model=model,
        latest_data=latest_data,
        data_processor=data_processor,
        device=config.device
    )
    
    return predictions

# Example of using the prediction functions
def update_predictions():
    """
    Update predictions using the latest data
    """
    # Load your latest data (adjust as needed)
    latest_data = pd.read_csv('latest_data.csv')  # Or however you get your data
    
    # Ensure you have enough historical data for the sequence
    if len(latest_data) < 20:  # sequence_length
        raise ValueError("Not enough historical data for prediction")
    
    # Get predictions
    predictions = get_latest_predictions(
        model_path='models/your_model.pt',
        config_path='models/your_config.json',
        latest_data=latest_data
    )
    
    # Print or store predictions
    print("\nPredictions for next 15 days:")
    print(predictions[['horizon', 'prediction_label', 'prob_flat', 'prob_up', 'prob_down']])
    
    # Optionally, save predictions
    predictions.to_csv('latest_predictions.csv', index=False)
    
    return predictions

# Example visualization of predictions
def plot_predictions(predictions):
    """
    Visualize predictions and their probabilities
    """
    plt.figure(figsize=(15, 6))
    
    # Plot prediction probabilities
    plt.subplot(1, 2, 1)
    plt.plot(predictions['horizon'], predictions['prob_flat'], 'b-', label='Flat')
    plt.plot(predictions['horizon'], predictions['prob_up'], 'g-', label='Up')
    plt.plot(predictions['horizon'], predictions['prob_down'], 'r-', label='Down')
    plt.xlabel('Days Ahead')
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities by Horizon')
    plt.legend()
    plt.grid(True)
    
    # Plot predicted classes
    plt.subplot(1, 2, 2)
    colors = predictions['prediction'].map({0: 'blue', 1: 'green', 2: 'red'})
    plt.scatter(predictions['horizon'], [1]*len(predictions), c=colors, s=100)
    plt.yticks([])
    plt.xlabel('Days Ahead')
    plt.title('Predicted Direction by Horizon')
    
    plt.tight_layout()
    plt.show()

# # Use it like this:
# if __name__ == "__main__":
#     # Get latest predictions
#     predictions = update_predictions()
    
#     # Visualize predictions
#     plot_predictions(predictions)