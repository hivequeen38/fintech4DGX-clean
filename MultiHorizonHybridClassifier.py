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

def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
            'short': {'up': 0.4, 'down': 0.4},     # 1-5 days
            'medium': {'up': 0.45, 'down': 0.45},  # 6-10 days
            'long': {'up': 0.5, 'down': 0.5}       # 11-15 days
        }
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. LSTM pathway
        lstm_out, _ = self.lstm(x)
        
        # 2. TCN pathway
        x_tcn = x.transpose(1, 2)
        tcn_features = self.tcn(x_tcn)
        tcn_features = torch.mean(tcn_features, dim=2)
        
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

# Modified training loop with regularization
def train_with_regularization(
    model, 
    train_loader, 
    val_loader, 
    num_epochs,
    learning_rate=0.001,
    l1_lambda=1e-5,    # L1 regularization strength
    l2_lambda=1e-4,    # L2 regularization strength
    validation_interval=1
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Optional: Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
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
            
            # Forward pass
            predictions = model(batch_x)
            
            # Calculate main loss
            main_loss = horizon_aware_loss(predictions, batch_y, range(15))
            
            # Calculate regularization losses
            l1_loss = model.get_l1_loss()
            l2_loss = model.get_l2_loss()
            
            # Combine all losses
            total_loss = main_loss + l1_lambda * l1_loss + l2_lambda * l2_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping (optional but recommended)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Store losses for monitoring
            epoch_losses.append(main_loss.item())
            epoch_l1_losses.append(l1_loss.item())
            epoch_l2_losses.append(l2_loss.item())
        
        # Calculate average losses for the epoch
        avg_loss = np.mean(epoch_losses)
        avg_l1_loss = np.mean(epoch_l1_losses)
        avg_l2_loss = np.mean(epoch_l2_losses)
        
        history['train_loss'].append(avg_loss)
        history['l1_loss'].append(avg_l1_loss)
        history['l2_loss'].append(avg_l2_loss)
        
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
            
            # Calculate metrics
            metrics = compute_metrics(model, val_loader, range(15))
            history['metrics'].append(metrics)
            
            # Update learning rate based on validation loss
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Training Loss: {avg_loss:.4f}")
            print(f"L1 Loss: {avg_l1_loss:.4f}")
            print(f"L2 Loss: {avg_l2_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print("Metrics:", metrics)
            print("-" * 50)
    
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

# Usage example
model = MultiHorizonHybridClassifier(input_dim=your_input_dim)

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

# Find best regularization parameters
best_params, reg_results = find_best_regularization(
    model, train_loader, val_loader
)

# Train with best parameters
history = train_with_regularization(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    l1_lambda=best_params['l1'],
    l2_lambda=best_params['l2']
)

# Plot results
plot_training_history(history)

# # Training loop example
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

# Advanced usage example
# Set random seed for reproducibility
set_seed(42)

# Initialize configuration
config = ModelConfig(
    input_dim=your_input_dim,  # Replace with your actual input dimension
    hidden_dim=128,
    num_horizons=15,
    num_classes=3,  # flat=0, up=1, down=2
    dropout=0.2,
    learning_rate=0.001,
    l1_lambda=1e-5,
    l2_lambda=1e-4
)

# Create datasets
train_dataset = StockDataset(train_data, train_targets, device=config.device)
val_dataset = StockDataset(val_data, val_targets, device=config.device)

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

# Initialize model
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

# Plot results
plot_training_history(history)

# Save model and config
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"models/model_{timestamp}"
os.makedirs(save_dir, exist_ok=True)

torch.save(model.state_dict(), f"{save_dir}/model.pt")
config.save(f"{save_dir}/config.json")
