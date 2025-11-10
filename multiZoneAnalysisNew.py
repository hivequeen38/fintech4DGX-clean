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


# class MultiZoneTransformer(nn.Module):
#     def __init__(self, input_dim, num_classes, num_heads, num_layers, dropout_rate, embedded_dim, num_zones=15):
#         super(MultiZoneTransformer, self).__init__()
#         self.num_zones = num_zones
#         self.embedding_dim = embedded_dim
        
#         # Input projection layer
#         self.input_projection = nn.Linear(input_dim, self.embedding_dim)
        
#         # Positional encoding with dropout
#         self.positional_encoding = PositionalEncoding(self.embedding_dim, dropout_rate)
        
#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.embedding_dim, 
#             nhead=num_heads, 
#             dropout=dropout_rate, 
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         # Dropout layer
#         self.dropout = nn.Dropout(dropout_rate)
        
#         # Create different embeddings for each zone
#         self.zone_embeddings = nn.Parameter(torch.randn(num_zones, self.embedding_dim))
        
#         # Output layers for different zones
#         self.zone_predictors = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(self.embedding_dim * 2, self.embedding_dim),
#                 nn.ReLU(),
#                 nn.Linear(self.embedding_dim, num_classes)
#             ) for _ in range(num_zones)
#         ])
    
#     def forward(self, src):
#         batch_size = src.size(0)
        
#         # Handle input shape
#         if src.dim() == 2:
#             src = src.unsqueeze(1)
        
#         # Project input
#         x = self.input_projection(src)  # [batch_size, seq_len, embedding_dim]
        
#         # Add positional encoding
#         x = self.positional_encoding(x)
        
#         # Apply transformer encoder
#         x = self.transformer_encoder(x)  # [batch_size, seq_len, embedding_dim]
        
#         # Get sequence representation (use mean pooling instead of squeeze)
#         x = x.mean(dim=1)  # [batch_size, embedding_dim]
        
#         # Generate predictions for each zone
#         zone_outputs = []
#         for i in range(self.num_zones):
#             # Combine sequence representation with zone-specific embedding
#             zone_embedding = self.zone_embeddings[i].expand(batch_size, -1)
#             combined = torch.cat([x, zone_embedding], dim=1)
            
#             # Apply zone-specific prediction
#             zone_output = self.zone_predictors[i](combined)
#             zone_outputs.append(zone_output)
        
#         return torch.stack(zone_outputs, dim=1)  # [batch_size, num_zones, num_classes]

class MultiZoneTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads, num_layers, dropout_rate, embedded_dim, num_zones=15):
        super(MultiZoneTransformer, self).__init__()
        self.num_zones = num_zones
        self.embedding_dim = embedded_dim
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, self.embedding_dim)
        
        # Positional encoding with dropout
        self.positional_encoding = PositionalEncoding(self.embedding_dim, dropout_rate)
        
        # Zone-specific transformer encoders
        self.zone_encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.embedding_dim,
                    nhead=num_heads,
                    dropout=dropout_rate,
                    batch_first=True
                ),
                num_layers=num_layers
            ) for _ in range(num_zones)
        ])
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Zone-specific embeddings
        self.zone_embeddings = nn.Parameter(torch.randn(num_zones, self.embedding_dim))
        
        # Zone predictors with independent processing
        self.zone_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.embedding_dim, self.embedding_dim // 2),
                nn.LayerNorm(self.embedding_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.embedding_dim // 2, num_classes)
            ) for _ in range(num_zones)
        ])
    
    def forward(self, src):
        batch_size = src.size(0)
        
        # Handle input shape
        if src.dim() == 2:
            src = src.unsqueeze(1)
        
        # Project input
        x = self.input_projection(src)  # [batch_size, seq_len, embedding_dim]
        x = self.positional_encoding(x)
        
        # Generate predictions for each zone using zone-specific processing
        zone_outputs = []
        for i in range(self.num_zones):
            # Apply zone-specific transformer
            zone_x = self.zone_encoders[i](x)
            
            # Global pooling with attention
            attention_weights = torch.softmax(
                torch.matmul(zone_x, self.zone_embeddings[i].unsqueeze(-1)).squeeze(-1),
                dim=-1
            )
            zone_context = torch.sum(zone_x * attention_weights.unsqueeze(-1), dim=1)
            
            # Combine with zone embedding
            zone_embedding = self.zone_embeddings[i].expand(batch_size, -1)
            combined = torch.cat([zone_context, zone_embedding], dim=1)
            
            # Apply zone-specific prediction
            zone_output = self.zone_predictors[i](combined)
            zone_outputs.append(zone_output)
        
        return torch.stack(zone_outputs, dim=1)  # [batch_size, num_zones, num_classes]

# Additional helper functions for gradient scaling
def scale_gradient(tensor, scale):
    """Scale gradients for multi-task learning"""
    return tensor * scale + tensor.detach() * (1 - scale)

# def calculate_multi_zone_label(df: DataFrame, param: dict[str]):
#     threshold = param['threshold']
#     num_zones = param['num_zones']  # Should be 15
    
#     df['adjusted close'] = df['adjusted close'].astype(float)
#     df.loc[:, 'label'] = [[] for _ in range(len(df))]
    
#     for zone in range(1, num_zones + 1):
#         price_change = (df['adjusted close'].shift(-zone) - df['adjusted close']) / df['adjusted close']
        
#         zone_labels = np.zeros(len(df))
#         zone_labels = np.where(price_change >= threshold, 1, zone_labels)   # 1 for increase
#         zone_labels = np.where(price_change <= -threshold, 2, zone_labels)  # 2 for decrease
        
#         # Store labels for each zone
#         for idx in range(len(df)):
#             if not pd.isna(price_change[idx]):
#                 if isinstance(df.at[idx, 'label'], list):
#                     df.at[idx, 'label'].append(int(zone_labels[idx]))
#                 else:
#                     df.at[idx, 'label'] = [int(zone_labels[idx])]
    
#     # Drop rows with incomplete labels
#     df = df.dropna(subset=['adjusted close'])
#     df = df[df['label'].map(len) == num_zones]
    
#     return df

def calculate_multi_zone_label(df: DataFrame, param: dict[str]):
    threshold = param['threshold']
    num_zones = param['num_zones']
    
    df['adjusted close'] = df['adjusted close'].astype(float)
    labels = []
    
    for idx in range(len(df) - num_zones):
        row_labels = []
        current_price = df['adjusted close'].iloc[idx]
        
        for zone in range(1, num_zones + 1):
            if idx + zone >= len(df):
                break
            future_price = df['adjusted close'].iloc[idx + zone]
            price_change = (future_price - current_price) / current_price
            
            if price_change >= threshold:
                label = 1  # UP
            elif price_change <= -threshold:
                label = 2  # DOWN
            else:
                label = 0  # NO CHANGE
                
            row_labels.append(label)
            
        if len(row_labels) == num_zones:
            labels.append(row_labels)
        else:
            labels.append([0] * num_zones)  # Padding for incomplete data
            
    df = df.iloc[:len(labels)].copy()
    df['label'] = labels
    
    return df

# def make_multi_zone_prediction(model, raw_df: DataFrame, param: dict[str], currentDateTime, symbol):
#     selected_columns = param['selected_columns']
#     new_df = raw_df[selected_columns].copy()
#     last_df = new_df.tail(param['batch_size'] + param['target_size']).copy()
#     last_df['label'] = [[0] * param['num_zones']] * len(last_df)
    
#     features_to_scale = last_df.drop(['label'], axis=1)
#     scaler = load('scaler.joblib')
#     features_array = scaler.transform(features_to_scale)
    
#     input_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)
    
#     model.eval()
#     with torch.no_grad():
#         predictions = model(input_tensor)  # [batch, zones, classes]
#         probabilities = torch.softmax(predictions, dim=2)
#         predicted_classes = torch.argmax(probabilities, dim=2)  # [batch, zones]
        
#         # Take the last prediction for each zone
#         predicted_classes = predicted_classes[-1]  # [zones]
        
#     class_labels = {0: "__", 1: "UP", 2: "DN"}
#     zone_predictions = []
    
#     # Convert predictions to labels
#     for zone in range(param['num_zones']):
#         pred_class = predicted_classes[zone].cpu().numpy().item()  # Convert to scalar
#         zone_predictions.append(class_labels[pred_class])
    
#     print(f"Predictions for next {param['num_zones']} days:", zone_predictions)
#     print("Raw predictions:", predicted_classes.cpu().numpy())  # Debug print
    
#     date_str = currentDateTime.strftime('%Y-%m-%d')
#     close = raw_df['adjusted close'].iloc[-1]
    
#     processPrediction.process_prediction_results(symbol, date_str, close, zone_predictions, param['num_zones'])

def OLD_make_multi_zone_prediction(model, raw_df: DataFrame, param: dict[str], currentDateTime, symbol, comment: str):
    selected_columns = param['selected_columns']
    new_df = raw_df[selected_columns].copy()
    last_df = new_df.tail(param['batch_size'] + param['target_size']).copy()
    last_df['label'] = [[0] * param['num_zones']] * len(last_df)
    
    features_to_scale = last_df.drop(['label'], axis=1)
    scaler = load('scaler.joblib')
    features_array = scaler.transform(features_to_scale)
    
    input_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)
    
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        # Get model predictions
        predictions = model(input_tensor)  # Shape: [1, 15, 47, 3]
        print("\nDebug - raw predictions shape:", predictions.shape)
        
        # Take first batch and mean over the sequence dimension (47)
        batch_predictions = predictions[0]  # Shape: [15, 47, 3]
        batch_predictions = torch.mean(batch_predictions, dim=1)  # Shape: [15, 3]
        
        # Apply softmax and get final predictions
        probabilities = torch.softmax(batch_predictions, dim=1)  # Shape: [15, 3]
        final_predictions = torch.argmax(probabilities, dim=1)  # Shape: [15]
        
        # Convert to numpy and handle predictions
        pred_array = final_predictions.cpu().numpy()
        print("Debug - final predictions shape:", pred_array.shape)
        print("Debug - prediction values:", pred_array)
        
        # Convert predictions to labels
        class_labels = {0: "__", 1: "UP", 2: "DN"}
        zone_predictions = [class_labels[int(p)] for p in pred_array]
    
    print(f"\nPredictions for next {param['num_zones']} days:")
    for i, pred in enumerate(zone_predictions):
        print(f"Day {i+1}: {pred}")
        
    # Print probabilities for each day
    prob_array = probabilities.cpu().numpy()
    print("\nProbabilities for each prediction:")
    for i in range(len(zone_predictions)):
        probs = prob_array[i]
        print(f"Day {i+1}: No Change: {probs[0]:.3f}, Up: {probs[1]:.3f}, Down: {probs[2]:.3f}")
    
    date_str = currentDateTime.strftime('%Y-%m-%d')
    close = raw_df['adjusted close'].iloc[-1]
    
    raw_predictions = zone_predictions  # need to save this away because next call will alter it by adding price
    processPrediction.process_prediction_results(symbol, date_str, close, zone_predictions, param['num_zones'])

    # now add to main db
    dateStr, closing_price = analysisUtil.fetchDateAndClosing(param)

    # Create DataFrame using dict comprehension
    # data = {f'p{i+1}': [pred] for i, pred in enumerate(raw_predictions)}
    # incr_df = pd.DataFrame(data)

    incr_df = pd.DataFrame([zone_predictions], columns=['close', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 
                                                   'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15'])
    analysisUtil.processDeltaFromTodayResults(param["symbol"], incr_df, dateStr, closing_price, comment)

def make_multi_zone_prediction(model, raw_df: DataFrame, param: dict[str], currentDateTime, symbol, comment: str):
    selected_columns = param['selected_columns']
    new_df = raw_df[selected_columns].copy()
    last_df = new_df.tail(param['batch_size'] + param['target_size']).copy()
    last_df['label'] = [[0] * param['num_zones']] * len(last_df)
    
    features_to_scale = last_df.drop(['label'], axis=1)
    scaler = load('scaler.joblib')
    features_array = scaler.transform(features_to_scale)
    
    input_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)
    
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        # Get model predictions
        predictions = model(input_tensor)  # Shape: [batch_size, num_zones, num_classes]
        print("\nDebug - raw predictions shape:", predictions.shape)
        
        # Since we have only one batch, squeeze it out
        predictions = predictions.squeeze(0)  # Shape: [num_zones, num_classes]
        
        # Apply softmax on the class dimension
        probabilities = torch.softmax(predictions, dim=-1)  # Shape: [num_zones, num_classes]
        final_predictions = torch.argmax(probabilities, dim=-1)  # Shape: [num_zones]
        
        # Convert to numpy and handle predictions
        pred_array = final_predictions.cpu().numpy()
        print("Debug - final predictions shape:", pred_array.shape)
        print("Debug - prediction values:", pred_array)
        
        # Convert predictions to labels
        class_labels = {0: "__", 1: "UP", 2: "DN"}
        zone_predictions = [class_labels[int(p)] for p in pred_array]
    
    print(f"\nPredictions for next {param['num_zones']} days:")
    for i, pred in enumerate(zone_predictions):
        print(f"Day {i+1}: {pred}")
        
    # Print probabilities for each day
    prob_array = probabilities.cpu().numpy()
    print("\nProbabilities for each prediction:")
    for i in range(len(zone_predictions)):
        probs = prob_array[i]
        print(f"Day {i+1}: No Change: {probs[0]:.3f}, Up: {probs[1]:.3f}, Down: {probs[2]:.3f}")
    
    date_str = currentDateTime.strftime('%Y-%m-%d')
    close = raw_df['adjusted close'].iloc[-1]
    
    raw_predictions = zone_predictions.copy()  # Store original predictions
    processPrediction.process_prediction_results(symbol, date_str, close, zone_predictions, param['num_zones'])

    # Add to main db
    dateStr, closing_price = analysisUtil.fetchDateAndClosing(param)
    
    # Create DataFrame with predictions
    incr_df = pd.DataFrame([raw_predictions], columns=['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 
                                                     'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15'])
    incr_df.insert(0, 'close', closing_price)  # Add closing price as first column
    
    analysisUtil.processDeltaFromTodayResults(param["symbol"], incr_df, dateStr, closing_price, comment)

def validate_multi_zone(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    zone_correct = torch.zeros(model.num_zones).to(device)
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            outputs = model(inputs)
            
            # Calculate metrics for each zone
            for zone in range(outputs.shape[1]):
                zone_output = outputs[:, zone, :]
                zone_labels = labels[:, zone]
                
                loss = criterion(zone_output, zone_labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(zone_output, 1)
                zone_correct[zone] += (predicted == zone_labels).sum().item()
            
            total_samples += labels.size(0)
    
    avg_loss = total_loss / (len(val_loader) * model.num_zones)
    zone_accuracies = zone_correct / total_samples
    
    return avg_loss, zone_accuracies

# def evaluate_multi_zone_model(model, data_loader, criterion, device):
#     model.eval()
#     total_loss = 0
#     predictions = [[] for _ in range(model.num_zones)]
#     true_labels = [[] for _ in range(model.num_zones)]
    
#     with torch.no_grad():
#         for inputs, labels in data_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
            
#             outputs = model(inputs)
            
#             for zone in range(outputs.shape[1]):
#                 zone_output = outputs[:, zone, :]
#                 zone_labels = labels[:, zone]
                
#                 loss = criterion(zone_output, zone_labels)
#                 total_loss += loss.item()
                
#                 _, predicted = torch.max(zone_output, 1)
#                 predictions[zone].extend(predicted.cpu().numpy())
#                 true_labels[zone].extend(zone_labels.cpu().numpy())
    
#     # Calculate metrics for each zone
#     metrics = []
#     for zone in range(model.num_zones):
#         zone_metrics = {
#             'precision': precision_score(true_labels[zone], predictions[zone], average='macro'),
#             'recall': recall_score(true_labels[zone], predictions[zone], average='macro'),
#             'f1': f1_score(true_labels[zone], predictions[zone], average='macro'),
#             'confusion_matrix': confusion_matrix(true_labels[zone], predictions[zone])
#         }
#         metrics.append(zone_metrics)
    
#     return metrics

def evaluate_multi_zone_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []  # For consolidated metrics
    all_labels = []      # For consolidated metrics
    per_zone_predictions = [[] for _ in range(model.num_zones)]
    per_zone_labels = [[] for _ in range(model.num_zones)]
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            for zone in range(outputs.shape[1]):
                zone_output = outputs[:, zone, :]
                zone_labels = labels[:, zone]
                
                loss = criterion(zone_output, zone_labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(zone_output, 1)
                
                # Store predictions and labels for both per-zone and consolidated metrics
                per_zone_predictions[zone].extend(predicted.cpu().numpy())
                per_zone_labels[zone].extend(zone_labels.cpu().numpy())
                
                # Add to consolidated metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(zone_labels.cpu().numpy())
    
    # Calculate per-zone metrics
    per_zone_metrics = []
    for zone in range(model.num_zones):
        zone_metrics = {
            'precision': precision_score(per_zone_labels[zone], per_zone_predictions[zone], average='macro'),
            'recall': recall_score(per_zone_labels[zone], per_zone_predictions[zone], average='macro'),
            'f1': f1_score(per_zone_labels[zone], per_zone_predictions[zone], average='macro'),
            'confusion_matrix': confusion_matrix(per_zone_labels[zone], per_zone_predictions[zone])
        }
        per_zone_metrics.append(zone_metrics)
    
    # Calculate consolidated metrics
    consolidated_metrics = {
        'precision': precision_score(all_labels, all_predictions, average='macro'),
        'recall': recall_score(all_labels, all_predictions, average='macro'),
        'f1': f1_score(all_labels, all_predictions, average='macro'),
        'confusion_matrix': confusion_matrix(all_labels, all_predictions)
    }
    
    return {
        'per_zone': per_zone_metrics,
        'consolidated': consolidated_metrics
    }

def print_evaluation_metrics(metrics):
    print("\n=== Consolidated Metrics (All Zones) ===")
    print(f"Overall F1 Score: {metrics['consolidated']['f1']:.4f}")
    print(f"Overall Precision: {metrics['consolidated']['precision']:.4f}")
    print(f"Overall Recall: {metrics['consolidated']['recall']:.4f}")
    print("\nConsolidated Confusion Matrix:")
    print(metrics['consolidated']['confusion_matrix'])
    
    # print("\n=== Per-Zone Metrics ===")
    # for zone_idx, zone_metric in enumerate(metrics['per_zone']):
    #     print(f"\nZone {zone_idx + 1} (Day {zone_idx + 1}):")
    #     print(f"F1 Score: {zone_metric['f1']:.4f}")
    #     print(f"Precision: {zone_metric['precision']:.4f}")
    #     print(f"Recall: {zone_metric['recall']:.4f}")
    #     print("Confusion Matrix:")
    #     print(zone_metric['confusion_matrix'])


def format_tensor_dataset(features, labels, num_zones):
    """Convert features and labels to appropriate tensor format for multi-zone prediction"""
    feature_tensor = torch.FloatTensor(features)
    
    # Convert labels to numpy array
    if isinstance(labels[0], (list, np.ndarray)):
        # If labels are already in the correct format (list of lists)
        label_array = np.array([np.array(label, dtype=np.int64) for label in labels])
    else:
        # If labels need to be expanded
        label_array = np.array([[label] * num_zones for label in labels], dtype=np.int64)
    
    # Ensure labels are in the correct shape [batch_size, num_zones]
    if len(label_array.shape) == 1:
        label_array = np.expand_dims(label_array, axis=-1)
    
    # Convert to tensor
    label_tensor = torch.LongTensor(label_array.astype(np.int64))
    
    return TensorDataset(feature_tensor, label_tensor)

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


def load_data_to_cache(config: dict[str, str], param: dict[str]):
    df, num_data_points, display_date_range = analysisUtil.download_data(config, param)    
    calculate_label(df, param)  # Call subroutine to calicalte the label 
    symbol = param['symbol']
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
# def calculate_class_weight(total_samples_df: DataFrame, num_classes: int):
#     total_sample_count = len(total_samples_df)
#     class_weights = []
#     for i in range(num_classes):
#         sample_count = len(total_samples_df[total_samples_df['label'] == i])
#         sample_weight = total_sample_count / (num_classes* sample_count)
#         class_weights.append(sample_weight)

#     # now we have the counts for all three calculate the weight using the formula
#     # WI = total_sample_count / (3* sample_count )
#     return class_weights

def calculate_class_weight(total_samples_df: DataFrame, num_classes: int):
    # Convert labels column to list format if it's not already
    labels = total_samples_df['label'].values
    if not isinstance(labels[0], list):
        labels = [[label] for label in labels]
    
    num_zones = len(labels[0])
    zone_weights = []
    
    for zone in range(num_zones):
        class_weights = []
        zone_labels = [label[zone] for label in labels]
        total_sample_count = len(zone_labels)
        
        for i in range(num_classes):
            sample_count = sum(1 for label in zone_labels if label == i)
            if sample_count == 0:
                sample_weight = 1.0
            else:
                sample_weight = total_sample_count / (num_classes * sample_count)
            class_weights.append(sample_weight)
            
        zone_weights.append(class_weights)
    
    # Average weights across all zones
    avg_weights = np.mean(zone_weights, axis=0)
    return avg_weights

    
def train_with_multi_zone_stopping(model, train_loader, val_loader, num_epochs, optimizer, 
                                 criterion, device, scheduler=None, l1_lambda=0):
    stopping = analysisUtil.TrendBasedStopping(window_size=10, threshold=0.02)
    min_epochs = 80  # Allow some minimum training

    warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs=10) # new for mz attention

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        if epoch < 10:  # During warmup period
            warmup_scheduler.step()

        for batch_idx, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)  # Shape: [batch_size, num_zones]
            
            optimizer.zero_grad()
            outputs = model(inputs)  # Shape: [batch_size, num_zones, num_classes]
            
            # Reshape for loss calculation
            batch_size, num_zones, num_classes = outputs.size()
            outputs = outputs.view(-1, num_classes)  # Shape: [batch_size * num_zones, num_classes]
            labels = labels.view(-1)  # Shape: [batch_size * num_zones]
            
            loss = criterion(outputs, labels)
            
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for name, p in model.named_parameters() 
                            if 'weight' in name)
                loss = loss + l1_lambda * l1_norm
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        if scheduler is not None:
            scheduler.step(avg_loss)
            
        if epoch >= min_epochs and stopping(avg_loss, model):
            print(f"Stopping at epoch {epoch+1} due to increasing trend")
            model.load_state_dict(stopping.best_model)
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return model
#
# Training and evaluation functions (same as before)

def print_metrics(val_metrics, test_metrics):
    print("\n=== Validation Metrics ===")
    for zone_idx, zone_metric in enumerate(val_metrics):
        print(f"\nZone {zone_idx + 1}:")
        print(f"Precision: {zone_metric['precision']:.4f}")
        print(f"Recall: {zone_metric['recall']:.4f}")
        print(f"F1 Score: {zone_metric['f1']:.4f}")
        print("\nConfusion Matrix:")
        print(zone_metric['confusion_matrix'])

    print("\n=== Test Metrics ===")
    for zone_idx, zone_metric in enumerate(test_metrics):
        print(f"\nZone {zone_idx + 1}:")
        print(f"Precision: {zone_metric['precision']:.4f}")
        print(f"Recall: {zone_metric['recall']:.4f}")
        print(f"F1 Score: {zone_metric['f1']:.4f}")
        print("\nConfusion Matrix:")
        print(zone_metric['confusion_matrix'])

class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs=10, initial_lr_factor=0.1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr_factor = initial_lr_factor
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
        
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            factor = self.initial_lr_factor + (1 - self.initial_lr_factor) * (self.current_epoch / self.warmup_epochs)
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * factor
        self.current_epoch += 1

def analyze_trend(config: dict[str, str], param: dict[str], comment : str, turn_random_on: bool, use_cached_data: bool):
    print('\nDeveloping multi-zone  for ' + param['symbol'])
    symbol = param['symbol']

    if turn_random_on:
        random_seed = random.randint(0, 2**32 - 1)
    else:
        random_seed = 42

    # Set random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load or cache data
    if use_cached_data:
        df = pd.read_csv(symbol + '_TMP.csv')
        print('>data loaded from cache')
    else:
        df = load_data_to_cache(config, param)

        df = calculate_multi_zone_label(df, param)  # New multi-zone label calculation
    raw_df = df.copy()

    # Filter by date
    df = df[df['date'] >= param['start_date']]
    if param.get('end_date'):
        df = df[df['date'] <= param['end_date']]

    selected_columns = param['selected_columns']
    df = df[selected_columns]

    # Split data
    train_df, test_df = train_test_split(df, test_size=param['test_size'], shuffle=True, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=param['validation_size'], shuffle=True, random_state=42)

    # Calculate class weights for each zone
    num_labels = 3  # Fixed for our case (no change, up, down)
    class_weights = calculate_class_weight(train_df, num_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # Scale features
    scaler = RobustScaler() if param['scaler_type'] == 'Robust' else \
             MinMaxScaler() if param['scaler_type'] == 'MinMax' else \
             StandardScaler()
    
    train_features = scaler.fit_transform(train_df.drop(['label'], axis=1))
    val_features = scaler.transform(val_df.drop(['label'], axis=1))
    test_features = scaler.transform(test_df.drop(['label'], axis=1))
    dump(scaler, 'scaler.joblib')

    # Prepare data loaders
    train_data = format_tensor_dataset(train_features, train_df['label'].values, param['num_zones'])
    val_data = format_tensor_dataset(val_features, val_df['label'].values, param['num_zones'])
    test_data = format_tensor_dataset(test_features, test_df['label'].values, param['num_zones'])

    train_loader = DataLoader(train_data, shuffle=param['shuffle'], batch_size=param['batch_size'])
    val_loader = DataLoader(val_data, shuffle=param['shuffle'], batch_size=param['batch_size'])
    test_loader = DataLoader(test_data, shuffle=param['shuffle'], batch_size=param['batch_size'])

    # Initialize model
    feature_count = train_features.shape[1]
    model = MultiZoneTransformer(
        input_dim=feature_count,
        num_classes=3,
        num_heads=param['headcount'],
        num_layers=param['num_layers'],
        dropout_rate=param['dropout_rate'],
        embedded_dim=param['embedded_dim'],
        num_zones=param['num_zones']
    )

    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    # optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'], weight_decay=param["l2_weight_decay"])

    # use new optimizer for zone specific processing
    optimizer = optim.Adam([
    {'params': model.input_projection.parameters(), 'lr': 0.0005},
    {'params': model.zone_encoders.parameters(), 'lr': 0.001},
    {'params': model.zone_predictors.parameters(), 'lr': 0.001},
    {'params': model.zone_embeddings, 'lr': 0.002}])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    # Train model
    model = train_with_multi_zone_stopping(
        model, train_loader, val_loader, param['num_epochs'],
        optimizer, criterion, device, scheduler, param["l1_lambda"]
    )

    # Save model
    currentDateTime = datetime.now()
    date = currentDateTime.strftime("%Y-%m-%d %H:%M:%S")
    torch.save(model, f'model_{date}.pth')

    # Evaluate model
    val_metrics = evaluate_multi_zone_model(model, val_loader, criterion, device)
    test_metrics = evaluate_multi_zone_model(model, test_loader, criterion, device)

    # Print results
    print("\nValidation Metrics:")
    print_evaluation_metrics(val_metrics)

    print("\nTest Metrics:")
    print_evaluation_metrics(test_metrics)

    # Make predictions
    make_multi_zone_prediction(model, raw_df, param, currentDateTime, symbol, comment)

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
