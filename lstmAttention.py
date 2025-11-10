import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Optional, List
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight


class StockMovementDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int,
        target_column: int = 3,  # Assuming 'Close' price is at index 3
        threshold: float = 0.03  # 3% threshold
    ):
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.threshold = threshold
        self.X, self.y = self._create_sequences(data)
        
    def _calculate_class(self, current_price: float, prev_price: float) -> int:
        """Calculate class based on percentage change
        0: No Change (-3% to 3%)
        1: Up (> 3%)
        2: Down (< -3%)
        """
        pct_change = (current_price - prev_price) / prev_price
        if pct_change >= self.threshold:
            return 1  # Up
        elif pct_change <= -self.threshold:
            return 2  # Down
        return 0  # No Change
        
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - 1):
            # Get sequence
            sequence = data[i:(i + self.sequence_length)]
            
            # Calculate class based on next day's price movement
            current_price = data[i + self.sequence_length, self.target_column]
            next_price = data[i + self.sequence_length + 1, self.target_column]
            movement_class = self._calculate_class(next_price, current_price)
            
            X.append(sequence)
            y.append(movement_class)
            
        return np.array(X), np.array(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.FloatTensor(self.X[idx]),
            torch.LongTensor([self.y[idx]])  # Changed to LongTensor for classification
        )

class StockDataPreprocessor:
    def __init__(
        self,
        sequence_length: int = 20,
        train_split: float = 0.8,
        val_split: float = 0.1,
        batch_size: int = 32,
        target_column: int = 3,
        threshold: float = 0.03
    ):
        self.sequence_length = sequence_length
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.target_column = target_column
        self.threshold = threshold
        self.scaler = MinMaxScaler()
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        if feature_columns is None:
            feature_columns = ['open', 'high', 'low', 'adjusted close', 'volume']
            
        data = df[feature_columns].values
        normalized_data = self.scaler.fit_transform(data)
        
        train_size = int(len(normalized_data) * self.train_split)
        val_size = int(len(normalized_data) * self.val_split)
        
        train_data = normalized_data[:train_size]
        val_data = normalized_data[train_size:train_size + val_size]
        test_data = normalized_data[train_size + val_size:]
        
        train_dataset = StockMovementDataset(
            train_data, self.sequence_length, self.target_column, self.threshold
        )
        val_dataset = StockMovementDataset(
            val_data, self.sequence_length, self.target_column, self.threshold
        )
        test_dataset = StockMovementDataset(
            test_data, self.sequence_length, self.target_column, self.threshold
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader

class StockMovementClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)  # Output 3 classes
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # LSTM forward pass
        lstm_output, _ = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        
        # Classification layer
        logits = self.fc(context_vector)
        return logits, attention_weights

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10
    ) -> Dict[str, list]:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_losses = []
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device).squeeze()
                
                optimizer.zero_grad()
                logits, _ = self(batch_x)
                loss = criterion(logits, batch_y)
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                _, predicted = torch.max(logits, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            self.eval()
            val_losses = []
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device).squeeze()
                    logits, _ = self(batch_x)
                    loss = criterion(logits, batch_y)
                    
                    val_losses.append(loss.item())
                    _, predicted = torch.max(logits, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate epoch metrics
            train_loss = np.mean(train_losses)
            train_acc = train_correct / train_total
            val_loss = np.mean(val_losses)
            val_acc = val_correct / val_total
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        return history

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model and return detailed metrics"""
        self.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                logits, _ = self(batch_x)
                _, predicted = torch.max(logits, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.numpy().flatten())
        
        # Calculate metrics
        class_names = ['No Change', 'Up', 'Down']
        report = classification_report(all_targets, all_predictions, 
                                    target_names=class_names, output_dict=True)
        
        # Create confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return report

def plot_training_history(history: Dict[str, list]) -> None:
    """Plot training history"""
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def analyze_class_distribution(dataset: StockMovementDataset) -> Dict[str, float]:
    """Analyze the distribution of classes in the dataset"""
    classes, counts = np.unique(dataset.y, return_counts=True)
    total = len(dataset.y)
    distribution = {
        'No Change': 0,
        'Up': 0,
        'Down': 0
    }
    class_names = ['No Change', 'Up', 'Down']
    
    for class_idx, count in zip(classes, counts):
        percentage = (count / total) * 100
        distribution[class_names[class_idx]] = percentage
        
    return distribution

class WeightedStockMovementClassifier(StockMovementClassifier):
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10
    ) -> Dict[str, list]:
        # Calculate class weights
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy().flatten())
        
        class_counts = np.bincount(all_labels)
        total_samples = len(all_labels)
        class_weights = torch.FloatTensor(total_samples / (len(class_counts) * class_counts))
        class_weights = class_weights.to(self.device)
        
        # Use weighted loss
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'class_predictions': []  # Track predictions per class
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_losses = []
            train_correct = 0
            train_total = 0
            class_correct = np.zeros(3)
            class_total = np.zeros(3)
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device).squeeze()
                
                optimizer.zero_grad()
                logits, _ = self(batch_x)
                loss = criterion(logits, batch_y)
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                _, predicted = torch.max(logits, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
                
                # Track per-class accuracy
                for class_idx in range(3):
                    mask = (batch_y == class_idx)
                    class_total[class_idx] += mask.sum().item()
                    class_correct[class_idx] += ((predicted == class_idx) & mask).sum().item()
            
            # Validation phase
            self.eval()
            val_losses = []
            val_correct = 0
            val_total = 0
            val_class_correct = np.zeros(3)
            val_class_total = np.zeros(3)
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device).squeeze()
                    logits, _ = self(batch_x)
                    loss = criterion(logits, batch_y)
                    
                    val_losses.append(loss.item())
                    _, predicted = torch.max(logits, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
                    
                    # Track per-class accuracy
                    for class_idx in range(3):
                        mask = (batch_y == class_idx)
                        val_class_total[class_idx] += mask.sum().item()
                        val_class_correct[class_idx] += ((predicted == class_idx) & mask).sum().item()
            
            # Calculate epoch metrics
            train_loss = np.mean(train_losses)
            train_acc = train_correct / train_total
            val_loss = np.mean(val_losses)
            val_acc = val_correct / val_total
            
            # Calculate per-class accuracies
            train_class_acc = np.divide(class_correct, class_total, 
                                      out=np.zeros_like(class_correct), where=class_total!=0)
            val_class_acc = np.divide(val_class_correct, val_class_total,
                                    out=np.zeros_like(val_class_correct), where=val_class_total!=0)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['class_predictions'].append(val_class_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print("\nPer-class Validation Accuracy:")
                print(f"No Change: {val_class_acc[0]:.4f}")
                print(f"Up: {val_class_acc[1]:.4f}")
                print(f"Down: {val_class_acc[2]:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
        
        return history

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model with handling for class imbalance"""
        self.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                logits, _ = self(batch_x)
                _, predicted = torch.max(logits, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.numpy().flatten())
        
        # Calculate metrics with handling for zero division
        class_names = ['No Change', 'Up', 'Down']
        report = classification_report(
            all_targets, 
            all_predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0  # Handle zero division explicitly
        )
        
        # Create confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return report

class ClassWeightCalculator:
    """Implements different strategies for calculating class weights"""
    
    @staticmethod
    def get_class_counts(labels: np.ndarray) -> Dict[int, int]:
        """Get count of samples for each class"""
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))
    
    @staticmethod
    def inverse_frequency(labels: np.ndarray, normalization: str = 'sum') -> torch.Tensor:
        """
        Calculate weights as inverse of class frequencies
        normalization: 'sum' or 'max' to control weight scaling
        """
        class_counts = ClassWeightCalculator.get_class_counts(labels)
        weights = np.array([1/count for count in class_counts.values()])
        
        if normalization == 'sum':
            weights = weights / weights.sum() * len(weights)
        elif normalization == 'max':
            weights = weights / weights.max()
            
        return torch.FloatTensor(weights)
    
    @staticmethod
    def squared_inverse_frequency(labels: np.ndarray) -> torch.Tensor:
        """More aggressive weighting for minority classes"""
        class_counts = ClassWeightCalculator.get_class_counts(labels)
        weights = np.array([1/(count**2) for count in class_counts.values()])
        weights = weights / weights.sum() * len(weights)
        return torch.FloatTensor(weights)
    
    @staticmethod
    def sklearn_balanced(labels: np.ndarray) -> torch.Tensor:
        """Use sklearn's balanced weight calculation"""
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        return torch.FloatTensor(weights)
    
    @staticmethod
    def custom_weights(labels: np.ndarray, weight_multipliers: Dict[int, float]) -> torch.Tensor:
        """Apply custom multipliers to base weights"""
        base_weights = ClassWeightCalculator.inverse_frequency(labels)
        for class_idx, multiplier in weight_multipliers.items():
            base_weights[class_idx] *= multiplier
        return base_weights

def analyze_class_weights(labels: np.ndarray) -> Dict[str, torch.Tensor]:
    """Compare different weighting strategies"""
    calculator = ClassWeightCalculator()
    
    weights = {
        'inverse_freq': calculator.inverse_frequency(labels),
        'inverse_freq_max_norm': calculator.inverse_frequency(labels, 'max'),
        'squared_inverse': calculator.squared_inverse_frequency(labels),
        'sklearn_balanced': calculator.sklearn_balanced(labels)
    }
    
    return weights

class WeightedClassifier(WeightedStockMovementClassifier):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
        weight_strategy: str = 'inverse_freq',
        custom_weights: Optional[Dict[int, float]] = None
    ):
        super().__init__(input_dim, hidden_dim, num_layers, num_classes, dropout)
        self.weight_strategy = weight_strategy
        self.custom_weights = custom_weights
    
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10
    ) -> Dict[str, list]:
        # Get all training labels
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy().flatten())
        all_labels = np.array(all_labels)
        
        # Calculate class distribution
        class_counts = ClassWeightCalculator.get_class_counts(all_labels)
        print("\nClass Distribution:")
        total_samples = sum(class_counts.values())
        for class_idx, count in class_counts.items():
            percentage = (count / total_samples) * 100
            print(f"Class {class_idx}: {count} samples ({percentage:.2f}%)")
        
        # Calculate weights based on selected strategy
        calculator = ClassWeightCalculator()
        if self.weight_strategy == 'inverse_freq':
            weights = calculator.inverse_frequency(all_labels)
        elif self.weight_strategy == 'squared_inverse':
            weights = calculator.squared_inverse_frequency(all_labels)
        elif self.weight_strategy == 'sklearn_balanced':
            weights = calculator.sklearn_balanced(all_labels)
        elif self.weight_strategy == 'custom' and self.custom_weights:
            weights = calculator.custom_weights(all_labels, self.custom_weights)
        else:
            weights = calculator.inverse_frequency(all_labels)
        
        print("\nSelected Class Weights:")
        for i, w in enumerate(weights):
            print(f"Class {i}: {w:.4f}")
        
        # Move weights to device
        weights = weights.to(self.device)
        
        # Use weighted loss
        criterion = nn.CrossEntropyLoss(weight=weights)
        
        # Continue with training as before...
              # Calculate class weights
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy().flatten())
        
        class_counts = np.bincount(all_labels)
        total_samples = len(all_labels)
        class_weights = torch.FloatTensor(total_samples / (len(class_counts) * class_counts))
        class_weights = class_weights.to(self.device)
        
        # Use weighted loss
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'class_predictions': []  # Track predictions per class
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_losses = []
            train_correct = 0
            train_total = 0
            class_correct = np.zeros(3)
            class_total = np.zeros(3)
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device).squeeze()
                
                optimizer.zero_grad()
                logits, _ = self(batch_x)
                loss = criterion(logits, batch_y)
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                _, predicted = torch.max(logits, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
                
                # Track per-class accuracy
                for class_idx in range(3):
                    mask = (batch_y == class_idx)
                    class_total[class_idx] += mask.sum().item()
                    class_correct[class_idx] += ((predicted == class_idx) & mask).sum().item()
            
            # Validation phase
            self.eval()
            val_losses = []
            val_correct = 0
            val_total = 0
            val_class_correct = np.zeros(3)
            val_class_total = np.zeros(3)
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device).squeeze()
                    logits, _ = self(batch_x)
                    loss = criterion(logits, batch_y)
                    
                    val_losses.append(loss.item())
                    _, predicted = torch.max(logits, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
                    
                    # Track per-class accuracy
                    for class_idx in range(3):
                        mask = (batch_y == class_idx)
                        val_class_total[class_idx] += mask.sum().item()
                        val_class_correct[class_idx] += ((predicted == class_idx) & mask).sum().item()
            
            # Calculate epoch metrics
            train_loss = np.mean(train_losses)
            train_acc = train_correct / train_total
            val_loss = np.mean(val_losses)
            val_acc = val_correct / val_total
            
            # Calculate per-class accuracies
            train_class_acc = np.divide(class_correct, class_total, 
                                      out=np.zeros_like(class_correct), where=class_total!=0)
            val_class_acc = np.divide(val_class_correct, val_class_total,
                                    out=np.zeros_like(val_class_correct), where=val_class_total!=0)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['class_predictions'].append(val_class_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print("\nPer-class Validation Accuracy:")
                print(f"No Change: {val_class_acc[0]:.4f}")
                print(f"Up: {val_class_acc[1]:.4f}")
                print(f"Down: {val_class_acc[2]:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
        
        return history

def compare_weight_strategies(train_loader: DataLoader) -> None:
    """Compare different weighting strategies on your data"""
    # Get all training labels
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy().flatten())
    all_labels = np.array(all_labels)
    
    # Get weights from different strategies
    weights = analyze_class_weights(all_labels)
    
    # Print comparison
    print("\nComparison of Class Weighting Strategies:")
    print("\nClass Distribution:")
    class_counts = ClassWeightCalculator.get_class_counts(all_labels)
    total_samples = sum(class_counts.values())
    
    for class_idx in range(3):
        count = class_counts.get(class_idx, 0)
        percentage = (count / total_samples) * 100
        print(f"\nClass {class_idx}:")
        print(f"Samples: {count} ({percentage:.2f}%)")
        print("Weights across strategies:")
        for strategy, weight_tensor in weights.items():
            print(f"  {strategy}: {weight_tensor[class_idx]:.4f}")


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    df = pd.read_csv('NVDA_TMP.csv')
    
    # Initialize preprocessor
    preprocessor = StockDataPreprocessor(
        sequence_length=20,
        batch_size=32,
        train_split=0.8,
        val_split=0.1,
        threshold=0.03  # 3% threshold
    )
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = preprocessor.prepare_data(df)
    
    # Initialize model
    # model = StockMovementClassifier(
    #     input_dim=5,  # Number of features (Open, High, Low, Close, Volume)
    #     hidden_dim=64,
    #     num_layers=2,
    #     num_classes=3,
    #     dropout=0.2
    # )
    
      # Analyze class distribution before training
    print("\nAnalyzing class distribution in training data:")
    train_distribution = analyze_class_distribution(train_loader.dataset)
    print("\nClass Distribution:")
    for class_name, percentage in train_distribution.items():
        print(f"{class_name}: {percentage:.2f}%")

           # Compare different weighting strategies
    compare_weight_strategies(train_loader)
    
    # Try different weighting strategies
    strategies = [
        ('inverse_freq', None),
        ('squared_inverse', None),
        ('sklearn_balanced', None),
        ('custom', {0: 1.0, 1: 2.0, 2: 1.5})  # Custom multipliers
    ]
    
    results = {}
    for strategy, custom_weights in strategies:
        print(f"\nTraining with {strategy} weighting strategy:")
        model = WeightedClassifier(
            input_dim=5,
            hidden_dim=64,
            num_layers=2,
            weight_strategy=strategy,
            custom_weights=custom_weights
        )
        
        history = model.train_model(
            train_loader, 
            val_loader, 
            epochs=100, 
            early_stopping_patience=15,
            #min_delta=0.0005
        )
        results[strategy] = history
    
    # Compare results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    for strategy, history in results.items():
        plt.plot(history['val_loss'], label=strategy)
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for strategy, history in results.items():
        plt.plot(history['val_acc'], label=strategy)
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # # Initialize model with class weighting
    # model = WeightedStockMovementClassifier(
    #     input_dim=5,
    #     hidden_dim=64,
    #     num_layers=2,
    #     num_classes=3,
    #     dropout=0.2
    # )

    # # Train model
    # history = model.train_model(
    #     train_loader,
    #     val_loader,
    #     epochs=100,
    #     learning_rate=0.001,
    #     early_stopping_patience=10
    # )
    
    # # Plot training history
    # plot_training_history(history)
    
    # # Evaluate model
    # test_metrics = model.evaluate(test_loader)
    # print("\nTest Set Metrics:")
    # for class_name, metrics in test_metrics.items():
    #     if class_name in ['No Change', 'Up', 'Down']:
    #         print(f"\n{class_name}:")
    #         print(f"Precision: {metrics['precision']:.4f}")
    #         print(f"Recall: {metrics['recall']:.4f}")
    #         print(f"F1-Score: {metrics['f1-score']:.4f}")

if __name__ == "__main__":
    main()