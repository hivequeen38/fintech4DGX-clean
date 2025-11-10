from torch.utils.data import Dataset, DataLoader, random_split

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# define k-fold
from sklearn.model_selection import KFold

num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# cross validation loop
import torch
import torch.nn as nn
import torch.optim as optim

# Assume model is your Transformer model class
accuracy_list = []

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    # Split data
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    # Define data loaders for training and testing data in this fold
    train_loader = DataLoader(
        dataset, 
        batch_size=10, 
        sampler=train_subsampler)
    test_loader = DataLoader(
        dataset,
        batch_size=10,
        sampler=test_subsampler)

    # Init the neural network
    model = StockTransformer(input_dim, num_classes, num_heads, num_layers, dropout_rate)
    model.train()

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    # Evaluation for this fold
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100. * correct / total
    accuracy_list.append(accuracy)
    print(f'Fold {fold+1}, Accuracy: {accuracy}%')

# Print fold results
print(f'K-Fold Cross Validation results: {np.mean(accuracy_list):.2f}%')
