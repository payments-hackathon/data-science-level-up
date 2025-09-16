#!/usr/bin/env python3

# flake8: noqa

# %% Data Import
from data import load_and_preprocess_data

print("Loading and preprocessing data...")
X_train, y_train, X_test, y_test = load_and_preprocess_data()

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Fraud rate: {y_train.mean():.4f}")

# %% Model Definition
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class FraudDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        self.y = torch.FloatTensor(y.values if hasattr(y, 'values') else y) if y is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class FraudNet(nn.Module):
    def __init__(self, input_size):
        super(FraudNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

# %% Model Training
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_model(X_train, y_train):
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    train_dataset = FraudDataset(X_tr, y_tr)
    val_dataset = FraudDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    model = FraudNet(X_train.shape[1])
    # Handle class imbalance
    pos_weight = torch.tensor([y_train.mean() / (1 - y_train.mean())])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = torch.sigmoid(model(batch_X)).squeeze()
                val_preds.extend(outputs.numpy())
                val_true.extend(batch_y.numpy())
        
        auc = roc_auc_score(val_true, val_preds)
        print(f'Epoch {epoch+1}/10, Loss: {total_loss/len(train_loader):.4f}, Val AUC: {auc:.4f}')
        model.train()
    
    return model

def evaluate_model(model, X_test, y_test):
    model.eval()
    test_dataset = FraudDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = torch.sigmoid(model(batch_X)).squeeze()
            predictions.extend(outputs.numpy())
            actuals.extend(batch_y.numpy())
    
    return roc_auc_score(actuals, predictions)

print("Training model...")
model = train_model(X_train, y_train)

print("Evaluating model...")
test_auc = evaluate_model(model, X_test, y_test)
print(f"Test AUC: {test_auc:.4f}")
