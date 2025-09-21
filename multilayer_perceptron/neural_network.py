#!/usr/bin/env python3

# flake8: noqa

# %% Data loading & preprocessing

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from collections import Counter
import pickle

from data import load_and_preprocess_data

print("Loading and preprocessing data...")
X_train, y_train, X_test, y_test = load_and_preprocess_data()

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Fraud rate: {y_train.mean():.4f}")

# Data is already preprocessed by data.py
X = X_train
y = y_train

print(f"Feature set size: {X.shape[1]} features")
print(f"Class distribution: {Counter(y)}")
print(f"Fraud rate: {y.mean():.4f}")

# Use the test split from data.py
X_val = X_test
y_val = y_test

X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

class EnhancedFraudNet(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super(EnhancedFraudNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)  
        return x

input_dim = X.shape[1]
model = EnhancedFraudNet(input_dim, dropout_rate=0.4)

try:
    model.load_state_dict(torch.load("enhanced_fraud_model.pth"))
    print("Loaded existing enhanced model weights. Continuing training...")
except FileNotFoundError:
    print("No saved enhanced model found. Starting fresh.")

num_pos = y.sum()
num_neg = len(y) - num_pos
pos_weight = torch.tensor([num_neg / num_pos * 0.7], dtype=torch.float32)  # Slightly reduce to prevent over-weighting

print(f"Positive class weight: {pos_weight.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

def find_optimal_threshold(y_true, y_pred_proba):
    """Find optimal threshold that maximizes F1 score"""
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1

EPOCHS = 5
best_f1 = 0
patience_counter = 0
patience = 3

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item()
    
    model.eval()
    val_probs, val_labels = [], []
    val_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch)
            val_loss += criterion(logits, y_batch).item()
            probs = torch.sigmoid(logits).cpu().numpy()
            val_probs.extend(probs.flatten())
            val_labels.extend(y_batch.cpu().numpy().flatten())
    
    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)
    
    optimal_threshold, max_f1 = find_optimal_threshold(val_labels, val_probs)
    
    val_preds = (val_probs >= optimal_threshold).astype(int)
    
    acc = accuracy_score(val_labels, val_preds)
    prec = precision_score(val_labels, val_preds, zero_division=0)
    rec = recall_score(val_labels, val_preds, zero_division=0)
    f1 = f1_score(val_labels, val_preds, zero_division=0)
    auc = roc_auc_score(val_labels, val_probs)
    
    print(f"Epoch {epoch+1}/{EPOCHS}:")
    print(f"  Train Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"  Threshold: {optimal_threshold:.3f}, AUC: {auc:.4f}")
    print(f"  Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    print("-" * 60)
    
    scheduler.step(f1)
    
    if f1 > best_f1:
        best_f1 = f1
        patience_counter = 0
        torch.save(model.state_dict(), "enhanced_fraud_model.pth")
        with open("model_metadata.pkl", "wb") as f:
            pickle.dump({
                'feature_names': list(X.columns),
                'numeric_features': list(X.columns),
                'label_encoders': {},
                'scaler_feature_names': list(X.columns)
            }, f)

        print(f"  New best F1: {best_f1:.4f} - Model saved!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

try:
    model.load_state_dict(torch.load("best_enhanced_fraud_model.pth", weights_only=True))
    with open("model_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    optimal_threshold = metadata['threshold']
    print("Loaded best model successfully!")
except:
    print("Using current model state for evaluation...")
    optimal_threshold = 0.850 

model.eval()
with torch.no_grad():
    val_logits = model(X_val_tensor)
    val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
    val_preds = (val_probs >= optimal_threshold).astype(int)

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"AUC-ROC: {roc_auc_score(y_val, val_probs):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, val_preds, target_names=['Non-Fraud', 'Fraud']))

with torch.no_grad():
    first_layer_weights = model.fc1.weight.abs().mean(dim=0).cpu().numpy()
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': first_layer_weights
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))

print(f"\nBest model saved as: enhanced_fraud_model.pth")
print(f"Final F1 Score: {f1_score(y_val, val_preds):.4f}")