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

transactions_df = pd.read_csv("transactions_train.csv", low_memory=False)
merchants_df = pd.read_csv("merchants.csv")
terminals_df = pd.read_csv("terminals.csv")

df = transactions_df.merge(terminals_df, on='TERMINAL_ID', how='left')
df = df.merge(merchants_df, on='MERCHANT_ID', how='left')

df['TX_TS'] = pd.to_datetime(df['TX_TS'])

df['hour'] = df['TX_TS'].dt.hour
df['day_of_week'] = df['TX_TS'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)


if 'x_customer_id' in df.columns and 'y_customer_id' in df.columns:
    df['distance'] = np.sqrt(
        (df['x_customer_id'] - df['x_terminal_id'])**2 + 
        (df['y_customer_id'] - df['y_terminal_id'])**2
    )

    df['log_distance'] = np.log1p(df['distance'])


df['log_amount'] = np.log1p(df['TX_AMOUNT'])
df['amount_risk_score'] = (df['TX_AMOUNT'] > df['TX_AMOUNT'].quantile(0.95)).astype(int)


if 'CUSTOMER_ID' in df.columns:
    
    df = df.sort_values(['CUSTOMER_ID', 'TX_TS'])
    
    customer_stats = df.groupby('CUSTOMER_ID').agg({
        'TX_AMOUNT': ['mean', 'std', 'count'],
        'TX_FRAUD': 'mean'  
    }).reset_index()
    
    customer_stats.columns = ['CUSTOMER_ID', 'customer_avg_amount', 'customer_std_amount', 
                             'customer_tx_count', 'customer_fraud_rate']
    
    df = df.merge(customer_stats, on='CUSTOMER_ID', how='left')
    
    df['amount_deviation'] = np.abs(df['TX_AMOUNT'] - df['customer_avg_amount']) / (df['customer_std_amount'] + 1e-6)

terminal_stats = df.groupby('TERMINAL_ID').agg({
    'TX_FRAUD': 'mean',  
    'TX_AMOUNT': 'count' 
}).reset_index()
terminal_stats.columns = ['TERMINAL_ID', 'terminal_fraud_rate', 'terminal_tx_volume']
df = df.merge(terminal_stats, on='TERMINAL_ID', how='left')

columns_to_drop = []
if 'CUSTOMER_ID' in df.columns:
    columns_to_drop.append('CUSTOMER_ID')
if 'MERCHANT_ID' in df.columns:
    columns_to_drop.append('MERCHANT_ID')
if 'TERMINAL_ID' in df.columns:
    columns_to_drop.append('TERMINAL_ID')
if 'TX_TS' in df.columns:
    columns_to_drop.append('TX_TS')

if columns_to_drop:
    df = df.drop(columns_to_drop, axis=1)

categorical_columns = [
    'CARD_BRAND', 
    'TRANSACTION_TYPE', 
    'TRANSACTION_STATUS',
    'CARD_COUNTRY_CODE',
    'IS_RECURRING_TRANSACTION',
    'TRANSACTION_CURRENCY'
]

merchant_categorical_features = ['BUSINESS_TYPE', 'OUTLET_TYPE']
for col in merchant_categorical_features:
    if col in df.columns:
        categorical_columns.append(col)

label_encoders = {}
for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = df[col].fillna('UNKNOWN')
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

base_numeric_columns = [
    'TX_AMOUNT',
    'TRANSACTION_GOODS_AND_SERVICES_AMOUNT',
    'TRANSACTION_CASHBACK_AMOUNT',
    'hour',
    'day_of_week',
    'is_weekend',
    'is_night',
    'is_business_hours',
    'log_amount',
    'amount_risk_score',
    'x_terminal_id',
    'y_terminal__id'  
]

enhanced_features = [
    'distance', 'log_distance', 'customer_avg_amount', 'customer_std_amount',
    'customer_tx_count', 'customer_fraud_rate', 'amount_deviation',
    'terminal_fraud_rate', 'terminal_tx_volume'
]

merchant_numeric_features = [
    'ANNUAL_TURNOVER', 'ANNUAL_TURNOVER_CARD', 'AVERAGE_TICKET_SALE_AMOUNT',
    'DEPOSIT_PERCENTAGE', 'DELIVERY_SAME_DAYS_PERCENTAGE'
]

numeric_columns = []
for col in base_numeric_columns + enhanced_features + merchant_numeric_features:
    if col in df.columns:
        numeric_columns.append(col)

for col in numeric_columns:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

all_features = numeric_columns + [col for col in categorical_columns if col in df.columns]
X = df[all_features]
y = df['TX_FRAUD']

print(f"Feature set size: {X.shape[1]} features")
print(f"Class distribution: {Counter(y)}")
print(f"Fraud rate: {y.mean():.4f}")

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
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

input_dim = X_train.shape[1]
model = EnhancedFraudNet(input_dim, dropout_rate=0.4)

try:
    model.load_state_dict(torch.load("enhanced_fraud_model.pth"))
    print("Loaded existing enhanced model weights. Continuing training...")
except FileNotFoundError:
    print("No saved enhanced model found. Starting fresh.")

num_pos = y_train.sum()
num_neg = len(y_train) - num_pos
pos_weight = torch.tensor([num_neg / num_pos * 0.7], dtype=torch.float32)  # Slightly reduce to prevent over-weighting

print(f"Positive class weight: {pos_weight.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

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
                'feature_names': all_features,
                'numeric_features': numeric_columns,
                'label_encoders': label_encoders,
                'scaler_feature_names': getattr(scaler, 'feature_names_in_', list(X.columns))
            }, f)
            with open("scaler.pkl", "wb") as fsc:
                pickle.dump(scaler, fsc)

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
        'feature': all_features,
        'importance': first_layer_weights
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))

print(f"\nBest model saved as: enhanced_fraud_model.pth")
print(f"Final F1 Score: {f1_score(y_val, val_preds):.4f}")