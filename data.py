#!/usr/bin/env python3
# flake8: noqa

# %% Imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


# %% Definitions
class FraudDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        
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
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)


# %% Data Loading & Preprocessing
def load_and_preprocess_data():
    # Load datasets
    customers = pd.read_csv('data/Payments Fraud DataSet/customers.csv')
    terminals = pd.read_csv('data/Payments Fraud DataSet/terminals.csv')
    merchants = pd.read_csv('data/Payments Fraud DataSet/merchants.csv')
    train_tx = pd.read_csv('data/Payments Fraud DataSet/transactions_train.csv')
    
    # Merge transaction data with customer and terminal info
    data = train_tx.merge(customers, on='CUSTOMER_ID', how='left')
    data = data.merge(terminals, on='TERMINAL_ID', how='left')
    data = data.merge(merchants, on='MERCHANT_ID', how='left')
    
    # Feature engineering
    def engineer_features(df):
        # Distance between customer and terminal
        df['distance'] = np.sqrt((df['x_customer_id'] - df['x_terminal_id'])**2 + 
                                (df['y_customer_id'] - df['y_terminal_id'])**2)
        
        # Time features
        df['TX_TS'] = pd.to_datetime(df['TX_TS'])
        df['hour'] = df['TX_TS'].dt.hour
        df['day_of_week'] = df['TX_TS'].dt.dayofweek
        
        # Amount ratios
        df['cashback_ratio'] = df['TRANSACTION_CASHBACK_AMOUNT'] / (df['TX_AMOUNT'] + 1e-8)
        df['goods_ratio'] = df['TRANSACTION_GOODS_AND_SERVICES_AMOUNT'] / (df['TX_AMOUNT'] + 1e-8)
        
        return df
    
    data = engineer_features(data)
    
    # Select numerical features
    num_features = ['TX_AMOUNT', 'TRANSACTION_GOODS_AND_SERVICES_AMOUNT', 'TRANSACTION_CASHBACK_AMOUNT',
                   'x_customer_id', 'y_customer_id', 'x_terminal_id', 'y_terminal_id', 'distance',
                   'hour', 'day_of_week', 'cashback_ratio', 'goods_ratio', 'ANNUAL_TURNOVER_CARD',
                   'ANNUAL_TURNOVER', 'AVERAGE_TICKET_SALE_AMOUNT']
    
    # Select categorical features
    cat_features = ['CARD_BRAND', 'TRANSACTION_TYPE', 'TRANSACTION_STATUS', 'TRANSACTION_CURRENCY',
                   'CARD_COUNTRY_CODE', 'IS_RECURRING_TRANSACTION', 'BUSINESS_TYPE', 'OUTLET_TYPE']
    
    # Handle missing values and encode categoricals
    for col in num_features:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    
    # Encode categorical variables
    for col in cat_features:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = data[col].fillna('unknown')
            data[col] = le.transform(data[col])
    
    # Prepare feature matrix
    feature_cols = [col for col in num_features + cat_features if col in data.columns]
    X = data[feature_cols].values
    y = data['TX_FRAUD'].values
    
    # Split into train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test


print("Loading and preprocessing data...")
X_train, y_train, X_test, y_test = load_and_preprocess_data()

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Fraud rate: {y_train.mean():.4f}")

# %% Training
def train_model(X_train, y_train):
    # Split data
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    # Create datasets
    train_dataset = FraudDataset(X_tr, y_tr)
    val_dataset = FraudDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    # Initialize model
    model = FraudNet(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
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
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X).squeeze()
                val_preds.extend(outputs.numpy())
                val_true.extend(batch_y.numpy())
        
        auc = roc_auc_score(val_true, val_preds)
        print(f'Epoch {epoch+1}/10, Loss: {total_loss/len(train_loader):.4f}, Val AUC: {auc:.4f}')
        model.train()
    
    return model

print("Training model...")
model = train_model(X_train, y_train)


# %% Evaluation
def evaluate_model(model, X_test, y_test):
    model.eval()
    test_dataset = FraudDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X).squeeze()
            predictions.extend(outputs.numpy())
            actuals.extend(batch_y.numpy())
    
    test_auc = roc_auc_score(actuals, predictions)
    return test_auc


print("Evaluating model...")
test_auc = evaluate_model(model, X_test, y_test)
print(f"Test AUC: {test_auc:.4f}")
