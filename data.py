#!/usr/bin/env python3

# flake8: noqa

# %% Data loading & preprocessing

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load and preprocess transaction data for neural network models."""
    # Load transaction data
    train_tx = pd.read_csv('data/Payments Fraud DataSet/transactions_train.csv')
    
    # Convert timestamp
    train_tx['TX_TS'] = pd.to_datetime(train_tx['TX_TS'])
    
    # Extract day of week (one-hot encoded)
    dow_dummies = pd.get_dummies(train_tx['TX_TS'].dt.dayofweek, prefix='dow')
    
    # Extract time of day (scaled 0.0-1.0)
    time_of_day = (train_tx['TX_TS'].dt.hour * 3600 + 
                   train_tx['TX_TS'].dt.minute * 60 + 
                   train_tx['TX_TS'].dt.second) / 86400
    
    # Numeric features (as-is)
    numeric_features = train_tx[['TX_AMOUNT', 'TRANSACTION_GOODS_AND_SERVICES_AMOUNT', 
                                'TRANSACTION_CASHBACK_AMOUNT']].copy()
    
    # Feature engineering
    train_tx['amount_log'] = np.log1p(train_tx['TX_AMOUNT'])
    train_tx['cashback_ratio'] = train_tx['TRANSACTION_CASHBACK_AMOUNT'] / (train_tx['TX_AMOUNT'] + 1e-8)
    train_tx['goods_ratio'] = train_tx['TRANSACTION_GOODS_AND_SERVICES_AMOUNT'] / (train_tx['TX_AMOUNT'] + 1e-8)
    train_tx['hour'] = train_tx['TX_TS'].dt.hour
    train_tx['is_weekend'] = (train_tx['TX_TS'].dt.dayofweek >= 5).astype(int)
    train_tx['is_night'] = ((train_tx['hour'] >= 22) | (train_tx['hour'] <= 6)).astype(int)
    
    # Add engineered features
    engineered_features = train_tx[['amount_log', 'cashback_ratio', 'goods_ratio', 'is_weekend', 'is_night']]
    
    # Boolean feature (convert Y/N to 1/0)
    recurring = (train_tx['IS_RECURRING_TRANSACTION'] == 'Y').astype(int)
    
    # Categorical features (one-hot encoded)
    cat_features = []
    for col in ['TRANSACTION_TYPE', 'TRANSACTION_STATUS', 'FAILURE_CODE', 'ACQUIRER_ID']:
        if col in train_tx.columns:
            dummies = pd.get_dummies(train_tx[col].fillna('unknown'), prefix=col.lower())
            cat_features.append(dummies)
    
    # Combine all features
    features = [dow_dummies, pd.DataFrame({'time_of_day': time_of_day}), 
                numeric_features, engineered_features, pd.DataFrame({'is_recurring': recurring})]
    features.extend(cat_features)
    
    X = pd.concat(features, axis=1).astype(float)
    y = train_tx['TX_FRAUD']
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numeric features only
    scaler = StandardScaler()
    numeric_cols = ['TX_AMOUNT', 'TRANSACTION_GOODS_AND_SERVICES_AMOUNT', 'TRANSACTION_CASHBACK_AMOUNT']
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    return X_train, y_train, X_test, y_test


# %% Test
xtrain, ytrain, xtest, ytest = load_and_preprocess_data()

# %% Pretty print

BOLD =   f'\033[1m       '
RED =    f'\033[91m{BOLD}'
GREEN =  f'\033[92m{BOLD}'
YELLOW = f'\033[93m{BOLD}'
BLUE =   f'\033[94m{BOLD}'
RESET =  f'\033[0m       '

print(f"\n{RED}----> X train features {xtrain.shape}:{RESET}")
print(xtrain.head())

print(f"\n{GREEN}----> Y train features {ytrain.shape}:{RESET}")
print(ytrain.head())

print(f"\n{YELLOW}----> X test features {xtest.shape}:{RESET}")
print(xtest.head())

print(f"\n{BLUE}----> Y test features {ytest.shape}:{RESET}")
print(ytest.head())