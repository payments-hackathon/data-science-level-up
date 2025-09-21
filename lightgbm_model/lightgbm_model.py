#!/usr/bin/env python3
# flake8: noqa

# %% Module Imports and Data Import
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import early_stopping, log_evaluation
import warnings
warnings.filterwarnings('ignore')

# from data import load_and_preprocess_data
def load_and_preprocess_data():
    # Load datasets
    customers = pd.read_csv('../data/Payments Fraud DataSet/customers.csv')
    terminals = pd.read_csv('../data/Payments Fraud DataSet/terminals.csv')
    merchants = pd.read_csv('../data/Payments Fraud DataSet/merchants.csv')
    train_tx = pd.read_csv('../data/Payments Fraud DataSet/transactions_train.csv')
    
    # Merge transaction data with customer and terminal info
    data = train_tx.merge(customers, on='CUSTOMER_ID', how='left')
    data = data.merge(terminals, on='TERMINAL_ID', how='left')
    data = data.merge(merchants, on='MERCHANT_ID', how='left')
    
    # Feature engineering
    def engineer_features(df):
        df['distance'] = np.sqrt(
            (df['x_customer_id'] - df['x_terminal_id'])**2 +
            (df['y_customer_id'] - df['y_terminal__id'])**2
        )
        df['TX_TS'] = pd.to_datetime(df['TX_TS'])
        df['hour'] = df['TX_TS'].dt.hour
        df['day_of_week'] = df['TX_TS'].dt.dayofweek
        df['cashback_ratio'] = df['TRANSACTION_CASHBACK_AMOUNT'] / (df['TX_AMOUNT'] + 1e-8)
        df['goods_ratio'] = df['TRANSACTION_GOODS_AND_SERVICES_AMOUNT'] / (df['TX_AMOUNT'] + 1e-8)
        return df
    
    data = engineer_features(data)
    
    # Numerical features
    num_features = [
        'TX_AMOUNT', 'TRANSACTION_GOODS_AND_SERVICES_AMOUNT', 'TRANSACTION_CASHBACK_AMOUNT',
        'x_customer_id', 'y_customer_id', 'x_terminal_id', 'y_terminal_id', 'distance',
        'hour', 'day_of_week', 'cashback_ratio', 'goods_ratio',
        'ANNUAL_TURNOVER_CARD', 'ANNUAL_TURNOVER', 'AVERAGE_TICKET_SALE_AMOUNT'
    ]
    
    # Categorical features
    cat_features = [
        'CARD_BRAND', 'TRANSACTION_TYPE', 'TRANSACTION_STATUS', 'TRANSACTION_CURRENCY',
        'CARD_COUNTRY_CODE', 'IS_RECURRING_TRANSACTION', 'BUSINESS_TYPE', 'OUTLET_TYPE'
    ]
    
    # Handle missing values
    for col in num_features:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    
    # Encode categorical variables
    for col in cat_features:
        if col in data.columns:
            if data[col].dtype == 'bool':
                data[col] = data[col].astype(int)  # True->1, False->0
            else:
                le = LabelEncoder()
                data[col] = data[col].fillna('unknown').astype(str)
                data[col] = le.fit_transform(data[col])
    
    # Final features
    feature_cols = [col for col in num_features + cat_features if col in data.columns]
    
    X = data[feature_cols].values
    y = data['TX_FRAUD'].values
    ids = data['TRANSACTION_ID'].values if 'TRANSACTION_ID' in data.columns else np.arange(len(data))
    
    # Split into train/test
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test, ids_test


# %% LightGBM Training

def train_lightgbm(X_train, y_train, X_test, y_test, ids_test):
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dtrain, dvalid],
        callbacks=[
            log_evaluation(period=50)
        ]
    )

    preds = model.predict(X_test, num_iteration=model.best_iteration)

    output_df = pd.DataFrame({
        "TRANSACTION_ID": ids_test,
        "prediction": preds,
        "actual": y_test
    })
    output_df.to_csv("lightgbm_predictions.csv", index=False)
    print("LightGBM predictions saved to lightgbm_predictions.csv")

    # Predict on train
    preds_train = model.predict(X_train, num_iteration=model.best_iteration)
    train_auc = roc_auc_score(y_train, preds_train)
    print(f"LightGBM Train AUC: {train_auc:.4f}")

    # Predict on test
    preds_test = model.predict(X_test, num_iteration=model.best_iteration)
    test_auc = roc_auc_score(y_test, preds_test)
    print(f"LightGBM Test AUC: {test_auc:.4f}")


    return model

# %% Main
if __name__ == "__main__":
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test, ids_test = load_and_preprocess_data()

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Fraud rate: {y_train.mean():.4f}")

    print("Training LightGBM model...")
    lgb_model = train_lightgbm(X_train, y_train, X_test, y_test, ids_test)
