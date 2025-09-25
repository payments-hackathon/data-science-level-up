#!/usr/bin/env python3
# flake8: noqa

# %% Module Import
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pickle
import warnings

warnings.filterwarnings("ignore")

# %% Data Loading and Preprocessing
def load_and_preprocess_data():
    """Load and merge all datasets"""
    print("Loading datasets...")
    customers = pd.read_csv("../data/Payments Fraud DataSet/customers.csv")
    terminals = pd.read_csv("../data/Payments Fraud DataSet/terminals.csv")
    merchants = pd.read_csv("../data/Payments Fraud DataSet/merchants.csv")
    train_tx = pd.read_csv("../data/Payments Fraud DataSet/transactions_train.csv")
    test_tx = pd.read_csv("../data/Payments Fraud DataSet/transactions_test.csv")
    
    print(f"Loaded {len(train_tx):,} training transactions")
    print(f"Loaded {len(test_tx):,} test transactions")
    
    # Merge datasets
    train_data = (train_tx.merge(customers, on="CUSTOMER_ID", how="left")
                 .merge(terminals, on="TERMINAL_ID", how="left")
                 .merge(merchants, on="MERCHANT_ID", how="left"))
    
    test_data = (test_tx.merge(customers, on="CUSTOMER_ID", how="left")
                .merge(terminals, on="TERMINAL_ID", how="left")
                .merge(merchants, on="MERCHANT_ID", how="left"))
    
    return train_data, test_data

# %% Fraudulent Terminal Detection
def detect_fraudulent_terminals(train_data, threshold=1.0):
    """Detect terminals with 100% fraud rate"""
    terminal_stats = train_data.groupby('TERMINAL_ID')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
    terminal_stats = terminal_stats[terminal_stats['count'] >= 5]  # Min 5 transactions
    fraudulent_terminals = terminal_stats[terminal_stats['mean'] >= threshold].index.tolist()
    
    print(f"Found {len(fraudulent_terminals)} terminals with 100% fraud rate")
    return fraudulent_terminals

# %% Feature Engineering (from original high-performing model)
def engineer_features(df, fraudulent_terminals):
    """Enhanced feature engineering combining original + new features"""
    df = df.copy()
    df["TX_TS"] = pd.to_datetime(df["TX_TS"])
    
    # Fraudulent terminal flag (NEW)
    df['is_fraudulent_terminal'] = df['TERMINAL_ID'].isin(fraudulent_terminals).astype(int)
    
    # Basic time features (ORIGINAL)
    df["hour"] = df["TX_TS"].dt.hour
    df["day_of_week"] = df["TX_TS"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
    
    # Time of day as continuous feature (ORIGINAL)
    df["time_of_day"] = (df["TX_TS"].dt.hour * 3600 + 
                         df["TX_TS"].dt.minute * 60 + 
                         df["TX_TS"].dt.second) / 86400
    
    # Distance features (ORIGINAL)
    df["distance"] = np.sqrt(
        (df["x_customer_id"] - df["x_terminal_id"]) ** 2
        + (df["y_customer_id"] - df["y_terminal__id"]) ** 2
    )
    df["distance_tx"] = df["distance"] * df["TX_AMOUNT"]
    
    # Amount ratios (ORIGINAL)
    df["cashback_ratio"] = df["TRANSACTION_CASHBACK_AMOUNT"] / (df["TX_AMOUNT"] + 1e-8)
    df["goods_ratio"] = df["TRANSACTION_GOODS_AND_SERVICES_AMOUNT"] / (df["TX_AMOUNT"] + 1e-8)
    
    # Log transformations (ORIGINAL)
    for col in ["TX_AMOUNT", "TRANSACTION_GOODS_AND_SERVICES_AMOUNT", "TRANSACTION_CASHBACK_AMOUNT"]:
        df[col + "_log"] = np.log1p(df[col])
    
    # Merchant features (ORIGINAL)
    df["tax_exempt"] = df["TAX_EXCEMPT_INDICATOR"].fillna(False).astype(int)
    df["annual_turnover_log"] = np.log1p(df["ANNUAL_TURNOVER"].fillna(0))
    df["avg_ticket_log"] = np.log1p(df["AVERAGE_TICKET_SALE_AMOUNT"].fillna(0))
    
    # MCC frequency encoding (ORIGINAL)
    mcc_freq = df["MCC_CODE"].value_counts(normalize=True)
    df["mcc_frequency"] = df["MCC_CODE"].map(mcc_freq).fillna(0)
    
    # Recurring transaction feature (ORIGINAL)
    df["is_recurring"] = (df["IS_RECURRING_TRANSACTION"] == "Y").astype(int)
    
    # Customer/terminal statistics (ORIGINAL)
    customer_stats = (
        df.groupby("CUSTOMER_ID")["TX_AMOUNT"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "customer_avg_tx", "std": "customer_std_tx"})
    )
    df = df.merge(customer_stats, on="CUSTOMER_ID", how="left")
    
    terminal_stats = (
        df.groupby("TERMINAL_ID")["TX_AMOUNT"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "terminal_avg_tx", "std": "terminal_std_tx"})
    )
    df = df.merge(terminal_stats, on="TERMINAL_ID", how="left")
    
    df["tx_to_turnover_ratio"] = df["TX_AMOUNT"] / (df["ANNUAL_TURNOVER_CARD"] + 1e-8)
    
    # NEW: Enhanced fraud-specific features
    df['is_fraud_amount_range'] = ((df['TX_AMOUNT'] >= 500) & (df['TX_AMOUNT'] <= 1000)).astype(int)
    df['is_ron_currency'] = (df['TRANSACTION_CURRENCY'] == 'RON').astype(int)
    df['is_ro_country'] = (df['CARD_COUNTRY_CODE'] == 'RO').astype(int)
    df['ron_ro_combined'] = df['is_ron_currency'] * df['is_ro_country']
    
    return df

# %% Historical and Lag Features (CRITICAL from original)
def add_historical_features(df, reference_df=None):
    """Add historical aggregation features"""
    ref = reference_df if reference_df is not None else df
    
    # Transaction counts
    df["cust_tx_count"] = df["CUSTOMER_ID"].map(
        ref.groupby("CUSTOMER_ID")["TX_ID"].count()
    )
    df["term_tx_count"] = df["TERMINAL_ID"].map(
        ref.groupby("TERMINAL_ID")["TX_ID"].count()
    )
    df["merch_tx_count"] = df["MERCHANT_ID"].map(
        ref.groupby("MERCHANT_ID")["TX_ID"].count()
    )

    # Mean amounts
    df["cust_tx_mean"] = df["CUSTOMER_ID"].map(
        ref.groupby("CUSTOMER_ID")["TX_AMOUNT"].mean()
    )
    df["term_tx_mean"] = df["TERMINAL_ID"].map(
        ref.groupby("TERMINAL_ID")["TX_AMOUNT"].mean()
    )
    df["merch_tx_mean"] = df["MERCHANT_ID"].map(
        ref.groupby("MERCHANT_ID")["TX_AMOUNT"].mean()
    )

    # Fraud rate per entity
    if "TX_FRAUD" in ref.columns:
        df["cust_fraud_rate"] = df["CUSTOMER_ID"].map(
            ref.groupby("CUSTOMER_ID")["TX_FRAUD"].mean()
        )
        df["term_fraud_rate"] = df["TERMINAL_ID"].map(
            ref.groupby("TERMINAL_ID")["TX_FRAUD"].mean()
        )
        df["merch_fraud_rate"] = df["MERCHANT_ID"].map(
            ref.groupby("MERCHANT_ID")["TX_FRAUD"].mean()
        )
    else:
        df["cust_fraud_rate"] = 0
        df["term_fraud_rate"] = 0
        df["merch_fraud_rate"] = 0

    # Fill missing
    df.fillna(0, inplace=True)
    return df

def add_lag_features(df):
    """Add temporal lag features"""
    df = df.sort_values("TX_TS")
    df["TX_TS_unix"] = df["TX_TS"].astype(np.int64) // 10**9

    # Time since previous transaction
    df["cust_prev_tx_time"] = df.groupby("CUSTOMER_ID")["TX_TS_unix"].diff().fillna(0)
    df["term_prev_tx_time"] = df.groupby("TERMINAL_ID")["TX_TS_unix"].diff().fillna(0)

    # Rolling sum of last 3 transactions
    df["cust_last3_tx_sum"] = (
        df.groupby("CUSTOMER_ID")["TX_AMOUNT"]
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(0, drop=True) 
    )
    df["term_last3_tx_sum"] = (
        df.groupby("TERMINAL_ID")["TX_AMOUNT"]
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(0, drop=True)
    )
    return df

# %% K-Fold Target Encoding (CRITICAL from original)
def kfold_target_encoding(train, y_col, cat_cols, n_splits=5):
    """K-fold target encoding for categorical features"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_encoded = train.copy()
    for col in cat_cols:
        train_encoded[col + "_te"] = 0
        for train_idx, val_idx in skf.split(train, train[y_col]):
            fold_mean = train.iloc[train_idx].groupby(col)[y_col].mean()
            train_encoded.iloc[val_idx, train_encoded.columns.get_loc(col + "_te")] = (
                train.iloc[val_idx][col].map(fold_mean)
            )
        train_encoded[col + "_te"].fillna(train[y_col].mean(), inplace=True)
    return train_encoded

# %% Enhanced LightGBM Training with Original Parameters
def train_lightgbm_enhanced(train_data, test_data, fraudulent_terminals, n_splits=5):
    """Train enhanced LightGBM with original high-performing features"""
    
    # Define features (from original model + enhancements)
    num_features = [
        "TX_AMOUNT_log",
        "TRANSACTION_GOODS_AND_SERVICES_AMOUNT_log",
        "TRANSACTION_CASHBACK_AMOUNT_log",
        "x_customer_id",
        "y_customer_id",
        "x_terminal_id",
        "y_terminal__id",
        "distance",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_night",
        "time_of_day",
        "cashback_ratio",
        "goods_ratio",
        "tax_exempt",
        "annual_turnover_log",
        "avg_ticket_log",
        "mcc_frequency",
        "is_recurring",
        "ANNUAL_TURNOVER_CARD",
        "ANNUAL_TURNOVER",
        "AVERAGE_TICKET_SALE_AMOUNT",
        "customer_avg_tx",
        "customer_std_tx",
        "terminal_avg_tx",
        "terminal_std_tx",
        "tx_to_turnover_ratio",
        "distance_tx",
        "cust_tx_count",
        "term_tx_count",
        "merch_tx_count",
        "cust_tx_mean",
        "term_tx_mean",
        "merch_tx_mean",
        "cust_fraud_rate",
        "term_fraud_rate",
        "merch_fraud_rate",
        "cust_prev_tx_time",
        "term_prev_tx_time",
        "cust_last3_tx_sum",
        "term_last3_tx_sum",
        # NEW enhanced features
        "is_fraudulent_terminal",
        "is_fraud_amount_range",
        "is_ron_currency",
        "is_ro_country",
        "ron_ro_combined"
    ]
    
    cat_features = [
        "CARD_BRAND",
        "TRANSACTION_TYPE",
        "TRANSACTION_STATUS",
        "TRANSACTION_CURRENCY",
        "CARD_COUNTRY_CODE",
        "IS_RECURRING_TRANSACTION",
        "BUSINESS_TYPE",
        "OUTLET_TYPE",
    ]
    
    # Prepare features
    X = train_data[num_features + [f + "_te" for f in cat_features]].values
    y = train_data["TX_FRAUD"].values
    X_test = test_data[num_features + [f + "_te" for f in cat_features]].values
    ids_test = test_data["TX_ID"].values
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    preds_test = np.zeros(len(X_test))
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"Training fold {fold+1}/{n_splits}...")
        
        X_tr, X_val = X[train_idx], X[valid_idx]
        y_tr, y_val = y[train_idx], y[valid_idx]
        
        # Class imbalance handling
        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos
        
        # Original high-performing parameters
        params = {
            "objective": "binary", 
            "metric": "auc",         
            "boosting_type": "gbdt",
            "learning_rate": 0.01,  
            "num_leaves": 128,      
            "min_data_in_leaf": 30,  
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8, 
            "bagging_freq": 5,  
            "scale_pos_weight": n_neg / n_pos,  
            "seed": 42,            
            "verbose": -1,          
        }
        
        # Create datasets
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        # Train model
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,               
            valid_sets=[dtrain, dvalid],         
            callbacks=[lgb.early_stopping(100), 
                       lgb.log_evaluation(200)], 
        )
        
        # Predict on test
        fold_pred = model.predict(X_test, num_iteration=model.best_iteration)
        
        # Override fraudulent terminals
        fraud_mask = test_data['is_fraudulent_terminal'] == 1
        fold_pred[fraud_mask] = 1.0
        
        preds_test += fold_pred / n_splits
        
        # Save model checkpoint
        model.save_model(f'lightgbm_enhanced_fold_{fold+1}.txt')
    
    return preds_test, ids_test

# %% Visualization
def plot_fraud_analysis(train_data):
    """Plot fraud analysis charts"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Amount distribution
    fraud_amounts = train_data[train_data['TX_FRAUD'] == 1]['TX_AMOUNT']
    legit_amounts = train_data[train_data['TX_FRAUD'] == 0]['TX_AMOUNT']
    
    axes[0,0].hist(legit_amounts, bins=50, alpha=0.7, label='Legitimate', density=True)
    axes[0,0].hist(fraud_amounts, bins=50, alpha=0.7, label='Fraudulent', density=True)
    axes[0,0].set_xlabel('Transaction Amount')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_title('Amount Distribution')
    axes[0,0].legend()
    axes[0,0].set_xlim(0, 2000)
    
    # Currency fraud rates
    currency_fraud = train_data.groupby('TRANSACTION_CURRENCY')['TX_FRAUD'].agg(['count', 'mean'])
    currency_fraud = currency_fraud[currency_fraud['count'] >= 100]
    currency_fraud['mean'].plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Fraud Rate by Currency')
    axes[0,1].set_ylabel('Fraud Rate')
    
    # Country fraud rates
    country_fraud = train_data.groupby('CARD_COUNTRY_CODE')['TX_FRAUD'].agg(['count', 'mean'])
    country_fraud = country_fraud[country_fraud['count'] >= 100]
    country_fraud['mean'].plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Fraud Rate by Country')
    axes[1,0].set_ylabel('Fraud Rate')
    
    # Terminal fraud distribution
    terminal_fraud = train_data.groupby('TERMINAL_ID')['TX_FRAUD'].mean()
    axes[1,1].hist(terminal_fraud, bins=50, alpha=0.7)
    axes[1,1].set_xlabel('Terminal Fraud Rate')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_title('Terminal Fraud Rate Distribution')
    
    plt.tight_layout()
    plt.savefig('fraud_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# %% Main Execution Pipeline
"""Enhanced main execution pipeline"""
print("Starting Enhanced LightGBM Fraud Detection Pipeline")
print("=" * 60)

# Load and preprocess data
train_data, test_data = load_and_preprocess_data()

# Detect fraudulent terminals
fraudulent_terminals = detect_fraudulent_terminals(train_data)

# Feature engineering
print("Engineering features...")
train_data = engineer_features(train_data, fraudulent_terminals)
test_data = engineer_features(test_data, fraudulent_terminals)

# Add historical features (CRITICAL)
print("Adding historical features...")
train_data = add_historical_features(train_data)
test_data = add_historical_features(test_data, reference_df=train_data)

# Add lag features (CRITICAL)
print("Adding lag features...")
train_data = add_lag_features(train_data)
test_data = add_lag_features(test_data)

# Define categorical features
cat_features = [
    "CARD_BRAND",
    "TRANSACTION_TYPE",
    "TRANSACTION_STATUS",
    "TRANSACTION_CURRENCY",
    "CARD_COUNTRY_CODE",
    "IS_RECURRING_TRANSACTION",
    "BUSINESS_TYPE",
    "OUTLET_TYPE",
]

# K-Fold target encoding (CRITICAL)
print("Applying K-Fold target encoding...")
train_data = kfold_target_encoding(train_data, "TX_FRAUD", cat_features)
for col in cat_features:
    test_data[col + "_te"] = (
        test_data[col]
        .map(train_data.groupby(col)[col + "_te"].mean())
        .fillna(train_data["TX_FRAUD"].mean())
    )

# Plot analysis
plot_fraud_analysis(train_data)

# Train enhanced model
print("Training enhanced LightGBM model...")
test_predictions, ids_test = train_lightgbm_enhanced(train_data, test_data, fraudulent_terminals)

# Save submission
submission = pd.DataFrame({"TX_ID": ids_test, "TX_FRAUD": test_predictions})
submission.to_csv("lightgbm_enhanced_predictions.csv", index=False)
print(f"Enhanced predictions saved: {submission.shape[0]} rows")

# Save artifacts
with open('enhanced_model_artifacts.pkl', 'wb') as f:
    pickle.dump({
        'fraudulent_terminals': fraudulent_terminals,
        'cat_features': cat_features
    }, f)

print("Enhanced pipeline completed successfully!")