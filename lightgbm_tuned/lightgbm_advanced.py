#!/usr/bin/env python3
# flake8: noqa

# %% Module Import
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")

# %% Advanced Feature Engineering
def create_advanced_features(df, fraudulent_terminals):
    """Advanced feature engineering for higher performance"""
    df = df.copy()
    df["TX_TS"] = pd.to_datetime(df["TX_TS"])
    
    # 1. VELOCITY FEATURES (simplified)
    df = df.sort_values(['CUSTOMER_ID', 'TX_TS'])
    df['cust_tx_velocity_1h'] = df.groupby('CUSTOMER_ID')['TX_ID'].rolling(10, min_periods=1).count().reset_index(0, drop=True) - 1
    df['cust_tx_velocity_24h'] = df.groupby('CUSTOMER_ID')['TX_ID'].rolling(50, min_periods=1).count().reset_index(0, drop=True) - 1
    
    df = df.sort_values(['TERMINAL_ID', 'TX_TS'])
    df['term_tx_velocity_1h'] = df.groupby('TERMINAL_ID')['TX_ID'].rolling(10, min_periods=1).count().reset_index(0, drop=True) - 1
    
    # 2. AMOUNT DEVIATION FEATURES
    df = df.sort_values(['CUSTOMER_ID', 'TX_TS'])
    df['cust_amount_rolling_mean'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(
        lambda x: x.rolling(10, min_periods=1).mean())
    df['cust_amount_deviation'] = np.abs(df['TX_AMOUNT'] - df['cust_amount_rolling_mean']) / (df['cust_amount_rolling_mean'] + 1e-8)
    
    # 3. CARD FEATURES
    df['card_expiry'] = pd.to_datetime(df['CARD_EXPIRY_DATE'], format='%m/%y', errors='coerce')
    # Ensure both are timezone-naive
    if hasattr(df['TX_TS'].dtype, 'tz') and df['TX_TS'].dtype.tz is not None:
        df['TX_TS'] = df['TX_TS'].dt.tz_localize(None)
    if hasattr(df['card_expiry'].dtype, 'tz') and df['card_expiry'].dtype.tz is not None:
        df['card_expiry'] = df['card_expiry'].dt.tz_localize(None)
    df['card_age_days'] = (df['card_expiry'] - df['TX_TS']).dt.days.fillna(365)
    df['card_near_expiry'] = (df['card_age_days'] <= 90).astype(int)
    
    # 4. MERCHANT RISK FEATURES
    df['merchant_foundation'] = pd.to_datetime(df['FOUNDATION_DATE'], errors='coerce').dt.tz_localize(None)
    df['merchant_age_days'] = (df['TX_TS'] - df['merchant_foundation']).dt.days.fillna(1000)
    df['merchant_young'] = (df['merchant_age_days'] <= 365).astype(int)
    
    # 5. CROSS-ENTITY FEATURES
    df['amount_vs_merchant_avg'] = df['TX_AMOUNT'] / (df['AVERAGE_TICKET_SALE_AMOUNT'] + 1e-8)
    df['amount_percentile_merchant'] = df.groupby('MERCHANT_ID')['TX_AMOUNT'].transform(
        lambda x: x.rank(pct=True))
    
    # 6. TIME-BASED RISK FEATURES
    df['is_business_hours'] = ((df['TX_TS'].dt.hour >= 9) & (df['TX_TS'].dt.hour <= 17)).astype(int)
    df['tx_hour_risk'] = df['TX_TS'].dt.hour.map({
        0:3, 1:3, 2:3, 3:3, 4:2, 5:1, 6:1, 7:1, 8:1, 9:0, 10:0, 11:0, 12:0,
        13:0, 14:0, 15:0, 16:0, 17:0, 18:1, 19:1, 20:1, 21:2, 22:2, 23:3
    })
    
    # 7. ENHANCED FRAUD INDICATORS
    df['is_fraudulent_terminal'] = df['TERMINAL_ID'].isin(fraudulent_terminals).astype(int)
    df['is_fraud_amount_range'] = ((df['TX_AMOUNT'] >= 500) & (df['TX_AMOUNT'] <= 1000)).astype(int)
    df['is_ron_currency'] = (df['TRANSACTION_CURRENCY'] == 'RON').astype(int)
    df['is_ro_country'] = (df['CARD_COUNTRY_CODE'] == 'RO').astype(int)
    df['high_risk_combo'] = df['is_ron_currency'] * df['is_ro_country'] * df['is_fraud_amount_range']
    
    # 8. ORIGINAL CORE FEATURES
    df["hour"] = df["TX_TS"].dt.hour
    df["day_of_week"] = df["TX_TS"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
    df["time_of_day"] = (df["TX_TS"].dt.hour * 3600 + df["TX_TS"].dt.minute * 60 + df["TX_TS"].dt.second) / 86400
    
    df["distance"] = np.sqrt((df["x_customer_id"] - df["x_terminal_id"]) ** 2 + (df["y_customer_id"] - df["y_terminal__id"]) ** 2)
    df["distance_tx"] = df["distance"] * df["TX_AMOUNT"]
    df["cashback_ratio"] = df["TRANSACTION_CASHBACK_AMOUNT"] / (df["TX_AMOUNT"] + 1e-8)
    df["goods_ratio"] = df["TRANSACTION_GOODS_AND_SERVICES_AMOUNT"] / (df["TX_AMOUNT"] + 1e-8)
    
    for col in ["TX_AMOUNT", "TRANSACTION_GOODS_AND_SERVICES_AMOUNT", "TRANSACTION_CASHBACK_AMOUNT"]:
        df[col + "_log"] = np.log1p(df[col])
    
    df["tax_exempt"] = df["TAX_EXCEMPT_INDICATOR"].fillna(False).astype(int)
    df["annual_turnover_log"] = np.log1p(df["ANNUAL_TURNOVER"].fillna(0))
    df["avg_ticket_log"] = np.log1p(df["AVERAGE_TICKET_SALE_AMOUNT"].fillna(0))
    
    mcc_freq = df["MCC_CODE"].value_counts(normalize=True)
    df["mcc_frequency"] = df["MCC_CODE"].map(mcc_freq).fillna(0)
    df["is_recurring"] = (df["IS_RECURRING_TRANSACTION"] == "Y").astype(int)
    
    return df

# %% Enhanced Historical Features
def add_enhanced_historical_features(df, reference_df=None):
    """Enhanced historical features with more granular statistics"""
    ref = reference_df if reference_df is not None else df
    
    # Basic counts and means
    for entity, id_col in [('cust', 'CUSTOMER_ID'), ('term', 'TERMINAL_ID'), ('merch', 'MERCHANT_ID')]:
        df[f"{entity}_tx_count"] = df[id_col].map(ref.groupby(id_col)["TX_ID"].count())
        df[f"{entity}_tx_mean"] = df[id_col].map(ref.groupby(id_col)["TX_AMOUNT"].mean())
        df[f"{entity}_tx_std"] = df[id_col].map(ref.groupby(id_col)["TX_AMOUNT"].std())
        df[f"{entity}_tx_median"] = df[id_col].map(ref.groupby(id_col)["TX_AMOUNT"].median())
        
        if "TX_FRAUD" in ref.columns:
            df[f"{entity}_fraud_rate"] = df[id_col].map(ref.groupby(id_col)["TX_FRAUD"].mean())
            df[f"{entity}_fraud_count"] = df[id_col].map(ref.groupby(id_col)["TX_FRAUD"].sum())
        else:
            df[f"{entity}_fraud_rate"] = 0
            df[f"{entity}_fraud_count"] = 0
    
    # Customer/terminal statistics
    customer_stats = ref.groupby("CUSTOMER_ID")["TX_AMOUNT"].agg(["mean", "std", "min", "max"])
    customer_stats.columns = ["customer_avg_tx", "customer_std_tx", "customer_min_tx", "customer_max_tx"]
    df = df.merge(customer_stats, on="CUSTOMER_ID", how="left")
    
    terminal_stats = ref.groupby("TERMINAL_ID")["TX_AMOUNT"].agg(["mean", "std", "min", "max"])
    terminal_stats.columns = ["terminal_avg_tx", "terminal_std_tx", "terminal_min_tx", "terminal_max_tx"]
    df = df.merge(terminal_stats, on="TERMINAL_ID", how="left")
    
    df["tx_to_turnover_ratio"] = df["TX_AMOUNT"] / (df["ANNUAL_TURNOVER_CARD"] + 1e-8)
    
    df.fillna(0, inplace=True)
    return df

# %% Advanced Lag Features
def add_advanced_lag_features(df):
    """Advanced temporal lag features"""
    df = df.sort_values("TX_TS")
    df["TX_TS_unix"] = df["TX_TS"].astype(np.int64) // 10**9

    # Time gaps
    df["cust_prev_tx_time"] = df.groupby("CUSTOMER_ID")["TX_TS_unix"].diff().fillna(0)
    df["term_prev_tx_time"] = df.groupby("TERMINAL_ID")["TX_TS_unix"].diff().fillna(0)
    
    # Rolling statistics
    for window in [3, 5, 10]:
        df[f"cust_last{window}_tx_sum"] = (
            df.groupby("CUSTOMER_ID")["TX_AMOUNT"].rolling(window, min_periods=1).sum().reset_index(0, drop=True))
        df[f"cust_last{window}_tx_mean"] = (
            df.groupby("CUSTOMER_ID")["TX_AMOUNT"].rolling(window, min_periods=1).mean().reset_index(0, drop=True))
        df[f"term_last{window}_tx_sum"] = (
            df.groupby("TERMINAL_ID")["TX_AMOUNT"].rolling(window, min_periods=1).sum().reset_index(0, drop=True))
    
    return df

# %% Optimized Target Encoding with Smoothing
def smoothed_target_encoding(train, y_col, cat_cols, n_splits=5, alpha=10):
    """Target encoding with Bayesian smoothing"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_encoded = train.copy()
    global_mean = train[y_col].mean()
    
    for col in cat_cols:
        train_encoded[col + "_te"] = 0
        for train_idx, val_idx in skf.split(train, train[y_col]):
            fold_stats = train.iloc[train_idx].groupby(col)[y_col].agg(['count', 'mean'])
            # Bayesian smoothing
            smoothed_means = (fold_stats['count'] * fold_stats['mean'] + alpha * global_mean) / (fold_stats['count'] + alpha)
            train_encoded.iloc[val_idx, train_encoded.columns.get_loc(col + "_te")] = (
                train.iloc[val_idx][col].map(smoothed_means).fillna(global_mean))
        
    return train_encoded

# %% Hyperparameter Optimization
def get_optimized_params(X_tr, y_tr):
    """Get optimized LightGBM parameters"""
    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "num_leaves": 128,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.05,
        "lambda_l2": 0.05,
        "min_gain_to_split": 0.01,
        "scale_pos_weight": n_neg / n_pos,
        "seed": 42,
        "verbose": -1,
        "force_row_wise": True
    }

# %% Advanced Training
def train_advanced_lightgbm(train_data, test_data, fraudulent_terminals, n_splits=5):
    """Advanced training with ensemble"""
    
    # Feature selection
    num_features = [
        "TX_AMOUNT_log", "TRANSACTION_GOODS_AND_SERVICES_AMOUNT_log", "TRANSACTION_CASHBACK_AMOUNT_log",
        "x_customer_id", "y_customer_id", "x_terminal_id", "y_terminal__id", "distance", "distance_tx",
        "hour", "day_of_week", "is_weekend", "is_night", "time_of_day", "tx_hour_risk", "is_business_hours",
        "cashback_ratio", "goods_ratio", "tax_exempt", "annual_turnover_log", "avg_ticket_log",
        "mcc_frequency", "is_recurring", "ANNUAL_TURNOVER_CARD", "ANNUAL_TURNOVER", "AVERAGE_TICKET_SALE_AMOUNT",
        "customer_avg_tx", "customer_std_tx", "customer_min_tx", "customer_max_tx",
        "terminal_avg_tx", "terminal_std_tx", "terminal_min_tx", "terminal_max_tx", "tx_to_turnover_ratio",
        "cust_tx_count", "term_tx_count", "merch_tx_count", "cust_tx_mean", "term_tx_mean", "merch_tx_mean",
        "cust_tx_std", "term_tx_std", "merch_tx_std", "cust_tx_median", "term_tx_median", "merch_tx_median",
        "cust_fraud_rate", "term_fraud_rate", "merch_fraud_rate", "cust_fraud_count", "term_fraud_count", "merch_fraud_count",
        "cust_prev_tx_time", "term_prev_tx_time",
        "cust_last3_tx_sum", "cust_last3_tx_mean", "cust_last5_tx_sum", "cust_last5_tx_mean", "cust_last10_tx_sum", "cust_last10_tx_mean",
        "term_last3_tx_sum", "term_last5_tx_sum", "term_last10_tx_sum",
        "cust_tx_velocity_1h", "cust_tx_velocity_24h", "term_tx_velocity_1h",
        "cust_amount_rolling_mean", "cust_amount_deviation", "card_age_days", "card_near_expiry",
        "merchant_age_days", "merchant_young", "amount_vs_merchant_avg", "amount_percentile_merchant",
        "is_fraudulent_terminal", "is_fraud_amount_range", "is_ron_currency", "is_ro_country", "high_risk_combo"
    ]
    
    cat_features = ["CARD_BRAND", "TRANSACTION_TYPE", "TRANSACTION_STATUS", "TRANSACTION_CURRENCY", 
                   "CARD_COUNTRY_CODE", "IS_RECURRING_TRANSACTION", "BUSINESS_TYPE", "OUTLET_TYPE"]
    
    # Prepare data
    feature_cols = [f for f in num_features if f in train_data.columns] + [f + "_te" for f in cat_features]
    X = train_data[feature_cols].values
    y = train_data["TX_FRAUD"].values
    X_test = test_data[feature_cols].values
    ids_test = test_data["TX_ID"].values
    
    # Cross-validation with multiple models
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    preds_test = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X))
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"Training fold {fold+1}/{n_splits}...")
        
        X_tr, X_val = X[train_idx], X[valid_idx]
        y_tr, y_val = y[train_idx], y[valid_idx]
        
        # Get optimized parameters
        params = get_optimized_params(X_tr, y_tr)
        
        # Create datasets
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        # Train model
        model = lgb.train(
            params, dtrain, num_boost_round=5000,
            valid_sets=[dtrain, dvalid],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
        )
        
        # OOF predictions
        oof_preds[valid_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Test predictions
        fold_pred = model.predict(X_test, num_iteration=model.best_iteration)
        
        # Override fraudulent terminals with high confidence
        fraud_mask = test_data['is_fraudulent_terminal'] == 1
        fold_pred[fraud_mask] = np.maximum(fold_pred[fraud_mask], 0.95)
        
        preds_test += fold_pred / n_splits
        
        # Save model
        model.save_model(f'lightgbm_advanced_fold_{fold+1}.txt')
    

    
    return preds_test, ids_test, oof_preds

# %% Main Pipeline
def main():
    """Advanced main pipeline"""
    print("Starting Advanced LightGBM Pipeline")
    print("=" * 50)
    
    # Load data
    customers = pd.read_csv("../data/Payments Fraud DataSet/customers.csv")
    terminals = pd.read_csv("../data/Payments Fraud DataSet/terminals.csv")
    merchants = pd.read_csv("../data/Payments Fraud DataSet/merchants.csv")
    train_tx = pd.read_csv("../data/Payments Fraud DataSet/transactions_train.csv")
    test_tx = pd.read_csv("../data/Payments Fraud DataSet/transactions_test.csv")
    
    # Merge datasets
    train_data = (train_tx.merge(customers, on="CUSTOMER_ID", how="left")
                 .merge(terminals, on="TERMINAL_ID", how="left")
                 .merge(merchants, on="MERCHANT_ID", how="left"))
    test_data = (test_tx.merge(customers, on="CUSTOMER_ID", how="left")
                .merge(terminals, on="TERMINAL_ID", how="left")
                .merge(merchants, on="MERCHANT_ID", how="left"))
    
    # Detect fraudulent terminals
    fraudulent_terminals = train_data.groupby('TERMINAL_ID')['TX_FRAUD'].agg(['count', 'mean'])
    fraudulent_terminals = fraudulent_terminals[(fraudulent_terminals['count'] >= 5) & (fraudulent_terminals['mean'] >= 1.0)].index.tolist()
    print(f"Found {len(fraudulent_terminals)} fraudulent terminals")
    
    # Advanced feature engineering
    print("Creating advanced features...")
    train_data = create_advanced_features(train_data, fraudulent_terminals)
    test_data = create_advanced_features(test_data, fraudulent_terminals)
    
    train_data = add_enhanced_historical_features(train_data)
    test_data = add_enhanced_historical_features(test_data, reference_df=train_data)
    
    train_data = add_advanced_lag_features(train_data)
    test_data = add_advanced_lag_features(test_data)
    
    # Target encoding
    cat_features = ["CARD_BRAND", "TRANSACTION_TYPE", "TRANSACTION_STATUS", "TRANSACTION_CURRENCY", 
                   "CARD_COUNTRY_CODE", "IS_RECURRING_TRANSACTION", "BUSINESS_TYPE", "OUTLET_TYPE"]
    
    print("Applying smoothed target encoding...")
    train_data = smoothed_target_encoding(train_data, "TX_FRAUD", cat_features)
    for col in cat_features:
        test_data[col + "_te"] = test_data[col].map(
            train_data.groupby(col)[col + "_te"].mean()).fillna(train_data["TX_FRAUD"].mean())
    
    # Train advanced model
    print("Training advanced model...")
    test_predictions, ids_test, oof_preds = train_advanced_lightgbm(train_data, test_data, fraudulent_terminals)
    
    # Evaluate
    from sklearn.metrics import roc_auc_score
    oof_auc = roc_auc_score(train_data['TX_FRAUD'], oof_preds)
    print(f"Out-of-fold AUC: {oof_auc:.6f}")
    
    # Save submission
    submission = pd.DataFrame({"TX_ID": ids_test, "TX_FRAUD": test_predictions})
    submission.to_csv("lightgbm_advanced_predictions.csv", index=False)
    print(f"Advanced predictions saved: {submission.shape[0]} rows")
    
    return submission

if __name__ == "__main__":
    submission = main()