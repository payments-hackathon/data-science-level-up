#!/usr/bin/env python3
# flake8: noqa

# %% Module Import
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")


# %% Load and preprocess data
def load_and_preprocess_data():
    # Load datasets
    customers = pd.read_csv("../data/Payments Fraud DataSet/customers.csv")
    terminals = pd.read_csv("../data/Payments Fraud DataSet/terminals.csv")
    merchants = pd.read_csv("../data/Payments Fraud DataSet/merchants.csv")
    train_tx = pd.read_csv("../data/Payments Fraud DataSet/transactions_train.csv")
    test_tx = pd.read_csv("../data/Payments Fraud DataSet/transactions_test.csv")

    # Merge train/test
    train_data = (
        train_tx.merge(customers, on="CUSTOMER_ID", how="left")
        .merge(terminals, on="TERMINAL_ID", how="left")
        .merge(merchants, on="MERCHANT_ID", how="left")
    )

    test_data = (
        test_tx.merge(customers, on="CUSTOMER_ID", how="left")
        .merge(terminals, on="TERMINAL_ID", how="left")
        .merge(merchants, on="MERCHANT_ID", how="left")
    )

    # --- Feature engineering ---
    def engineer_features(df):
        df["distance"] = np.sqrt(
            (df["x_customer_id"] - df["x_terminal_id"]) ** 2
            + (df["y_customer_id"] - df["y_terminal__id"]) ** 2
        )
        df["TX_TS"] = pd.to_datetime(df["TX_TS"])
        df["hour"] = df["TX_TS"].dt.hour
        df["day_of_week"] = df["TX_TS"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        df["cashback_ratio"] = df["TRANSACTION_CASHBACK_AMOUNT"] / (
            df["TX_AMOUNT"] + 1e-8
        )
        df["goods_ratio"] = df["TRANSACTION_GOODS_AND_SERVICES_AMOUNT"] / (
            df["TX_AMOUNT"] + 1e-8
        )

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

        df["tx_to_turnover_ratio"] = df["TX_AMOUNT"] / (
            df["ANNUAL_TURNOVER_CARD"] + 1e-8
        )
        df["distance_tx"] = df["distance"] * df["TX_AMOUNT"]

        for col in [
            "TX_AMOUNT",
            "TRANSACTION_GOODS_AND_SERVICES_AMOUNT",
            "TRANSACTION_CASHBACK_AMOUNT",
        ]:
            df[col + "_log"] = np.log1p(df[col])
        return df

    # Apply feature engineering
    train_data = engineer_features(train_data)
    test_data = engineer_features(test_data)

    return train_data, test_data


# %% Historical / temporal features
def add_historical_features(df, reference_df=None):
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

# Adding features to help capture recent behavior patterns, which are very useful for detecting anomalies like fraud.
def add_lag_features(df):
    # Sort the dataframe by transaction timestamp so that lag/rolling features make sense
    df = df.sort_values("TX_TS")
    
    df["TX_TS_unix"] = df["TX_TS"].astype(np.int64) // 10**9

    # --- Time since previous transaction features ---

    # For each customer, compute the time difference between the current and previous transaction
    df["cust_prev_tx_time"] = df.groupby("CUSTOMER_ID")["TX_TS_unix"].diff().fillna(0)

    df["term_prev_tx_time"] = df.groupby("TERMINAL_ID")["TX_TS_unix"].diff().fillna(0)

    # --- Rolling sum of last 3 transactions ---

    # For each customer, compute the rolling sum of the last 3 transaction amounts
    df["cust_last3_tx_sum"] = (
        df.groupby("CUSTOMER_ID")["TX_AMOUNT"]
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(0, drop=True) 
    )

    # Similarly, for each terminal, compute rolling sum of last 3 transaction amounts
    df["term_last3_tx_sum"] = (
        df.groupby("TERMINAL_ID")["TX_AMOUNT"]
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(0, drop=True)
    )

    # Return the dataframe with new features
    return df



# %% K-Fold target encoding
def kfold_target_encoding(train, y_col, cat_cols, n_splits=5):
    from sklearn.model_selection import StratifiedKFold

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


# %% Train LightGBM with cross-validation
def train_with_cv(train_data, test_data, num_features, cat_features, n_splits=5):
    # Prepare the feature matrix X and target y for training
    # Concatenate numeric features and target-encoded categorical features
    X = train_data[num_features + [f + "_te" for f in cat_features]].values
    y = train_data["TX_FRAUD"].values

    # Prepare the test feature matrix
    X_test = test_data[num_features + [f + "_te" for f in cat_features]].values
    # Extract the transaction IDs for the submission
    ids_test = test_data["TX_ID"].values

    # Initialize an array to store averaged predictions for the test set
    preds_test = np.zeros(len(X_test))

    # Define Stratified K-Fold cross-validation
    # Stratified ensures each fold has roughly the same proportion of fraud/non-fraud
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Loop over each fold
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"Training fold {fold+1}/{n_splits}...")

        # Split training data into training and validation sets for this fold
        X_tr, X_val = X[train_idx], X[valid_idx]
        y_tr, y_val = y[train_idx], y[valid_idx]

        # Compute number of positive and negative samples in training
        # Used to handle class imbalance
        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos

        # LightGBM parameters
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

        # Create LightGBM datasets
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Train the model with early stopping and logging
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,               
            valid_sets=[dtrain, dvalid],         
            callbacks=[lgb.early_stopping(100), 
                       lgb.log_evaluation(200)], 
        )

        # Predict on the test set using the best iteration
        # Average predictions across folds
        preds_test += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

    # Prepare submission DataFrame
    submission = pd.DataFrame({"TX_ID": ids_test, "TX_FRAUD": preds_test})

    # Save predictions to CSV
    submission.to_csv("submission_lgbm.csv", index=False)
    print(f"Submission saved: {submission.shape[0]} rows")

    # Return the submission DataFrame
    return submission



# %% Main
if __name__ == "__main__":
    print("Loading and processing data...")
    train_data, test_data = load_and_preprocess_data()

    # Add historical/lag features
    print("Adding historical features...")
    train_data = add_historical_features(train_data)
    test_data = add_historical_features(test_data, reference_df=train_data)

    print("Adding lag features...")
    train_data = add_lag_features(train_data)
    test_data = add_lag_features(test_data)

    # Define features
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
        "cashback_ratio",
        "goods_ratio",
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

    # K-Fold target encoding
    print("Applying K-Fold target encoding...")
    train_data = kfold_target_encoding(train_data, "TX_FRAUD", cat_features)
    for col in cat_features:
        test_data[col + "_te"] = (
            test_data[col]
            .map(train_data.groupby(col)[col + "_te"].mean())
            .fillna(train_data["TX_FRAUD"].mean())
        )

    print("Training LightGBM with CV...")
    submission = train_with_cv(train_data, test_data, num_features, cat_features)

# %%
