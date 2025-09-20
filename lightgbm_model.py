#!/usr/bin/env python3
# flake8: noqa

# %% Module Imports and Data Import
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import early_stopping, log_evaluation
import warnings
warnings.filterwarnings('ignore')

from data import load_and_preprocess_data


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
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dtrain, dvalid],
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(period=50)
        ]
    )

    preds = model.predict(X_test, num_iteration=model.best_iteration)

    output_df = pd.DataFrame({
        "TRANSACTION_ID": ids_test.values,
        "prediction": preds,
        "actual": y_test.values
    })
    output_df.to_csv("lightgbm_predictions.csv", index=False)
    print("LightGBM predictions saved to lightgbm_predictions.csv")

    auc = roc_auc_score(y_test, preds)
    print(f"LightGBM Test AUC: {auc:.4f}")

    return model

# %% Main
if __name__ == "__main__":
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    # Reload raw data to extract TRANSACTION_ID
    raw_data = pd.read_csv('data/Payments Fraud DataSet/transactions_train.csv')

    # Split IDs using same random_state and stratify as in preprocessing
    _, ids_test = train_test_split(
        raw_data['TX_ID'],
        test_size=0.2,
        random_state=42,
        stratify=raw_data['TX_FRAUD']
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Fraud rate: {y_train.mean():.4f}")

    print("Training LightGBM model...")
    lgb_model = train_lightgbm(X_train, y_train, X_test, y_test, ids_test)

