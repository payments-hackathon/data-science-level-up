# flake8: noqa

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import random
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

test_df = pd.read_csv("transactions_test.csv", low_memory=False)

with open("model_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

feature_names = metadata['feature_names']
numeric_features = metadata['numeric_features']  
fraud_map = metadata.get('fraud_map', {})  
label_encoders = metadata.get('label_encoders', {})
scaler_feature_names = metadata.get('scaler_feature_names', None)

if os.path.exists("scaler.pkl"):
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("Loaded scaler.pkl")
else:
    scaler = None
    print("scaler.pkl not found — test script will fit a fallback StandardScaler on available numeric features")

def process_test_data(df, feature_names, numeric_features, fraud_map, scaler):
    df['TX_TS'] = pd.to_datetime(df['TX_TS'])
    df['hour'] = df['TX_TS'].dt.hour
    df['day_of_week'] = df['TX_TS'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)

    df['log_amount'] = np.log1p(df['TX_AMOUNT'])
    df['amount_risk_score'] = (df['TX_AMOUNT'] > df['TX_AMOUNT'].quantile(0.95)).astype(int)

    if 'CUSTOMER_ID' in df.columns:
        customer_stats = df.groupby('CUSTOMER_ID').agg({
            'TX_AMOUNT': ['mean', 'std', 'count']
        }).reset_index()
        customer_stats.columns = ['CUSTOMER_ID', 'customer_avg_amount', 'customer_std_amount', 'customer_tx_count']
        df = df.merge(customer_stats, on='CUSTOMER_ID', how='left')
        df['amount_deviation'] = np.abs(df['TX_AMOUNT'] - df['customer_avg_amount']) / (df['customer_std_amount'] + 1e-6)

    df['customer_fraud_rate'] = df['CUSTOMER_ID'].map(fraud_map).fillna(0.0)

    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].fillna('UNKNOWN')
            try:
                df[col] = le.transform(df[col].astype(str))
            except Exception:
                classes = list(le.classes_)
                df[col] = df[col].apply(lambda v: classes.index(v) if v in classes else len(classes))


    drop_cols = ['TX_TS', 'CUSTOMER_ID', 'TERMINAL_ID', 'MERCHANT_ID']
    df = df.drop([c for c in drop_cols if c in df.columns], axis=1)

    
    for col in numeric_features:
        if col not in df.columns:
            df[col] = 0.0

    if scaler is None:
        from sklearn.preprocessing import StandardScaler as _SS
        _sc = _SS()
        _sc.fit(df[numeric_features].values)
        df[numeric_features] = _sc.transform(df[numeric_features].values)
    else:
        scaler_cols = getattr(scaler, 'feature_names_in_', None)
        if scaler_cols is not None:
            cols = list(scaler_cols)
        elif scaler_feature_names is not None:
            cols = list(scaler_feature_names)
        else:
            cols = list(numeric_features)

        for col in cols:
            if col not in df.columns:
                df[col] = 0.0

        transformed = scaler.transform(df[cols])
        df_trans = pd.DataFrame(transformed, columns=cols, index=df.index)
        for col in cols:
            df[col] = df_trans[col].values
    df = df.reindex(columns=feature_names, fill_value=0.0)

    df = df.fillna(0.0)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

    return df

X_test_scaled = process_test_data(test_df, feature_names, numeric_features, fraud_map, scaler)
X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)


class EnhancedFraudNet(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.4):
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

model = EnhancedFraudNet(input_dim=X_test_tensor.shape[1])


candidate_models = [
    "fraud_model.pth",
    "enhanced_fraud_model.pth",
]

loaded_model_file = None
for mfile in candidate_models:
    if os.path.exists(mfile):
        try:
            model.load_state_dict(torch.load(mfile))
            loaded_model_file = mfile
            print(f"Loaded model weights from: {mfile}")
            break
        except Exception as e:
            print(f"Found model file {mfile} but failed to load: {e}")

if loaded_model_file is None:
    print("No model weights found among candidates; skipping prediction step. Please run training to produce a .pth file (e.g., enhanced_fraud_model.pth).")

    test_df["Fraud_Probability"] = 0.0
    test_df["Fraud_Prediction"] = 0
    test_df.to_csv("test_predictions.csv", index=False)
    print("Wrote fallback predictions to test_predictions.csv (all zeros)")

    raise SystemExit(0)

model.eval()


with torch.no_grad():
    test_outputs = torch.sigmoid(model(X_test_tensor)).squeeze().numpy()

test_df["Fraud_Probability"] = test_outputs
test_df["Fraud_Prediction"] = (test_df["Fraud_Probability"] >= 0.5).astype(int)

test_df.to_csv("test_predictions.csv", index=False)
print("Predictions saved to test_predictions.csv")
