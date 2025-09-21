#!/usr/bin/env python3

# flake8: noqa

# %% Imports and setup
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from collections import Counter
import pickle
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

pd.set_option('display.max_columns', None)

# %% Data loading and preprocessing
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

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

# %% Neural network model definition
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

# %% Model initialization and training setup
input_dim = X.shape[1]
model = EnhancedFraudNet(input_dim, dropout_rate=0.4)

# Try to load existing model
try:
    model.load_state_dict(torch.load("enhanced_fraud_model.pth"))
    print("Loaded existing enhanced model weights. Continuing training...")
except FileNotFoundError:
    print("No saved enhanced model found. Starting fresh.")

# Calculate class weights for imbalanced data
num_pos = y.sum()
num_neg = len(y) - num_pos
pos_weight = torch.tensor([num_neg / num_pos * 0.7], dtype=torch.float32)  # Slightly reduce to prevent over-weighting

print(f"Positive class weight: {pos_weight.item():.2f}")

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# %% Training utilities
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

# %% Model training
EPOCHS = 5
best_f1 = 0
patience_counter = 0
patience = 3

# Training history tracking
train_losses, val_losses = [], []
train_f1s, val_f1s = [], []

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
    
    # Track training history
    train_losses.append(running_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    val_f1s.append(f1)
    
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
                'scaler_feature_names': list(X.columns),
                'threshold': optimal_threshold
            }, f)

        print(f"  New best F1: {best_f1:.4f} - Model saved!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# %% Final evaluation
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

# Feature importance analysis
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

# Training visualization
from stats import plot_training_history, plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix

# Plot training progress
plot_training_history(train_losses, val_losses, None, val_f1s)

# Plot model performance curves
plot_roc_curve(y_val, val_probs)
plot_precision_recall_curve(y_val, val_probs)
plot_confusion_matrix(y_val, val_preds, labels=['Non-Fraud', 'Fraud'])

# %% Test data preprocessing utilities
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

# %% Test data prediction
test_df = pd.read_csv("data/Payments Fraud DataSet/transactions_test.csv", low_memory=False)

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

X_test_scaled = process_test_data(test_df, feature_names, numeric_features, fraud_map, scaler)
X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)

# Load model for prediction
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
else:
    model.eval()
    with torch.no_grad():
        test_outputs = torch.sigmoid(model(X_test_tensor)).squeeze().numpy()

    # Use a more conservative threshold based on expected fraud rate (~2-5%)
    # Sort probabilities and use 95th percentile as threshold
    conservative_threshold = np.percentile(test_outputs, 95)
    print(f"Using conservative threshold (95th percentile): {conservative_threshold:.3f}")
    
    test_df["Fraud_Probability"] = test_outputs
    test_df["Fraud_Prediction"] = (test_df["Fraud_Probability"] >= conservative_threshold).astype(int)

    test_df.to_csv("test_predictions.csv", index=False)
    print("Predictions saved to test_predictions.csv")

# %% Results visualization
df = pd.read_csv('test_predictions.csv')
print('Rows:', len(df))
print('\nPrediction counts:')
print(df['Fraud_Prediction'].value_counts(dropna=False))
print('\nProbability stats:')
print(df['Fraud_Probability'].describe())
print('\nTop 10 by probability:')
print(df.sort_values('Fraud_Probability', ascending=False).head(10))

out_dir = 'plots'
os.makedirs(out_dir, exist_ok=True)

# Fraud probability histogram
plt.figure(figsize=(8,4))
sns.histplot(data=df, x='Fraud_Probability', bins=50, kde=False)
plt.title('Predicted Fraud Probability Distribution')
plt.xlabel('Fraud Probability')
plt.ylabel('Count')
plt.tight_layout()
hist_path = os.path.join(out_dir, 'fraud_probability_histogram.png')
plt.savefig(hist_path)
plt.close()
print(f'Saved: {hist_path}')

# Prediction counts
plt.figure(figsize=(4,4))
ax = sns.countplot(x='Fraud_Prediction', data=df)
plt.title('Predicted Fraud vs Non-Fraud Counts')
plt.xlabel('Prediction (0=Non-Fraud,1=Fraud)')
plt.ylabel('Count')
plt.tight_layout()
counts_path = os.path.join(out_dir, 'fraud_prediction_counts.png')

from matplotlib.patches import Rectangle
for p in ax.patches:
    if isinstance(p, Rectangle):
        height = int(p.get_height())
        ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, color='black')

plt.savefig(counts_path)
plt.close()
print(f'Saved: {counts_path}')

# Summary text
summary_text = f"Total rows: {len(df)}\nNon-Fraud: {df['Fraud_Prediction'].value_counts().get(0,0)}\nFraud: {df['Fraud_Prediction'].value_counts().get(1,0)}"
plt.figure(figsize=(4,2))
plt.axis('off')
plt.text(0.01, 0.5, summary_text, fontsize=12, va='center')
summary_path = os.path.join(out_dir, 'prediction_summary.png')
plt.savefig(summary_path, bbox_inches='tight')
plt.close()
print(f'Saved: {summary_path}')

# Amount vs probability scatter plot
if 'TX_AMOUNT' in df.columns:
    plt.figure(figsize=(8,4))
    sns.scatterplot(x='TX_AMOUNT', y='Fraud_Probability', data=df, alpha=0.6)
    plt.xscale('symlog')
    plt.title('Transaction Amount vs Predicted Fraud Probability')
    plt.xlabel('TX_AMOUNT (symlog)')
    plt.ylabel('Fraud Probability')
    plt.tight_layout()
    amt_path = os.path.join(out_dir, 'amount_vs_probability.png')
    plt.savefig(amt_path)
    plt.close()
    print(f'Saved: {amt_path}')

# Hourly analysis
if 'TX_TS' in df.columns:
    try:
        df['TX_TS'] = pd.to_datetime(df['TX_TS'])
        df['hour'] = df['TX_TS'].dt.hour
        hourly = df.groupby('hour')['Fraud_Probability'].mean().reset_index()
        plt.figure(figsize=(8,3))
        sns.lineplot(x='hour', y='Fraud_Probability', data=hourly, marker='o')
        plt.title('Mean Predicted Fraud Probability by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Mean Fraud Probability')
        plt.tight_layout()
        hour_path = os.path.join(out_dir, 'hourly_mean_probability.png')
        plt.savefig(hour_path)
        plt.close()
        print(f'Saved: {hour_path}')
    except Exception as e:
        print('Failed to compute hourly summary:', e)

print('All visuals saved in ./plots')