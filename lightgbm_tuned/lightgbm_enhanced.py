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
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# %% Data Loading and Initial Processing
def load_data():
    """Load all datasets"""
    print("Loading datasets...")
    customers = pd.read_csv("../data/Payments Fraud DataSet/customers.csv")
    terminals = pd.read_csv("../data/Payments Fraud DataSet/terminals.csv")
    merchants = pd.read_csv("../data/Payments Fraud DataSet/merchants.csv")
    train_tx = pd.read_csv("../data/Payments Fraud DataSet/transactions_train.csv")
    test_tx = pd.read_csv("../data/Payments Fraud DataSet/transactions_test.csv")
    
    print(f"Loaded {len(train_tx):,} training transactions")
    print(f"Loaded {len(test_tx):,} test transactions")
    
    return customers, terminals, merchants, train_tx, test_tx

# %% Fraudulent Terminal Detection
def detect_fraudulent_terminals(train_tx, threshold=1.0):
    """Detect terminals with 100% fraud rate"""
    terminal_stats = train_tx.groupby('TERMINAL_ID')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
    terminal_stats = terminal_stats[terminal_stats['count'] >= 5]  # Min 5 transactions
    fraudulent_terminals = terminal_stats[terminal_stats['mean'] >= threshold].index.tolist()
    
    print(f"Found {len(fraudulent_terminals)} terminals with 100% fraud rate")
    return fraudulent_terminals

class SpatialNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def create_spatial_fraud_maps(customers, terminals, train_tx, fraudulent_terminals):
    """Create spatial probability maps using PyTorch neural networks"""
    print("Creating spatial fraud probability maps...")
    
    # Customer fraud mapping
    customer_tx = train_tx[['CUSTOMER_ID', 'TX_FRAUD']].merge(
        customers[['CUSTOMER_ID', 'x_customer_id', 'y_customer_id']], on='CUSTOMER_ID')
    customer_fraud = (customer_tx.groupby(['x_customer_id', 'y_customer_id'])['TX_FRAUD']
                     .agg(['count', 'sum']).reset_index())
    customer_fraud = customer_fraud[customer_fraud['count'] >= 3]
    customer_fraud['fraud_rate'] = customer_fraud['sum'] / customer_fraud['count']
    customer_fraud['log_fraud_rate'] = np.log1p(customer_fraud['fraud_rate'])
    del customer_tx
    
    # Terminal fraud mapping - exclude 100% fraud terminals
    terminal_tx = train_tx[~train_tx['TERMINAL_ID'].isin(fraudulent_terminals)][['TERMINAL_ID', 'TX_FRAUD']].merge(
        terminals[['TERMINAL_ID', 'x_terminal_id', 'y_terminal__id']], on='TERMINAL_ID')
    terminal_fraud = (terminal_tx.groupby(['x_terminal_id', 'y_terminal__id'])['TX_FRAUD']
                     .agg(['count', 'sum']).reset_index())
    terminal_fraud = terminal_fraud[terminal_fraud['count'] >= 3]
    terminal_fraud['fraud_rate'] = terminal_fraud['sum'] / terminal_fraud['count']
    terminal_fraud['log_fraud_rate'] = np.log1p(terminal_fraud['fraud_rate'])
    del terminal_tx
    
    # Normalize coordinates and targets
    customer_scaler = StandardScaler()
    terminal_scaler = StandardScaler()
    customer_target_scaler = StandardScaler()
    terminal_target_scaler = StandardScaler()
    
    customer_X = customer_scaler.fit_transform(customer_fraud[['x_customer_id', 'y_customer_id']])
    customer_y = customer_target_scaler.fit_transform(customer_fraud[['log_fraud_rate']])
    terminal_X = terminal_scaler.fit_transform(terminal_fraud[['x_terminal_id', 'y_terminal__id']])
    terminal_y = terminal_target_scaler.fit_transform(terminal_fraud[['log_fraud_rate']])
    
    # Train customer model
    customer_model = SpatialNN()
    optimizer = optim.Adam(customer_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    X_tensor = torch.FloatTensor(customer_X)
    y_tensor = torch.FloatTensor(customer_y)
    
    customer_model.train()
    for epoch in range(500):
        optimizer.zero_grad()
        pred = customer_model(X_tensor)
        loss = nn.MSELoss()(pred, y_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if epoch % 100 == 0:
            print(f"Customer model epoch {epoch}, loss: {loss.item():.6f}")
    
    # Train terminal model
    terminal_model = SpatialNN()
    optimizer = optim.Adam(terminal_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    X_tensor = torch.FloatTensor(terminal_X)
    y_tensor = torch.FloatTensor(terminal_y)
    
    terminal_model.train()
    for epoch in range(500):
        optimizer.zero_grad()
        pred = terminal_model(X_tensor)
        loss = nn.MSELoss()(pred, y_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if epoch % 100 == 0:
            print(f"Terminal model epoch {epoch}, loss: {loss.item():.6f}")
    
    customer_model.eval()
    terminal_model.eval()
    
    return (customer_model, customer_scaler, customer_target_scaler, 
            terminal_model, terminal_scaler, terminal_target_scaler, 
            customer_fraud, terminal_fraud)

# %% Spatial Visualization
def plot_spatial_heatmaps(models, customer_fraud, terminal_fraud):
    """Plot spatial fraud probability heatmaps"""
    print("Generating spatial heatmaps...")
    customer_model, customer_scaler, customer_target_scaler, terminal_model, terminal_scaler, terminal_target_scaler = models[:6]
    
    # Create grid for predictions
    x_grid = np.linspace(0, 100, 50)
    y_grid = np.linspace(0, 100, 50)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    # Customer predictions
    customer_grid_norm = customer_scaler.transform(grid_points)
    with torch.no_grad():
        customer_pred_norm = customer_model(torch.FloatTensor(customer_grid_norm)).numpy()
    customer_pred = customer_target_scaler.inverse_transform(customer_pred_norm).reshape(X_grid.shape)
    
    # Terminal predictions
    terminal_grid_norm = terminal_scaler.transform(grid_points)
    with torch.no_grad():
        terminal_pred_norm = terminal_model(torch.FloatTensor(terminal_grid_norm)).numpy()
    terminal_pred = terminal_target_scaler.inverse_transform(terminal_pred_norm).reshape(X_grid.shape)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Customer heatmap
    im1 = axes[0,0].contourf(X_grid, Y_grid, customer_pred, levels=20, cmap='plasma')
    axes[0,0].set_title('Customer Fraud Probability (NN)')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Customer actual points
    scatter1 = axes[0,1].scatter(customer_fraud['x_customer_id'], customer_fraud['y_customer_id'], 
                                c=customer_fraud['log_fraud_rate'], cmap='plasma', s=20)
    axes[0,1].set_title('Customer Actual Fraud Points')
    plt.colorbar(scatter1, ax=axes[0,1])
    
    # Terminal heatmap
    im2 = axes[1,0].contourf(X_grid, Y_grid, terminal_pred, levels=20, cmap='plasma')
    axes[1,0].set_title('Terminal Fraud Probability (NN)')
    plt.colorbar(im2, ax=axes[1,0])
    
    # Terminal actual points
    scatter2 = axes[1,1].scatter(terminal_fraud['x_terminal_id'], terminal_fraud['y_terminal__id'], 
                                c=terminal_fraud['log_fraud_rate'], cmap='plasma', s=20, marker='s')
    axes[1,1].set_title('Terminal Actual Fraud Points')
    plt.colorbar(scatter2, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig('spatial_fraud_maps.png', dpi=300, bbox_inches='tight')
    plt.show()

# %% Feature Engineering
def engineer_features(df, models, fraudulent_terminals):
    """Enhanced feature engineering with spatial probabilities"""
    df = df.copy()
    df['TX_TS'] = pd.to_datetime(df['TX_TS'])
    
    # Fraudulent terminal flag
    df['is_fraudulent_terminal'] = df['TERMINAL_ID'].isin(fraudulent_terminals).astype(int)
    
    # Time features
    df['hour'] = df['TX_TS'].dt.hour
    df['day_of_week'] = df['TX_TS'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    # Amount features with fraud range weighting
    df['TX_AMOUNT_log'] = np.log1p(df['TX_AMOUNT'])
    df['is_fraud_amount_range'] = ((df['TX_AMOUNT'] >= 500) & (df['TX_AMOUNT'] <= 1000)).astype(int)
    df['amount_fraud_weight'] = np.where(df['is_fraud_amount_range'], 2.0, 1.0)
    
    # Currency and country features (RON/RO outliers)
    df['is_ron_currency'] = (df['TRANSACTION_CURRENCY'] == 'RON').astype(int)
    df['is_ro_country'] = (df['CARD_COUNTRY_CODE'] == 'RO').astype(int)
    df['ron_ro_combined'] = df['is_ron_currency'] * df['is_ro_country']
    
    # Cashback and goods ratios
    df['cashback_ratio'] = df['TRANSACTION_CASHBACK_AMOUNT'] / (df['TX_AMOUNT'] + 1e-8)
    df['goods_ratio'] = df['TRANSACTION_GOODS_AND_SERVICES_AMOUNT'] / (df['TX_AMOUNT'] + 1e-8)
    
    # Recurring transaction
    df['is_recurring'] = (df['IS_RECURRING_TRANSACTION'].isin(['Y', 'True'])).astype(int)
    
    return df

def add_spatial_features(df, customers, terminals, models):
    """Add spatial probability features using neural networks"""
    customer_model, customer_scaler, customer_target_scaler, terminal_model, terminal_scaler, terminal_target_scaler = models[:6]
    
    # Merge coordinates efficiently
    df = df.merge(customers[['CUSTOMER_ID', 'x_customer_id', 'y_customer_id']], 
                  on='CUSTOMER_ID', how='left')
    df = df.merge(terminals[['TERMINAL_ID', 'x_terminal_id', 'y_terminal__id']], 
                  on='TERMINAL_ID', how='left')
    
    # Distance
    df['distance'] = np.sqrt((df['x_customer_id'] - df['x_terminal_id'])**2 + 
                            (df['y_customer_id'] - df['y_terminal__id'])**2)
    
    # Neural network predictions
    customer_coords = df[['x_customer_id', 'y_customer_id']].fillna(50).values
    terminal_coords = df[['x_terminal_id', 'y_terminal__id']].fillna(50).values
    
    # Normalize and predict
    customer_coords_norm = customer_scaler.transform(customer_coords)
    terminal_coords_norm = terminal_scaler.transform(terminal_coords)
    
    customer_model.eval()
    terminal_model.eval()
    with torch.no_grad():
        customer_pred_norm = customer_model(torch.FloatTensor(customer_coords_norm)).numpy()
        terminal_pred_norm = terminal_model(torch.FloatTensor(terminal_coords_norm)).numpy()
    
    # Denormalize predictions
    customer_pred = customer_target_scaler.inverse_transform(customer_pred_norm).flatten()
    terminal_pred = terminal_target_scaler.inverse_transform(terminal_pred_norm).flatten()
    
    # Convert log space back to probability space
    df['customer_fraud_prob'] = np.expm1(customer_pred)
    df['terminal_fraud_prob'] = np.expm1(terminal_pred)
    df['spatial_fraud_score'] = df['customer_fraud_prob'] + df['terminal_fraud_prob']
    
    return df


# %% Categorical Encoding for LightGBM
def prepare_categorical_features(df):
    """Prepare categorical features for LightGBM native handling"""
    categorical_cols = [
        'CARD_BRAND', 'TRANSACTION_TYPE', 'TRANSACTION_STATUS', 
        'TRANSACTION_CURRENCY', 'CARD_COUNTRY_CODE', 'CARDHOLDER_AUTH_METHOD',
        'ACQUIRER_ID'
    ]
    
    # Convert to category dtype for LightGBM
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    return df, categorical_cols

# %% Model Training with Cross-Validation
def train_lightgbm_model(train_data, fraudulent_terminals, n_splits=5):
    """Train LightGBM with enhanced features and cross-validation"""
    
    # Define feature columns
    numeric_features = [
        'TX_AMOUNT_log', 'hour', 'day_of_week', 'is_weekend', 'is_night',
        'distance', 'cashback_ratio', 'goods_ratio', 'is_recurring',
        'is_fraudulent_terminal', 'is_fraud_amount_range', 'is_ron_currency',
        'is_ro_country', 'ron_ro_combined', 'customer_fraud_prob', 
        'terminal_fraud_prob', 'spatial_fraud_score'
    ]
    
    categorical_features = [
        'CARD_BRAND', 'TRANSACTION_TYPE', 'TRANSACTION_STATUS',
        'TRANSACTION_CURRENCY', 'CARD_COUNTRY_CODE', 'CARDHOLDER_AUTH_METHOD',
        'ACQUIRER_ID'
    ]
    
    # Prepare features
    feature_cols = numeric_features + categorical_features
    available_features = [col for col in feature_cols if col in train_data.columns]
    available_categorical = [col for col in categorical_features if col in train_data.columns]
    
    X = train_data[available_features]
    y = train_data['TX_FRAUD']
    
    # Handle fraudulent terminals (always predict fraud)
    fraud_terminal_mask = train_data['is_fraudulent_terminal'] == 1
    
    # Cross-validation setup
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []
    oof_predictions = np.zeros(len(X))
    
    # Sample weights (higher for fraud amount range)
    sample_weights = train_data['amount_fraud_weight'].values
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Training fold {fold+1}/{n_splits}...")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_train = sample_weights[train_idx]
        
        # LightGBM datasets
        train_dataset = lgb.Dataset(X_train, label=y_train, weight=w_train,
                                   categorical_feature=available_categorical)
        val_dataset = lgb.Dataset(X_val, label=y_val, 
                                 categorical_feature=available_categorical, reference=train_dataset)
        
        # Parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'random_state': 42,
            'verbosity': -1
        }
        
        # Train model
        model = lgb.train(
            params,
            train_dataset,
            valid_sets=[val_dataset],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100)
            ]
        )
        
        # Predictions
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_predictions[val_idx] = val_pred
        
        # Override fraudulent terminals
        fraud_mask_val = fraud_terminal_mask.iloc[val_idx]
        oof_predictions[val_idx][fraud_mask_val] = 1.0
        
        models.append(model)
        
        # Save model checkpoint
        model.save_model(f'lightgbm_fold_{fold+1}.txt')
    
    return models, oof_predictions, available_features, available_categorical

# %% Prediction and Submission
def make_predictions(models, test_data, fraudulent_terminals, feature_cols, categorical_cols):
    """Make predictions on test set"""
    print("Making predictions on test set...")
    
    available_features = [col for col in feature_cols if col in test_data.columns]
    X_test = test_data[available_features]
    
    # Average predictions across folds
    test_predictions = np.zeros(len(X_test))
    for model in models:
        pred = model.predict(X_test, num_iteration=model.best_iteration)
        test_predictions += pred
    test_predictions /= len(models)
    
    # Override fraudulent terminals
    fraud_terminal_mask = test_data['is_fraudulent_terminal'] == 1
    test_predictions[fraud_terminal_mask] = 1.0
    
    return test_predictions

def save_submission(test_data, predictions, filename='lightgbm_enhanced_predictions.csv'):
    """Save predictions to CSV"""
    submission = pd.DataFrame({
        'TX_ID': test_data['TX_ID'],
        'TX_FRAUD': predictions
    })
    submission.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

# %% Main Execution Pipeline
print("Starting Enhanced LightGBM Fraud Detection Pipeline")
print("=" * 60)

if __name__ == "__main__":
    # Load data
    customers, terminals, merchants, train_tx, test_tx = load_data()
    
    # Detect fraudulent terminals
    fraudulent_terminals = detect_fraudulent_terminals(train_tx)
    
    # Create spatial fraud maps
    spatial_models = create_spatial_fraud_maps(customers, terminals, train_tx, fraudulent_terminals)
    
    # Plot spatial heatmaps
    plot_spatial_heatmaps(spatial_models, spatial_models[6], spatial_models[7])
    
    # Feature engineering for training data
    print("Engineering features for training data...")
    train_data = engineer_features(train_tx, spatial_models, fraudulent_terminals)
    train_data = add_spatial_features(train_data, customers, terminals, spatial_models)
    train_data, categorical_cols = prepare_categorical_features(train_data)
    
    # Feature engineering for test data
    print("Engineering features for test data...")
    test_data = engineer_features(test_tx, spatial_models, fraudulent_terminals)
    test_data = add_spatial_features(test_data, customers, terminals, spatial_models)
    test_data, _ = prepare_categorical_features(test_data)
    
    # Train model
    print("Training LightGBM model...")
    lgb_models, oof_predictions, feature_cols, categorical_cols = train_lightgbm_model(
        train_data, fraudulent_terminals)
    
    # Evaluate OOF performance
    from sklearn.metrics import roc_auc_score, classification_report
    oof_auc = roc_auc_score(train_data['TX_FRAUD'], oof_predictions)
    print(f"\nOut-of-fold AUC: {oof_auc:.4f}")
    
    # Make predictions
    test_predictions = make_predictions(lgb_models, test_data, fraudulent_terminals, 
                                        feature_cols, categorical_cols)
    
    # Save submission
    save_submission(test_data, test_predictions)
    
    # Save model artifacts
    with open('model_artifacts.pkl', 'wb') as f:
        pickle.dump({
            'spatial_models': spatial_models,
            'fraudulent_terminals': fraudulent_terminals,
            'feature_cols': feature_cols,
            'categorical_cols': categorical_cols
        }, f)
    
    print("Pipeline completed successfully!")
    print(f"Final model performance - AUC: {oof_auc:.4f}")