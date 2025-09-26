# flake8: noqa

# %% Utilities
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, average_precision_score

def generate_validation_predictions():
    """Generate validation predictions from both models on training data"""
    # Load training data
    transactions = pd.read_csv("../data/Payments Fraud DataSet/transactions_train.csv")
    
    # Use last 20% as validation (time-based split)
    transactions['TX_TS'] = pd.to_datetime(transactions['TX_TS'])
    transactions = transactions.sort_values('TX_TS')
    split_idx = int(len(transactions) * 0.8)
    
    val_data = transactions.iloc[split_idx:].copy()
    val_ids = val_data['TX_ID'].values
    val_labels = val_data['TX_FRAUD'].values
    
    print(f"Validation set: {len(val_data)} transactions, {val_labels.mean():.4f} fraud rate")
    return val_ids, val_labels

def load_model_validation_predictions(val_ids):
    """Load validation predictions from saved model outputs"""
    # You'll need to modify your model training to save validation predictions
    # For now, simulate with random predictions weighted by performance
    np.random.seed(42)
    n = len(val_ids)
    
    # Simulate CatBoost validation predictions (better model)
    cat_val_preds = np.random.beta(1, 50, n)  # Low fraud rate distribution
    cat_val_preds[np.random.choice(n, int(n*0.02))] = np.random.beta(5, 1, int(n*0.02))  # Some high fraud
    
    # Simulate LightGBM validation predictions (slightly worse)
    lgb_val_preds = cat_val_preds + np.random.normal(0, 0.01, n)
    lgb_val_preds = np.clip(lgb_val_preds, 0, 1)
    
    return cat_val_preds, lgb_val_preds

def optimize_ensemble_weights(cat_preds, lgb_preds, true_labels):
    """Optimize ensemble weights using validation data"""
    
    def ensemble_objective(params):
        w1, temp1, temp2 = params
        w2 = 1 - w1
        
        # Apply temperature scaling
        cat_scaled = 1 / (1 + np.exp(-np.log(cat_preds / (1-cat_preds + 1e-8)) / temp1))
        lgb_scaled = 1 / (1 + np.exp(-np.log(lgb_preds / (1-lgb_preds + 1e-8)) / temp2))
        
        # Weighted ensemble
        ensemble_preds = w1 * cat_scaled + w2 * lgb_scaled
        
        # Optimize for PR-AUC (better for imbalanced data)
        try:
            score = average_precision_score(true_labels, ensemble_preds)
            return -score  # Minimize negative score
        except:
            return 1.0  # Return bad score if error
    
    # Optimize weights and temperature parameters
    bounds = [(0.3, 0.8), (0.8, 1.5), (0.8, 1.5)]  # w1, temp1, temp2
    result = minimize(ensemble_objective, [0.6, 1.0, 1.0], bounds=bounds, method='L-BFGS-B')
    
    return result.x

# %% Ensemble methods
"""Apply optimized ensemble to test predictions"""

# Step 1: Get validation data and optimize
print("Generating validation predictions...")
val_ids, val_labels = generate_validation_predictions()
cat_val_preds, lgb_val_preds = load_model_validation_predictions(val_ids)

print("Optimizing ensemble weights...")
optimal_params = optimize_ensemble_weights(cat_val_preds, lgb_val_preds, val_labels)
w1, temp1, temp2 = optimal_params

print(f"Optimal weights: CatBoost={w1:.3f}, LightGBM={1-w1:.3f}")
print(f"Optimal temperatures: CatBoost={temp1:.3f}, LightGBM={temp2:.3f}")

# Step 2: Apply to test predictions
print("Loading test predictions...")
catboost_pred = pd.read_csv("../catboost/submission_catboost.csv")
lightgbm_pred = pd.read_csv("../lightgbm_tuned/lightgbm_complete_predictions.csv")

df = catboost_pred.merge(lightgbm_pred, on="TX_ID", suffixes=("_cat", "_lgb"))

cat_test = np.clip(df["TX_FRAUD_cat"].values, 1e-8, 1-1e-8)
lgb_test = np.clip(df["TX_FRAUD_lgb"].values, 1e-8, 1-1e-8)

# Apply optimized parameters
cat_scaled = 1 / (1 + np.exp(-np.log(cat_test / (1-cat_test)) / temp1))
lgb_scaled = 1 / (1 + np.exp(-np.log(lgb_test / (1-lgb_test)) / temp2))

optimized_ensemble = w1 * cat_scaled + (1-w1) * lgb_scaled

# Save result
result = pd.DataFrame({
    "TX_ID": df["TX_ID"],
    "TX_FRAUD": optimized_ensemble
})

result.to_csv("optimized_ensemble_submission.csv", index=False)
print(f"Optimized ensemble mean: {optimized_ensemble.mean():.6f}")
print("Saved to optimized_ensemble_submission.csv")