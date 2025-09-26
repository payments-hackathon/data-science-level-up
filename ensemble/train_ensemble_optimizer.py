# flake8: noqa

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score

def extract_validation_predictions():
    """Extract validation predictions by re-running models on validation split"""
    
    # Load and prepare data
    transactions = pd.read_csv("data/Payments Fraud DataSet/transactions_train.csv")
    transactions['TX_TS'] = pd.to_datetime(transactions['TX_TS'])
    transactions = transactions.sort_values('TX_TS')
    
    # Time-based split (last 20% for validation)
    split_idx = int(len(transactions) * 0.8)
    val_data = transactions.iloc[split_idx:].copy()
    
    print(f"Validation set: {len(val_data)} samples")
    
    # Save validation IDs and labels for model prediction
    val_subset = val_data[['TX_ID', 'TX_FRAUD']].copy()
    val_subset.to_csv("validation_subset.csv", index=False)
    
    print("Run your CatBoost and LightGBM models on validation_subset.csv")
    print("Save outputs as:")
    print("- catboost_validation_preds.csv")  
    print("- lightgbm_validation_preds.csv")
    
    return val_subset

def optimize_with_validation_preds():
    """Optimize ensemble using actual validation predictions"""
    
    # Load validation predictions (you need to generate these first)
    try:
        val_labels = pd.read_csv("validation_subset.csv")
        cat_val = pd.read_csv("catboost_validation_preds.csv")
        lgb_val = pd.read_csv("lightgbm_validation_preds.csv")
        
        # Merge all data
        val_data = val_labels.merge(cat_val, on='TX_ID').merge(lgb_val, on='TX_ID', suffixes=('_cat', '_lgb'))
        
    except FileNotFoundError:
        print("Validation prediction files not found. Run extract_validation_predictions() first.")
        return None
    
    y_true = val_data['TX_FRAUD'].values
    cat_preds = val_data['TX_FRAUD_cat'].values  
    lgb_preds = val_data['TX_FRAUD_lgb'].values
    
    def objective(params):
        w1, w2, w3, temp1, temp2 = params
        w_total = w1 + w2 + w3
        w1, w2, w3 = w1/w_total, w2/w_total, w3/w_total  # Normalize
        
        # Temperature scaling
        cat_scaled = 1 / (1 + np.exp(-np.log(cat_preds/(1-cat_preds+1e-8)+1e-8) / temp1))
        lgb_scaled = 1 / (1 + np.exp(-np.log(lgb_preds/(1-lgb_preds+1e-8)+1e-8) / temp2))
        
        # Multiple ensemble methods
        linear = w1 * cat_scaled + w2 * lgb_scaled
        geometric = np.power(cat_scaled, w1) * np.power(lgb_scaled, w2)
        harmonic = 1 / (w1/cat_scaled + w2/lgb_scaled + 1e-8)
        
        # Meta-ensemble
        ensemble = w1 * linear + w2 * geometric + w3 * harmonic
        
        # Optimize PR-AUC
        try:
            score = average_precision_score(y_true, ensemble)
            return -score
        except:
            return 1.0
    
    # Optimize parameters
    bounds = [(0.1, 0.8), (0.1, 0.8), (0.05, 0.3), (0.8, 1.5), (0.8, 1.5)]
    result = minimize(objective, [0.5, 0.4, 0.1, 1.0, 1.0], bounds=bounds)
    
    optimal_params = result.x
    print(f"Optimal parameters: {optimal_params}")
    print(f"Best validation PR-AUC: {-result.fun:.6f}")
    
    # Save optimal parameters
    np.save("optimal_ensemble_params.npy", optimal_params)
    return optimal_params

def apply_optimal_ensemble():
    """Apply optimized parameters to test predictions"""
    
    # Load optimal parameters
    try:
        params = np.load("optimal_ensemble_params.npy")
    except:
        print("No optimal parameters found. Run optimization first.")
        return
    
    w1, w2, w3, temp1, temp2 = params
    w_total = w1 + w2 + w3
    w1, w2, w3 = w1/w_total, w2/w_total, w3/w_total
    
    # Load test predictions
    catboost_pred = pd.read_csv("../catboost/submission_catboost.csv")
    lightgbm_pred = pd.read_csv("../lightgbm_tuned/lightgbm_complete_predictions.csv")
    
    df = catboost_pred.merge(lightgbm_pred, on="TX_ID", suffixes=("_cat", "_lgb"))
    
    cat_test = np.clip(df["TX_FRAUD_cat"].values, 1e-8, 1-1e-8)
    lgb_test = np.clip(df["TX_FRAUD_lgb"].values, 1e-8, 1-1e-8)
    
    # Apply optimal transformations
    cat_scaled = 1 / (1 + np.exp(-np.log(cat_test/(1-cat_test)) / temp1))
    lgb_scaled = 1 / (1 + np.exp(-np.log(lgb_test/(1-lgb_test)) / temp2))
    
    # Apply optimal ensemble
    linear = w1 * cat_scaled + w2 * lgb_scaled
    geometric = np.power(cat_scaled, w1) * np.power(lgb_scaled, w2)
    harmonic = 1 / (w1/cat_scaled + w2/lgb_scaled + 1e-8)
    
    final_ensemble = w1 * linear + w2 * geometric + w3 * harmonic
    
    result = pd.DataFrame({
        "TX_ID": df["TX_ID"],
        "TX_FRAUD": final_ensemble
    })
    
    result.to_csv("final_optimized_ensemble.csv", index=False)
    print(f"Final ensemble mean: {final_ensemble.mean():.6f}")
    
if __name__ == "__main__":
    # Step 1: Extract validation data
    extract_validation_predictions()
    
    # Step 2: Generate validation predictions with your models
    print("\nNow run your models on validation_subset.csv and save predictions")
    print("Then run: optimize_with_validation_preds() and apply_optimal_ensemble()")