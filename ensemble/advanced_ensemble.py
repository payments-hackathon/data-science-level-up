# flake8: noqa

import pandas as pd
import numpy as np
from scipy.optimize import minimize

def load_predictions():
    catboost_pred = pd.read_csv("../catboost/submission_catboost.csv")
    lightgbm_pred = pd.read_csv("../lightgbm_tuned/lightgbm_complete_predictions.csv")
    return catboost_pred.merge(lightgbm_pred, on="TX_ID", suffixes=("_cat", "_lgb"))

def geometric_mean_ensemble(cat_probs, lgb_probs, cat_weight=0.6):
    return np.power(cat_probs, cat_weight) * np.power(lgb_probs, 1-cat_weight)

def harmonic_mean_ensemble(cat_probs, lgb_probs, cat_weight=0.6):
    return 1 / (cat_weight/cat_probs + (1-cat_weight)/lgb_probs + 1e-8)

def bayesian_ensemble(cat_probs, lgb_probs):
    # Estimate model reliability from prediction entropy
    cat_entropy = -cat_probs * np.log(cat_probs + 1e-8) - (1-cat_probs) * np.log(1-cat_probs + 1e-8)
    lgb_entropy = -lgb_probs * np.log(lgb_probs + 1e-8) - (1-lgb_probs) * np.log(1-lgb_probs + 1e-8)
    
    # Lower entropy = higher confidence = higher weight
    cat_reliability = 1 / (cat_entropy + 1e-8)
    lgb_reliability = 1 / (lgb_entropy + 1e-8)
    
    total_reliability = cat_reliability + lgb_reliability
    cat_weight = cat_reliability / total_reliability
    
    return cat_weight * cat_probs + (1 - cat_weight) * lgb_probs

def stacking_ensemble(cat_probs, lgb_probs):
    """Simple stacking with logistic transformation"""
    # Transform to logits for better blending
    cat_logits = np.log(cat_probs / (1 - cat_probs + 1e-8) + 1e-8)
    lgb_logits = np.log(lgb_probs / (1 - lgb_probs + 1e-8) + 1e-8)
    
    # Weighted logit combination
    combined_logits = 0.65 * cat_logits + 0.35 * lgb_logits
    
    # Convert back to probabilities
    return 1 / (1 + np.exp(-combined_logits))

def create_advanced_ensemble():
    df = load_predictions()
    cat_probs = df["TX_FRAUD_cat"].values
    lgb_probs = df["TX_FRAUD_lgb"].values
    
    # Clip extreme values for stability
    cat_probs = np.clip(cat_probs, 1e-8, 1-1e-8)
    lgb_probs = np.clip(lgb_probs, 1e-8, 1-1e-8)
    
    # Multiple ensemble methods
    geometric = geometric_mean_ensemble(cat_probs, lgb_probs, 0.62)
    harmonic = harmonic_mean_ensemble(cat_probs, lgb_probs, 0.58)
    bayesian = bayesian_ensemble(cat_probs, lgb_probs)
    stacking = stacking_ensemble(cat_probs, lgb_probs)
    
    # Meta-ensemble: combine the ensemble methods
    final_ensemble = (
        0.35 * geometric +
        0.25 * stacking +
        0.25 * bayesian +
        0.15 * harmonic
    )
    
    result = pd.DataFrame({
        "TX_ID": df["TX_ID"],
        "TX_FRAUD": final_ensemble
    })
    
    result.to_csv("advanced_ensemble_submission.csv", index=False)
    print(f"Advanced ensemble mean: {final_ensemble.mean():.6f}")
    return result

if __name__ == "__main__":
    create_advanced_ensemble()