# flake8: noqa

# %% Utilities
import pandas as pd
import numpy as np

def load_predictions():
    """Load predictions from both models"""
    catboost_pred = pd.read_csv("../catboost/submission_catboost.csv")
    lightgbm_pred = pd.read_csv("../lightgbm_tuned/lightgbm_complete_predictions.csv")
    
    # Merge on TX_ID to ensure alignment
    merged = catboost_pred.merge(lightgbm_pred, on="TX_ID", suffixes=("_cat", "_lgb"))
    return merged

def rank_averaging_ensemble(cat_probs, lgb_probs, cat_weight=0.6):
    """Rank averaging - converts probabilities to ranks then averages"""
    cat_ranks = pd.Series(cat_probs).rank(pct=True)
    lgb_ranks = pd.Series(lgb_probs).rank(pct=True)
    
    combined_ranks = cat_weight * cat_ranks + (1 - cat_weight) * lgb_ranks
    return combined_ranks

def confidence_weighted_ensemble(cat_probs, lgb_probs, base_cat_weight=0.6):
    """Weight based on prediction confidence - more confident predictions get higher weight"""
    cat_confidence = np.abs(cat_probs - 0.5) * 2  # Distance from 0.5, normalized to [0,1]
    lgb_confidence = np.abs(lgb_probs - 0.5) * 2
    
    # Adjust weights based on relative confidence
    total_confidence = cat_confidence + lgb_confidence + 1e-8
    dynamic_cat_weight = base_cat_weight + 0.2 * (cat_confidence - lgb_confidence) / total_confidence
    dynamic_cat_weight = np.clip(dynamic_cat_weight, 0.4, 0.8)  # Keep within reasonable bounds
    
    return dynamic_cat_weight * cat_probs + (1 - dynamic_cat_weight) * lgb_probs

def power_averaging_ensemble(cat_probs, lgb_probs, cat_weight=0.6, power=2):
    """Power averaging - emphasizes higher probabilities"""
    cat_powered = np.power(cat_probs, power)
    lgb_powered = np.power(lgb_probs, power)
    
    combined = cat_weight * cat_powered + (1 - cat_weight) * lgb_powered
    return np.power(combined, 1/power)

# %%
"""Create ensemble predictions using multiple methods"""
print("Loading predictions...")
df = load_predictions()

cat_probs = df["TX_FRAUD_cat"].values
lgb_probs = df["TX_FRAUD_lgb"].values

print(f"CatBoost mean probability: {cat_probs.mean():.6f}")
print(f"LightGBM mean probability: {lgb_probs.mean():.6f}")

# Method 1: Weighted average (baseline)
weighted_avg = 0.6 * cat_probs + 0.4 * lgb_probs

# Method 2: Rank averaging
rank_avg = rank_averaging_ensemble(cat_probs, lgb_probs, 0.6)

# Method 3: Confidence-weighted ensemble
conf_weighted = confidence_weighted_ensemble(cat_probs, lgb_probs, 0.6)

# Method 4: Power averaging
power_avg = power_averaging_ensemble(cat_probs, lgb_probs, 0.6, 1.5)

# Create final ensemble (combination of methods)
final_ensemble = (
    0.4 * weighted_avg + 
    0.25 * rank_avg + 
    0.25 * conf_weighted + 
    0.1 * power_avg
)

# Create output dataframe
result = pd.DataFrame({
    "TX_ID": df["TX_ID"],
    "TX_FRAUD": final_ensemble
})

print(f"Final ensemble mean probability: {final_ensemble.mean():.6f}")

# Save result
result.to_csv("ensemble_submission.csv", index=False)
print("Ensemble predictions saved to ensemble/ensemble_submission.csv")