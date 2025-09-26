import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

def load_predictions():
    catboost_pred = pd.read_csv("../catboost/submission_catboost.csv")
    lightgbm_pred = pd.read_csv("../lightgbm_tuned/lightgbm_complete_predictions.csv")
    return catboost_pred.merge(lightgbm_pred, on="TX_ID", suffixes=("_cat", "_lgb"))

def load_validation_data():
    """Load validation predictions if available for calibration"""
    try:
        # Assuming you have validation predictions saved
        val_cat = pd.read_csv("../catboost/validation_predictions.csv")
        val_lgb = pd.read_csv("../lightgbm_tuned/validation_predictions.csv")
        val_true = pd.read_csv("../validation_labels.csv")  # True labels
        return val_cat, val_lgb, val_true
    except:
        return None, None, None

def isotonic_calibration(probs):
    """Apply isotonic regression for probability calibration"""
    # Use quantile-based calibration without true labels
    sorted_probs = np.sort(probs)
    n = len(sorted_probs)
    
    # Create synthetic calibration curve
    expected_probs = np.linspace(0.001, 0.999, n)
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(sorted_probs, expected_probs)
    
    return iso_reg.transform(probs)

def temperature_scaling(probs, temperature=1.2):
    """Apply temperature scaling to calibrate probabilities"""
    logits = np.log(probs / (1 - probs + 1e-8) + 1e-8)
    scaled_logits = logits / temperature
    return 1 / (1 + np.exp(-scaled_logits))

def platt_scaling(probs):
    """Platt scaling using sigmoid transformation"""
    # Estimate parameters based on distribution
    mean_prob = np.mean(probs)
    std_prob = np.std(probs)
    
    # Simple heuristic for A and B parameters
    A = 1.0 / std_prob if std_prob > 0 else 1.0
    B = -A * np.log(mean_prob / (1 - mean_prob + 1e-8) + 1e-8)
    
    logits = np.log(probs / (1 - probs + 1e-8) + 1e-8)
    return 1 / (1 + np.exp(A * logits + B))

def create_calibrated_ensemble():
    df = load_predictions()
    cat_probs = np.clip(df["TX_FRAUD_cat"].values, 1e-8, 1-1e-8)
    lgb_probs = np.clip(df["TX_FRAUD_lgb"].values, 1e-8, 1-1e-8)
    
    # Apply different calibration methods
    cat_temp = temperature_scaling(cat_probs, 1.15)
    lgb_temp = temperature_scaling(lgb_probs, 1.25)
    
    cat_platt = platt_scaling(cat_probs)
    lgb_platt = platt_scaling(lgb_probs)
    
    # Ensemble calibrated predictions
    calibrated_ensemble = (
        0.4 * (0.65 * cat_temp + 0.35 * lgb_temp) +
        0.35 * (0.62 * cat_platt + 0.38 * lgb_platt) +
        0.25 * (0.6 * cat_probs + 0.4 * lgb_probs)  # Original uncalibrated
    )
    
    result = pd.DataFrame({
        "TX_ID": df["TX_ID"],
        "TX_FRAUD": calibrated_ensemble
    })
    
    result.to_csv("calibrated_ensemble_submission.csv", index=False)
    print(f"Calibrated ensemble mean: {calibrated_ensemble.mean():.6f}")
    return result

if __name__ == "__main__":
    create_calibrated_ensemble()