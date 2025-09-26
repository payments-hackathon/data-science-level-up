# flake8: noqa

# %% Utilities
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
import warnings
import time
import pickle
import os
from datetime import timedelta
warnings.filterwarnings('ignore')

class ProgressTracker:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []
        
    def update(self, step_name=""):
        self.current_step += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if self.current_step > 1:
            avg_time_per_step = elapsed / (self.current_step - 1)
            remaining_steps = self.total_steps - self.current_step
            eta = remaining_steps * avg_time_per_step
            eta_str = str(timedelta(seconds=int(eta)))
        else:
            eta_str = "calculating..."
            
        progress_pct = (self.current_step / self.total_steps) * 100
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        print(f"[{self.current_step}/{self.total_steps}] {step_name}")
        print(f"Progress: {progress_pct:.1f}% | Elapsed: {elapsed_str} | ETA: {eta_str}")
        print("-" * 60)
        
class ModelCheckpoint:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save(self, data, filename):
        filepath = os.path.join(self.checkpoint_dir, filename)
        temp_filepath = filepath + ".tmp"
        try:
            with open(temp_filepath, 'wb') as f:
                pickle.dump(data, f)
            os.rename(temp_filepath, filepath)
            print(f"Checkpoint saved: {filepath}")
        except Exception as e:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            print(f"Failed to save checkpoint {filename}: {e}")
            raise
        
    def load(self, filename):
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Corrupted checkpoint detected: {filename}. Removing...")
                os.remove(filepath)
                return None
        return None
        
    def exists(self, filename):
        return os.path.exists(os.path.join(self.checkpoint_dir, filename))
    
    def list_checkpoints(self):
        """List all checkpoint files"""
        files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pkl')]
        return sorted(files)
    
    def get_training_progress(self, total_folds=5):
        """Get current training progress"""
        completed_folds = 0
        for fold in range(total_folds):
            if self.exists(f"fold_{fold}_complete.pkl"):
                completed_folds = fold + 1
            else:
                break
        return completed_folds, total_folds

def load_data():
    """Load all datasets"""
    transactions = pd.read_csv("../data/Payments Fraud DataSet/transactions_train.csv")
    merchants = pd.read_csv("../data/Payments Fraud DataSet/merchants.csv")
    customers = pd.read_csv("../data/Payments Fraud DataSet/customers.csv")
    terminals = pd.read_csv("../data/Payments Fraud DataSet/terminals.csv")
    return transactions, merchants, customers, terminals

def detect_fraudulent_terminals(train_tx, threshold=1.0, min_count=5):
    """Detect fraudulent terminals from training data"""
    terminal_stats = train_tx.groupby('TERMINAL_ID')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
    terminal_stats = terminal_stats[terminal_stats['count'] >= min_count]
    fraudulent_terminals = terminal_stats[terminal_stats['mean'] >= threshold].index.tolist()
    print(f"Found {len(fraudulent_terminals)} terminals with >= {int(threshold*100)}% fraud rate (min {min_count} txns).")
    return fraudulent_terminals

def create_comprehensive_features(transactions, merchants, customers, terminals, fraudulent_terminals, is_test=False):
    """Complete feature engineering combining ALL features from both models"""
    
    # Merge all datasets
    df = (transactions.merge(merchants, on="MERCHANT_ID", how="left")
          .merge(customers, on="CUSTOMER_ID", how="left")
          .merge(terminals, on="TERMINAL_ID", how="left"))
    
    df["TX_TS"] = pd.to_datetime(df["TX_TS"])
    df = df.sort_values(["CUSTOMER_ID", "TX_TS"])

    # === CATBOOST FEATURES ===
    
    # Terminal fraud indicator
    df['is_fraudulent_terminal'] = df['TERMINAL_ID'].isin(fraudulent_terminals).astype(int)

    # Distance features
    df['distance'] = np.sqrt((df['x_customer_id'] - df['x_terminal_id'])**2 + 
                            (df['y_customer_id'] - df['y_terminal__id'])**2)
    df['distance'] = df['distance'].fillna(0)

    # Time features
    df["hour"] = df["TX_TS"].dt.hour
    df["dayofweek"] = df["TX_TS"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    df['TX_AMOUNT_log'] = np.log1p(df['TX_AMOUNT'])
    
    # Fraud-specific features
    df['is_fraud_amount_range'] = ((df['TX_AMOUNT'] >= 500) & (df['TX_AMOUNT'] <= 1000)).astype(int)
    df['is_ron_currency'] = (df['TRANSACTION_CURRENCY'] == 'RON').astype(int)
    if 'CARD_COUNTRY_CODE' in df.columns:
        df['is_ro_country'] = (df['CARD_COUNTRY_CODE'] == 'RO').astype(int)
        df['ron_ro_combined'] = df['is_ron_currency'] * df['is_ro_country']
    else:
        df['is_ro_country'] = 0
        df['ron_ro_combined'] = 0

    # Temporal features
    df["time_since_last_tx"] = (df.groupby("CUSTOMER_ID")["TX_TS"].diff().dt.total_seconds().fillna(3600 * 24))
    df["time_since_last_tx_merchant"] = (df.groupby(["CUSTOMER_ID", "MERCHANT_ID"])["TX_TS"].diff().dt.total_seconds().fillna(3600 * 24 * 30))
    df["cust_tx_count_cumm"] = df.groupby("CUSTOMER_ID").cumcount() + 1
    df["cust_merchant_pair_count_cumm"] = df.groupby(["CUSTOMER_ID", "MERCHANT_ID"]).cumcount()

    # Amount aggregations
    median_amount = df["TX_AMOUNT"].median()
    df["cust_avg_amount_cumm"] = (df.groupby("CUSTOMER_ID")["TX_AMOUNT"].transform(lambda x: x.expanding().mean().shift(1)).fillna(median_amount))
    df["cust_std_amount_cumm"] = (df.groupby("CUSTOMER_ID")["TX_AMOUNT"].transform(lambda x: x.expanding().std().shift(1)).fillna(0))
    df["cust_max_amount_cumm"] = (df.groupby("CUSTOMER_ID")["TX_AMOUNT"].transform(lambda x: x.expanding().max().shift(1)).fillna(median_amount))

    # Rolling window features
    for window in [3, 5, 10]:
        df[f"cust_amount_mean_last_{window}"] = (df.groupby("CUSTOMER_ID")["TX_AMOUNT"].transform(lambda x: x.shift().rolling(window).mean()).fillna(median_amount))
        df[f"cust_amount_std_last_{window}"] = (df.groupby("CUSTOMER_ID")["TX_AMOUNT"].transform(lambda x: x.shift().rolling(window).std()).fillna(0))
        df[f"cust_tx_count_last_{window}"] = (df.groupby("CUSTOMER_ID")["TX_TS"].transform(lambda x: x.shift().rolling(window).count()).fillna(0))

    # Amount deviation features
    df["amount_deviation"] = ((df["TX_AMOUNT"] - df["cust_avg_amount_cumm"]) / df["cust_std_amount_cumm"].replace(0, 1)).fillna(0)
    df["amount_to_max_ratio"] = df["TX_AMOUNT"] / df["cust_max_amount_cumm"].replace(0, 1)

    # === LIGHTGBM FEATURES ===
    
    # Additional amount features
    for col in ["TRANSACTION_GOODS_AND_SERVICES_AMOUNT", "TRANSACTION_CASHBACK_AMOUNT"]:
        if col in df.columns:
            df[col + "_log"] = np.log1p(df[col])
    
    # Enhanced distance features
    df["distance_tx"] = df["distance"] * df["TX_AMOUNT"]
    if "TRANSACTION_CASHBACK_AMOUNT" in df.columns:
        df["cashback_ratio"] = df["TRANSACTION_CASHBACK_AMOUNT"] / (df["TX_AMOUNT"] + 1e-8)
    if "TRANSACTION_GOODS_AND_SERVICES_AMOUNT" in df.columns:
        df["goods_ratio"] = df["TRANSACTION_GOODS_AND_SERVICES_AMOUNT"] / (df["TX_AMOUNT"] + 1e-8)
    
    # Time-based features
    df["time_of_day"] = (df["TX_TS"].dt.hour * 3600 + df["TX_TS"].dt.minute * 60 + df["TX_TS"].dt.second) / 86400
    df["is_business_hours"] = ((df["TX_TS"].dt.hour >= 9) & (df["TX_TS"].dt.hour <= 17)).astype(int)
    df['tx_hour_risk'] = df['TX_TS'].dt.hour.map({
        0:3, 1:3, 2:3, 3:3, 4:2, 5:1, 6:1, 7:1, 8:1, 9:0, 10:0, 11:0, 12:0,
        13:0, 14:0, 15:0, 16:0, 17:0, 18:1, 19:1, 20:1, 21:2, 22:2, 23:3
    })
    
    # Merchant features
    if "TAX_EXCEMPT_INDICATOR" in df.columns:
        df["tax_exempt"] = df["TAX_EXCEMPT_INDICATOR"].fillna(False).astype(int)
    if "ANNUAL_TURNOVER" in df.columns:
        df["annual_turnover_log"] = np.log1p(df["ANNUAL_TURNOVER"].fillna(0))
    if "AVERAGE_TICKET_SALE_AMOUNT" in df.columns:
        df["avg_ticket_log"] = np.log1p(df["AVERAGE_TICKET_SALE_AMOUNT"].fillna(0))
        df["amount_vs_merchant_avg"] = df["TX_AMOUNT"] / (df["AVERAGE_TICKET_SALE_AMOUNT"] + 1e-8)
    
    # MCC frequency
    if "MCC_CODE" in df.columns:
        mcc_freq = df["MCC_CODE"].value_counts(normalize=True)
        df["mcc_frequency"] = df["MCC_CODE"].map(mcc_freq).fillna(0)
    
    # Recurring transaction indicator
    if "IS_RECURRING_TRANSACTION" in df.columns:
        df["is_recurring"] = (df["IS_RECURRING_TRANSACTION"] == "Y").astype(int)
    
    # Velocity features
    df = df.sort_values(['CUSTOMER_ID', 'TX_TS'])
    df['cust_tx_velocity_1h'] = df.groupby('CUSTOMER_ID')['TX_ID'].rolling(10, min_periods=1).count().reset_index(0, drop=True) - 1
    df['cust_tx_velocity_24h'] = df.groupby('CUSTOMER_ID')['TX_ID'].rolling(50, min_periods=1).count().reset_index(0, drop=True) - 1
    
    df = df.sort_values(['TERMINAL_ID', 'TX_TS'])
    df['term_tx_velocity_1h'] = df.groupby('TERMINAL_ID')['TX_ID'].rolling(10, min_periods=1).count().reset_index(0, drop=True) - 1
    
    # Amount deviation (rolling)
    df = df.sort_values(['CUSTOMER_ID', 'TX_TS'])
    df['cust_amount_rolling_mean'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df['cust_amount_deviation_rolling'] = np.abs(df['TX_AMOUNT'] - df['cust_amount_rolling_mean']) / (df['cust_amount_rolling_mean'] + 1e-8)
    
    # Card features
    if 'CARD_EXPIRY_DATE' in df.columns:
        df['card_expiry'] = pd.to_datetime(df['CARD_EXPIRY_DATE'], format='%m/%y', errors='coerce')
        if hasattr(df['TX_TS'].dtype, 'tz') and df['TX_TS'].dtype.tz is not None:
            df['TX_TS'] = df['TX_TS'].dt.tz_localize(None)
        if hasattr(df['card_expiry'].dtype, 'tz') and df['card_expiry'].dtype.tz is not None:
            df['card_expiry'] = df['card_expiry'].dt.tz_localize(None)
        df['card_age_days'] = (df['card_expiry'] - df['TX_TS']).dt.days.fillna(365)
        df['card_near_expiry'] = (df['card_age_days'] <= 90).astype(int)
    
    # Merchant age features
    if 'FOUNDATION_DATE' in df.columns:
        df['merchant_foundation'] = pd.to_datetime(df['FOUNDATION_DATE'], errors='coerce')
        if hasattr(df['merchant_foundation'].dtype, 'tz') and df['merchant_foundation'].dtype.tz is not None:
            df['merchant_foundation'] = df['merchant_foundation'].dt.tz_localize(None)
        df['merchant_age_days'] = (df['TX_TS'] - df['merchant_foundation']).dt.days.fillna(1000)
        df['merchant_young'] = (df['merchant_age_days'] <= 365).astype(int)
    
    # Enhanced fraud indicators
    df['high_risk_combo'] = df['is_ron_currency'] * df['is_ro_country'] * df['is_fraud_amount_range']
    
    # Turnover ratio
    if "ANNUAL_TURNOVER_CARD" in df.columns:
        df["tx_to_turnover_ratio"] = df["TX_AMOUNT"] / (df["ANNUAL_TURNOVER_CARD"] + 1e-8)
    
    # === NEW HIGH-CORRELATION FEATURES FROM ANALYSIS ===
    
    # 1. CASHBACK PATTERNS (Strong correlations: 0.0307, 0.0282, 0.0211)
    if 'TRANSACTION_CASHBACK_AMOUNT' in df.columns:
        df['has_cashback'] = (df['TRANSACTION_CASHBACK_AMOUNT'] > 0).astype(int)
        df['high_cashback_ratio'] = (df['TRANSACTION_CASHBACK_AMOUNT'] / (df['TX_AMOUNT'] + 1e-8) > 0.1).astype(int)
        df['cashback_amount_log'] = np.log1p(df['TRANSACTION_CASHBACK_AMOUNT'])
    
    # 2. GOODS VS TOTAL MISMATCH (Correlation: 0.0266)
    if 'TRANSACTION_GOODS_AND_SERVICES_AMOUNT' in df.columns:
        df['goods_vs_total_mismatch'] = (np.abs(df['TRANSACTION_GOODS_AND_SERVICES_AMOUNT'] - df['TX_AMOUNT']) > 1).astype(int)
    
    # 3. AUTHENTICATION PATTERNS (Correlations: 0.0242, -0.0187)
    if 'CARDHOLDER_AUTH_METHOD' in df.columns:
        df['no_auth'] = (df['CARDHOLDER_AUTH_METHOD'] == 'No CVM performed').astype(int)
        df['pin_auth'] = df['CARDHOLDER_AUTH_METHOD'].str.contains('PIN', na=False).astype(int)
        df['signature_only'] = (df['CARDHOLDER_AUTH_METHOD'] == 'Signature').astype(int)
        df['offline_auth'] = df['CARDHOLDER_AUTH_METHOD'].str.contains('Offline', na=False).astype(int)
    
    # 4. GEOGRAPHIC CENTER FEATURES (Correlations: 0.0125, -0.0124)
    df['distance_from_center_cust'] = np.sqrt((df['x_customer_id'] - 50)**2 + (df['y_customer_id'] - 50)**2)
    df['distance_from_center_term'] = np.sqrt((df['x_terminal_id'] - 50)**2 + (df['y_terminal__id'] - 50)**2)
    df['customer_in_center'] = ((df['x_customer_id'] > 25) & (df['x_customer_id'] < 75) & 
                               (df['y_customer_id'] > 25) & (df['y_customer_id'] < 75)).astype(int)
    df['terminal_in_center'] = ((df['x_terminal_id'] > 25) & (df['x_terminal_id'] < 75) & 
                               (df['y_terminal__id'] > 25) & (df['y_terminal__id'] < 75)).astype(int)
    df['both_in_center'] = df['customer_in_center'] * df['terminal_in_center']
    
    # 5. TRANSACTION TYPE RISK (Refund: 6.67%, Purchase with cashback: 4.92%)
    if 'TRANSACTION_TYPE' in df.columns:
        df['is_refund'] = (df['TRANSACTION_TYPE'] == 'Refund').astype(int)
        df['is_purchase_cashback'] = (df['TRANSACTION_TYPE'] == 'Purchase with cashback').astype(int)
        df['is_cash_advance'] = (df['TRANSACTION_TYPE'] == 'Cash Advance/Withdrawal').astype(int)
    
    # 6. HIGH-RISK CURRENCY/COUNTRY (RON: 9.09% fraud rate!)
    df['is_high_risk_currency'] = df.get('TRANSACTION_CURRENCY', '').isin(['RON', 'AED']).astype(int)
    if 'CARD_COUNTRY_CODE' in df.columns:
        df['is_high_risk_country'] = df['CARD_COUNTRY_CODE'].isin(['AE', 'RO', 'FI']).astype(int)
    
    # 7. ACQUIRER RISK (ACQ5, ACQ6 higher risk)
    if 'ACQUIRER_ID' in df.columns:
        df['is_high_risk_acquirer'] = df['ACQUIRER_ID'].isin(['ACQ5', 'ACQ6']).astype(int)
    
    # 8. CRITICAL AMOUNT RANGES (100-200: 4.13%, 200-500: 100%!)
    df['is_medium_high_amount'] = ((df['TX_AMOUNT'] >= 100) & (df['TX_AMOUNT'] < 200)).astype(int)
    df['is_very_high_amount'] = (df['TX_AMOUNT'] >= 200).astype(int)
    df['is_fraud_critical_amount'] = ((df['TX_AMOUNT'] >= 100) & (df['TX_AMOUNT'] <= 500)).astype(int)
    
    # 9. ROUND AMOUNT PATTERNS
    df['is_round_amount'] = (df['TX_AMOUNT'] % 10 == 0).astype(int)
    df['is_very_round_amount'] = (df['TX_AMOUNT'] % 100 == 0).astype(int)
    df['amount_has_cents'] = (df['TX_AMOUNT'] % 1 != 0).astype(int)
    
    # 10. HIGH-RISK HOURS (4am: 3.03%, 10am: 3.20%)
    df['is_high_risk_hour'] = df['hour'].isin([4, 8, 10, 11, 18]).astype(int)
    df['is_low_risk_hour'] = df['hour'].isin([12, 20, 23]).astype(int)
    df['is_early_morning'] = ((df['hour'] >= 3) & (df['hour'] <= 7)).astype(int)
    
    # 11. BUSINESS TYPE RISK (S Corp slightly higher)
    if 'BUSINESS_TYPE' in df.columns:
        df['is_s_corp'] = (df['BUSINESS_TYPE'] == 'S Corporations').astype(int)
        df['is_llc'] = (df['BUSINESS_TYPE'] == 'Limited Liability Company (LLC)').astype(int)
    
    # 12. MERCHANT DEPOSIT RISK
    if all(col in df.columns for col in ['DEPOSIT_REQUIRED_PERCENTAGE', 'DEPOSIT_PERCENTAGE']):
        df['merchant_deposit_risk'] = (df['DEPOSIT_REQUIRED_PERCENTAGE'] > df['DEPOSIT_PERCENTAGE']).astype(int)
        df['deposit_gap'] = df['DEPOSIT_REQUIRED_PERCENTAGE'] - df['DEPOSIT_PERCENTAGE']
    
    # 13. CARD BRAND RISK (Visa slightly higher: 2.72%)
    if 'CARD_BRAND' in df.columns:
        df['is_visa'] = (df['CARD_BRAND'] == 'Visa').astype(int)
        df['is_amex'] = (df['CARD_BRAND'] == 'AMEX').astype(int)
    
    # 14. HIGH-RISK COMBINATIONS
    df['ron_ro_high_amount'] = (df.get('is_ron_currency', 0) * df.get('is_ro_country', 0) * 
                               df.get('is_fraud_critical_amount', 0))
    df['cashback_no_auth'] = df.get('has_cashback', 0) * df.get('no_auth', 0)
    df['refund_high_amount'] = df.get('is_refund', 0) * df.get('is_very_high_amount', 0)
    
    # 15. DELIVERY PATTERNS
    if 'DELIVERY_OVER_TWO_WEEKS_PERCENTAGE' in df.columns:
        df['slow_delivery_risk'] = (df['DELIVERY_OVER_TWO_WEEKS_PERCENTAGE'] > 30).astype(int)
    if 'DELIVERY_SAME_DAYS_PERCENTAGE' in df.columns:
        df['same_day_delivery'] = (df['DELIVERY_SAME_DAYS_PERCENTAGE'] > 50).astype(int)
    
    # 16. OUTLET TYPE PATTERNS
    if 'OUTLET_TYPE' in df.columns:
        df['is_ecommerce_only'] = (df['OUTLET_TYPE'] == 'Ecommerce').astype(int)
        df['is_face_to_face_only'] = (df['OUTLET_TYPE'] == 'Face to Face').astype(int)
        df['is_mixed_outlet'] = (df['OUTLET_TYPE'] == 'Face to Face and Ecommerce').astype(int)
    
    # Cross-entity percentiles
    if "MERCHANT_ID" in df.columns:
        df['amount_percentile_merchant'] = df.groupby('MERCHANT_ID')['TX_AMOUNT'].transform(lambda x: x.rank(pct=True))
    
    # Time gaps
    df["TX_TS_unix"] = df["TX_TS"].astype(np.int64) // 10**9
    df["cust_prev_tx_time"] = df.groupby("CUSTOMER_ID")["TX_TS_unix"].diff().fillna(0)
    df["term_prev_tx_time"] = df.groupby("TERMINAL_ID")["TX_TS_unix"].diff().fillna(0)
    
    # Final feature selection
    final_cols = [
        "TX_ID", "TX_TS", "TX_AMOUNT", "TX_AMOUNT_log", "CUSTOMER_ID", "MERCHANT_ID",
        "MCC_CODE", "TERMINAL_ID", "CARD_BRAND", "TRANSACTION_CURRENCY", 
        "x_customer_id", "y_customer_id", "is_fraudulent_terminal", "distance", "distance_tx",
        "is_night", "is_fraud_amount_range", "is_ron_currency", "is_ro_country", "ron_ro_combined",
        "hour", "dayofweek", "is_weekend", "time_of_day", "is_business_hours", "tx_hour_risk",
        "time_since_last_tx", "time_since_last_tx_merchant", "cust_tx_count_cumm", 
        "cust_merchant_pair_count_cumm", "cust_avg_amount_cumm", "cust_std_amount_cumm", 
        "cust_max_amount_cumm", "amount_deviation", "amount_to_max_ratio",
        "cust_tx_velocity_1h", "cust_tx_velocity_24h", "term_tx_velocity_1h",
        "cust_amount_rolling_mean", "cust_amount_deviation_rolling", "high_risk_combo",
        "cust_prev_tx_time", "term_prev_tx_time", "TX_TS_unix"
    ]

    # Add rolling window features
    for w in [3, 5, 10]:
        final_cols += [f"cust_amount_mean_last_{w}", f"cust_amount_std_last_{w}", f"cust_tx_count_last_{w}"]

    # Add optional features if they exist
    optional_cols = ["cashback_ratio", "goods_ratio", "tax_exempt", "annual_turnover_log", 
                    "avg_ticket_log", "mcc_frequency", "is_recurring", "tx_to_turnover_ratio",
                    "card_age_days", "card_near_expiry", "merchant_age_days", "merchant_young",
                    "amount_vs_merchant_avg", "amount_percentile_merchant"]
    
    # Add new high-correlation features
    new_features = ["has_cashback", "high_cashback_ratio", "cashback_amount_log", "goods_vs_total_mismatch",
                   "no_auth", "pin_auth", "signature_only", "offline_auth", "distance_from_center_cust",
                   "distance_from_center_term", "customer_in_center", "terminal_in_center", "both_in_center",
                   "is_refund", "is_purchase_cashback", "is_cash_advance", "is_high_risk_currency",
                   "is_high_risk_country", "is_high_risk_acquirer", "is_medium_high_amount", "is_very_high_amount",
                   "is_fraud_critical_amount", "is_round_amount", "is_very_round_amount", "amount_has_cents",
                   "is_high_risk_hour", "is_low_risk_hour", "is_early_morning", "is_s_corp", "is_llc",
                   "merchant_deposit_risk", "deposit_gap", "is_visa", "is_amex", "ron_ro_high_amount",
                   "cashback_no_auth", "refund_high_amount", "slow_delivery_risk", "same_day_delivery",
                   "is_ecommerce_only", "is_face_to_face_only", "is_mixed_outlet"]
    
    for col in optional_cols + new_features:
        if col in df.columns:
            final_cols.append(col)

    if not is_test:
        final_cols.append("TX_FRAUD")

    return df[[c for c in final_cols if c in df.columns]].drop_duplicates(subset="TX_ID")

def get_categorical_features(df):
    """Get categorical features for both models including new ones"""
    cat = ["CUSTOMER_ID", "hour", "dayofweek", "TRANSACTION_CURRENCY", 
           "MCC_CODE", "TERMINAL_ID", "CARD_BRAND", "MERCHANT_ID",
           "is_fraudulent_terminal", "is_night", "is_weekend", "is_business_hours",
           "is_fraud_amount_range", "is_ron_currency", "is_ro_country", "ron_ro_combined",
           "has_cashback", "no_auth", "is_refund", "is_purchase_cashback", "customer_in_center",
           "terminal_in_center", "is_high_risk_currency", "is_high_risk_country", "is_high_risk_acquirer",
           "is_fraud_critical_amount", "is_high_risk_hour", "is_s_corp", "is_visa", "is_ecommerce_only"]
    
    available = [c for c in cat if c in df.columns]
    for c in available:
        if df[c].isnull().any():
            df[c] = df[c].fillna("MISSING")
        df[c] = df[c].astype(str)
    return available

def prepare_data(df, is_test=False):
    """Prepare data for training"""
    y = df["TX_FRAUD"] if not is_test else None
    
    drop_cols = ["TX_ID", "TX_TS", "x_customer_id", "y_customer_id", "TX_TS_unix"] + ([] if is_test else ["TX_FRAUD"])
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Fill missing values
    for c in X.select_dtypes(include=[np.number]).columns:
        if X[c].isnull().any():
            X[c] = X[c].fillna(X[c].median())
            
    cat_features = get_categorical_features(X)
    return X, y, cat_features

class UnifiedEnsembleModel:
    def __init__(self, catboost_weight=0.6):
        self.catboost_weight = catboost_weight
        self.lightgbm_weight = 1 - catboost_weight
        self.catboost_models = []
        self.lightgbm_models = []
        self.cat_features = []
        
    def _get_catboost_params(self, scale_pos_weight):
        return {
            'iterations': 1500,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 6,
            'bagging_temperature': 0.5,
            'border_count': 254,
            'loss_function': 'Logloss',
            'eval_metric': 'PRAUC',
            'scale_pos_weight': scale_pos_weight,
            'random_seed': 42,
            'od_type': 'Iter',
            'od_wait': 150,
            'verbose': False,
            'thread_count': -1
        }
    
    def _get_lightgbm_params(self, scale_pos_weight):
        return {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.01,
            'num_leaves': 128,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.05,
            'lambda_l2': 0.05,
            'min_gain_to_split': 0.01,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'verbosity': -1,
            'force_row_wise': True
        }
    
    def fit(self, X, y, cv_folds=5, checkpoint=None):
        """Train unified ensemble with cross-validation and progress tracking"""
        print("Training unified ensemble model...")
        
        # Initialize progress tracking
        total_steps = cv_folds * 2 + 2  # 2 models per fold + setup + evaluation
        progress = ProgressTracker(total_steps)
        
        if checkpoint is None:
            checkpoint = ModelCheckpoint()
            
        # Check for existing progress - look for individual fold checkpoints
        start_fold = 0
        self.cat_features = [col for col in X.columns if X[col].dtype == 'object']
        oof_catboost = np.zeros(len(X))
        oof_lightgbm = np.zeros(len(X))
        
        # Find the last completed fold
        for fold in range(cv_folds):
            if checkpoint.exists(f"fold_{fold}_complete.pkl"):
                fold_data = checkpoint.load(f"fold_{fold}_complete.pkl")
                if fold >= len(self.catboost_models):
                    self.catboost_models.extend([None] * (fold + 1 - len(self.catboost_models)))
                if fold >= len(self.lightgbm_models):
                    self.lightgbm_models.extend([None] * (fold + 1 - len(self.lightgbm_models)))
                
                self.catboost_models[fold] = fold_data['catboost_model']
                self.lightgbm_models[fold] = fold_data['lightgbm_model']
                oof_catboost[fold_data['val_idx']] = fold_data['oof_catboost']
                oof_lightgbm[fold_data['val_idx']] = fold_data['oof_lightgbm']
                start_fold = fold + 1
                progress.current_step = fold * 2 + 2
                print(f"Resumed from fold {fold}")
        
        if start_fold == 0:
            progress.update("Initializing training")
        else:
            print(f"Resuming training from fold {start_fold}")
            progress.current_step = start_fold * 2 + 1
        
        # Prepare data
        X_processed = X.copy()
        for col in self.cat_features:
            X_processed[col] = X_processed[col].astype(str)
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_splits = list(skf.split(X_processed, y))
        
        for fold in range(start_fold, cv_folds):
            train_idx, val_idx = fold_splits[fold]
            
            X_train, X_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            fraud_ratio = y_train.mean()
            scale_pos_weight = (1 - fraud_ratio) / fraud_ratio
            
            # Train CatBoost
            progress.update(f"Training CatBoost fold {fold + 1}/{cv_folds}")
            train_pool = Pool(X_train, y_train, cat_features=self.cat_features)
            val_pool = Pool(X_val, y_val, cat_features=self.cat_features)
            
            catboost_model = CatBoostClassifier(**self._get_catboost_params(scale_pos_weight))
            catboost_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            
            if fold >= len(self.catboost_models):
                self.catboost_models.append(catboost_model)
            else:
                self.catboost_models[fold] = catboost_model
            
            # Train LightGBM
            progress.update(f"Training LightGBM fold {fold + 1}/{cv_folds}")
            X_train_lgb = X_train.copy()
            X_val_lgb = X_val.copy()
            for col in self.cat_features:
                X_train_lgb[col] = pd.Categorical(X_train_lgb[col]).codes
                X_val_lgb[col] = pd.Categorical(X_val_lgb[col]).codes
            
            dtrain = lgb.Dataset(X_train_lgb, label=y_train)
            dvalid = lgb.Dataset(X_val_lgb, label=y_val, reference=dtrain)
            
            lightgbm_model = lgb.train(
                self._get_lightgbm_params(scale_pos_weight), 
                dtrain, 
                num_boost_round=5000,
                valid_sets=[dtrain, dvalid],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            
            if fold >= len(self.lightgbm_models):
                self.lightgbm_models.append(lightgbm_model)
            else:
                self.lightgbm_models[fold] = lightgbm_model
            
            # Out-of-fold predictions
            val_pool_pred = Pool(X_val, cat_features=self.cat_features)
            fold_oof_catboost = catboost_model.predict_proba(val_pool_pred)[:, 1]
            fold_oof_lightgbm = lightgbm_model.predict(X_val_lgb, num_iteration=lightgbm_model.best_iteration)
            
            oof_catboost[val_idx] = fold_oof_catboost
            oof_lightgbm[val_idx] = fold_oof_lightgbm
            
            # Save individual fold checkpoint
            checkpoint.save({
                'catboost_model': catboost_model,
                'lightgbm_model': lightgbm_model,
                'val_idx': val_idx,
                'oof_catboost': fold_oof_catboost,
                'oof_lightgbm': fold_oof_lightgbm,
                'fold_number': fold
            }, f"fold_{fold}_complete.pkl")
            
            # Save progress checkpoint
            checkpoint.save({
                'completed_folds': fold + 1,
                'total_folds': cv_folds
            }, f"progress_fold_{fold}.pkl")
        
        # Evaluate ensemble performance
        progress.update("Evaluating ensemble performance")
        ensemble_pred = self._combine_predictions(oof_catboost, oof_lightgbm)
        
        print(f"CatBoost OOF AUC: {roc_auc_score(y, oof_catboost):.6f}")
        print(f"LightGBM OOF AUC: {roc_auc_score(y, oof_lightgbm):.6f}")
        print(f"Ensemble OOF AUC: {roc_auc_score(y, ensemble_pred):.6f}")
        print(f"Ensemble OOF PR-AUC: {average_precision_score(y, ensemble_pred):.6f}")
        
        # Save final model and OOF predictions
        checkpoint.save({
            'catboost_models': self.catboost_models,
            'lightgbm_models': self.lightgbm_models,
            'cat_features': self.cat_features,
            'oof_catboost': oof_catboost,
            'oof_lightgbm': oof_lightgbm,
            'training_complete': True
        }, "final_model.pkl")
        
        # Save evaluation results
        checkpoint.save({
            'catboost_auc': roc_auc_score(y, oof_catboost),
            'lightgbm_auc': roc_auc_score(y, oof_lightgbm),
            'ensemble_auc': roc_auc_score(y, ensemble_pred),
            'ensemble_prauc': average_precision_score(y, ensemble_pred)
        }, "evaluation_results.pkl")
        
        return self
    
    def _combine_predictions(self, cat_preds, lgb_preds):
        """Advanced ensemble combination"""
        cat_preds = np.clip(cat_preds, 1e-8, 1-1e-8)
        lgb_preds = np.clip(lgb_preds, 1e-8, 1-1e-8)
        
        # Method 1: Weighted average
        weighted_avg = self.catboost_weight * cat_preds + self.lightgbm_weight * lgb_preds
        
        # Method 2: Geometric mean
        geometric_mean = np.power(cat_preds, self.catboost_weight) * np.power(lgb_preds, self.lightgbm_weight)
        
        # Method 3: Rank averaging
        cat_ranks = pd.Series(cat_preds).rank(pct=True).values
        lgb_ranks = pd.Series(lgb_preds).rank(pct=True).values
        rank_avg = self.catboost_weight * cat_ranks + self.lightgbm_weight * lgb_ranks
        
        # Method 4: Logit space combination
        cat_logits = np.log(cat_preds / (1 - cat_preds))
        lgb_logits = np.log(lgb_preds / (1 - lgb_preds))
        combined_logits = self.catboost_weight * cat_logits + self.lightgbm_weight * lgb_logits
        logit_ensemble = 1 / (1 + np.exp(-combined_logits))
        
        # Meta-ensemble
        final_ensemble = (0.4 * weighted_avg + 0.25 * geometric_mean + 0.2 * rank_avg + 0.15 * logit_ensemble)
        
        return final_ensemble
    
    def predict_proba(self, X):
        """Predict using ensemble of all trained models"""
        X_processed = X.copy()
        for col in self.cat_features:
            if col in X_processed.columns:
                X_processed[col] = X_processed[col].astype(str)
        
        # CatBoost predictions
        catboost_preds = []
        for model in self.catboost_models:
            test_pool = Pool(X_processed, cat_features=self.cat_features)
            catboost_preds.append(model.predict_proba(test_pool)[:, 1])
        
        # LightGBM predictions
        lightgbm_preds = []
        X_lgb = X_processed.copy()
        for col in self.cat_features:
            if col in X_lgb.columns:
                X_lgb[col] = pd.Categorical(X_lgb[col]).codes
        
        for model in self.lightgbm_models:
            lightgbm_preds.append(model.predict(X_lgb, num_iteration=model.best_iteration))
        
        # Average across folds
        avg_catboost = np.mean(catboost_preds, axis=0)
        avg_lightgbm = np.mean(lightgbm_preds, axis=0)
        
        return self._combine_predictions(avg_catboost, avg_lightgbm)

# %% Comprehensive unified ensemble model
print("=== COMPREHENSIVE UNIFIED ENSEMBLE MODEL ===")

# Initialize checkpoint system
checkpoint = ModelCheckpoint()
total_pipeline_steps = 8  # Load, detect, features, prepare, train, test_features, predict, save
pipeline_progress = ProgressTracker(total_pipeline_steps)

# Load data
pipeline_progress.update("Loading datasets")
if checkpoint.exists("data_cache.pkl"):
    print("Loading cached data...")
    cached_data = checkpoint.load("data_cache.pkl")
    transactions, merchants, customers, terminals = cached_data
else:
    transactions, merchants, customers, terminals = load_data()
    checkpoint.save((transactions, merchants, customers, terminals), "data_cache.pkl")

# Detect fraudulent terminals
pipeline_progress.update("Detecting fraudulent terminals")
if checkpoint.exists("fraudulent_terminals.pkl"):
    fraudulent_terminals = checkpoint.load("fraudulent_terminals.pkl")
    print(f"Loaded {len(fraudulent_terminals)} fraudulent terminals from cache")
else:
    fraudulent_terminals = detect_fraudulent_terminals(transactions)
    checkpoint.save(fraudulent_terminals, "fraudulent_terminals.pkl")

# Create comprehensive features
pipeline_progress.update("Creating comprehensive feature set")
if checkpoint.exists("train_features.pkl"):
    print("Loading cached training features...")
    train_data = checkpoint.load("train_features.pkl")
else:
    train_data = create_comprehensive_features(transactions, merchants, customers, terminals, fraudulent_terminals, is_test=False)
    checkpoint.save(train_data, "train_features.pkl")

# Prepare features and target
pipeline_progress.update("Preparing features and target")
X, y, cat_features = prepare_data(train_data, is_test=False)

print(f"Training data: {len(X)} samples, {y.mean():.4f} fraud rate")
print(f"Features: {len(X.columns)} total, {len(cat_features)} categorical")

# Print new feature impact
new_features = ['has_cashback', 'goods_vs_total_mismatch', 'no_auth', 'is_refund', 
               'is_fraud_critical_amount', 'ron_ro_high_amount', 'cashback_no_auth']
available_new = [f for f in new_features if f in train_data.columns]

if available_new:
    print(f"\nNew High-Impact Features Added: {len(available_new)}")
    for feature in available_new:
        if train_data[feature].dtype in ['int64', 'float64']:
            fraud_rate = train_data[train_data[feature] == 1]['TX_FRAUD'].mean()
            count = train_data[train_data[feature] == 1].shape[0]
            print(f"  {feature}: {fraud_rate:.4f} fraud rate ({count} samples)")

# Train unified ensemble
pipeline_progress.update("Training unified ensemble")
ensemble = UnifiedEnsembleModel(catboost_weight=0.6)

completed, total = checkpoint.get_training_progress()
print(f"Training progress: {completed}/{total} folds completed")

# Check if model already trained
if checkpoint.exists("final_model.pkl"):
    print("Loading pre-trained model...")
    model_state = checkpoint.load("final_model.pkl")
    if model_state and model_state.get('training_complete', False):
        ensemble.catboost_models = model_state['catboost_models']
        ensemble.lightgbm_models = model_state['lightgbm_models']
        ensemble.cat_features = model_state['cat_features']
        print("Model loaded successfully!")
        
        # Load evaluation results if available
        eval_results = checkpoint.load("evaluation_results.pkl")
        if eval_results:
            print(f"Previous results - Ensemble AUC: {eval_results['ensemble_auc']:.6f}")
    else:
        print("Final model checkpoint corrupted, retraining...")
        ensemble.fit(X, y, cv_folds=5, checkpoint=checkpoint)
else:
    ensemble.fit(X, y, cv_folds=5, checkpoint=checkpoint)

# Prepare test data
pipeline_progress.update("Preparing test data")
if checkpoint.exists("test_features.pkl"):
    print("Loading cached test features...")
    test_data = checkpoint.load("test_features.pkl")
else:
    test_transactions = pd.read_csv("../data/Payments Fraud DataSet/transactions_test.csv")
    test_data = create_comprehensive_features(test_transactions, merchants, customers, terminals, fraudulent_terminals, is_test=True)
    checkpoint.save(test_data, "test_features.pkl")

# Predict on test data
pipeline_progress.update("Generating predictions")
X_test, _, _ = prepare_data(test_data, is_test=True)
test_predictions = ensemble.predict_proba(X_test)

# Save predictions
pipeline_progress.update("Saving results")
result = pd.DataFrame({
    'TX_ID': test_data['TX_ID'],
    'TX_FRAUD': test_predictions
})

result.to_csv("full_submission.csv", index=False)
print(f"Enhanced predictions saved. Mean fraud probability: {test_predictions.mean():.6f}")
print(f"Total features used: {len(X.columns)}")
print(f"New analysis-based features: {len([f for f in ['has_cashback', 'no_auth', 'is_refund', 'is_fraud_critical_amount'] if f in X.columns])}")

# # Clean up intermediate checkpoints (keep final model and evaluation)
# print("\nCleaning up intermediate checkpoints...")
# cleanup_files = []
# for fold in range(5):  # Assuming 5 folds
#     cleanup_files.extend([f"fold_{fold}_complete.pkl", f"progress_fold_{fold}.pkl"])

# for filename in cleanup_files:
#     filepath = os.path.join(checkpoint.checkpoint_dir, filename)
#     if os.path.exists(filepath):
#         os.remove(filepath)
#         print(f"Removed: {filename}")
        
# print(f"Kept: final_model.pkl, evaluation_results.pkl, and data cache files")
        
print("\n=== TRAINING COMPLETE ===")