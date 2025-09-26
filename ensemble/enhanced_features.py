import pandas as pd
import numpy as np

def add_enhanced_features(df):
    """Add new high-correlation features discovered from analysis"""
    
    # === TOP CORRELATION FEATURES ===
    
    # 1. CASHBACK PATTERNS (Strong correlations: 0.0307, 0.0282, 0.0211)
    if 'TRANSACTION_CASHBACK_AMOUNT' in df.columns:
        df['has_cashback'] = (df['TRANSACTION_CASHBACK_AMOUNT'] > 0).astype(int)
        df['high_cashback_ratio'] = (df['TRANSACTION_CASHBACK_AMOUNT'] / (df['TX_AMOUNT'] + 1e-8) > 0.1).astype(int)
        df['cashback_amount_log'] = np.log1p(df['TRANSACTION_CASHBACK_AMOUNT'])
    
    # 2. GOODS VS TOTAL MISMATCH (Correlation: 0.0266)
    if 'TRANSACTION_GOODS_AND_SERVICES_AMOUNT' in df.columns:
        df['goods_vs_total_mismatch'] = (np.abs(df['TRANSACTION_GOODS_AND_SERVICES_AMOUNT'] - df['TX_AMOUNT']) > 1).astype(int)
        df['goods_amount_log'] = np.log1p(df['TRANSACTION_GOODS_AND_SERVICES_AMOUNT'])
    
    # 3. AUTHENTICATION PATTERNS (Correlations: 0.0242, -0.0187)
    if 'CARDHOLDER_AUTH_METHOD' in df.columns:
        df['no_auth'] = (df['CARDHOLDER_AUTH_METHOD'] == 'No CVM performed').astype(int)
        df['pin_auth'] = df['CARDHOLDER_AUTH_METHOD'].str.contains('PIN', na=False).astype(int)
        df['signature_only'] = (df['CARDHOLDER_AUTH_METHOD'] == 'Signature').astype(int)
        df['offline_auth'] = df['CARDHOLDER_AUTH_METHOD'].str.contains('Offline', na=False).astype(int)
    
    # 4. GEOGRAPHIC CENTER FEATURES (Correlations: 0.0125, -0.0124)
    if all(col in df.columns for col in ['x_customer_id', 'y_customer_id', 'x_terminal_id', 'y_terminal__id']):
        df['distance_from_center_cust'] = np.sqrt((df['x_customer_id'] - 50)**2 + (df['y_customer_id'] - 50)**2)
        df['distance_from_center_term'] = np.sqrt((df['x_terminal_id'] - 50)**2 + (df['y_terminal__id'] - 50)**2)
        df['customer_in_center'] = ((df['x_customer_id'] > 25) & (df['x_customer_id'] < 75) & 
                                   (df['y_customer_id'] > 25) & (df['y_customer_id'] < 75)).astype(int)
        df['terminal_in_center'] = ((df['x_terminal_id'] > 25) & (df['x_terminal_id'] < 75) & 
                                   (df['y_terminal__id'] > 25) & (df['y_terminal__id'] < 75)).astype(int)
        df['both_in_center'] = df['customer_in_center'] * df['terminal_in_center']
    
    # === HIGH-RISK CATEGORICAL PATTERNS ===
    
    # 5. TRANSACTION TYPE RISK (Refund: 6.67%, Purchase with cashback: 4.92%)
    if 'TRANSACTION_TYPE' in df.columns:
        df['is_refund'] = (df['TRANSACTION_TYPE'] == 'Refund').astype(int)
        df['is_purchase_cashback'] = (df['TRANSACTION_TYPE'] == 'Purchase with cashback').astype(int)
        df['is_cash_advance'] = (df['TRANSACTION_TYPE'] == 'Cash Advance/Withdrawal').astype(int)
    
    # 6. CURRENCY RISK (RON: 9.09% fraud rate!)
    if 'TRANSACTION_CURRENCY' in df.columns:
        df['is_ron_currency'] = (df['TRANSACTION_CURRENCY'] == 'RON').astype(int)
        df['is_aed_currency'] = (df['TRANSACTION_CURRENCY'] == 'AED').astype(int)
        df['is_high_risk_currency'] = df['TRANSACTION_CURRENCY'].isin(['RON', 'AED']).astype(int)
    
    # 7. COUNTRY RISK (AE, RO, FI high risk)
    if 'CARD_COUNTRY_CODE' in df.columns:
        df['is_high_risk_country'] = df['CARD_COUNTRY_CODE'].isin(['AE', 'RO', 'FI']).astype(int)
        df['is_ro_country'] = (df['CARD_COUNTRY_CODE'] == 'RO').astype(int)
        df['ron_ro_combined'] = df.get('is_ron_currency', 0) * df['is_ro_country']
    
    # 8. ACQUIRER RISK (ACQ5, ACQ6 higher risk)
    if 'ACQUIRER_ID' in df.columns:
        df['is_high_risk_acquirer'] = df['ACQUIRER_ID'].isin(['ACQ5', 'ACQ6']).astype(int)
    
    # === AMOUNT PATTERN FEATURES ===
    
    # 9. CRITICAL AMOUNT RANGES (100-200: 4.13%, 200-500: 100%!)
    df['is_medium_high_amount'] = ((df['TX_AMOUNT'] >= 100) & (df['TX_AMOUNT'] < 200)).astype(int)
    df['is_very_high_amount'] = (df['TX_AMOUNT'] >= 200).astype(int)
    df['is_fraud_critical_amount'] = ((df['TX_AMOUNT'] >= 100) & (df['TX_AMOUNT'] <= 500)).astype(int)
    
    # 10. ROUND AMOUNT PATTERNS
    df['is_round_amount'] = (df['TX_AMOUNT'] % 10 == 0).astype(int)
    df['is_very_round_amount'] = (df['TX_AMOUNT'] % 100 == 0).astype(int)
    df['amount_has_cents'] = (df['TX_AMOUNT'] % 1 != 0).astype(int)
    
    # === TEMPORAL RISK FEATURES ===
    
    # 11. HIGH-RISK HOURS (4am: 3.03%, 10am: 3.20%)
    if 'TX_TS' in df.columns:
        df['hour'] = df['TX_TS'].dt.hour
        df['is_high_risk_hour'] = df['hour'].isin([4, 8, 10, 11, 18]).astype(int)
        df['is_low_risk_hour'] = df['hour'].isin([12, 20, 23]).astype(int)
        df['is_early_morning'] = ((df['hour'] >= 3) & (df['hour'] <= 7)).astype(int)
    
    # === MERCHANT RISK FEATURES ===
    
    # 12. BUSINESS TYPE RISK (S Corp slightly higher)
    if 'BUSINESS_TYPE' in df.columns:
        df['is_s_corp'] = (df['BUSINESS_TYPE'] == 'S Corporations').astype(int)
        df['is_llc'] = (df['BUSINESS_TYPE'] == 'Limited Liability Company (LLC)').astype(int)
    
    # 13. MERCHANT DEPOSIT RISK
    if all(col in df.columns for col in ['DEPOSIT_REQUIRED_PERCENTAGE', 'DEPOSIT_PERCENTAGE']):
        df['merchant_deposit_risk'] = (df['DEPOSIT_REQUIRED_PERCENTAGE'] > df['DEPOSIT_PERCENTAGE']).astype(int)
        df['deposit_gap'] = df['DEPOSIT_REQUIRED_PERCENTAGE'] - df['DEPOSIT_PERCENTAGE']
    
    # === CARD BRAND PATTERNS ===
    
    # 14. CARD BRAND RISK (Visa slightly higher: 2.72% vs others)
    if 'CARD_BRAND' in df.columns:
        df['is_visa'] = (df['CARD_BRAND'] == 'Visa').astype(int)
        df['is_amex'] = (df['CARD_BRAND'] == 'AMEX').astype(int)  # Lowest risk
    
    # === INTERACTION FEATURES ===
    
    # 15. HIGH-RISK COMBINATIONS
    df['ron_ro_high_amount'] = (df.get('is_ron_currency', 0) * df.get('is_ro_country', 0) * 
                               df.get('is_fraud_critical_amount', 0))
    df['cashback_no_auth'] = df.get('has_cashback', 0) * df.get('no_auth', 0)
    df['refund_high_amount'] = df.get('is_refund', 0) * df.get('is_very_high_amount', 0)
    
    # === DELIVERY PATTERNS ===
    
    # 16. DELIVERY RISK INDICATORS
    if 'DELIVERY_OVER_TWO_WEEKS_PERCENTAGE' in df.columns:
        df['slow_delivery_risk'] = (df['DELIVERY_OVER_TWO_WEEKS_PERCENTAGE'] > 30).astype(int)
    
    if 'DELIVERY_SAME_DAYS_PERCENTAGE' in df.columns:
        df['same_day_delivery'] = (df['DELIVERY_SAME_DAYS_PERCENTAGE'] > 50).astype(int)
    
    # === PAYMENT METHOD PATTERNS ===
    
    # 17. OUTLET TYPE COMBINATIONS
    if 'OUTLET_TYPE' in df.columns:
        df['is_ecommerce_only'] = (df['OUTLET_TYPE'] == 'Ecommerce').astype(int)
        df['is_face_to_face_only'] = (df['OUTLET_TYPE'] == 'Face to Face').astype(int)
        df['is_mixed_outlet'] = (df['OUTLET_TYPE'] == 'Face to Face and Ecommerce').astype(int)
    
    return df

def get_enhanced_categorical_features(df):
    """Get enhanced categorical features including new ones"""
    base_cat = ["CUSTOMER_ID", "MERCHANT_ID", "TERMINAL_ID", "CARD_BRAND", "TRANSACTION_CURRENCY", "MCC_CODE"]
    
    # Add new categorical features
    new_cat = ["is_ron_currency", "is_high_risk_country", "is_high_risk_acquirer", "no_auth", 
              "is_refund", "is_purchase_cashback", "has_cashback", "is_s_corp", "is_visa",
              "customer_in_center", "terminal_in_center", "is_fraud_critical_amount",
              "is_high_risk_hour", "merchant_deposit_risk", "is_ecommerce_only"]
    
    available = [c for c in base_cat + new_cat if c in df.columns]
    
    for c in available:
        if df[c].isnull().any():
            df[c] = df[c].fillna("MISSING")
        df[c] = df[c].astype(str)
    
    return available

def print_feature_summary(df):
    """Print summary of enhanced features"""
    enhanced_features = [
        'has_cashback', 'high_cashback_ratio', 'goods_vs_total_mismatch', 'no_auth', 'pin_auth',
        'distance_from_center_cust', 'distance_from_center_term', 'customer_in_center', 'terminal_in_center',
        'is_refund', 'is_purchase_cashback', 'is_ron_currency', 'is_high_risk_country', 'is_high_risk_acquirer',
        'is_fraud_critical_amount', 'is_very_high_amount', 'is_high_risk_hour', 'merchant_deposit_risk',
        'ron_ro_high_amount', 'cashback_no_auth', 'refund_high_amount'
    ]
    
    available_features = [f for f in enhanced_features if f in df.columns]
    
    print(f"\nENHANCED FEATURES SUMMARY:")
    print(f"Added {len(available_features)} new high-correlation features")
    
    if 'TX_FRAUD' in df.columns:
        print(f"\nTop Enhanced Features by Fraud Rate:")
        for feature in available_features[:10]:
            if df[feature].dtype in ['int64', 'float64']:
                fraud_rate = df[df[feature] == 1]['TX_FRAUD'].mean()
                print(f"  {feature}: {fraud_rate:.4f}")

# Example usage function
def apply_enhanced_features_to_model_data(transactions, merchants, customers, terminals, is_test=False):
    """Apply enhanced features to existing model pipeline"""
    
    # Merge datasets (same as existing pipeline)
    df = (transactions.merge(merchants, on="MERCHANT_ID", how="left")
          .merge(customers, on="CUSTOMER_ID", how="left")
          .merge(terminals, on="TERMINAL_ID", how="left"))
    
    df["TX_TS"] = pd.to_datetime(df["TX_TS"])
    
    # Add enhanced features
    df = add_enhanced_features(df)
    
    # Print summary
    if not is_test:
        print_feature_summary(df)
    
    return df