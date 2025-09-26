import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pearsonr
import warnings
warnings.filterwarnings('ignore')

def load_sample_data(n_samples=100000):
    """Load sample of data for analysis"""
    print("Loading sample data for analysis...")
    transactions = pd.read_csv("data/Payments Fraud DataSet/transactions_train.csv").sample(n_samples, random_state=42)
    merchants = pd.read_csv("data/Payments Fraud DataSet/merchants.csv")
    customers = pd.read_csv("data/Payments Fraud DataSet/customers.csv")
    terminals = pd.read_csv("data/Payments Fraud DataSet/terminals.csv")
    
    # Merge datasets
    df = (transactions.merge(merchants, on="MERCHANT_ID", how="left")
          .merge(customers, on="CUSTOMER_ID", how="left")
          .merge(terminals, on="TERMINAL_ID", how="left"))
    
    df["TX_TS"] = pd.to_datetime(df["TX_TS"])
    return df

def create_potential_features(df):
    """Create potential new features for analysis"""
    print("Creating potential features...")
    
    # === BASIC DISTANCE CALCULATION ===
    df['distance'] = np.sqrt((df['x_customer_id'] - df['x_terminal_id'])**2 + 
                            (df['y_customer_id'] - df['y_terminal__id'])**2)
    df['distance'] = df['distance'].fillna(0)
    
    # === PAYMENT METHOD PATTERNS ===
    df['card_brand_risk'] = df['CARD_BRAND'].map({'AMEX': 0, 'Visa': 1, 'MasterCard': 2, 'Discover': 3}).fillna(1)
    df['is_amex'] = (df['CARD_BRAND'] == 'AMEX').astype(int)
    df['is_discover'] = (df['CARD_BRAND'] == 'Discover').astype(int)
    
    # === TRANSACTION STATUS PATTERNS ===
    df['is_rejected'] = (df['TRANSACTION_STATUS'] == 'Rejected').astype(int)
    df['is_authorized_only'] = (df['TRANSACTION_STATUS'] == 'Authorized').astype(int)
    df['has_failure'] = df['FAILURE_CODE'].notna().astype(int)
    
    # === MERCHANT RISK INDICATORS ===
    df['merchant_deposit_risk'] = (df['DEPOSIT_REQUIRED_PERCENTAGE'] > df['DEPOSIT_PERCENTAGE']).astype(int)
    df['merchant_ecom_heavy'] = (df['PAYMENT_PERCENTAGE_ECOM'] > 50).astype(int)
    df['merchant_face_to_face_heavy'] = (df['PAYMENT_PERCENTAGE_FACE_TO_FACE'] > 80).astype(int)
    df['merchant_moto_risk'] = (df['PAYMENT_PERCENTAGE_MOTO'] > 20).astype(int)
    df['merchant_same_day_delivery'] = (df['DELIVERY_SAME_DAYS_PERCENTAGE'] > 50).astype(int)
    df['merchant_slow_delivery'] = (df['DELIVERY_OVER_TWO_WEEKS_PERCENTAGE'] > 30).astype(int)
    
    # === CARD DATA PATTERNS ===
    df['card_first_digit'] = df['CARD_DATA'].str[0].astype(str)
    df['card_issuer_type'] = df['card_first_digit'].map({'3': 'amex', '4': 'visa', '5': 'mc', '6': 'discover'}).fillna('other')
    df['is_visa_card'] = (df['card_first_digit'] == '4').astype(int)
    df['is_mc_card'] = (df['card_first_digit'] == '5').astype(int)
    
    # === GEOGRAPHIC RISK ===
    df['customer_terminal_same_quadrant'] = ((df['x_customer_id'] > 50) == (df['x_terminal_id'] > 50)) & ((df['y_customer_id'] > 50) == (df['y_terminal__id'] > 50))
    df['customer_in_center'] = ((df['x_customer_id'] > 25) & (df['x_customer_id'] < 75) & (df['y_customer_id'] > 25) & (df['y_customer_id'] < 75)).astype(int)
    df['terminal_in_center'] = ((df['x_terminal_id'] > 25) & (df['x_terminal_id'] < 75) & (df['y_terminal__id'] > 25) & (df['y_terminal__id'] < 75)).astype(int)
    df['distance_from_center_cust'] = np.sqrt((df['x_customer_id'] - 50)**2 + (df['y_customer_id'] - 50)**2)
    df['distance_from_center_term'] = np.sqrt((df['x_terminal_id'] - 50)**2 + (df['y_terminal__id'] - 50)**2)
    
    # === TEMPORAL RISK PATTERNS ===
    df['is_early_morning'] = ((df['TX_TS'].dt.hour >= 3) & (df['TX_TS'].dt.hour <= 7)).astype(int)
    df['is_late_night'] = ((df['TX_TS'].dt.hour >= 23) | (df['TX_TS'].dt.hour <= 2)).astype(int)
    df['is_lunch_time'] = ((df['TX_TS'].dt.hour >= 11) & (df['TX_TS'].dt.hour <= 14)).astype(int)
    df['is_rush_hour'] = (((df['TX_TS'].dt.hour >= 7) & (df['TX_TS'].dt.hour <= 9)) | ((df['TX_TS'].dt.hour >= 17) & (df['TX_TS'].dt.hour <= 19))).astype(int)
    df['day_of_month'] = df['TX_TS'].dt.day
    df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
    df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
    
    # === AMOUNT PATTERNS ===
    df['is_round_amount'] = (df['TX_AMOUNT'] % 10 == 0).astype(int)
    df['is_very_round_amount'] = (df['TX_AMOUNT'] % 100 == 0).astype(int)
    df['amount_has_cents'] = (df['TX_AMOUNT'] % 1 != 0).astype(int)
    df['is_small_amount'] = (df['TX_AMOUNT'] < 10).astype(int)
    df['is_large_amount'] = (df['TX_AMOUNT'] > 1000).astype(int)
    df['is_medium_amount'] = ((df['TX_AMOUNT'] >= 50) & (df['TX_AMOUNT'] <= 200)).astype(int)
    
    # === CASHBACK PATTERNS ===
    df['has_cashback'] = (df['TRANSACTION_CASHBACK_AMOUNT'] > 0).astype(int)
    df['high_cashback_ratio'] = (df['TRANSACTION_CASHBACK_AMOUNT'] / (df['TX_AMOUNT'] + 1e-8) > 0.1).astype(int)
    df['goods_vs_total_mismatch'] = (np.abs(df['TRANSACTION_GOODS_AND_SERVICES_AMOUNT'] - df['TX_AMOUNT']) > 1).astype(int)
    
    # === ACQUIRER PATTERNS ===
    df['acquirer_risk'] = df['ACQUIRER_ID'].map({'ACQ1': 0, 'ACQ2': 1, 'ACQ3': 2, 'ACQ4': 3, 'ACQ5': 4, 'ACQ6': 5}).fillna(2)
    
    # === AUTHENTICATION PATTERNS ===
    df['no_auth'] = (df['CARDHOLDER_AUTH_METHOD'] == 'No CVM performed').astype(int)
    df['signature_only'] = (df['CARDHOLDER_AUTH_METHOD'] == 'Signature').astype(int)
    df['pin_auth'] = df['CARDHOLDER_AUTH_METHOD'].str.contains('PIN', na=False).astype(int)
    
    # === MCC RISK CATEGORIES ===
    high_risk_mccs = ['5999', '5967', '5964', '5962', '5960', '5947', '5945', '5943', '5941']  # Common fraud MCCs
    df['high_risk_mcc'] = df['MCC_CODE'].astype(str).isin(high_risk_mccs).astype(int)
    
    # === BUSINESS TYPE RISK ===
    df['is_llc'] = (df['BUSINESS_TYPE'] == 'Limited Liability Company (LLC)').astype(int)
    df['is_sole_prop'] = (df['BUSINESS_TYPE'] == 'Sole Proprietorships').astype(int)
    
    return df

def analyze_correlations(df):
    """Analyze correlations with fraud"""
    print("\n=== CORRELATION ANALYSIS WITH FRAUD ===")
    
    # Get all potential features
    potential_features = [col for col in df.columns if col not in ['TX_ID', 'TX_TS', 'TX_FRAUD', 'CUSTOMER_ID', 'MERCHANT_ID', 'TERMINAL_ID']]
    
    correlations = []
    
    for feature in potential_features:
        if df[feature].dtype in ['int64', 'float64'] and df[feature].notna().sum() > 100:
            try:
                corr, p_value = pearsonr(df[feature].fillna(0), df['TX_FRAUD'])
                correlations.append({
                    'feature': feature,
                    'correlation': corr,
                    'abs_correlation': abs(corr),
                    'p_value': p_value,
                    'fraud_rate_when_1': df[df[feature] == 1]['TX_FRAUD'].mean() if feature.startswith('is_') or feature.startswith('has_') else None,
                    'fraud_rate_when_0': df[df[feature] == 0]['TX_FRAUD'].mean() if feature.startswith('is_') or feature.startswith('has_') else None
                })
            except:
                pass
    
    corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
    
    print("TOP 20 FEATURES BY ABSOLUTE CORRELATION:")
    print("=" * 80)
    for _, row in corr_df.head(20).iterrows():
        print(f"{row['feature']:<35} | Corr: {row['correlation']:>7.4f} | P-val: {row['p_value']:>8.2e}")
        if row['fraud_rate_when_1'] is not None:
            print(f"{'':>35} | When 1: {row['fraud_rate_when_1']:>6.4f} | When 0: {row['fraud_rate_when_0']:>6.4f}")
        print("-" * 80)
    
    return corr_df

def analyze_categorical_patterns(df):
    """Analyze categorical variable patterns"""
    print("\n=== CATEGORICAL PATTERN ANALYSIS ===")
    
    categorical_cols = ['CARD_BRAND', 'TRANSACTION_TYPE', 'TRANSACTION_STATUS', 'TRANSACTION_CURRENCY', 
                       'CARD_COUNTRY_CODE', 'ACQUIRER_ID', 'CARDHOLDER_AUTH_METHOD', 'BUSINESS_TYPE', 'OUTLET_TYPE']
    
    for col in categorical_cols:
        if col in df.columns:
            fraud_by_cat = df.groupby(col)['TX_FRAUD'].agg(['count', 'mean']).sort_values('mean', ascending=False)
            fraud_by_cat = fraud_by_cat[fraud_by_cat['count'] >= 50]  # Min 50 samples
            
            if len(fraud_by_cat) > 1:
                print(f"\n{col} - Fraud Rate by Category:")
                print("-" * 50)
                for category, stats in fraud_by_cat.head(10).iterrows():
                    print(f"{str(category):<25} | Count: {stats['count']:>6} | Fraud Rate: {stats['mean']:>7.4f}")

def analyze_amount_patterns(df):
    """Analyze transaction amount patterns"""
    print("\n=== AMOUNT PATTERN ANALYSIS ===")
    
    # Amount bins
    df['amount_bin'] = pd.cut(df['TX_AMOUNT'], bins=[0, 10, 50, 100, 200, 500, 1000, 2000, float('inf')], 
                             labels=['<10', '10-50', '50-100', '100-200', '200-500', '500-1000', '1000-2000', '>2000'])
    
    amount_fraud = df.groupby('amount_bin')['TX_FRAUD'].agg(['count', 'mean'])
    print("Fraud Rate by Amount Range:")
    print("-" * 40)
    for bin_name, stats in amount_fraud.iterrows():
        print(f"{str(bin_name):<12} | Count: {stats['count']:>7} | Fraud Rate: {stats['mean']:>7.4f}")
    
    # Round amounts
    print(f"\nRound Amount Analysis:")
    print(f"Round amounts (divisible by 10): {df[df['is_round_amount'] == 1]['TX_FRAUD'].mean():.4f}")
    print(f"Very round amounts (divisible by 100): {df[df['is_very_round_amount'] == 1]['TX_FRAUD'].mean():.4f}")
    print(f"Amounts with cents: {df[df['amount_has_cents'] == 1]['TX_FRAUD'].mean():.4f}")

def analyze_temporal_patterns(df):
    """Analyze temporal fraud patterns"""
    print("\n=== TEMPORAL PATTERN ANALYSIS ===")
    
    # Hour analysis
    hour_fraud = df.groupby(df['TX_TS'].dt.hour)['TX_FRAUD'].agg(['count', 'mean'])
    print("Fraud Rate by Hour:")
    print("-" * 30)
    for hour, stats in hour_fraud.iterrows():
        print(f"Hour {hour:>2} | Count: {stats['count']:>6} | Fraud Rate: {stats['mean']:>7.4f}")
    
    # Day of week
    dow_fraud = df.groupby(df['TX_TS'].dt.dayofweek)['TX_FRAUD'].agg(['count', 'mean'])
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(f"\nFraud Rate by Day of Week:")
    print("-" * 35)
    for dow, stats in dow_fraud.iterrows():
        print(f"{dow_names[dow]} | Count: {stats['count']:>7} | Fraud Rate: {stats['mean']:>7.4f}")

def analyze_geographic_patterns(df):
    """Analyze geographic fraud patterns"""
    print("\n=== GEOGRAPHIC PATTERN ANALYSIS ===")
    
    # Distance analysis
    df['distance_bin'] = pd.cut(df['distance'], bins=[0, 5, 10, 20, 50, 100, float('inf')], 
                               labels=['<5', '5-10', '10-20', '20-50', '50-100', '>100'])
    
    distance_fraud = df.groupby('distance_bin')['TX_FRAUD'].agg(['count', 'mean'])
    print("Fraud Rate by Distance:")
    print("-" * 35)
    for dist, stats in distance_fraud.iterrows():
        print(f"{str(dist):<8} | Count: {stats['count']:>7} | Fraud Rate: {stats['mean']:>7.4f}")
    
    # Quadrant analysis
    print(f"\nGeographic Position Analysis:")
    print(f"Same quadrant (cust/term): {df[df['customer_terminal_same_quadrant']]['TX_FRAUD'].mean():.4f}")
    print(f"Customer in center: {df[df['customer_in_center'] == 1]['TX_FRAUD'].mean():.4f}")
    print(f"Terminal in center: {df[df['terminal_in_center'] == 1]['TX_FRAUD'].mean():.4f}")

def main():
    """Main analysis function"""
    print("FRAUD DETECTION FEATURE ANALYSIS")
    print("=" * 50)
    
    # Load sample data
    df = load_sample_data(100000)
    print(f"Loaded {len(df)} transactions, fraud rate: {df['TX_FRAUD'].mean():.4f}")
    
    # Create potential features
    df = create_potential_features(df)
    
    # Run analyses
    corr_df = analyze_correlations(df)
    analyze_categorical_patterns(df)
    analyze_amount_patterns(df)
    analyze_temporal_patterns(df)
    analyze_geographic_patterns(df)
    
    # Save correlation results
    corr_df.to_csv("feature_correlation_analysis.csv", index=False)
    print(f"\nCorrelation analysis saved to feature_correlation_analysis.csv")
    
    # Summary of top insights
    print("\n" + "=" * 60)
    print("TOP INSIGHTS FOR NEW FEATURES:")
    print("=" * 60)
    
    top_features = corr_df.head(10)
    for _, row in top_features.iterrows():
        if abs(row['correlation']) > 0.01:  # Only significant correlations
            print(f"• {row['feature']}: correlation = {row['correlation']:.4f}")
    
    return corr_df

if __name__ == "__main__":
    results = main()