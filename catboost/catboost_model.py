import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

def load_data():
    transactions = pd.read_csv('data/Payments Fraud DataSet/transactions_train.csv')
    merchants = pd.read_csv('data/Payments Fraud DataSet/merchants.csv')
    customers = pd.read_csv('data/Payments Fraud DataSet/customers.csv')
    return transactions, merchants, customers

def create_features_no_leakage(transactions, merchants, customers):
    transactions['TX_TS'] = pd.to_datetime(transactions['TX_TS'])
    
    transactions['hour'] = transactions['TX_TS'].dt.hour
    transactions['day_of_week'] = transactions['TX_TS'].dt.dayofweek
    transactions['day_of_month'] = transactions['TX_TS'].dt.day
    transactions['month'] = transactions['TX_TS'].dt.month
    transactions['is_weekend'] = transactions['day_of_week'].isin([5, 6]).astype(int)
    transactions['is_night'] = ((transactions['hour'] >= 22) | (transactions['hour'] <= 6)).astype(int)
    
    transactions = transactions.sort_values(['CUSTOMER_ID', 'TX_TS'])
    
    transactions['time_since_last_tx'] = transactions.groupby('CUSTOMER_ID')['TX_TS'].diff().dt.total_seconds().fillna(24*3600)
    
    customer_stats = transactions.groupby('CUSTOMER_ID').agg({
        'TX_AMOUNT': ['mean', 'std', 'max', 'min', 'count'],
        'TX_TS': lambda x: (x.max() - x.min()).total_seconds() if len(x) > 1 else 0
    }).reset_index()
    customer_stats.columns = ['CUSTOMER_ID', 'cust_avg_amount', 'cust_std_amount', 
                             'cust_max_amount', 'cust_min_amount', 'cust_tx_count', 'cust_activity_period']
    
    transactions = transactions.merge(customer_stats, on='CUSTOMER_ID', how='left')
    
    transactions['amount_deviation'] = (transactions['TX_AMOUNT'] - transactions['cust_avg_amount']) / transactions['cust_std_amount'].replace(0, 1)
    transactions['amount_deviation'] = transactions['amount_deviation'].fillna(0)
    transactions['amount_to_max_ratio'] = transactions['TX_AMOUNT'] / transactions['cust_max_amount'].replace(0, 1)
    transactions['is_high_amount'] = (transactions['TX_AMOUNT'] > transactions['cust_avg_amount'] * 3).astype(int)
    
    transactions['card_expiry_year'] = transactions['CARD_EXPIRY_DATE'].str.split('/').str[1].astype(int) + 2000
    transactions['card_age'] = 2024 - transactions['card_expiry_year']
    
    transactions['tx_hour_of_week'] = transactions['day_of_week'] * 24 + transactions['hour']
    
    transactions = transactions.merge(merchants, on='MERCHANT_ID', how='left')
    
    transactions = transactions.merge(customers, on='CUSTOMER_ID', how='left')
    
    transactions['customer_activity_intensity'] = transactions['cust_tx_count'] / (transactions['cust_activity_period'] / 3600 + 1)
    transactions['amount_time_ratio'] = transactions['TX_AMOUNT'] / (transactions['time_since_last_tx'] + 1)
    
    return transactions

def get_categorical_features(transactions):
    categorical_features = [
        'CARD_BRAND', 'TRANSACTION_TYPE', 'TRANSACTION_STATUS', 
        'CARDHOLDER_AUTH_METHOD', 'BUSINESS_TYPE', 'OUTLET_TYPE',
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_night'
    ]
    
    available_categorical = [col for col in categorical_features if col in transactions.columns]
    
    for col in available_categorical:
        if transactions[col].isnull().any():
            transactions[col] = transactions[col].fillna('MISSING')
        transactions[col] = transactions[col].astype(str)
    
    return available_categorical

def prepare_data(transactions):
    feature_cols = [
        'TX_AMOUNT',
        'hour', 'day_of_week', 'day_of_month', 'month',
        'is_weekend', 'is_night', 'time_since_last_tx', 'tx_hour_of_week',
        'amount_deviation', 'cust_avg_amount', 'cust_std_amount', 
        'cust_max_amount', 'cust_tx_count', 'cust_activity_period',
        'amount_to_max_ratio', 'is_high_amount', 'customer_activity_intensity',
        'amount_time_ratio',
        'card_age',
        'x_customer_id', 'y_customer_id',
        'MCC_CODE', 'ANNUAL_TURNOVER', 'AVERAGE_TICKET_SALE_AMOUNT',
        'CARD_BRAND', 'TRANSACTION_TYPE', 'TRANSACTION_STATUS', 
        'CARDHOLDER_AUTH_METHOD', 'BUSINESS_TYPE', 'OUTLET_TYPE'
    ]
    
    available_cols = [col for col in feature_cols if col in transactions.columns]
    X = transactions[available_cols]
    y = transactions['TX_FRAUD']
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    
    cat_features = get_categorical_features(X)
    
    return X, y, cat_features, available_cols

def train_catboost_model(X, y, cat_features):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    fraud_ratio = y_train.mean()
    scale_pos_weight = (1 - fraud_ratio) / fraud_ratio
    
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)
    
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        border_count=254,
        loss_function='Logloss',
        eval_metric='AUC',
        scale_pos_weight=scale_pos_weight,
        random_seed=42,
        od_type='Iter',
        od_wait=50,
        verbose=False,
        thread_count=-1,
    )
    
    model.fit(
        train_pool,
        eval_set=test_pool,
        use_best_model=True,
        plot=False
    )
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test, cat_features):
    test_pool = Pool(X_test, cat_features=cat_features)
    
    y_pred_proba = model.predict_proba(test_pool)[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    return optimal_threshold, roc_auc, avg_precision

def analyze_feature_importance(model, feature_names):
    feature_importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    return importance_df

def perform_cross_validation(X, y, cat_features):
    fraud_ratio = y.mean()
    scale_pos_weight = (1 - fraud_ratio) / fraud_ratio
    
    pool = Pool(X, y, cat_features=cat_features)
    
    params = {
        'iterations': 200,
        'learning_rate': 0.05,
        'depth': 6,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'scale_pos_weight': scale_pos_weight,
        'random_seed': 42,
        'verbose': False
    }
    
    cv_data = cv(
        pool=pool,
        params=params,
        fold_count=5,
        stratified=True,
        shuffle=True,
        partition_random_seed=42
    )
    
    cv_scores = cv_data['test-AUC-mean'].values
    best_iteration = cv_data['test-AUC-mean'].idxmax() + 1
    best_score = cv_data['test-AUC-mean'].max()
    
    return cv_scores, best_score

def predict_on_test_catboost(model, cat_features):
    test_transactions = pd.read_csv('data/Payments Fraud DataSet/transactions_test.csv')
    merchants = pd.read_csv('data/Payments Fraud DataSet/merchants.csv')
    customers = pd.read_csv('data/Payments Fraud DataSet/customers.csv')
    
    test_transactions['TX_TS'] = pd.to_datetime(test_transactions['TX_TS'])
    
    test_transactions['hour'] = test_transactions['TX_TS'].dt.hour
    test_transactions['day_of_week'] = test_transactions['TX_TS'].dt.dayofweek
    test_transactions['day_of_month'] = test_transactions['TX_TS'].dt.day
    test_transactions['month'] = test_transactions['TX_TS'].dt.month
    test_transactions['is_weekend'] = test_transactions['day_of_week'].isin([5, 6]).astype(int)
    test_transactions['is_night'] = ((test_transactions['hour'] >= 22) | (test_transactions['hour'] <= 6)).astype(int)
    
    test_transactions = test_transactions.sort_values(['CUSTOMER_ID', 'TX_TS'])
    test_transactions['time_since_last_tx'] = test_transactions.groupby('CUSTOMER_ID')['TX_TS'].diff().dt.total_seconds().fillna(24*3600)
    
    customer_stats = test_transactions.groupby('CUSTOMER_ID').agg({
        'TX_AMOUNT': ['mean', 'std', 'max', 'count'],
        'TX_TS': lambda x: (x.max() - x.min()).total_seconds() if len(x) > 1 else 0
    }).reset_index()
    customer_stats.columns = ['CUSTOMER_ID', 'cust_avg_amount', 'cust_std_amount', 'cust_max_amount', 'cust_tx_count', 'cust_activity_period']
    
    test_transactions = test_transactions.merge(customer_stats, on='CUSTOMER_ID', how='left')
    test_transactions['amount_deviation'] = (test_transactions['TX_AMOUNT'] - test_transactions['cust_avg_amount']) / test_transactions['cust_std_amount'].replace(0, 1)
    test_transactions['amount_deviation'] = test_transactions['amount_deviation'].fillna(0)
    test_transactions['amount_to_max_ratio'] = test_transactions['TX_AMOUNT'] / test_transactions['cust_max_amount'].replace(0, 1)
    test_transactions['is_high_amount'] = (test_transactions['TX_AMOUNT'] > test_transactions['cust_avg_amount'] * 3).astype(int)
    test_transactions['customer_activity_intensity'] = test_transactions['cust_tx_count'] / (test_transactions['cust_activity_period'] / 3600 + 1)
    test_transactions['amount_time_ratio'] = test_transactions['TX_AMOUNT'] / (test_transactions['time_since_last_tx'] + 1)
    
    test_transactions['card_expiry_year'] = test_transactions['CARD_EXPIRY_DATE'].str.split('/').str[1].astype(int) + 2000
    test_transactions['card_age'] = 2024 - test_transactions['card_expiry_year']
    test_transactions['tx_hour_of_week'] = test_transactions['day_of_week'] * 24 + test_transactions['hour']
    
    test_transactions = test_transactions.merge(merchants, on='MERCHANT_ID', how='left')
    test_transactions = test_transactions.merge(customers, on='CUSTOMER_ID', how='left')
    
    feature_cols = [
        'TX_AMOUNT', 'hour', 'day_of_week', 'day_of_month', 'month',
        'is_weekend', 'is_night', 'time_since_last_tx', 'tx_hour_of_week',
        'amount_deviation', 'cust_avg_amount', 'cust_std_amount', 
        'cust_max_amount', 'cust_tx_count', 'cust_activity_period',
        'amount_to_max_ratio', 'is_high_amount', 'customer_activity_intensity',
        'amount_time_ratio', 'card_age', 'x_customer_id', 'y_customer_id',
        'MCC_CODE', 'ANNUAL_TURNOVER', 'AVERAGE_TICKET_SALE_AMOUNT',
        'CARD_BRAND', 'TRANSACTION_TYPE', 'TRANSACTION_STATUS', 
        'CARDHOLDER_AUTH_METHOD', 'BUSINESS_TYPE', 'OUTLET_TYPE'
    ]
    
    available_cols = [col for col in feature_cols if col in test_transactions.columns]
    X_test = test_transactions[available_cols]
    
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X_test[col].isnull().any():
            X_test[col] = X_test[col].fillna(X_test[col].median())
    
    for cat_feature in cat_features:
        if cat_feature in X_test.columns:
            if X_test[cat_feature].isnull().any():
                X_test[cat_feature] = X_test[cat_feature].fillna('MISSING')
            X_test[cat_feature] = X_test[cat_feature].astype(str)
    
    test_pool = Pool(X_test, cat_features=cat_features)
    
    predictions_proba = model.predict_proba(test_pool)[:, 1]
    
    output = pd.DataFrame({
        'TX_ID': test_transactions['TX_ID'],
        'TX_FRAUD': predictions_proba
    }).sort_values('TX_FRAUD', ascending=False)
    
    output.to_csv('predictions_catboost.csv', index=False)
    
    return output, predictions_proba.mean()

def main_catboost():
    transactions, merchants, customers = load_data()
    
    transactions_with_features = create_features_no_leakage(transactions, merchants, customers)
    
    X, y, cat_features, feature_names = prepare_data(transactions_with_features)
    
    try:
        cv_scores, best_score = perform_cross_validation(X, y, cat_features)
    except Exception as e:
        cv_scores, best_score = None, None
    
    model, X_test, y_test = train_catboost_model(X, y, cat_features)
    
    optimal_threshold, roc_auc, avg_precision = evaluate_model(model, X_test, y_test, cat_features)
    
    feature_importance = analyze_feature_importance(model, feature_names)
    
    predictions, avg_fraud_prob = predict_on_test_catboost(model, cat_features)
    
    feature_importance.to_csv('catboost_feature_importance.csv', index=False)
    
    print("=" * 60)
    print("CATBOOST PROCESS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Final Model Performance:")
    print(f"ROC AUC: {roc_auc:.6f}")
    print(f"Average Precision: {avg_precision:.6f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Average Fraud Probability on Test Set: {avg_fraud_prob:.4f}")
    print(f"Predictions saved to: predictions_catboost.csv")
    print(f"Feature importance saved to: catboost_feature_importance.csv")

if __name__ == "__main__":
    main_catboost()