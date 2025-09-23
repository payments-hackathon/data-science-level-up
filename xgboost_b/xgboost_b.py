import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_data():
    transactions = pd.read_csv('data/Payments Fraud DataSet/transactions_train.csv')
    merchants = pd.read_csv('data/Payments Fraud DataSet/merchants.csv')
    customers = pd.read_csv('data/Payments Fraud DataSet/customers.csv')
    return transactions, merchants, customers

def create_features(transactions, merchants, customers):
    transactions['TX_TS'] = pd.to_datetime(transactions['TX_TS'])
    transactions['hour'] = transactions['TX_TS'].dt.hour
    transactions['day_of_week'] = transactions['TX_TS'].dt.dayofweek
    transactions['day_of_month'] = transactions['TX_TS'].dt.day
    transactions['month'] = transactions['TX_TS'].dt.month

    transactions = transactions.sort_values(['CUSTOMER_ID', 'TX_TS'])
    transactions['time_since_last_tx'] = transactions.groupby('CUSTOMER_ID')['TX_TS'].diff().dt.total_seconds().fillna(0)

    transactions['amount_to_avg_ratio'] = transactions.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(lambda x: x / x.mean())

    transactions = transactions.merge(merchants, on='MERCHANT_ID', how='left')
    transactions = transactions.merge(customers, on='CUSTOMER_ID', how='left')

    transactions['card_expiry_year'] = transactions['CARD_EXPIRY_DATE'].str.split('/').str[1].astype(int)
    transactions['card_expiry_month'] = transactions['CARD_EXPIRY_DATE'].str.split('/').str[0].astype(int)
    transactions['card_first_digits'] = transactions['CARD_DATA'].str[:4]
    transactions['card_last_digits'] = transactions['CARD_DATA'].str[-4:]

    categorical_cols = ['CARD_BRAND', 'TRANSACTION_TYPE', 'TRANSACTION_STATUS', 'CARDHOLDER_AUTH_METHOD', 'BUSINESS_TYPE', 'OUTLET_TYPE', 'card_first_digits', 'card_last_digits']
    label_encoders = {}
    for col in categorical_cols:
        if col in transactions.columns:
            le = LabelEncoder()
            transactions[col] = le.fit_transform(transactions[col].astype(str))
            label_encoders[col] = le

    return transactions, label_encoders

# Prepare data for modeling
def prepare_data(transactions):
    feature_cols = ['TX_AMOUNT', 'hour', 'day_of_week', 'day_of_month', 'month', 'time_since_last_tx', 'amount_to_avg_ratio', 'x_customer_id', 'y_customer_id', 'MCC_CODE', 'ANNUAL_TURNOVER', 'AVERAGE_TICKET_SALE_AMOUNT', 'CARD_BRAND', 'TRANSACTION_TYPE', 'TRANSACTION_STATUS', 'CARDHOLDER_AUTH_METHOD', 'BUSINESS_TYPE', 'OUTLET_TYPE', 'card_expiry_year', 'card_expiry_month']
    available_cols = [col for col in feature_cols if col in transactions.columns]
    X = transactions[available_cols]
    y = transactions['TX_FRAUD']
    return X, y

# Train and evaluate model
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    fraud_ratio = y_train.mean()
    scale_pos_weight = (1 - fraud_ratio) / fraud_ratio
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='aucpr')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_pred_proba)
    avg_prec = average_precision_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return model

# Predict on test data
def predict_on_test(model, label_encoders):
    test_transactions = pd.read_csv('data/Payments Fraud DataSet/transactions_test.csv')
    merchants = pd.read_csv('data/Payments Fraud DataSet/merchants.csv')
    customers = pd.read_csv('data/Payments Fraud DataSet/customers.csv')

    test_transactions['TX_TS'] = pd.to_datetime(test_transactions['TX_TS'])
    test_transactions['hour'] = test_transactions['TX_TS'].dt.hour
    test_transactions['day_of_week'] = test_transactions['TX_TS'].dt.dayofweek
    test_transactions['day_of_month'] = test_transactions['TX_TS'].dt.day
    test_transactions['month'] = test_transactions['TX_TS'].dt.month

    test_transactions = test_transactions.sort_values(['CUSTOMER_ID', 'TX_TS'])
    test_transactions['time_since_last_tx'] = test_transactions.groupby('CUSTOMER_ID')['TX_TS'].diff().dt.total_seconds().fillna(0)

    test_transactions['amount_to_avg_ratio'] = test_transactions.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(lambda x: x / x.mean())

    test_transactions = test_transactions.merge(merchants, on='MERCHANT_ID', how='left')
    test_transactions = test_transactions.merge(customers, on='CUSTOMER_ID', how='left')

    test_transactions['card_expiry_year'] = test_transactions['CARD_EXPIRY_DATE'].str.split('/').str[1].astype(int)
    test_transactions['card_expiry_month'] = test_transactions['CARD_EXPIRY_DATE'].str.split('/').str[0].astype(int)
    test_transactions['card_first_digits'] = test_transactions['CARD_DATA'].str[:4]
    test_transactions['card_last_digits'] = test_transactions['CARD_DATA'].str[-4:]

    categorical_cols = ['CARD_BRAND', 'TRANSACTION_TYPE', 'TRANSACTION_STATUS', 'CARDHOLDER_AUTH_METHOD', 'BUSINESS_TYPE', 'OUTLET_TYPE', 'card_first_digits', 'card_last_digits']
    for col in categorical_cols:
        if col in test_transactions.columns and col in label_encoders:
            test_transactions[col] = test_transactions[col].astype(str)
            unseen_mask = ~test_transactions[col].isin(label_encoders[col].classes_)
            test_transactions.loc[unseen_mask, col] = label_encoders[col].classes_[0]
            test_transactions[col] = label_encoders[col].transform(test_transactions[col])

    feature_cols = ['TX_AMOUNT', 'hour', 'day_of_week', 'day_of_month', 'month', 'time_since_last_tx', 'amount_to_avg_ratio', 'x_customer_id', 'y_customer_id', 'MCC_CODE', 'ANNUAL_TURNOVER', 'AVERAGE_TICKET_SALE_AMOUNT', 'CARD_BRAND', 'TRANSACTION_TYPE', 'TRANSACTION_STATUS', 'CARDHOLDER_AUTH_METHOD', 'BUSINESS_TYPE', 'OUTLET_TYPE', 'card_expiry_year', 'card_expiry_month']
    available_cols = [col for col in feature_cols if col in test_transactions.columns]
    X_test = test_transactions[available_cols]

    predictions = model.predict_proba(X_test)[:, 1]
    output = pd.DataFrame({'TX_ID': test_transactions['TX_ID'], 'TX_FRAUD': predictions}).sort_values('TX_FRAUD', ascending=False)
    output.to_csv('predictions.csv', index=False)
    return output

# Main execution
def main():
    transactions, merchants, customers = load_data()
    transactions_with_features, label_encoders = create_features(transactions, merchants, customers)
    X, y = prepare_data(transactions_with_features)
    model = train_and_evaluate(X, y)
    predictions = predict_on_test(model, label_encoders)
    # process completed

if __name__ == "__main__":
    main()