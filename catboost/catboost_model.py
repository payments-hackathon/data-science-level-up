import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

def load_data():
    transactions = pd.read_csv("data/Payments Fraud DataSet/transactions_train.csv")
    merchants = pd.read_csv("data/Payments Fraud DataSet/merchants.csv")
    customers = pd.read_csv("data/Payments Fraud DataSet/customers.csv")
    terminals = pd.read_csv("data/Payments Fraud DataSet/terminals.csv")
    return transactions, merchants, customers, terminals

def detect_fraudulent_terminals(train_tx, threshold=1.0, min_count=5):
    terminal_stats = train_tx.groupby('TERMINAL_ID')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
    terminal_stats = terminal_stats[terminal_stats['count'] >= min_count]
    fraudulent_terminals = terminal_stats[terminal_stats['mean'] >= threshold].index.tolist()
    print(f"Found {len(fraudulent_terminals)} terminals with >= {int(threshold*100)}% fraud rate (min {min_count} txns).")
    return fraudulent_terminals

def create_features_v6(transactions, merchants, customers, terminals, fraudulent_terminals, is_test=False):
    transactions["TX_TS"] = pd.to_datetime(transactions["TX_TS"])
    transactions = transactions.sort_values(["CUSTOMER_ID", "TX_TS"])

    transactions = transactions.merge(terminals[['TERMINAL_ID', 'x_terminal_id', 'y_terminal__id']], 
                                      on='TERMINAL_ID', how='left')
    
    transactions['is_fraudulent_terminal'] = transactions['TERMINAL_ID'].isin(fraudulent_terminals).astype(int)

    transactions['distance'] = np.sqrt(
        (transactions['x_customer_id'] - transactions['x_terminal_id'])**2 + 
        (transactions['y_customer_id'] - transactions['y_terminal__id'])**2
    )
    transactions['distance'] = transactions['distance'].fillna(0)

    transactions["hour"] = transactions["TX_TS"].dt.hour
    transactions["dayofweek"] = transactions["TX_TS"].dt.dayofweek
    transactions["is_weekend"] = transactions["dayofweek"].isin([5, 6]).astype(int)
    transactions['is_night'] = ((transactions['hour'] >= 22) | (transactions['hour'] <= 6)).astype(int)
    transactions['TX_AMOUNT_log'] = np.log1p(transactions['TX_AMOUNT'])
    transactions['is_fraud_amount_range'] = ((transactions['TX_AMOUNT'] >= 500) & (transactions['TX_AMOUNT'] <= 1000)).astype(int)

    transactions['is_ron_currency'] = (transactions['TRANSACTION_CURRENCY'] == 'RON').astype(int)
    if 'CARD_COUNTRY_CODE' in transactions.columns:
        transactions['is_ro_country'] = (transactions['CARD_COUNTRY_CODE'] == 'RO').astype(int)
        transactions['ron_ro_combined'] = transactions['is_ron_currency'] * transactions['is_ro_country']
    else:
        transactions['is_ro_country'] = 0
        transactions['ron_ro_combined'] = 0

    transactions["time_since_last_tx"] = (
        transactions.groupby("CUSTOMER_ID")["TX_TS"]
        .diff()
        .dt.total_seconds()
        .fillna(3600 * 24)
    )
    transactions["time_since_last_tx_merchant"] = (
        transactions.groupby(["CUSTOMER_ID", "MERCHANT_ID"])["TX_TS"]
        .diff()
        .dt.total_seconds()
        .fillna(3600 * 24 * 30)
    )
    transactions["cust_tx_count_cumm"] = transactions.groupby("CUSTOMER_ID").cumcount() + 1
    transactions["cust_merchant_pair_count_cumm"] = transactions.groupby(
        ["CUSTOMER_ID", "MERCHANT_ID"]
    ).cumcount()

    median_amount = transactions["TX_AMOUNT"].median()
    transactions["cust_avg_amount_cumm"] = (
        transactions.groupby("CUSTOMER_ID")["TX_AMOUNT"]
        .transform(lambda x: x.expanding().mean().shift(1))
        .fillna(median_amount)
    )
    transactions["cust_std_amount_cumm"] = (
        transactions.groupby("CUSTOMER_ID")["TX_AMOUNT"]
        .transform(lambda x: x.expanding().std().shift(1))
        .fillna(0)
    )
    transactions["cust_max_amount_cumm"] = (
        transactions.groupby("CUSTOMER_ID")["TX_AMOUNT"]
        .transform(lambda x: x.expanding().max().shift(1))
        .fillna(median_amount)
    )

    for window in [3, 5, 10]:
        transactions[f"cust_amount_mean_last_{window}"] = (
            transactions.groupby("CUSTOMER_ID")["TX_AMOUNT"]
            .transform(lambda x: x.shift().rolling(window).mean())
            .fillna(median_amount)
        )
        transactions[f"cust_amount_std_last_{window}"] = (
            transactions.groupby("CUSTOMER_ID")["TX_AMOUNT"]
            .transform(lambda x: x.shift().rolling(window).std())
            .fillna(0)
        )
        transactions[f"cust_tx_count_last_{window}"] = (
            transactions.groupby("CUSTOMER_ID")["TX_TS"]
            .transform(lambda x: x.shift().rolling(window).count())
            .fillna(0)
        )

    transactions["amount_deviation"] = (
        (transactions["TX_AMOUNT"] - transactions["cust_avg_amount_cumm"])
        / transactions["cust_std_amount_cumm"].replace(0, 1)
    ).fillna(0)
    transactions["amount_to_max_ratio"] = transactions["TX_AMOUNT"] / transactions[
        "cust_max_amount_cumm"
    ].replace(0, 1)

    final_cols = [
        "TX_ID", "TX_TS", "TX_AMOUNT", "TX_AMOUNT_log", "CUSTOMER_ID", "MERCHANT_ID",
        "MCC_CODE", "TERMINAL_ID", "CARD_BRAND", "TRANSACTION_CURRENCY", 
        "x_customer_id", "y_customer_id",
        "is_fraudulent_terminal", "distance", "is_night", "is_fraud_amount_range",
        "is_ron_currency", "is_ro_country", "ron_ro_combined",
        "hour", "dayofweek", "is_weekend", "time_since_last_tx", 
        "time_since_last_tx_merchant", "cust_tx_count_cumm", 
        "cust_merchant_pair_count_cumm", "cust_avg_amount_cumm", 
        "cust_std_amount_cumm", "cust_max_amount_cumm", "amount_deviation", 
        "amount_to_max_ratio"
    ]

    for w in [3, 5, 10]:
        final_cols += [
            f"cust_amount_mean_last_{w}", f"cust_amount_std_last_{w}", f"cust_tx_count_last_{w}",
        ]

    if not is_test:
        final_cols.append("TX_FRAUD")

    return transactions[[c for c in final_cols if c in transactions.columns]].drop_duplicates(
        subset="TX_ID"
    )

def get_categorical_features(df):
    cat = [
        "CUSTOMER_ID", "hour", "dayofweek", "TRANSACTION_CURRENCY", 
        "MCC_CODE", "TERMINAL_ID", "CARD_BRAND", "MERCHANT_ID",
        "is_fraudulent_terminal", "is_night", "is_weekend",
        "is_fraud_amount_range", "is_ron_currency", "is_ro_country", "ron_ro_combined"
    ]
    available = [c for c in cat if c in df.columns]
    for c in available:
        if df[c].isnull().any():
            df[c] = df[c].fillna("MISSING")
        df[c] = df[c].astype(str)
    return available

def prepare_data(df, is_test=False):
    y = df["TX_FRAUD"] if not is_test else None
    
    drop_cols = ["TX_ID", "TX_TS", "x_customer_id", "y_customer_id"] + ([] if is_test else ["TX_FRAUD"])
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    for c in X.select_dtypes(include=[np.number]).columns:
        if X[c].isnull().any():
            X[c] = X[c].fillna(X[c].median())
            
    cat_features = get_categorical_features(X)
    return X, y, cat_features

def train_catboost_v6(X, y, df_time, cat_features):
    df = X.copy()
    df["TX_FRAUD"] = y
    df["TX_TS"] = df_time
    df = df.sort_values("TX_TS")

    split_point = int(len(df) * 0.8)
    X_train, X_val = (
        df.iloc[:split_point].drop(columns=["TX_FRAUD", "TX_TS"]),
        df.iloc[split_point:].drop(columns=["TX_FRAUD", "TX_TS"]),
    )
    y_train, y_val = df.iloc[:split_point]["TX_FRAUD"], df.iloc[split_point:]["TX_FRAUD"]

    fraud_ratio = y_train.mean()
    scale_pos_weight = (1 - fraud_ratio) / fraud_ratio

    print(
        f"Time split:\n  Train {df['TX_TS'].min()} -> {df.iloc[split_point-1]['TX_TS']}\n"
        f"  Valid {df.iloc[split_point]['TX_TS']} -> {df['TX_TS'].max()}"
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=6,
        bagging_temperature=0.5,
        border_count=254,
        loss_function="Logloss",
        eval_metric="PRAUC",
        scale_pos_weight=scale_pos_weight,
        random_seed=42,
        od_type="Iter",
        od_wait=150,
        verbose=200,
        thread_count=-1,
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True, plot=False)
    return model, X_val, y_val

def evaluate_model(model, X_val, y_val, cat_features):
    val_pool = Pool(X_val, cat_features=cat_features)
    y_pred = model.predict_proba(val_pool)[:, 1]
    precision, recall, thr = precision_recall_curve(y_val, y_pred)
    f1 = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    best_idx = np.argmax(f1)
    return {
        "ROC_AUC": roc_auc_score(y_val, y_pred),
        "PR_AUC": average_precision_score(y_val, y_pred),
        "Best_Threshold": thr[best_idx],
    }

def feature_importance_df(model, feature_names):
    return pd.DataFrame(
        {"feature": feature_names, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False)

def predict_on_test(model, train_tx_for_fraud_terminals):
    print("\n--- Predicting on Test Set (V6) ---")
    
    test_tx = pd.read_csv("data/Payments Fraud DataSet/transactions_test.csv")
    merchants = pd.read_csv("data/Payments Fraud DataSet/merchants.csv")
    customers = pd.read_csv("data/Payments Fraud DataSet/customers.csv")
    terminals = pd.read_csv("data/Payments Fraud DataSet/terminals.csv")

    fraudulent_terminals = detect_fraudulent_terminals(train_tx_for_fraud_terminals)

    test_merged = test_tx.merge(merchants, on="MERCHANT_ID", how="left").merge(
        customers, on="CUSTOMER_ID", how="left"
    )

    test_feat = create_features_v6(test_merged, merchants, customers, terminals, fraudulent_terminals, is_test=True)
    X_test, _, cat_features = prepare_data(test_feat, is_test=True)
    test_pool = Pool(X_test, cat_features=cat_features)
    proba = model.predict_proba(test_pool)[:, 1]

    out = pd.DataFrame({"TX_ID": test_feat["TX_ID"], "TX_FRAUD": proba})
    out.to_csv("predictions_catboost_v6.csv", index=False)
    print(f"Saved predictions -> predictions_catboost_v6.csv")
    print(f"Mean fraud probability: {proba.mean():.6f}")
    return out

def main():
    print("=== CATBOOST FRAUD V6: V5 + Enhanced Spatial/Outlier Features ===")
    
    transactions, merchants, customers, terminals = load_data()
    
    fraudulent_terminals = detect_fraudulent_terminals(transactions) 

    train_merged = (
        transactions.merge(merchants, on="MERCHANT_ID", how="left")
        .merge(customers, on="CUSTOMER_ID", how="left")
    )

    feat_df = create_features_v6(train_merged, merchants, customers, terminals, fraudulent_terminals, is_test=False)
    
    X, y, cat_features = prepare_data(feat_df, is_test=False)
    
    model, X_val, y_val = train_catboost_v6(X, y, feat_df["TX_TS"], cat_features)

    metrics = evaluate_model(model, X_val, y_val, cat_features)
    print("\nValidation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if "Threshold" not in k else f"{k}: {v:.4f}")

    fi = feature_importance_df(model, X.columns)
    fi.to_csv("catboost_feature_importance_v6.csv", index=False)
    print("Feature importance saved -> catboost_feature_importance_v6.csv")

    predict_on_test(model, transactions)

if __name__ == "__main__":
    main()