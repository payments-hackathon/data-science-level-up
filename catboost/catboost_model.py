import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AdvancedFraudDetectionModel:
    
    def __init__(self, random_state=42):
        self.model = None
        self.cat_features = None
        self.feature_names = None
        self.random_state = random_state
        self.feature_importance = None
        
    def load_data(self, transactions_path, merchants_path=None, customers_path=None):

        transactions = pd.read_csv(transactions_path)
        
        # Load additional data if paths provided
        merchants = pd.read_csv(merchants_path) if merchants_path else None
        customers = pd.read_csv(customers_path) if customers_path else None
        
        return transactions, merchants, customers
    
    def merge_datasets(self, transactions, merchants=None, customers=None):

        merged_data = transactions.copy()
        
        if merchants is not None and 'MERCHANT_ID' in transactions.columns:
            merged_data = merged_data.merge(merchants, on='MERCHANT_ID', how='left')
            print(f"Merged with merchants data: {len(merchants)} merchants")
            
        if customers is not None and 'CUSTOMER_ID' in transactions.columns:
            merged_data = merged_data.merge(customers, on='CUSTOMER_ID', how='left')
            print(f"Merged with customers data: {len(customers)} customers")
            
        return merged_data
    
    def create_advanced_features(self, df, is_test=False):

        data = df.copy()
        
        # Convert timestamp and sort by customer and time
        data['TX_TS'] = pd.to_datetime(data['TX_TS'])
        data = data.sort_values(['CUSTOMER_ID', 'TX_TS'])
        
        print("Creating time-based features...")
        # Time-based features
        data['hour'] = data['TX_TS'].dt.hour
        data['day_of_week'] = data['TX_TS'].dt.dayofweek  
        data['day_of_month'] = data['TX_TS'].dt.day
        data['month'] = data['TX_TS'].dt.month
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        data['is_night'] = ((data['hour'] >= 22) | (data['hour'] <= 6)).astype(int)
        
        print("Creating sequential features...")
        # Sequential features (velocity and timing)
        data['time_since_last_tx'] = (data.groupby('CUSTOMER_ID')['TX_TS']
                                     .diff().dt.total_seconds().fillna(86400))  # 24 hours default
        
        # Customer transaction count (cumulative)
        data['cust_tx_count'] = data.groupby('CUSTOMER_ID').cumcount() + 1
        
        print("Creating customer behavioral features...")
        # Customer behavioral features (expanding window to avoid leakage)
        data['cust_avg_amount'] = (data.groupby('CUSTOMER_ID')['TX_AMOUNT']
                                  .transform(lambda x: x.expanding().mean().shift(1))
                                  .fillna(data['TX_AMOUNT'].median()))
        
        data['cust_std_amount'] = (data.groupby('CUSTOMER_ID')['TX_AMOUNT']
                                  .transform(lambda x: x.expanding().std().shift(1))
                                  .fillna(0))
        
        data['cust_max_amount'] = (data.groupby('CUSTOMER_ID')['TX_AMOUNT']
                                  .transform(lambda x: x.expanding().max().shift(1))
                                  .fillna(data['TX_AMOUNT'].median()))
        
        # Amount-based anomaly features
        data['amount_deviation'] = ((data['TX_AMOUNT'] - data['cust_avg_amount']) 
                                   / data['cust_std_amount'].replace(0, 1)).fillna(0)
        
        data['amount_to_max_ratio'] = (data['TX_AMOUNT'] 
                                      / data['cust_max_amount'].replace(0, 1))
        
        data['is_high_amount'] = (data['TX_AMOUNT'] > data['cust_avg_amount'] * 3).astype(int)
        
        print("Creating rolling window features...")
        # Rolling window features for recent behavior
        for window in [3, 5, 10]:
            data[f'amount_mean_last_{window}'] = (
                data.groupby('CUSTOMER_ID')['TX_AMOUNT']
                .transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
                .fillna(data['TX_AMOUNT'].median())
            )
            
            data[f'tx_count_last_{window}'] = (
                data.groupby('CUSTOMER_ID')['TX_TS']
                .transform(lambda x: x.shift().rolling(window, min_periods=1).count())
                .fillna(0)
            )
        
        # Merchant-specific features if available
        if 'MERCHANT_ID' in data.columns:
            print("Creating merchant-specific features...")
            data['time_since_last_tx_merchant'] = (
                data.groupby(['CUSTOMER_ID', 'MERCHANT_ID'])['TX_TS']
                .diff().dt.total_seconds().fillna(86400 * 30)  # 30 days default
            )
            
            data['cust_merchant_tx_count'] = (
                data.groupby(['CUSTOMER_ID', 'MERCHANT_ID']).cumcount()
            )
        
        # Card-related features
        if 'CARD_EXPIRY_DATE' in data.columns:
            print("Creating card-related features...")
            # Extract card expiry information
            expiry_parts = data['CARD_EXPIRY_DATE'].str.split('/')
            data['card_expiry_month'] = expiry_parts.str[0].astype(int, errors='ignore')
            data['card_expiry_year'] = (2000 + expiry_parts.str[1].astype(int, errors='ignore'))
            data['card_age'] = 2024 - data['card_expiry_year']
        
        # Activity intensity
        data['activity_intensity'] = (data['cust_tx_count'] 
                                     / (data['time_since_last_tx'] / 3600 + 1))
        
        return data
    
    def prepare_features_and_target(self, data, is_test=False):

        # Define feature columns
        base_features = [
            'TX_AMOUNT', 'hour', 'day_of_week', 'day_of_month', 'month',
            'is_weekend', 'is_night', 'time_since_last_tx', 'cust_tx_count',
            'cust_avg_amount', 'cust_std_amount', 'cust_max_amount',
            'amount_deviation', 'amount_to_max_ratio', 'is_high_amount',
            'activity_intensity'
        ]
        
        # Add rolling features
        for window in [3, 5, 10]:
            base_features.extend([
                f'amount_mean_last_{window}',
                f'tx_count_last_{window}'
            ])
        
        # Add merchant features if available
        if 'MERCHANT_ID' in data.columns:
            base_features.extend(['time_since_last_tx_merchant', 'cust_merchant_tx_count'])
        
        # Add card features if available
        if 'card_expiry_month' in data.columns:
            base_features.extend(['card_expiry_month', 'card_expiry_year', 'card_age'])
        
        # Add location features if available
        location_features = ['x_customer_id', 'y_customer_id']
        base_features.extend([f for f in location_features if f in data.columns])
        
        # Categorical features
        categorical_features = []
        potential_categoricals = [
            'CUSTOMER_ID', 'TERMINAL_ID', 'CARD_BRAND', 'TRANSACTION_TYPE', 
            'TRANSACTION_STATUS', 'TRANSACTION_CURRENCY', 'CARD_COUNTRY_CODE',
            'MERCHANT_ID', 'CARDHOLDER_AUTH_METHOD', 'ACQUIRER_ID'
        ]
        
        for feat in potential_categoricals:
            if feat in data.columns:
                categorical_features.append(feat)
                base_features.append(feat)
        
        # Add business/outlet type features if available
        business_features = ['BUSINESS_TYPE', 'OUTLET_TYPE', 'MCC_CODE']
        for feat in business_features:
            if feat in data.columns:
                categorical_features.append(feat)
                base_features.append(feat)
        
        # Filter to available columns
        available_features = [col for col in base_features if col in data.columns]
        
        # Prepare feature matrix
        X = data[available_features].copy()
        
        # Numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())
        
        # Categorical features
        available_categoricals = [col for col in categorical_features if col in X.columns]
        for col in available_categoricals:
            if X[col].isnull().any():
                X[col] = X[col].fillna('MISSING')
            X[col] = X[col].astype(str)
        
        # Prepare target
        y = None if is_test else data['TX_FRAUD']
        
        return X, y, available_categoricals
    
    def train_model(self, X, y, cat_features, use_time_split=True, test_size=0.2):

        self.cat_features = cat_features
        self.feature_names = list(X.columns)
        
        if use_time_split and 'TX_TS' in X.index or hasattr(X, 'TX_TS'):
            # Time-based split (simulate real-world deployment)
            print("Using time-based validation split...")
            # This assumes data is already sorted by time
            split_point = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        else:
            # Random split
            print("Using random validation split...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_test)}")
        print(f"Fraud rate in training: {y_train.mean():.4f}")
        print(f"Fraud rate in validation: {y_test.mean():.4f}")
        
        # Handle class imbalance
        fraud_ratio = y_train.mean()
        scale_pos_weight = (1 - fraud_ratio) / fraud_ratio
        
        # Create CatBoost pools
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        test_pool = Pool(X_test, y_test, cat_features=cat_features)
        
        # Configure model with optimized hyperparameters for fraud detection
        self.model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=5,
            border_count=254,
            loss_function='Logloss',
            eval_metric='PRAUC',  # Precision-Recall AUC better for imbalanced data
            scale_pos_weight=scale_pos_weight,
            random_seed=self.random_state,
            od_type='Iter',
            od_wait=100,
            verbose=100,
            thread_count=-1,
            bootstrap_type='Bayesian',
            bagging_temperature=1.0
        )
        
        # Train model
        print("Training CatBoost model...")
        self.model.fit(
            train_pool,
            eval_set=test_pool,
            use_best_model=True,
            plot=False
        )
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.get_feature_importance()
        }).sort_values('importance', ascending=False)
        
        return self._evaluate_model(X_test, y_test)
    
    def _evaluate_model(self, X_test, y_test):

        test_pool = Pool(X_test, cat_features=self.cat_features)
        y_pred_proba = self.model.predict_proba(test_pool)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Binary predictions with optimal threshold
        y_pred_binary = (y_pred_proba >= optimal_threshold).astype(int)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"ROC-AUC Score: {roc_auc:.6f}")
        print(f"PR-AUC Score: {pr_auc:.6f}")
        print(f"Optimal Threshold: {optimal_threshold:.4f}")
        print(f"Best Iteration: {self.model.get_best_iteration()}")
        
        print(f"\nClassification Report (threshold={optimal_threshold:.4f}):")
        print(classification_report(y_test, y_pred_binary))
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc, 
            'optimal_threshold': optimal_threshold,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba
        }
    
    def predict_fraud_probability(self, X):

        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        test_pool = Pool(X, cat_features=self.cat_features)
        return self.model.predict_proba(test_pool)[:, 1]
    
    def predict_binary(self, X, threshold=0.5):
        probabilities = self.predict_fraud_probability(X)
        return (probabilities >= threshold).astype(int)
    
    def get_feature_importance(self, top_n=20):

        if self.feature_importance is None:
            raise ValueError("Model not trained yet.")
        
        return self.feature_importance.head(top_n)
    
    def plot_feature_importance(self, top_n=15, figsize=(10, 8)):

        if self.feature_importance is None:
            raise ValueError("Model not trained yet.")
        
        top_features = self.feature_importance.head(top_n)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    
    def save_predictions(self, X, tx_ids, output_path='fraud_predictions.csv'):
   
        probabilities = self.predict_fraud_probability(X)
        
        predictions_df = pd.DataFrame({
            'TX_ID': tx_ids,
            'TX_FRAUD': probabilities
        }).sort_values('TX_FRAUD', ascending=False)
        
        predictions_df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
        print(f"Mean fraud probability: {probabilities.mean():.6f}")
        print(f"Max fraud probability: {probabilities.max():.6f}")
        
        return predictions_df

def run_fraud_detection_pipeline(
    train_transactions_path,
    test_transactions_path=None,
    merchants_path=None,
    customers_path=None
):

    # Initialize model
    fraud_model = AdvancedFraudDetectionModel(random_state=42)
    
    # Load training data
    print("Loading training data...")
    transactions, merchants, customers = fraud_model.load_data(
        train_transactions_path, merchants_path, customers_path
    )
    
    # Merge datasets
    print("Merging datasets...")
    merged_data = fraud_model.merge_datasets(transactions, merchants, customers)
    
    # Create features
    print("Creating features...")
    featured_data = fraud_model.create_advanced_features(merged_data)
    
    # Prepare for training
    print("Preparing features and target...")
    X, y, cat_features = fraud_model.prepare_features_and_target(featured_data)
    
    # Train model
    print("Training model...")
    eval_results = fraud_model.train_model(X, y, cat_features)
    
    # Show feature importance
    print("\nTop 10 Most Important Features:")
    print(fraud_model.get_feature_importance(10))
    
    # Save feature importance
    fraud_model.feature_importance.to_csv('feature_importance.csv', index=False)
    print("Feature importance saved to: feature_importance.csv")
    
    # Make predictions on test set if provided
    if test_transactions_path:
        test_transactions, _, _ = fraud_model.load_data(test_transactions_path, merchants_path, customers_path)
        test_merged = fraud_model.merge_datasets(test_transactions, merchants, customers)
        test_featured = fraud_model.create_advanced_features(test_merged, is_test=True)
        X_test, _, _ = fraud_model.prepare_features_and_target(test_featured, is_test=True)
        
        # Save predictions
        fraud_model.save_predictions(X_test, test_featured['TX_ID'], 'catboost_predictions.csv')
    
    return fraud_model, eval_results

if __name__ == "__main__":\
    
    model, results = run_fraud_detection_pipeline(
        train_transactions_path='data/Payments Fraud DataSet/transactions_train.csv',
        test_transactions_path='data/Payments Fraud DataSet/transactions_test.csv', 
        merchants_path='data/Payments Fraud DataSet/merchants.csv',
        customers_path='data/Payments Fraud DataSet/customers.csv'
    )
    