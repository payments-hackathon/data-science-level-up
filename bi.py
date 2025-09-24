# flake8: noqa

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import random
warnings.filterwarnings('ignore')

catppuccin = [
    '#f5e0dc', '#f2cdcd', '#f5c2e7', '#cba6f7',
    '#f38ba8', '#eba0ac', '#fab387', '#f9e2af',
    '#a6e3a1', '#94e2d5', '#89dceb', '#74c7ec',
    '#89b4fa', '#b4befe',
]

# Set dark theme with catppuccin base
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1e1e2e'
plt.rcParams['axes.facecolor'] = '#1e1e2e'
plt.rcParams['text.color'] = '#cdd6f4'
plt.rcParams['axes.labelcolor'] = '#cdd6f4'
plt.rcParams['xtick.color'] = '#cdd6f4'
plt.rcParams['ytick.color'] = '#cdd6f4'
plt.rcParams['axes.titlecolor'] = '#cdd6f4'
plt.rcParams['legend.facecolor'] = '#181825'
plt.rcParams['legend.edgecolor'] = '#cdd6f4'
plt.rcParams['legend.labelcolor'] = '#cdd6f4'

def get_catppuccin_colors(n=1):
    return random.sample(catppuccin, min(n, len(catppuccin)))

# Load data
print("Loading datasets...")
customers = pd.read_csv('data/Payments Fraud DataSet/customers.csv')
terminals = pd.read_csv('data/Payments Fraud DataSet/terminals.csv')
merchants = pd.read_csv('data/Payments Fraud DataSet/merchants.csv')
train_tx = pd.read_csv('data/Payments Fraud DataSet/transactions_train.csv')

# Convert timestamp
train_tx['TX_TS'] = pd.to_datetime(train_tx['TX_TS'])

print(f"Loaded {len(train_tx):,} training transactions")

# %% Overall Fraud Rate
fraud_rate = train_tx['TX_FRAUD'].mean()
fraud_count = train_tx['TX_FRAUD'].sum()
total_count = len(train_tx)

plt.figure(figsize=(8, 6))
labels = ['Legitimate', 'Fraudulent']
sizes = [total_count - fraud_count, fraud_count]
colors = get_catppuccin_colors(2)
pie_result = plt.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', startangle=90)
for text in pie_result[1]:  text.set_color('#cdd6f4')
if len(pie_result) > 2:
    for autotext in pie_result[2]: autotext.set_color('#11111b')
plt.title(f'Overall Fraud Rate: {fraud_rate:.3f}% ({fraud_count:,} fraudulent transactions)')
plt.show()

# %% Transaction Volume by Hour
train_tx['hour'] = train_tx['TX_TS'].dt.hour
hourly_volume = train_tx.groupby('hour').size()

plt.figure(figsize=(12, 6))
hourly_volume.plot(kind='bar', color=get_catppuccin_colors(1)[0])
plt.ylim(hourly_volume.min() * 0.99, hourly_volume.max() * 1.01)
plt.title('Transaction Volume by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=0)
plt.show()

# %% Fraud Rate by Hour
hourly_fraud = train_tx.groupby('hour')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
hourly_fraud['fraud_rate'] = hourly_fraud['mean'] * 100

plt.figure(figsize=(12, 6))
hourly_fraud['fraud_rate'].plot(kind='line', marker='o', color=get_catppuccin_colors(1)[0])
plt.title('Fraud Rate by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Fraud Rate (%)')
plt.grid(True, alpha=0.3)
plt.show()

# %% Transaction Amount Distribution
plt.figure(figsize=(12, 6))
fraud_amounts = train_tx[train_tx['TX_FRAUD'] == 1]['TX_AMOUNT']
legit_amounts = train_tx[train_tx['TX_FRAUD'] == 0]['TX_AMOUNT']

colors = get_catppuccin_colors(2)
plt.hist(legit_amounts, bins=50, alpha=0.7, label='Legitimate', density=True, color=colors[0])
plt.hist(fraud_amounts, bins=50, alpha=0.7, label='Fraudulent', density=True, color=colors[1])
plt.xlabel('Transaction Amount ($)')
plt.ylabel('Density')
plt.title('Transaction Amount Distribution')
plt.legend()
plt.xlim(0, 1000)
plt.show()

# %% Geographic Distribution - Customer Locations
plt.figure(figsize=(10, 8))
colors = get_catppuccin_colors(2)
plt.scatter(customers['x_customer_id'], customers['y_customer_id'], 
           alpha=0.5, s=1, c=colors[0], label='Customers')
plt.scatter(terminals['x_terminal_id'], terminals['y_terminal__id'], 
           alpha=0.8, s=20, c=colors[1], marker='s', label='Terminals')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Geographic Distribution: Customers and Terminals')
plt.legend()
plt.show()

# %% Customer-Terminal Distance Analysis
merged_data = train_tx.merge(customers, on='CUSTOMER_ID').merge(terminals, on='TERMINAL_ID')
merged_data['distance'] = np.sqrt((merged_data['x_customer_id'] - merged_data['x_terminal_id'])**2 + 
                                 (merged_data['y_customer_id'] - merged_data['y_terminal__id'])**2)

plt.figure(figsize=(12, 6))
fraud_dist = merged_data[merged_data['TX_FRAUD'] == 1]['distance']
legit_dist = merged_data[merged_data['TX_FRAUD'] == 0]['distance']

colors = get_catppuccin_colors(2)
plt.hist(legit_dist, bins=50, alpha=0.7, label='Legitimate', density=True, color=colors[0])
plt.hist(fraud_dist, bins=50, alpha=0.7, label='Fraudulent', density=True, color=colors[1])
plt.xlabel('Customer-Terminal Distance')
plt.ylabel('Density')
plt.title('Distance Distribution: Customer to Terminal')
plt.legend()
plt.show()

# %% Card Brand Fraud Analysis
brand_fraud = train_tx.groupby('CARD_BRAND')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
brand_fraud['fraud_rate'] = brand_fraud['mean'] * 100

plt.figure(figsize=(10, 6))
brand_fraud['fraud_rate'].plot(kind='bar', color=get_catppuccin_colors(len(brand_fraud)))
plt.ylim(brand_fraud['fraud_rate'].min() * 0.95, brand_fraud['fraud_rate'].max() * 1.05)
plt.title('Fraud Rate by Card Brand')
plt.xlabel('Card Brand')
plt.ylabel('Fraud Rate (%)')
plt.xticks(rotation=45)
plt.show()

# %% Transaction Type Analysis
tx_type_fraud = train_tx.groupby('TRANSACTION_TYPE')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
tx_type_fraud['fraud_rate'] = tx_type_fraud['mean'] * 100

plt.figure(figsize=(10, 6))
tx_type_fraud['fraud_rate'].plot(kind='bar', color=get_catppuccin_colors(len(tx_type_fraud)))
plt.ylim(tx_type_fraud['fraud_rate'].min() * 0.95, tx_type_fraud['fraud_rate'].max() * 1.05)
plt.title('Fraud Rate by Transaction Type')
plt.xlabel('Transaction Type')
plt.ylabel('Fraud Rate (%)')
plt.xticks(rotation=45)
plt.show()

# %% Top 20 MCC Codes by Fraud Rate
merged_merchant = train_tx.merge(merchants, on='MERCHANT_ID', how='left')
mcc_fraud = merged_merchant.groupby('MCC_CODE')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
mcc_fraud = mcc_fraud[mcc_fraud['count'] >= 100]  # Filter for significance
mcc_fraud['fraud_rate'] = mcc_fraud['mean'] * 100
top_mcc = mcc_fraud.nlargest(20, 'fraud_rate')

plt.figure(figsize=(12, 8))
top_mcc['fraud_rate'].plot(kind='barh', color=get_catppuccin_colors(len(top_mcc)))
plt.title('Top 20 MCC Codes by Fraud Rate (min 100 transactions)')
plt.xlabel('Fraud Rate (%)')
plt.show()

# %% Currency Analysis
currency_fraud = train_tx.groupby('TRANSACTION_CURRENCY')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
currency_fraud['fraud_rate'] = currency_fraud['mean'] * 100

plt.figure(figsize=(10, 6))
currency_fraud['fraud_rate'].plot(kind='bar', color=get_catppuccin_colors(len(currency_fraud)))
plt.ylim(currency_fraud['fraud_rate'].min() * 0.95, currency_fraud['fraud_rate'].max() * 1.05)
plt.title('Fraud Rate by Transaction Currency')
plt.xlabel('Currency')
plt.ylabel('Fraud Rate (%)')
plt.xticks(rotation=45)
plt.show()

# %% Authentication Method Effectiveness
auth_fraud = train_tx.groupby('CARDHOLDER_AUTH_METHOD')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
auth_fraud['fraud_rate'] = auth_fraud['mean'] * 100

plt.figure(figsize=(12, 6))
auth_fraud['fraud_rate'].plot(kind='bar', color=get_catppuccin_colors(len(auth_fraud)))
plt.ylim(auth_fraud['fraud_rate'].min() * 0.95, auth_fraud['fraud_rate'].max() * 1.05)
plt.title('Fraud Rate by Authentication Method')
plt.xlabel('Authentication Method')
plt.ylabel('Fraud Rate (%)')
plt.xticks(rotation=45)
plt.show()

# %% Terminal Fraud Hotspots
terminal_fraud = train_tx.groupby('TERMINAL_ID')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
terminal_fraud = terminal_fraud[terminal_fraud['count'] >= 50]  # Min transactions
terminal_fraud['fraud_rate'] = terminal_fraud['mean'] * 100
top_terminals = terminal_fraud.nlargest(20, 'fraud_rate')

plt.figure(figsize=(12, 6))
plt.hist(terminal_fraud['fraud_rate'], bins=30, color=get_catppuccin_colors(1)[0], alpha=0.7)
plt.title('Distribution of Terminal Fraud Rates')
plt.xlabel('Fraud Rate (%)')
plt.ylabel('Number of Terminals')
plt.show()

# %% Customer Risk Profiles
customer_fraud = train_tx.groupby('CUSTOMER_ID')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
customer_fraud['fraud_rate'] = customer_fraud['mean'] * 100
high_risk_customers = customer_fraud[customer_fraud['sum'] >= 2]  # 2+ fraud incidents

plt.figure(figsize=(10, 6))
plt.hist(customer_fraud['fraud_rate'], bins=50, color=get_catppuccin_colors(1)[0], alpha=0.7)
plt.title(f'Customer Fraud Rate Distribution\n{len(high_risk_customers)} customers with 2+ fraud incidents')
plt.xlabel('Customer Fraud Rate (%)')
plt.ylabel('Number of Customers')
plt.show()

# %% Acquirer Performance
acquirer_fraud = train_tx.groupby('ACQUIRER_ID')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
acquirer_fraud['fraud_rate'] = acquirer_fraud['mean'] * 100

plt.figure(figsize=(10, 6))
acquirer_fraud['fraud_rate'].plot(kind='bar', color=get_catppuccin_colors(len(acquirer_fraud)))
plt.ylim(acquirer_fraud['fraud_rate'].min() * 0.95, acquirer_fraud['fraud_rate'].max() * 1.05)
plt.title('Fraud Rate by Acquirer')
plt.xlabel('Acquirer ID')
plt.ylabel('Fraud Rate (%)')
plt.show()

# %% Transaction Status Analysis
status_fraud = train_tx.groupby('TRANSACTION_STATUS')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
status_fraud['fraud_rate'] = status_fraud['mean'] * 100

plt.figure(figsize=(10, 6))
status_fraud['fraud_rate'].plot(kind='bar', color=get_catppuccin_colors(len(status_fraud)))
plt.ylim(status_fraud['fraud_rate'].min() * 0.95, status_fraud['fraud_rate'].max() * 1.05)
plt.title('Fraud Rate by Transaction Status')
plt.xlabel('Transaction Status')
plt.ylabel('Fraud Rate (%)')
plt.xticks(rotation=45)
plt.show()

# %% Recurring Transaction Analysis
recurring_fraud = train_tx.groupby('IS_RECURRING_TRANSACTION')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
recurring_fraud['fraud_rate'] = recurring_fraud['mean'] * 100

plt.figure(figsize=(8, 6))
recurring_fraud['fraud_rate'].plot(kind='bar', color=get_catppuccin_colors(len(recurring_fraud)))
plt.ylim(recurring_fraud['fraud_rate'].min() * 0.95, recurring_fraud['fraud_rate'].max() * 1.05)
plt.title('Fraud Rate: Recurring vs One-time Transactions')
plt.xlabel('Is Recurring Transaction')
plt.ylabel('Fraud Rate (%)')
plt.show()

# %% Business Type Risk Analysis
business_fraud = merged_merchant.groupby('BUSINESS_TYPE')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
business_fraud['fraud_rate'] = business_fraud['mean'] * 100

plt.figure(figsize=(10, 6))
business_fraud['fraud_rate'].plot(kind='bar', color=get_catppuccin_colors(len(business_fraud)))
plt.ylim(business_fraud['fraud_rate'].min() * 0.95, business_fraud['fraud_rate'].max() * 1.05)
plt.title('Fraud Rate by Business Type')
plt.xlabel('Business Type')
plt.ylabel('Fraud Rate (%)')
plt.xticks(rotation=45)
plt.show()

# %% Outlet Type Analysis
outlet_fraud = merged_merchant.groupby('OUTLET_TYPE')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
outlet_fraud['fraud_rate'] = outlet_fraud['mean'] * 100

plt.figure(figsize=(10, 6))
outlet_fraud['fraud_rate'].plot(kind='bar', color=get_catppuccin_colors(len(outlet_fraud)))
plt.ylim(outlet_fraud['fraud_rate'].min() * 0.95, outlet_fraud['fraud_rate'].max() * 1.05)
plt.title('Fraud Rate by Outlet Type')
plt.xlabel('Outlet Type')
plt.ylabel('Fraud Rate (%)')
plt.xticks(rotation=45)
plt.show()

# %% Monthly Fraud Trends
train_tx['month'] = train_tx['TX_TS'].dt.month
monthly_fraud = train_tx.groupby('month')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
monthly_fraud['fraud_rate'] = monthly_fraud['mean'] * 100

plt.figure(figsize=(12, 6))
monthly_fraud['fraud_rate'].plot(kind='line', marker='o', color=get_catppuccin_colors(1)[0])
plt.title('Monthly Fraud Rate Trends')
plt.xlabel('Month')
plt.ylabel('Fraud Rate (%)')
plt.grid(True, alpha=0.3)
plt.show()

# %% Amount Threshold Analysis
train_tx['amount_bucket'] = pd.cut(train_tx['TX_AMOUNT'], 
                                  bins=[0, 10, 50, 100, 500, 1000, float('inf')],
                                  labels=['$0-10', '$10-50', '$50-100', '$100-500', '$500-1000', '$1000+'])

bucket_fraud = train_tx.groupby('amount_bucket')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
bucket_fraud['fraud_rate'] = bucket_fraud['mean'] * 100

plt.figure(figsize=(10, 6))
bucket_fraud['fraud_rate'].plot(kind='bar', color=get_catppuccin_colors(len(bucket_fraud)))
plt.ylim(bucket_fraud['fraud_rate'].min() * 0.95, bucket_fraud['fraud_rate'].max() * 1.05)
plt.title('Fraud Rate by Transaction Amount Buckets')
plt.xlabel('Amount Bucket')
plt.ylabel('Fraud Rate (%)')
plt.xticks(rotation=45)
plt.show()

# %% Country Code Risk Analysis
country_fraud = train_tx.groupby('CARD_COUNTRY_CODE')['TX_FRAUD'].agg(['count', 'sum', 'mean'])
country_fraud = country_fraud[country_fraud['count'] >= 1000]  # Filter for significance
country_fraud['fraud_rate'] = country_fraud['mean'] * 100

plt.figure(figsize=(12, 6))
country_fraud['fraud_rate'].plot(kind='bar', color=get_catppuccin_colors(len(country_fraud)))
plt.ylim(country_fraud['fraud_rate'].min() * 0.95, country_fraud['fraud_rate'].max() * 1.05)
plt.title('Fraud Rate by Card Country Code (min 1000 transactions)')
plt.xlabel('Country Code')
plt.ylabel('Fraud Rate (%)')
plt.show()

# %% Summary Statistics Table
print("\n\033[94m===== FRAUD DETECTION DATASET SUMMARY =====\033[0m")
print(f"Total Transactions:\t\t{len(train_tx):,}")
print(f"Fraudulent Transactions:\t{fraud_count:,}")
print(f"Overall Fraud Rate:\t\t{fraud_rate:.4f}%")
print(f"Average Transaction Amount:\t${train_tx['TX_AMOUNT'].mean():.2f}")
print(f"Median Transaction Amount:\t${train_tx['TX_AMOUNT'].median():.2f}")
print(f"Average Fraud Amount:\t\t${train_tx[train_tx['TX_FRAUD']==1]['TX_AMOUNT'].mean():.2f}")
print(f"Average Legitimate Amount:\t${train_tx[train_tx['TX_FRAUD']==0]['TX_AMOUNT'].mean():.2f}")
print(f"Unique Customers:\t\t{train_tx['CUSTOMER_ID'].nunique():,}")
print(f"Unique Terminals:\t\t{train_tx['TERMINAL_ID'].nunique():,}")
print(f"Unique Merchants:\t\t{train_tx['MERCHANT_ID'].nunique():,}")
print(f"Date Range:\t\t\t{train_tx['TX_TS'].min()}   -   {train_tx['TX_TS'].max()}")