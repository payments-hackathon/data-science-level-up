#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

def aggregate_ordinal_values():
    """Aggregate all possible values for ordinal/categorical attributes across all datasets."""
    
    data_dir = Path('data/Payments Fraud DataSet')
    datasets = {
        'customers': data_dir / 'customers.csv',
        'terminals': data_dir / 'terminals.csv', 
        'merchants': data_dir / 'merchants.csv',
        'transactions_train': data_dir / 'transactions_train.csv',
        'transactions_test': data_dir / 'transactions_test.csv'
    }
    
    # Define ordinal/categorical columns for each dataset
    ordinal_columns = {
        'customers': ['CUSTOMER_ID'],
        'terminals': ['TERMINAL_ID'],
        'merchants': ['MERCHANT_ID', 'BUSINESS_TYPE', 'MCC_CODE', 'LEGAL_NAME', 'TAX_EXCEMPT_INDICATOR', 'OUTLET_TYPE'],
        'transactions_train': ['CUSTOMER_ID', 'TERMINAL_ID', 'CARD_EXPIRY_DATE', 'CARD_DATA', 'CARD_BRAND', 
                              'TRANSACTION_TYPE', 'TRANSACTION_STATUS', 'FAILURE_CODE', 'FAILURE_REASON',
                              'TRANSACTION_CURRENCY', 'CARD_COUNTRY_CODE', 'MERCHANT_ID', 'IS_RECURRING_TRANSACTION',
                              'ACQUIRER_ID', 'CARDHOLDER_AUTH_METHOD'],
        'transactions_test': ['CUSTOMER_ID', 'TERMINAL_ID', 'CARD_EXPIRY_DATE', 'CARD_DATA', 'CARD_BRAND',
                             'TRANSACTION_TYPE', 'TRANSACTION_STATUS', 'FAILURE_CODE', 'FAILURE_REASON', 
                             'TRANSACTION_CURRENCY', 'CARD_COUNTRY_CODE', 'MERCHANT_ID', 'IS_RECURRING_TRANSACTION',
                             'ACQUIRER_ID']
    }
    
    aggregated_values = {}
    
    for dataset_name, file_path in datasets.items():
        print(f"Processing {dataset_name}...")
        
        # Read dataset in chunks to handle large files
        chunk_size = 10000
        column_values = {}
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            for col in ordinal_columns[dataset_name]:
                if col in chunk.columns:
                    unique_vals = chunk[col].dropna().astype(str).unique()
                    if col not in column_values:
                        column_values[col] = set()
                    column_values[col].update(unique_vals)
        
        # Merge with global aggregated values
        for col, values in column_values.items():
            if col not in aggregated_values:
                aggregated_values[col] = set()
            aggregated_values[col].update(values)
    
    # Convert sets to sorted lists and print results
    print("\n" + "="*50)
    print("AGGREGATED ORDINAL VALUES")
    print("="*50)
    
    for col in sorted(aggregated_values.keys()):
        values = list(aggregated_values[col])
        # Convert all values to strings for consistent sorting
        str_values = [str(v) for v in values]
        try:
            sorted_values = sorted(str_values)
        except:
            sorted_values = str_values
            
        print(f"\n{col} ({len(sorted_values)} unique values):")
        if len(sorted_values) <= 20:
            for val in sorted_values:
                print(f"  - {val}")
        else:
            print(f"  - {sorted_values[0]} ... {sorted_values[-1]} (showing first and last)")
            print(f"  - Total unique values: {len(sorted_values)}")
    
    return aggregated_values

if __name__ == "__main__":
    result = aggregate_ordinal_values()