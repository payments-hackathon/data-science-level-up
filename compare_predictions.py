import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import argparse

def load_and_preprocess_file(file_path, threshold=0.5):
    df = pd.read_csv(file_path)
    
    if 'TX_ID' not in df.columns or 'TX_FRAUD' not in df.columns:
        raise ValueError(f"File {file_path} must contain 'TX_ID' and 'TX_FRAUD' columns")
    
    df['TX_FRAUD_cleaned'] = np.clip(df['TX_FRAUD'], 0, 1)
    
    df['TX_FRAUD_binary'] = (df['TX_FRAUD_cleaned'] >= threshold).astype(int)
    
    return df

def calculate_agreement_stats(df1, df2, threshold=0.5):
    merged = pd.merge(df1[['TX_ID', 'TX_FRAUD_cleaned', 'TX_FRAUD_binary']], 
                      df2[['TX_ID', 'TX_FRAUD_cleaned', 'TX_FRAUD_binary']], 
                      on='TX_ID', 
                      suffixes=('_file1', '_file2'))
    
    if len(merged) == 0:
        raise ValueError("No common TX_IDs found between the two files")
    
    binary_agreement = (merged['TX_FRAUD_binary_file1'] == merged['TX_FRAUD_binary_file2']).mean()
    exact_agreement = (merged['TX_FRAUD_cleaned_file1'] == merged['TX_FRAUD_cleaned_file2']).mean()
    
    correlation = merged['TX_FRAUD_cleaned_file1'].corr(merged['TX_FRAUD_cleaned_file2'])
    mean_absolute_error = (merged['TX_FRAUD_cleaned_file1'] - merged['TX_FRAUD_cleaned_file2']).abs().mean()
    root_mean_squared_error = np.sqrt(((merged['TX_FRAUD_cleaned_file1'] - merged['TX_FRAUD_cleaned_file2']) ** 2).mean())
    
    tn = ((merged['TX_FRAUD_binary_file1'] == 0) & (merged['TX_FRAUD_binary_file2'] == 0)).sum()
    fp = ((merged['TX_FRAUD_binary_file1'] == 0) & (merged['TX_FRAUD_binary_file2'] == 1)).sum()
    fn = ((merged['TX_FRAUD_binary_file1'] == 1) & (merged['TX_FRAUD_binary_file2'] == 0)).sum()
    tp = ((merged['TX_FRAUD_binary_file1'] == 1) & (merged['TX_FRAUD_binary_file2'] == 1)).sum()
    
    cohen_kappa = cohen_kappa_score(merged['TX_FRAUD_binary_file1'], merged['TX_FRAUD_binary_file2'])
    
    prob_stats = {
        'file1_mean': merged['TX_FRAUD_cleaned_file1'].mean(),
        'file2_mean': merged['TX_FRAUD_cleaned_file2'].mean(),
        'file1_std': merged['TX_FRAUD_cleaned_file1'].std(),
        'file2_std': merged['TX_FRAUD_cleaned_file2'].std(),
        'probability_difference_mean': (merged['TX_FRAUD_cleaned_file1'] - merged['TX_FRAUD_cleaned_file2']).mean(),
        'probability_difference_std': (merged['TX_FRAUD_cleaned_file1'] - merged['TX_FRAUD_cleaned_file2']).std()
    }
    
    return {
        'merged_data': merged,
        'common_transactions': len(merged),
        'binary_agreement_percentage': binary_agreement * 100,
        'exact_agreement_percentage': exact_agreement * 100,
        'correlation': correlation,
        'mean_absolute_error': mean_absolute_error,
        'root_mean_squared_error': root_mean_squared_error,
        'confusion_matrix': {
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        },
        'cohen_kappa': cohen_kappa,
        'probability_stats': prob_stats,
        'threshold_used': threshold
    }

def print_comprehensive_report(stats, file1_path, file2_path):
    print("=" * 80)
    print("FRAUD DETECTION PREDICTION AGREEMENT ANALYSIS")
    print("=" * 80)
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}")
    print(f"Common transactions analyzed: {stats['common_transactions']:,}")
    print(f"Classification threshold: {stats['threshold_used']}")
    print()
    
    print("BINARY CLASSIFICATION AGREEMENT:")
    print("-" * 40)
    print(f"Binary Agreement: {stats['binary_agreement_percentage']:.2f}%")
    print(f"Cohen's Kappa: {stats['cohen_kappa']:.4f}")
    print()
    
    cm = stats['confusion_matrix']
    total = stats['common_transactions']
    print("CONFUSION MATRIX (File1 vs File2):")
    print("-" * 40)
    print(f"True Negatives (Both Non-Fraud):  {cm['true_negatives']:>6} ({cm['true_negatives']/total*100:5.1f}%)")
    print(f"False Positives (File1: Non-Fraud, File2: Fraud): {cm['false_positives']:>6} ({cm['false_positives']/total*100:5.1f}%)")
    print(f"False Negatives (File1: Fraud, File2: Non-Fraud): {cm['false_negatives']:>6} ({cm['false_negatives']/total*100:5.1f}%)")
    print(f"True Positives (Both Fraud):       {cm['true_positives']:>6} ({cm['true_positives']/total*100:5.1f}%)")
    print()
    
    print("PROBABILITY-BASED METRICS:")
    print("-" * 40)
    print(f"Exact Probability Agreement: {stats['exact_agreement_percentage']:.2f}%")
    print(f"Correlation Coefficient: {stats['correlation']:.4f}")
    print(f"Mean Absolute Error: {stats['mean_absolute_error']:.6f}")
    print(f"Root Mean Squared Error: {stats['root_mean_squared_error']:.6f}")
    print()
    
    prob_stats = stats['probability_stats']
    print("PROBABILITY DISTRIBUTION COMPARISON:")
    print("-" * 40)
    print(f"Mean Fraud Probability - File 1: {prob_stats['file1_mean']:.4f}")
    print(f"Mean Fraud Probability - File 2: {prob_stats['file2_mean']:.4f}")
    print(f"Std Dev Fraud Probability - File 1: {prob_stats['file1_std']:.4f}")
    print(f"Std Dev Fraud Probability - File 2: {prob_stats['file2_std']:.4f}")
    print(f"Mean Probability Difference (File1 - File2): {prob_stats['probability_difference_mean']:.6f}")
    print(f"Std Dev of Probability Differences: {prob_stats['probability_difference_std']:.6f}")
    print()
    
    agreement_level = "Excellent" if stats['binary_agreement_percentage'] >= 95 else \
                      "Good" if stats['binary_agreement_percentage'] >= 85 else \
                      "Moderate" if stats['binary_agreement_percentage'] >= 70 else \
                      "Poor"
    
    kappa_level = "Almost Perfect" if stats['cohen_kappa'] >= 0.81 else \
                  "Substantial" if stats['cohen_kappa'] >= 0.61 else \
                  "Moderate" if stats['cohen_kappa'] >= 0.41 else \
                  "Fair" if stats['cohen_kappa'] >= 0.21 else \
                  "Slight"
    
    print("AGREEMENT INTERPRETATION:")
    print("-" * 40)
    print(f"Binary Agreement Level: {agreement_level}")
    print(f"Cohen's Kappa Level: {kappa_level}")
    print(f"Correlation Strength: {'Strong' if abs(stats['correlation']) >= 0.7 else 'Moderate' if abs(stats['correlation']) >= 0.5 else 'Weak'}")

def main():
    parser = argparse.ArgumentParser(description='Compare two fraud detection prediction files')
    parser.add_argument('file1', help='First prediction file (CSV format)')
    parser.add_argument('file2', help='Second prediction file (CSV format)')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('--output', help='Output file for detailed comparison results (CSV)')
    
    args = parser.parse_args()
    
    try:
        df1 = load_and_preprocess_file(args.file1, args.threshold)
        df2 = load_and_preprocess_file(args.file2, args.threshold)
        
        stats = calculate_agreement_stats(df1, df2, args.threshold)
        
        print_comprehensive_report(stats, args.file1, args.file2)
        
        if args.output:
            stats['merged_data'].to_csv(args.output, index=False)
            print(f"\nDetailed comparison saved to: {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure both files exist and have the correct format:")
        print("Required columns: TX_ID, TX_FRAUD")
        return 1
    
    return 0

if __name__ == "__main__":
    main()