# flake8: noqa

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import argparse
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

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

def get_color_for_percentage(value, thresholds=(70, 85, 95)):
    if value >= thresholds[2]: return Fore.GREEN
    elif value >= thresholds[1]: return Fore.YELLOW
    elif value >= thresholds[0]: return Fore.CYAN
    else: return Fore.RED

def get_color_for_kappa(value):
    if value >= 0.81: return Fore.GREEN
    elif value >= 0.61: return Fore.YELLOW
    elif value >= 0.41: return Fore.CYAN
    else: return Fore.RED

def print_comprehensive_report(stats, file1_path, file2_path):
    print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 80}")
    print(f"{Fore.CYAN}{Style.BRIGHT}FRAUD DETECTION PREDICTION AGREEMENT ANALYSIS")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}File 1:{Style.RESET_ALL} {file1_path}")
    print(f"{Fore.WHITE}File 2:{Style.RESET_ALL} {file2_path}")
    print(f"{Fore.WHITE}Common transactions analyzed:{Style.RESET_ALL} {stats['common_transactions']:,}")
    print(f"{Fore.WHITE}Classification threshold:{Style.RESET_ALL} {stats['threshold_used']}")
    print()
    
    # Binary Classification Agreement
    print(f"{Fore.MAGENTA}{Style.BRIGHT}BINARY CLASSIFICATION AGREEMENT:{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'-' * 50}{Style.RESET_ALL}")
    
    agreement_color = get_color_for_percentage(stats['binary_agreement_percentage'])
    kappa_color = get_color_for_kappa(stats['cohen_kappa'])
    
    print(f"{'Binary Agreement:':<25} {agreement_color}{stats['binary_agreement_percentage']:>6.2f}%{Style.RESET_ALL}")
    print(f"{'Cohen\'s Kappa:':<25} {kappa_color}{stats['cohen_kappa']:>6.4f}{Style.RESET_ALL}")
    print()
    
    # Confusion Matrix
    cm = stats['confusion_matrix']
    total = stats['common_transactions']
    print(f"{Fore.MAGENTA}{Style.BRIGHT}CONFUSION MATRIX (File1 vs File2):{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'-' * 50}{Style.RESET_ALL}")
    
    print(f"{Fore.GREEN}{'True Negatives:':<25} {cm['true_negatives']:>8,} ({cm['true_negatives']/total*100:>5.1f}%){Style.RESET_ALL}")
    print(f"{Fore.RED}{'False Positives:':<25} {cm['false_positives']:>8,} ({cm['false_positives']/total*100:>5.1f}%){Style.RESET_ALL}")
    print(f"{Fore.RED}{'False Negatives:':<25} {cm['false_negatives']:>8,} ({cm['false_negatives']/total*100:>5.1f}%){Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'True Positives:':<25} {cm['true_positives']:>8,} ({cm['true_positives']/total*100:>5.1f}%){Style.RESET_ALL}")
    print()
    
    # Probability-based Metrics
    print(f"{Fore.MAGENTA}{Style.BRIGHT}PROBABILITY-BASED METRICS:{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'-' * 50}{Style.RESET_ALL}")
    
    exact_color = get_color_for_percentage(stats['exact_agreement_percentage'])
    corr_color = Fore.GREEN if abs(stats['correlation']) >= 0.7 else Fore.YELLOW if abs(stats['correlation']) >= 0.5 else Fore.RED
    
    print(f"{'Exact Agreement:':<25} {exact_color}{stats['exact_agreement_percentage']:>6.2f}%{Style.RESET_ALL}")
    print(f"{'Correlation:':<25} {corr_color}{stats['correlation']:>6.4f}{Style.RESET_ALL}")
    print(f"{'Mean Absolute Error:':<25} {stats['mean_absolute_error']:>8.6f}")
    print(f"{'Root Mean Squared Error:':<25} {stats['root_mean_squared_error']:>8.6f}")
    print()
    
    # Probability Distribution
    prob_stats = stats['probability_stats']
    print(f"{Fore.MAGENTA}{Style.BRIGHT}PROBABILITY DISTRIBUTION:{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'-' * 50}{Style.RESET_ALL}")
    
    print(f"{'File 1 Mean:':<25} {prob_stats['file1_mean']:>8.4f}")
    print(f"{'File 2 Mean:':<25} {prob_stats['file2_mean']:>8.4f}")
    print(f"{'File 1 Std Dev:':<25} {prob_stats['file1_std']:>8.4f}")
    print(f"{'File 2 Std Dev:':<25} {prob_stats['file2_std']:>8.4f}")
    
    diff_color = Fore.GREEN if abs(prob_stats['probability_difference_mean']) < 0.01 else Fore.YELLOW if abs(prob_stats['probability_difference_mean']) < 0.05 else Fore.RED
    print(f"{'Mean Difference:':<25} {diff_color}{prob_stats['probability_difference_mean']:>8.6f}{Style.RESET_ALL}")
    print(f"{'Std Dev Difference:':<25} {prob_stats['probability_difference_std']:>8.6f}")
    print()
    
    # Agreement Interpretation
    agreement_level = "Excellent" if stats['binary_agreement_percentage'] >= 95 else \
                      "Good" if stats['binary_agreement_percentage'] >= 85 else \
                      "Moderate" if stats['binary_agreement_percentage'] >= 70 else \
                      "Poor"
    
    kappa_level = "Almost Perfect" if stats['cohen_kappa'] >= 0.81 else \
                  "Substantial" if stats['cohen_kappa'] >= 0.61 else \
                  "Moderate" if stats['cohen_kappa'] >= 0.41 else \
                  "Fair" if stats['cohen_kappa'] >= 0.21 else \
                  "Slight"
    
    corr_strength = "Strong" if abs(stats['correlation']) >= 0.7 else "Moderate" if abs(stats['correlation']) >= 0.5 else "Weak"
    
    print(f"{Fore.MAGENTA}{Style.BRIGHT}AGREEMENT SUMMARY:{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'-' * 50}{Style.RESET_ALL}")
    
    level_color = get_color_for_percentage(stats['binary_agreement_percentage'])
    kappa_interp_color = get_color_for_kappa(stats['cohen_kappa'])
    corr_interp_color = Fore.GREEN if corr_strength == "Strong" else Fore.YELLOW if corr_strength == "Moderate" else Fore.RED
    
    print(f"{'Agreement Level:':<25} {level_color}{agreement_level}{Style.RESET_ALL}")
    print(f"{'Kappa Interpretation:':<25} {kappa_interp_color}{kappa_level}{Style.RESET_ALL}")
    print(f"{'Correlation Strength:':<25} {corr_interp_color}{corr_strength}{Style.RESET_ALL}")

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