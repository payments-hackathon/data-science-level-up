import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
pd.set_option('display.max_columns', None)

df = pd.read_csv('test_predictions.csv')
print('Rows:', len(df))
print('\nPrediction counts:')
print(df['Fraud_Prediction'].value_counts(dropna=False))
print('\nProbability stats:')
print(df['Fraud_Probability'].describe())
print('\nTop 10 by probability:')
print(df.sort_values('Fraud_Probability', ascending=False).head(10))


out_dir = 'plots'
os.makedirs(out_dir, exist_ok=True)


plt.figure(figsize=(8,4))
sns.histplot(df['Fraud_Probability'], bins=50, kde=False)
plt.title('Predicted Fraud Probability Distribution')
plt.xlabel('Fraud Probability')
plt.ylabel('Count')
plt.tight_layout()
hist_path = os.path.join(out_dir, 'fraud_probability_histogram.png')
plt.savefig(hist_path)
plt.close()
print(f'Saved: {hist_path}')

plt.figure(figsize=(4,4))
ax = sns.countplot(x='Fraud_Prediction', data=df)
plt.title('Predicted Fraud vs Non-Fraud Counts')
plt.xlabel('Prediction (0=Non-Fraud,1=Fraud)')
plt.ylabel('Count')
plt.tight_layout()
counts_path = os.path.join(out_dir, 'fraud_prediction_counts.png')

for p in ax.patches:
	height = int(p.get_height())
	ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
				ha='center', va='bottom', fontsize=10, color='black')

plt.savefig(counts_path)
plt.close()
print(f'Saved: {counts_path}')

summary_text = f"Total rows: {len(df)}\nNon-Fraud: {df['Fraud_Prediction'].value_counts().get(0,0)}\nFraud: {df['Fraud_Prediction'].value_counts().get(1,0)}"
plt.figure(figsize=(4,2))
plt.axis('off')
plt.text(0.01, 0.5, summary_text, fontsize=12, va='center')
summary_path = os.path.join(out_dir, 'prediction_summary.png')
plt.savefig(summary_path, bbox_inches='tight')
plt.close()
print(f'Saved: {summary_path}')

if 'TX_AMOUNT' in df.columns:
	plt.figure(figsize=(8,4))
	sns.scatterplot(x='TX_AMOUNT', y='Fraud_Probability', data=df, alpha=0.6)
	plt.xscale('symlog')
	plt.title('Transaction Amount vs Predicted Fraud Probability')
	plt.xlabel('TX_AMOUNT (symlog)')
	plt.ylabel('Fraud Probability')
	plt.tight_layout()
	amt_path = os.path.join(out_dir, 'amount_vs_probability.png')
	plt.savefig(amt_path)
	plt.close()
	print(f'Saved: {amt_path}')

if 'TX_TS' in df.columns:
	try:
		df['TX_TS'] = pd.to_datetime(df['TX_TS'])
		df['hour'] = df['TX_TS'].dt.hour
		hourly = df.groupby('hour')['Fraud_Probability'].mean().reset_index()
		plt.figure(figsize=(8,3))
		sns.lineplot(x='hour', y='Fraud_Probability', data=hourly, marker='o')
		plt.title('Mean Predicted Fraud Probability by Hour of Day')
		plt.xlabel('Hour of Day')
		plt.ylabel('Mean Fraud Probability')
		plt.tight_layout()
		hour_path = os.path.join(out_dir, 'hourly_mean_probability.png')
		plt.savefig(hour_path)
		plt.close()
		print(f'Saved: {hour_path}')
	except Exception as e:
		print('Failed to compute hourly summary:', e)

print('All visuals saved in ./plots')