#!/usr/bin/env python3
# flake8: noqa

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

pd.set_option('display.max_columns', None)

# --- Load predictions ---
df = pd.read_csv("submission_lgbm.csv")
print("Rows:", len(df))

# Add binary prediction column
df["Fraud_Probability"] = df["TX_FRAUD"]  # just rename for clarity
df["Fraud_Prediction"] = (df["Fraud_Probability"] > 0.5).astype(int)

# --- Create output folder ---
out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)

# --- Fraud probability distribution ---
plt.figure(figsize=(8, 4))
sns.histplot(df["Fraud_Probability"], bins=50, kde=True, color="crimson")
plt.title("Distribution of Predicted Fraud Probability")
plt.xlabel("Fraud Probability")
plt.ylabel("Transactions")
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(out_dir, "fraud_probability_distribution.png"))
plt.close()


# --- Prediction counts ---
plt.figure(figsize=(4, 4))
ax = sns.countplot(
    x="Fraud_Prediction",
    data=df,
    hue="Fraud_Prediction",
    palette=["#4caf50", "#f44336"],
    legend=False
)

plt.title("Predicted Fraud vs Non-Fraud Counts")
plt.xlabel("Prediction (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Count")
for p in ax.patches:
    height = int(p.get_height())
    ax.annotate(f"{height}", (p.get_x() + p.get_width() / 2., height),
                ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(out_dir, "fraud_prediction_counts.png"))
plt.close()

# --- Transaction Amount vs Fraud Probability ---
if "TX_AMOUNT" in df.columns:
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x="TX_AMOUNT", y="Fraud_Probability", data=df, alpha=0.5)
    plt.xscale("log")  # log scale for skewed amounts
    plt.title("Transaction Amount vs Fraud Probability")
    plt.xlabel("Transaction Amount (log scale)")
    plt.ylabel("Fraud Probability")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(out_dir, "amount_vs_probability.png"))
    plt.close()

# --- Hourly fraud risk profile ---
if "TX_TS" in df.columns:
    df["TX_TS"] = pd.to_datetime(df["TX_TS"], errors="coerce")
    df["hour"] = df["TX_TS"].dt.hour
    hourly = df.groupby("hour")["Fraud_Probability"].mean().reset_index()
    plt.figure(figsize=(8, 4))
    sns.lineplot(x="hour", y="Fraud_Probability", data=hourly, marker="o", color="navy")
    plt.title("Average Fraud Probability by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Avg Fraud Probability")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(out_dir, "hourly_fraud_profile.png"))
    plt.close()

# --- Fraud by Transaction Type (if available) ---
if "TRANSACTION_TYPE" in df.columns:
    plt.figure(figsize=(8, 4))
    type_stats = df.groupby("TRANSACTION_TYPE")["Fraud_Probability"].mean().sort_values(ascending=False)
    sns.barplot(x=type_stats.index, y=type_stats.values, palette="coolwarm")
    plt.title("Average Fraud Probability by Transaction Type")
    plt.xticks(rotation=45)
    plt.ylabel("Avg Fraud Probability")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(out_dir, "fraud_by_transaction_type.png"))
    plt.close()

# --- Fraud by Card Brand (if available) ---
if "CARD_BRAND" in df.columns:
    plt.figure(figsize=(6, 4))
    brand_stats = df.groupby("CARD_BRAND")["Fraud_Probability"].mean().sort_values(ascending=False)
    sns.barplot(x=brand_stats.index, y=brand_stats.values, palette="viridis")
    plt.title("Average Fraud Probability by Card Brand")
    plt.ylabel("Avg Fraud Probability")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(out_dir, "fraud_by_card_brand.png"))
    plt.close()

summary_text = (
    f"Total Transactions: {len(df)}\n"
    f"Non-Fraud Predictions: {df['Fraud_Prediction'].value_counts().get(0,0)}\n"
    f"Fraud Predictions: {df['Fraud_Prediction'].value_counts().get(1,0)}\n"
    f"Fraud Rate (Predicted): {df['Fraud_Prediction'].mean()*100:.2f}%"
)

plt.figure(figsize=(4, 2))
plt.axis("off")
plt.text(0.01, 0.5, summary_text, fontsize=12, va="center")
plt.show()
plt.savefig(os.path.join(out_dir, "prediction_summary.png"), bbox_inches="tight")
plt.close()

print("All visuals saved in ./plots")
# %%
