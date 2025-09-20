#!/usr/bin/env python3
# flake8: noqa

# %% [markdown]
# Notebook: Data Loading and Cleaning
#
# What this script does:
# - Load the 5 CSV data files
# - Clean each dataset
# - Drop low-value or leakage-prone columns
# - Save cleaned CSVs under data/processed/

# %% [markdown]
# Setup and imports

# %%
import os
from typing import Tuple
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 160)

# %% [markdown]
# Utilities: path inference, safe CSV reading, and parsing helpers

# %%

def read_csv_safe(path: str, **kwargs) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {os.path.abspath(path)}")
    return pd.read_csv(path, **kwargs)


def parse_card_expiry_mm_yy(mm_yy: str) -> Tuple[float, float]:
    try:
        if not isinstance(mm_yy, str) or "/" not in mm_yy:
            return np.nan, np.nan
        mm, yy = mm_yy.split("/")
        mm = int(mm)
        yy = 2000 + int(yy)
        if not (1 <= mm <= 12):
            return np.nan, np.nan
        return float(mm), float(yy)
    except Exception:
        return np.nan, np.nan


def parse_bool_fuzzy(val) -> bool:
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    if s.startswith(("t", "y")):
        return True
    if s.startswith(("f", "n")):
        return False
    return False

# %% [markdown]
# Load raw datasets

# %%
base_path = os.path.join("data", "Payments Fraud DataSet")
print("Using data path:", base_path)

dtypes_tx = {
    "TX_ID": "string",
    "CUSTOMER_ID": "string",
    "TERMINAL_ID": "string",
    "TX_AMOUNT": "float32",
    "TRANSACTION_GOODS_AND_SERVICES_AMOUNT": "float32",
    "TRANSACTION_CASHBACK_AMOUNT": "float32",
    "CARD_EXPIRY_DATE": "string",
    "CARD_DATA": "string",
    "CARD_BRAND": "string",
    "TRANSACTION_TYPE": "string",
    "TRANSACTION_STATUS": "string",
    "FAILURE_CODE": "string",
    "FAILURE_REASON": "string",
    "TRANSACTION_CURRENCY": "string",
    "CARD_COUNTRY_CODE": "string",
    "MERCHANT_ID": "string",
    "IS_RECURRING_TRANSACTION": "string",
    "ACQUIRER_ID": "string",
}

train_raw = read_csv_safe(
    os.path.join(base_path, "transactions_train.csv"),
    dtype=dtypes_tx,
    parse_dates=["TX_TS"],
)
test_raw = read_csv_safe(
    os.path.join(base_path, "transactions_test.csv"),
    dtype=dtypes_tx,
    parse_dates=["TX_TS"],
)
customers_raw = read_csv_safe(
    os.path.join(base_path, "customers.csv"),
    dtype={"CUSTOMER_ID": "string", "x_customer_id": "float32", "y_customer_id": "float32"},
)
terminals_raw = read_csv_safe(
    os.path.join(base_path, "terminals.csv"),
    dtype={"TERMINAL_ID": "string", "x_terminal_id": "float32", "y_terminal__id": "float32"},
)
merchants_raw = read_csv_safe(
    os.path.join(base_path, "merchants.csv"),
    dtype={
        "MERCHANT_ID": "string",
        "BUSINESS_TYPE": "string",
        "MCC_CODE": "string",
        "LEGAL_NAME": "string",
        "TAX_EXCEMPT_INDICATOR": "string",
        "OUTLET_TYPE": "string",
        "ANNUAL_TURNOVER_CARD": "float32",
        "ANNUAL_TURNOVER": "float32",
        "AVERAGE_TICKET_SALE_AMOUNT": "float32",
        "PAYMENT_PERCENTAGE_FACE_TO_FACE": "float32",
        "PAYMENT_PERCENTAGE_ECOM": "float32",
        "PAYMENT_PERCENTAGE_MOTO": "float32",
        "DEPOSIT_REQUIRED_PERCENTAGE": "float32",
        "DEPOSIT_PERCENTAGE": "float32",
        "DELIVERY_SAME_DAYS_PERCENTAGE": "float32",
        "DELIVERY_WEEK_ONE_PERCENTAGE": "float32",
        "DELIVERY_WEEK_TWO_PERCENTAGE": "float32",
        "DELIVERY_OVER_TWO_WEEKS_PERCENTAGE": "float32",
    },
    parse_dates=["FOUNDATION_DATE", "ACTIVE_FROM", "TRADING_FROM"],
    infer_datetime_format=True,
)

print("Shapes:")
print("  train_raw:", train_raw.shape)
print("  test_raw:", test_raw.shape)
print("  customers_raw:", customers_raw.shape)
print("  terminals_raw:", terminals_raw.shape)
print("  merchants_raw:", merchants_raw.shape)

# %% [markdown]
# Cleaning functions

# %%
def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["CUSTOMER_ID"])
    df = df.dropna(subset=["CUSTOMER_ID", "x_customer_id", "y_customer_id"])
    for c in ["x_customer_id", "y_customer_id"]:
        df[c] = df[c].clip(lower=0.0, upper=100.0)
    return df.reset_index(drop=True)


def clean_terminals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["TERMINAL_ID"])
    df = df.dropna(subset=["TERMINAL_ID", "x_terminal_id", "y_terminal__id"])
    for c in ["x_terminal_id", "y_terminal__id"]:
        df[c] = df[c].clip(lower=0.0, upper=100.0)
    return df.reset_index(drop=True)


def clean_merchants(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["MERCHANT_ID"])

    pct_cols = [
        "PAYMENT_PERCENTAGE_FACE_TO_FACE",
        "PAYMENT_PERCENTAGE_ECOM",
        "PAYMENT_PERCENTAGE_MOTO",
        "DEPOSIT_REQUIRED_PERCENTAGE",
        "DEPOSIT_PERCENTAGE",
        "DELIVERY_SAME_DAYS_PERCENTAGE",
        "DELIVERY_WEEK_ONE_PERCENTAGE",
        "DELIVERY_WEEK_TWO_PERCENTAGE",
        "DELIVERY_OVER_TWO_WEEKS_PERCENTAGE",
    ]
    for c in pct_cols:
        df[c] = df[c].clip(lower=0, upper=100)

    for c in ["ANNUAL_TURNOVER_CARD", "ANNUAL_TURNOVER", "AVERAGE_TICKET_SALE_AMOUNT"]:
        df[c] = df[c].fillna(0).clip(lower=0)

    df["TAX_EXCEMPT_INDICATOR"] = df["TAX_EXCEMPT_INDICATOR"].apply(parse_bool_fuzzy).astype(bool)

    drop_cols = ["LEGAL_NAME", "FOUNDATION_DATE", "ACTIVE_FROM", "TRADING_FROM"]
    df = df.drop(columns=drop_cols)

    return df.reset_index(drop=True)


def clean_transactions(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.sort_values("TX_TS")
    df = df.drop_duplicates(subset=["TX_ID"], keep="last")

    df = df.dropna(subset=["TX_ID", "CUSTOMER_ID", "TERMINAL_ID", "TX_TS"])

    df["IS_RECURRING_TRANSACTION"] = df["IS_RECURRING_TRANSACTION"].apply(parse_bool_fuzzy).astype(bool)

    for c in ["TX_AMOUNT", "TRANSACTION_GOODS_AND_SERVICES_AMOUNT", "TRANSACTION_CASHBACK_AMOUNT"]:
        df[c] = df[c].fillna(0).clip(lower=0)

    total = df["TRANSACTION_GOODS_AND_SERVICES_AMOUNT"] + df["TRANSACTION_CASHBACK_AMOUNT"]
    over = total > df["TX_AMOUNT"]
    total_safe = total.copy()
    total_safe[total_safe == 0] = 1
    scale = df["TX_AMOUNT"] / total_safe
    df.loc[over, "TRANSACTION_GOODS_AND_SERVICES_AMOUNT"] = (
            df.loc[over, "TRANSACTION_GOODS_AND_SERVICES_AMOUNT"] * scale[over]
    ).clip(lower=0)
    df.loc[over, "TRANSACTION_CASHBACK_AMOUNT"] = (
            df.loc[over, "TRANSACTION_CASHBACK_AMOUNT"] * scale[over]
    ).clip(lower=0)

    if "CARD_EXPIRY_DATE" in df.columns:
        parsed = [parse_card_expiry_mm_yy(s) for s in df["CARD_EXPIRY_DATE"].fillna("")]
        df["card_expiry_month_clean"] = pd.Series([m for m, _ in parsed], index=df.index).astype("float32")
        df["card_expiry_year_clean"] = pd.Series([y for _, y in parsed], index=df.index).astype("float32")

    if is_train and "TX_FRAUD" in df.columns:
        df["TX_FRAUD"] = df["TX_FRAUD"].fillna(0).astype(int).clip(0, 1)

    drop_common = [c for c in ["CARD_EXPIRY_DATE", "TRANSACTION_STATUS", "FAILURE_CODE", "FAILURE_REASON"] if c in df.columns]
    df = df.drop(columns=drop_common)
    if is_train and "TX_ID" in df.columns:
        df = df.drop(columns=["TX_ID"])

    return df.reset_index(drop=True)

# %% [markdown]
# Apply cleaning and inspect

# %%
customers_clean = clean_customers(customers_raw)
terminals_clean = clean_terminals(terminals_raw)
merchants_clean = clean_merchants(merchants_raw)
train_clean = clean_transactions(train_raw, is_train=True)
test_clean = clean_transactions(test_raw, is_train=False)

# %% [markdown]
# Save cleaned CSVs to data/processed

# %%
out_dir = os.path.join("data", "processed")
os.makedirs(out_dir, exist_ok=True)

paths = {
    "customers_clean": os.path.join(out_dir, "customers_clean.csv"),
    "terminals_clean": os.path.join(out_dir, "terminals_clean.csv"),
    "merchants_clean": os.path.join(out_dir, "merchants_clean.csv"),
    "transactions_train_clean": os.path.join(out_dir, "transactions_train_clean.csv"),
    "transactions_test_clean": os.path.join(out_dir, "transactions_test_clean.csv"),
}

customers_clean.to_csv(paths["customers_clean"], index=False)
terminals_clean.to_csv(paths["terminals_clean"], index=False)
merchants_clean.to_csv(paths["merchants_clean"], index=False)
train_clean.to_csv(paths["transactions_train_clean"], index=False)
test_clean.to_csv(paths["transactions_test_clean"], index=False)