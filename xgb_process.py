#!/usr/bin/env python3
# flake8: noqa

# %% [markdown]
# Notebook: Feature Engineering (Fraud Detection)
#
# What this script does:
# - Loads cleaned CSVs
# - Builds model-ready features
# - Encodes categoricals with train-fitted mappings
# - Saves features to data/features/*.csv

# %% [markdown]
# Setup and helpers

# %%
import os
import json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 160)

# %% [markdown]
# Load cleaned datasets

# %%
def load_cleaned(base_dir: str = os.path.join("data", "processed")) -> Dict[str, pd.DataFrame]:
    paths = {
        "train": os.path.join(base_dir, "transactions_train_clean.csv"),
        "test": os.path.join(base_dir, "transactions_test_clean.csv"),
        "customers": os.path.join(base_dir, "customers_clean.csv"),
        "terminals": os.path.join(base_dir, "terminals_clean.csv"),
        "merchants": os.path.join(base_dir, "merchants_clean.csv"),
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing cleaned file: {p}. Run the cleaning script first.")
    train = pd.read_csv(paths["train"], parse_dates=["TX_TS"])
    test = pd.read_csv(paths["test"], parse_dates=["TX_TS"])
    customers = pd.read_csv(paths["customers"])
    merchants = pd.read_csv(paths["merchants"])
    terminals = pd.read_csv(paths["terminals"])

    terminals = terminals.rename(columns={"y_terminal__id": "y_terminal_id"})

    return {"train": train, "test": test, "customers": customers, "terminals": terminals, "merchants": merchants}

# %% [markdown]
# Feature engineering utilities

# %%
def compute_geo_distance(x1: pd.Series, y1: pd.Series, x2: pd.Series, y2: pd.Series) -> pd.Series:
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def months_between(d1: pd.Series, d2: pd.Series) -> pd.Series:
    return (d2.dt.year - d1.dt.year) * 12 + (d2.dt.month - d1.dt.month) + (d2.dt.day - d1.dt.day) / 30.0


def encode_categorical_train(series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    vals = series.fillna("__NA__").astype(str)
    uniques = pd.Index(sorted(vals.unique()))
    mapping = {v: i for i, v in enumerate(uniques)}
    codes = vals.map(mapping).astype("int32")
    return codes, mapping


def encode_categorical_test(series: pd.Series, mapping: Dict[str, int]) -> pd.Series:
    vals = series.fillna("__NA__").astype(str)
    codes = vals.map(mapping).fillna(-1).astype("int32")
    return codes

# %% [markdown]
# Build features

# %%
def build_features(
    df: pd.DataFrame,
    customers: pd.DataFrame,
    terminals: pd.DataFrame,
    merchants: pd.DataFrame,
    is_train: bool,
    cat_mappings: Dict[str, Dict[str, int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    if cat_mappings is None:
        cat_mappings = {}

    df = df.merge(customers, on="CUSTOMER_ID", how="left")
    df = df.merge(terminals, on="TERMINAL_ID", how="left")
    df = df.merge(merchants, on="MERCHANT_ID", how="left")

    df["tx_hour"] = df["TX_TS"].dt.hour.astype("int16")
    df["tx_dow"] = df["TX_TS"].dt.dayofweek.astype("int16")
    df["tx_day"] = df["TX_TS"].dt.day.astype("int16")
    df["tx_month"] = df["TX_TS"].dt.month.astype("int16")

    total = df["TX_AMOUNT"].fillna(0.0)
    goods = df.get("TRANSACTION_GOODS_AND_SERVICES_AMOUNT", pd.Series(0.0, index=df.index)).fillna(0.0)
    cashback = df.get("TRANSACTION_CASHBACK_AMOUNT", pd.Series(0.0, index=df.index)).fillna(0.0)
    df["amount_log1p"] = np.log1p(total.clip(lower=0)).astype("float32")
    denom = (total + 1e-6)
    df["goods_ratio"] = (goods / denom).astype("float32")
    df["cashback_ratio"] = (cashback / denom).astype("float32")

    if {"card_expiry_month_clean", "card_expiry_year_clean"}.issubset(df.columns):
        valid = df["card_expiry_month_clean"].notna() & df["card_expiry_year_clean"].notna()
        exp_dates = pd.to_datetime(
            {
                "year": df.loc[valid, "card_expiry_year_clean"].astype(int),
                "month": df.loc[valid, "card_expiry_month_clean"].astype(int),
                "day": 28,
            },
            errors="coerce",
        )
        df["months_to_expiry"] = np.nan
        df.loc[valid, "months_to_expiry"] = months_between(df.loc[valid, "TX_TS"], exp_dates)
        df["months_to_expiry"] = df["months_to_expiry"].astype("float32")
    else:
        df["months_to_expiry"] = np.float32(np.nan)

    df["cust_term_dist"] = compute_geo_distance(
        df["x_customer_id"], df["y_customer_id"], df["x_terminal_id"], df["y_terminal_id"]
    ).astype("float32")

    for col in ["ANNUAL_TURNOVER_CARD", "AVERAGE_TICKET_SALE_AMOUNT"]:
        if col in df.columns:
            df[f"{col.lower()}_log1p"] = np.log1p(df[col].fillna(0).clip(lower=0)).astype("float32")

    df["merchant_tax_exempt_int"] = df.get("TAX_EXCEMPT_INDICATOR", False)
    df["merchant_tax_exempt_int"] = df["merchant_tax_exempt_int"].fillna(False).astype(int).astype("int8")
    df["is_recurring_int"] = df.get("IS_RECURRING_TRANSACTION", False)
    df["is_recurring_int"] = df["is_recurring_int"].fillna(False).astype(int).astype("int8")

    cat_cols = [
        "CARD_BRAND",
        "TRANSACTION_TYPE",
        "TRANSACTION_CURRENCY",
        "CARD_COUNTRY_CODE",
        "ACQUIRER_ID",
        "BUSINESS_TYPE",
        "MCC_CODE",
        "OUTLET_TYPE"
    ]

    for col in cat_cols:
        dst = f"{col}_code"
        if is_train:
            codes, mapping = encode_categorical_train(df[col] if col in df.columns else pd.Series(["__NA__"] * len(df)))
            df[dst] = codes
            cat_mappings[col] = mapping
        else:
            mapping = cat_mappings.get(col, {})
            df[dst] = encode_categorical_test(df[col] if col in df.columns else pd.Series(["__NA__"] * len(df)), mapping)

    feat_cols: List[str] = [
        "tx_hour", "tx_dow", "tx_day", "tx_month",
        "months_to_expiry",
        "amount_log1p", "goods_ratio", "cashback_ratio",
        "x_customer_id", "y_customer_id", "x_terminal_id", "y_terminal__id", "cust_term_dist",
        "annual_turnover_card_log1p" if "annual_turnover_card_log1p" in df.columns else "ANNUAL_TURNOVER_CARD_log1p",
        "average_ticket_sale_amount_log1p" if "average_ticket_sale_amount_log1p" in df.columns else "AVERAGE_TICKET_SALE_AMOUNT_log1p",
        "PAYMENT_PERCENTAGE_FACE_TO_FACE", "PAYMENT_PERCENTAGE_ECOM", "PAYMENT_PERCENTAGE_MOTO",
        "DEPOSIT_REQUIRED_PERCENTAGE", "DEPOSIT_PERCENTAGE",
        "DELIVERY_SAME_DAYS_PERCENTAGE", "DELIVERY_WEEK_ONE_PERCENTAGE", "DELIVERY_WEEK_TWO_PERCENTAGE", "DELIVERY_OVER_TWO_WEEKS_PERCENTAGE",
        "merchant_tax_exempt_int", "is_recurring_int",
    ] + [f"{c}_code" for c in cat_cols]

    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0
    df[feat_cols] = df[feat_cols].astype("float32")

    cols_out = feat_cols.copy()
    if "TX_FRAUD" in df.columns:
        cols_out.append("TX_FRAUD")
    if "TX_ID" in df.columns:
        cols_out.append("TX_ID")

    return df[cols_out].reset_index(drop=True), cat_mappings

# %% [markdown]
# Build and save features

# %%
data = load_cleaned()
train_raw = data["train"].sort_values("TX_TS").reset_index(drop=True)
test_raw = data["test"].sort_values("TX_TS").reset_index(drop=True)

train_feats, cat_maps = build_features(
    train_raw, data["customers"], data["terminals"], data["merchants"], is_train=True, cat_mappings=None
)
test_feats, _ = build_features(
    test_raw, data["customers"], data["terminals"], data["merchants"], is_train=False, cat_mappings=cat_maps
)

out_dir = os.path.join("data", "features")
os.makedirs(out_dir, exist_ok=True)
train_path = os.path.join(out_dir, "train_features.csv")
test_path = os.path.join(out_dir, "test_features.csv")
meta_path = os.path.join(out_dir, "cat_mappings.json")

train_feats.to_csv(train_path, index=False)
test_feats.to_csv(test_path, index=False)
with open(meta_path, "w") as f:
    json.dump({k: v for k, v in cat_maps.items()}, f)