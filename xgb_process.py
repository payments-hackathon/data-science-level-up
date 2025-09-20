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
# Rolling, frequency, and core feature builders

# %%
def add_time_recency_features(df: pd.DataFrame, key: str, ts_col: str = "TX_TS") -> pd.DataFrame:
    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    ts = ts.dt.tz_convert(None)
    g = df[[key]].copy()
    g[ts_col] = ts

    g = g.sort_values([key, ts_col])
    idx_sorted = g.index

    tsl = g.groupby(key, sort=False)[ts_col].diff().dt.total_seconds().astype("float32")
    tsl = tsl.fillna(np.float32(1e9))

    g["__ones__"] = 1.0

    c24 = (
        g.set_index(ts_col)
         .groupby(key)["__ones__"]
         .rolling(window="24h")
         .sum()
         .reset_index(level=0, drop=True)
         .astype("float32") - 1.0
    ).clip(lower=0)

    c7d = (
        g.set_index(ts_col)
         .groupby(key)["__ones__"]
         .rolling(window="7d")
         .sum()
         .reset_index(level=0, drop=True)
         .astype("float32") - 1.0
    ).clip(lower=0)

    out = df.copy()
    out.loc[idx_sorted, f"time_since_last_{key}_sec"] = tsl.values
    out.loc[idx_sorted, f"count_24h_{key}"] = c24.values
    out.loc[idx_sorted, f"count_7d_{key}"] = c7d.values

    out[f"time_since_last_{key}_sec"] = out[f"time_since_last_{key}_sec"].fillna(np.float32(1e9)).astype("float32")
    out[f"count_24h_{key}"] = out[f"count_24h_{key}"].fillna(0).astype("float32")
    out[f"count_7d_{key}"] = out[f"count_7d_{key}"].fillna(0).astype("float32")

    return out


def add_frequency_encoding(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for col in cols:
        cnt = train_df[col].fillna("__NA__").astype(str).value_counts()
        mapping = cnt.to_dict()
        train_df[f"freq_{col}"] = np.log1p(train_df[col].fillna("__NA__").astype(str).map(mapping).fillna(0)).astype("float32")
        test_df[f"freq_{col}"] = np.log1p(test_df[col].fillna("__NA__").astype(str).map(mapping).fillna(0)).astype("float32")
    return train_df, test_df

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

    for key in ["CUSTOMER_ID", "TERMINAL_ID"]:
        tmp = add_time_recency_features(df[[key, "TX_TS"]].copy(), key=key, ts_col="TX_TS")
        df[f"time_since_last_{key}_sec"] = tmp[f"time_since_last_{key}_sec"].astype("float32")
        df[f"count_24h_{key}"] = tmp[f"count_24h_{key}"].astype("float32")
        df[f"count_7d_{key}"] = tmp[f"count_7d_{key}"].astype("float32")

    feat_cols: List[str] = [
        "tx_hour", "tx_dow", "tx_day", "tx_month",
        "months_to_expiry",
        "amount_log1p", "goods_ratio", "cashback_ratio",
        "x_customer_id", "y_customer_id", "x_terminal_id", "y_terminal_id", "cust_term_dist",
        "ANNUAL_TURNOVER_CARD_log1p", "AVERAGE_TICKET_SALE_AMOUNT_log1p",
        "PAYMENT_PERCENTAGE_FACE_TO_FACE", "PAYMENT_PERCENTAGE_ECOM", "PAYMENT_PERCENTAGE_MOTO",
        "DEPOSIT_REQUIRED_PERCENTAGE", "DEPOSIT_PERCENTAGE",
        "DELIVERY_SAME_DAYS_PERCENTAGE", "DELIVERY_WEEK_ONE_PERCENTAGE", "DELIVERY_WEEK_TWO_PERCENTAGE", "DELIVERY_OVER_TWO_WEEKS_PERCENTAGE",
        "merchant_tax_exempt_int", "is_recurring_int",
        "time_since_last_CUSTOMER_ID_sec", "count_24h_CUSTOMER_ID", "count_7d_CUSTOMER_ID",
        "time_since_last_TERMINAL_ID_sec", "count_24h_TERMINAL_ID", "count_7d_TERMINAL_ID",
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

    for raw_col in ["CUSTOMER_ID", "TERMINAL_ID", "MERCHANT_ID", "CARD_DATA", "MCC_CODE", "BUSINESS_TYPE", "ACQUIRER_ID"]:
        if raw_col in df.columns and raw_col not in cols_out:
            cols_out.append(raw_col)

    return df[cols_out].reset_index(drop=True), cat_mappings

# %% [markdown]
# Build, add frequency encodings, and save features

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

freq_cols = ["CUSTOMER_ID", "TERMINAL_ID", "MERCHANT_ID", "CARD_DATA", "MCC_CODE", "BUSINESS_TYPE", "ACQUIRER_ID"]
train_feats, test_feats = add_frequency_encoding(train_feats, test_feats, cols=[c for c in freq_cols if c in train_feats.columns])

drop_ids = [c for c in freq_cols if c in train_feats.columns]
train_feats = train_feats.drop(columns=drop_ids, errors="ignore")
test_feats = test_feats.drop(columns=drop_ids, errors="ignore")

out_dir = os.path.join("data", "features")
os.makedirs(out_dir, exist_ok=True)
train_path = os.path.join(out_dir, "train_features.csv")
test_path = os.path.join(out_dir, "test_features.csv")
meta_path = os.path.join(out_dir, "cat_mappings.json")

train_feats.to_csv(train_path, index=False)
test_feats.to_csv(test_path, index=False)
with open(meta_path, "w") as f:
    json.dump({k: v for k, v in cat_maps.items()}, f)