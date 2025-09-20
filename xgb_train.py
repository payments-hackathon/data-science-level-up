#!/usr/bin/env python3
# flake8: noqa

# %% [markdown]
# Notebook: Train XGBoost and Create Submission
#
# What this script does:
# - Loads features from CSV
# - Trains XGBoost
# - Writes submission.

# %% [markdown]
# Setup and utilities

# %%
import os
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception as e:
    raise SystemExit(
        "xgboost is required for training but not installed.\n"
        "Install with:\n"
        "  pip install xgboost\n"
        f"Import error: {e}"
    )

# %% [markdown]
# Load features

# %%
def load_features(feature_dir: str = os.path.join("data", "features")):
    train_path = os.path.join(feature_dir, "train_features.csv")
    test_path = os.path.join(feature_dir, "test_features.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Missing feature files. Run the feature engineering script first.")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

# %% [markdown]
# Simple AUC and split helpers

# %%
def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    y_true = y_true[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    cum_neg = np.cumsum(1 - y_true)
    auc = (y_true * (cum_neg)).sum() / (n_pos * n_neg)
    return float(auc)


def train_valid_split_indices(n_rows: int, valid_fraction: float = 0.1):
    split_idx = int((1.0 - valid_fraction) * n_rows)
    tr_idx = np.arange(0, split_idx, dtype=int)
    va_idx = np.arange(split_idx, n_rows, dtype=int)
    return tr_idx, va_idx

# %% [markdown]
# Train, evaluate, and create submission

# %%
train, test = load_features()

label_col = "TX_FRAUD"
feature_cols = [c for c in train.columns if c not in {label_col, "TX_ID"}]
X = train[feature_cols].to_numpy(dtype=np.float32)
y = train[label_col].astype(np.int32).to_numpy()

tr_idx, va_idx = train_valid_split_indices(len(train), valid_fraction=0.1)
X_tr, y_tr = X[tr_idx], y[tr_idx]
X_va, y_va = X[va_idx], y[va_idx]

pos = max(1, int(y_tr.sum()))
neg = max(1, int((y_tr == 0).sum()))
spw = float(neg) / float(pos)

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dvalid = xgb.DMatrix(X_va, label=y_va)

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "eta": 0.05,
    "max_depth": 8,
    "min_child_weight": 2.0,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "lambda": 2.0,
    "gamma": 0.0,
    "scale_pos_weight": spw,
    "seed": 1337,
}

evals = [(dtrain, "train"), (dvalid, "valid")]
num_boost_round = 2000
early_stopping_rounds = 100

bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    evals=evals,
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=50,
)

best_ntree_limit = getattr(bst, "best_ntree_limit", None)
va_pred = bst.predict(dvalid, ntree_limit=best_ntree_limit) if best_ntree_limit else bst.predict(dvalid)
auc = fast_auc(y_va.astype(np.int32), va_pred.astype(np.float32))
print(f"Validation AUC (fast_auc): {auc:.6f}")

X_test = test[feature_cols].to_numpy(dtype=np.float32)
dtest = xgb.DMatrix(X_test)
test_pred = bst.predict(dtest, ntree_limit=best_ntree_limit) if best_ntree_limit else bst.predict(dtest)

sub = pd.DataFrame({"TX_ID": test["TX_ID"].astype(str).values, "TX_FRAUD": test_pred.astype(float)})
os.makedirs("submissions", exist_ok=True)
out_path = os.path.join("submissions", "submission_xgb.csv")
sub.to_csv(out_path, index=False)
print(f"Saved submission to: {out_path}")