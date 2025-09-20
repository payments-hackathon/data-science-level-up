#!/usr/bin/env python3
# flake8: noqa

# %% [markdown]
# Notebook: Train XGBoost and Create Submission
#
# What this script does:
# - Loads features from CSVs
# - Performs time-based CV with simple random search and bagging
# - Trains XGBoost
# - Writes submission.

# %% [markdown]
# Setup and utilities

# %%
import os
import random
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
# Metrics and splitting

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


def time_kfold_indices(n_rows: int, n_folds: int = 5):
    fold_sizes = [n_rows // n_folds] * n_folds
    for i in range(n_rows % n_folds):
        fold_sizes[i] += 1
    indices = np.arange(n_rows, dtype=int)
    folds = []
    start = 0
    for fs in fold_sizes:
        end = start + fs
        folds.append(indices[start:end])
        start = end
    return folds

# %% [markdown]
# CV training, random search, and bagging

# %%
def train_eval_xgb(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, params: dict,
                   num_boost_round: int = 3000, early_stopping_rounds: int = 150, verbose_eval: int = 0):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va)
    evals = [(dtrain, "train"), (dvalid, "valid")]
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )
    best_ntree_limit = getattr(bst, "best_ntree_limit", None)
    va_pred = bst.predict(dvalid, ntree_limit=best_ntree_limit) if best_ntree_limit else bst.predict(dvalid)
    auc = fast_auc(y_va.astype(np.int32), va_pred.astype(np.float32))
    best_iter = getattr(bst, "best_iteration", None)
    return auc, int(best_iter if best_iter is not None else num_boost_round), bst


def random_param_samples(seed: int = 1337, n_samples: int = 10):
    rng = random.Random(seed)
    for _ in range(n_samples):
        yield {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "eta": rng.uniform(0.03, 0.07),
            "max_depth": rng.randint(6, 10),
            "min_child_weight": rng.uniform(1.0, 5.0),
            "subsample": rng.uniform(0.7, 0.95),
            "colsample_bytree": rng.uniform(0.6, 0.9),
            "lambda": rng.uniform(1.0, 8.0),
            "gamma": 0.0,
            "scale_pos_weight": None,
            "max_bin": rng.choice([256, 512, 768, 1024]),
            "seed": rng.randint(1, 10_000),
        }

# %% [markdown]
# Train with time-based CV, select best params, bag models, and create submission

# %%
train, test = load_features()

label_col = "TX_FRAUD"
feature_cols = [c for c in train.columns if c not in {label_col, "TX_ID"}]
X_all = train[feature_cols].to_numpy(dtype=np.float32)
y_all = train[label_col].astype(np.int32).to_numpy()

n = len(train)
folds = time_kfold_indices(n_rows=n, n_folds=5)

best_score = -1.0
best_params = None
best_rounds = None

for ps in random_param_samples(seed=2025, n_samples=10):
    fold_scores = []
    fold_rounds = []
    for i, va_idx in enumerate(folds):
        start_va = va_idx[0]
        tr_idx = np.arange(0, start_va, dtype=int)
        if len(tr_idx) == 0:
            continue
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_va, y_va = X_all[va_idx], y_all[va_idx]
        pos = max(1, int(y_tr.sum()))
        neg = max(1, int((y_tr == 0).sum()))
        spw = float(neg) / float(pos)
        params = ps.copy()
        params["scale_pos_weight"] = spw
        auc, best_iter, _ = train_eval_xgb(X_tr, y_tr, X_va, y_va, params, verbose_eval=50 if i == 0 else 0)
        fold_scores.append(auc)
        fold_rounds.append(best_iter)
    if not fold_scores:
        continue
    mean_auc = float(np.mean(fold_scores))
    mean_rounds = int(np.mean(fold_rounds))
    if mean_auc > best_score:
        best_score, best_params, best_rounds = mean_auc, ps.copy(), mean_rounds
    print(f"Tried params -> AUC: {mean_auc:.5f}, rounds: {mean_rounds}, params(seed={ps['seed']}, depth={ps['max_depth']}, eta={ps['eta']:.4f}, max_bin={ps['max_bin']})")

print(f"\nBest CV AUC: {best_score:.6f} with rounds ~ {best_rounds} and params: {best_params}")

pos_all = max(1, int(y_all.sum()))
neg_all = max(1, int((y_all == 0).sum()))
best_params["scale_pos_weight"] = float(neg_all) / float(pos_all)

X_test = test[feature_cols].to_numpy(dtype=np.float32)
dtest = xgb.DMatrix(X_test)

bag_seeds = [101, 202, 303, 404, 505]
test_preds = []
for s in bag_seeds:
    p = best_params.copy()
    p["seed"] = s
    dall = xgb.DMatrix(X_all, label=y_all)
    final_bst = xgb.train(
        params=p,
        dtrain=dall,
        num_boost_round=int(best_rounds * 1.1) if best_rounds else 3000,
        evals=[(dall, "train")],
        verbose_eval=200,
    )
    test_pred = final_bst.predict(dtest)
    test_preds.append(test_pred)

avg_test_pred = np.mean(test_preds, axis=0)

sub = pd.DataFrame({"TX_ID": test["TX_ID"].astype(str).values, "TX_FRAUD": avg_test_pred.astype(float)})
os.makedirs("submissions", exist_ok=True)
out_path = os.path.join("submissions", "submission_xgb.csv")
sub.to_csv(out_path, index=False)
print(f"Saved submission to: {out_path}")