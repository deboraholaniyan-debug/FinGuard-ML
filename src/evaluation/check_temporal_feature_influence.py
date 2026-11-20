
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score
from src.features.build_transaction_features import (
    build_transaction_features
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "data/raw/paysim_transactions.csv"
MODEL_PATH = PROJECT_ROOT / "models/class_weight_experiments/unweighted.txt"
SCALER_PATH = PROJECT_ROOT / "models/preprocessing/standard_scaler.joblib"

FEATURES = [
    "amount",
    "origin_balance_before",
    "destination_balance_before",
    "step",
    "hour",
    "log_amount",
    "transaction_type_CASH_IN",
    "transaction_type_CASH_OUT",
    "transaction_type_DEBIT",
    "transaction_type_PAYMENT",
    "transaction_type_TRANSFER"
]

NUMERICAL = [
    "amount",
    "origin_balance_before",
    "destination_balance_before",
    "step",
    "hour",
    "log_amount"
]

CATEGORIES = [
    "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"
]

print("Loading model and scaler...")

model = lgb.Booster(model_file=str(MODEL_PATH))
scaler = joblib.load(SCALER_PATH)

print("Loading chronological test transactions...")

raw = pd.read_csv(RAW_PATH)

raw = raw[
    raw["step"].between(632, 743)
].copy()

# Identify non-sparse time steps.
step_counts = raw.groupby("step").size()
non_sparse_steps = step_counts[step_counts >= 100].index

raw = raw[raw["step"].isin(non_sparse_steps)].copy()

print("Non-sparse transactions:", f"{len(raw):,}")
print("Fraud cases:", f"{int(raw['isFraud'].sum()):,}")

# Build the same features used during training.
features = build_transaction_features(raw)

numerical = features[NUMERICAL].copy()
scaled = scaler.transform(numerical)

numerical_df = pd.DataFrame(
    scaled,
    columns=NUMERICAL,
    index=features.index
)

categorical = pd.get_dummies(
    features["transaction_type"],
    prefix="transaction_type",
    dtype=np.float32
)

categorical = categorical.reindex(
    columns=[
        f"transaction_type_{category}"
        for category in CATEGORIES
    ],
    fill_value=0
)

X = pd.concat([numerical_df, categorical], axis=1)
X = X[FEATURES].astype(np.float32)

y = raw["isFraud"].to_numpy()

# Baseline performance.
baseline_scores = model.predict(X)
baseline_pr_auc = average_precision_score(y, baseline_scores)

print("\nBaseline PR-AUC:", round(baseline_pr_auc, 6))

# Shuffle one feature at a time.
rng = np.random.default_rng(42)
results = []

for feature in ["step", "hour"]:

    X_permuted = X.copy()

    X_permuted[feature] = rng.permutation(
        X_permuted[feature].to_numpy()
    )

    permuted_scores = model.predict(X_permuted)

    permuted_pr_auc = average_precision_score(
        y,
        permuted_scores
    )

    results.append({
        "feature": feature,
        "baseline_pr_auc": baseline_pr_auc,
        "permuted_pr_auc": permuted_pr_auc,
        "pr_auc_drop": baseline_pr_auc - permuted_pr_auc
    })

results_df = pd.DataFrame(results)

print("\nTEMPORAL FEATURE PERMUTATION RESULTS")
print(results_df.to_string(index=False))

output_path = (
    PROJECT_ROOT
    / "reports/evaluation/temporal_feature_influence.csv"
)

results_df.to_csv(output_path, index=False)

print("\nResults saved to:", output_path)
