
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score

from src.features.build_transaction_features import (
    build_transaction_features
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "paysim_transactions.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "class_weight_experiments"
    / "unweighted.txt"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "models"
    / "preprocessing"
    / "standard_scaler.joblib"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 50_000

TEST_START_STEP = 632
TEST_END_STEP = 743

NUMERICAL_COLUMNS = [
    "amount",
    "origin_balance_before",
    "destination_balance_before",
    "step",
    "hour",
    "log_amount"
]

CATEGORIES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER"
]

FEATURE_COLUMNS = NUMERICAL_COLUMNS + [
    f"transaction_type_{category}"
    for category in CATEGORIES
]


# ==========================================
# 2. LOAD MODEL AND SCALER
# ==========================================

print("Loading model and preprocessing scaler...")

model = lgb.Booster(
    model_file=str(MODEL_PATH)
)

scaler = joblib.load(SCALER_PATH)


# ==========================================
# 3. PROCESS CHRONOLOGICAL TEST PERIOD
# ==========================================

prediction_parts = []

total_transactions = 0
total_fraud = 0

print("\nProcessing chronological test period...")

for chunk_number, chunk in enumerate(
    pd.read_csv(RAW_DATA, chunksize=CHUNK_SIZE),
    start=1
):

    # Select only chronological test transactions.
    chunk = chunk[
        chunk["step"].between(
            TEST_START_STEP,
            TEST_END_STEP
        )
    ].copy()

    if chunk.empty:
        continue

    # Build pre-transaction features.
    features = build_transaction_features(chunk)

    # Separate numerical features.
    numerical = features[NUMERICAL_COLUMNS].copy()

    # Apply the previously fitted scaler.
    numerical_scaled = scaler.transform(numerical)

    numerical_df = pd.DataFrame(
        numerical_scaled,
        columns=NUMERICAL_COLUMNS,
        index=features.index
    )

    # One-hot encode transaction type.
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

    # Combine features in the model's expected order.
    X = pd.concat(
        [numerical_df, categorical],
        axis=1
    )

    X = X[FEATURE_COLUMNS].astype(np.float32)

    # Generate risk scores.
    predictions = model.predict(X)

    # Preserve actual labels for evaluation.
    labels = chunk["isFraud"].to_numpy(dtype=np.int8)

    prediction_parts.append(
        pd.DataFrame({
            "step": chunk["step"].to_numpy(),
            "actual": labels,
            "risk_score": predictions
        })
    )

    total_transactions += len(chunk)
    total_fraud += int(labels.sum())

    print(
        f"Processed chunk {chunk_number} | "
        f"Transactions: {total_transactions:,}"
    )


# ==========================================
# 4. COMBINE PREDICTIONS
# ==========================================

results = pd.concat(
    prediction_parts,
    ignore_index=True
)

y_test = results["actual"].to_numpy()
scores = results["risk_score"].to_numpy()

print("\nChronological test transactions:", f"{len(results):,}")
print("Fraud cases:", f"{total_fraud:,}")


# ==========================================
# 5. EVALUATE MODEL
# ==========================================

pr_auc = average_precision_score(
    y_test,
    scores
)

roc_auc = roc_auc_score(
    y_test,
    scores
)

random_baseline = float(y_test.mean())

top_k = min(500, len(results))

top_indices = np.argsort(-scores)[:top_k]

detected_fraud = int(y_test[top_indices].sum())

precision_at_k = detected_fraud / top_k

recall_at_k = detected_fraud / total_fraud


# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 55)
print("CHRONOLOGICAL TEST EVALUATION")
print("=" * 55)

print(f"Test steps:          {TEST_START_STEP}-{TEST_END_STEP}")
print(f"Transactions:        {len(results):,}")
print(f"Fraud cases:         {total_fraud:,}")
print(f"PR-AUC:              {pr_auc:.6f}")
print(f"ROC-AUC:             {roc_auc:.6f}")
print(f"Random PR baseline:  {random_baseline:.6f}")
print(f"Precision@{top_k}:       {precision_at_k:.4f}")
print(f"Recall@{top_k}:          {recall_at_k:.4f}")
print(f"Fraud in top {top_k}:    {detected_fraud:,}")


# ==========================================
# 7. SAVE RESULTS
# ==========================================

output_path = (
    OUTPUT_DIR / "chronological_test_predictions.parquet"
)

results.to_parquet(
    output_path,
    index=False
)

print("\nPredictions saved to:", output_path)
