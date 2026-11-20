
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformed"
    / "test.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "class_weight_experiments"
    / "unweighted.txt"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
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

TARGET_COLUMN = "target"
TOP_K = 500


# Load test data
print("Loading test data...")

test_df = pd.read_parquet(TEST_PATH)

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df[TARGET_COLUMN].to_numpy()

print("Test transactions:", f"{len(test_df):,}")
print("Fraud cases:", f"{int(y_test.sum()):,}")


# Load selected model
print("\nLoading unweighted model...")

model = lgb.Booster(
    model_file=str(MODEL_PATH)
)

# Predict
predictions = model.predict(X_test)

# Overall metrics
pr_auc = average_precision_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    predictions
)

random_baseline = float(y_test.mean())


# Top-K evaluation
top_indices = np.argsort(-predictions)[:TOP_K]

detected_fraud = int(y_test[top_indices].sum())

precision_at_k = detected_fraud / TOP_K
recall_at_k = detected_fraud / int(y_test.sum())


# Report
print("\n" + "=" * 55)
print("UNWEIGHTED MODEL — TEST EVALUATION")
print("=" * 55)

print(f"PR-AUC:             {pr_auc:.6f}")
print(f"ROC-AUC:            {roc_auc:.6f}")
print(f"Random PR baseline: {random_baseline:.6f}")
print(f"Precision@{TOP_K}:      {precision_at_k:.4f}")
print(f"Recall@{TOP_K}:         {recall_at_k:.4f}")
print(f"Fraud in top {TOP_K}:   {detected_fraud:,}")

# Save predictions
prediction_df = pd.DataFrame({
    "actual": y_test,
    "risk_score": predictions
})

prediction_path = (
    OUTPUT_DIR / "unweighted_test_predictions.parquet"
)

prediction_df.to_parquet(
    prediction_path,
    index=False
)

print("\nPredictions saved to:", prediction_path)
