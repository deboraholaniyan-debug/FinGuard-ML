
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformed"
    / "validation.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lightgbm_baseline.txt"
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


# ==========================================
# 2. LOAD MODEL AND VALIDATION DATA
# ==========================================

print("Loading model...")

model = lgb.Booster(model_file=str(MODEL_PATH))

print("Loading validation data...")

df = pd.read_parquet(DATA_PATH)

X = df[FEATURE_COLUMNS]
y = df["target"].to_numpy()

print("Validation transactions:", f"{len(y):,}")
print("Fraud cases:", f"{y.sum():,}")


# ==========================================
# 3. GENERATE PREDICTIONS
# ==========================================

predictions = model.predict(
    X,
    num_iteration=model.best_iteration
)

# Save predictions for subsequent analysis.
prediction_df = pd.DataFrame({
    "actual": y,
    "risk_score": predictions
})

prediction_df.to_parquet(
    OUTPUT_DIR / "validation_predictions.parquet",
    index=False
)


# ==========================================
# 4. RANKING METRICS
# ==========================================

pr_auc = average_precision_score(y, predictions)
roc_auc = roc_auc_score(y, predictions)

print("\nRANKING PERFORMANCE")
print("=" * 45)
print(f"PR-AUC:  {pr_auc:.6f}")
print(f"ROC-AUC: {roc_auc:.6f}")
print(f"Random-ranking PR baseline: {y.mean():.6f}")


# ==========================================
# 5. THRESHOLD ANALYSIS
# ==========================================

thresholds = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]

results = []

for threshold in thresholds:

    predicted = (predictions >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predicted,
        labels=[0, 1]
    ).ravel()

    results.append({
        "threshold": threshold,
        "precision": precision_score(
            y, predicted, zero_division=0
        ),
        "recall": recall_score(
            y, predicted, zero_division=0
        ),
        "f1": f1_score(
            y, predicted, zero_division=0
        ),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "alerts": tp + fp
    })

threshold_df = pd.DataFrame(results)

print("\nTHRESHOLD PERFORMANCE")
print("=" * 90)
print(threshold_df.to_string(index=False))

threshold_df.to_csv(
    OUTPUT_DIR / "validation_threshold_results.csv",
    index=False
)


# ==========================================
# 6. TOP-K INVESTIGATION CAPACITY
# ==========================================

ranking = np.argsort(-predictions)

top_k_values = [100, 500, 1000, 2000, 5000]

top_k_results = []

for k in top_k_values:

    selected = ranking[:k]

    detected = int(y[selected].sum())

    top_k_results.append({
        "investigation_capacity": k,
        "fraud_detected": detected,
        "precision_at_k": detected / k,
        "recall_at_k": detected / y.sum()
    })

top_k_df = pd.DataFrame(top_k_results)

print("\nTOP-K INVESTIGATION PERFORMANCE")
print("=" * 75)
print(top_k_df.to_string(index=False))

top_k_df.to_csv(
    OUTPUT_DIR / "validation_top_k_results.csv",
    index=False
)


print("\nEvaluation completed.")
print("Results saved to:", OUTPUT_DIR)
