
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "evaluation"
    / "chronological_test_predictions.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "evaluation"
    / "sparse_period_comparison.csv"
)

SPARSE_THRESHOLD = 100


# ==========================================
# 1. LOAD CHRONOLOGICAL PREDICTIONS
# ==========================================

print("Loading chronological predictions...")

df = pd.read_parquet(PREDICTIONS_PATH)


# ==========================================
# 2. CLASSIFY TIME STEPS
# ==========================================

step_counts = df.groupby("step").size()

sparse_steps = step_counts[
    step_counts < SPARSE_THRESHOLD
].index

df["period_type"] = np.where(
    df["step"].isin(sparse_steps),
    "Sparse",
    "Non-sparse"
)

print("\nTime-step classification:")
print("Sparse steps:", len(sparse_steps))
print("Non-sparse steps:", len(step_counts) - len(sparse_steps))


# ==========================================
# 3. EVALUATE EACH PERIOD
# ==========================================

results = []

for period in ["Sparse", "Non-sparse"]:

    subset = df[df["period_type"] == period]

    y_true = subset["actual"].to_numpy()
    scores = subset["risk_score"].to_numpy()

    transactions = len(subset)
    fraud_cases = int(y_true.sum())

    if transactions == 0 or fraud_cases == 0:
        print(f"\nSkipping {period}: insufficient data.")
        continue

    pr_auc = average_precision_score(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)

    top_k = min(500, transactions)

    top_indices = np.argsort(-scores)[:top_k]

    detected_fraud = int(y_true[top_indices].sum())

    precision_at_k = detected_fraud / top_k
    recall_at_k = detected_fraud / fraud_cases

    results.append({
        "period": period,
        "transactions": transactions,
        "fraud_cases": fraud_cases,
        "fraud_rate_percent": fraud_cases / transactions * 100,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "top_k": top_k,
        "fraud_in_top_k": detected_fraud,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k
    })


# ==========================================
# 4. DISPLAY COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("SPARSE VS NON-SPARSE PERIOD EVALUATION")
print("=" * 75)

print(results_df.to_string(index=False))


# ==========================================
# 5. SAVE RESULTS
# ==========================================

results_df.to_csv(OUTPUT_PATH, index=False)

print("\nResults saved to:", OUTPUT_PATH)
