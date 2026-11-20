
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "evaluation"
    / "validation_predictions.parquet"
)

df = pd.read_parquet(PREDICTION_PATH)

scores = df["risk_score"].to_numpy()
labels = df["actual"].to_numpy()

print("PREDICTION SCORE DIAGNOSTICS")
print("=" * 55)

print("\nScore distribution:")
print(pd.Series(scores).describe(
    percentiles=[
        0.01, 0.05, 0.10, 0.25,
        0.50, 0.75, 0.90, 0.95, 0.99
    ]
))

print("\nMinimum score:", scores.min())
print("Maximum score:", scores.max())
print("Unique score values:", len(np.unique(scores)))

print("\nPrediction counts at selected thresholds:")

for threshold in [0.001, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 0.99]:

    predicted = scores >= threshold

    print(
        f"Threshold {threshold:.3f}: "
        f"{predicted.sum():,} predicted positives"
    )

print("\nScore distribution by actual class:")

for label in [0, 1]:

    class_scores = scores[labels == label]

    print(f"\nClass {label}")
    print(pd.Series(class_scores).describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.99]
    ))

print("\nMost frequent prediction scores:")

print(
    pd.Series(scores)
    .value_counts()
    .head(15)
    .to_string()
)
