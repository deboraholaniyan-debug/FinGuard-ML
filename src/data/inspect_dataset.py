
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/paysim_transactions.csv")
CHUNK_SIZE = 100_000

total_rows = 0
fraud_counts = {0: 0, 1: 0}
flagged_counts = {0: 0, 1: 0}
transaction_types = {}
missing_values = {}
min_step = float("inf")
max_step = float("-inf")

for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):

    total_rows += len(chunk)

    fraud = chunk["isFraud"].value_counts()
    flagged = chunk["isFlaggedFraud"].value_counts()

    for key, value in fraud.items():
        fraud_counts[int(key)] += int(value)

    for key, value in flagged.items():
        flagged_counts[int(key)] += int(value)

    for key, value in chunk["type"].value_counts().items():
        transaction_types[key] = (
            transaction_types.get(key, 0) + int(value)
        )

    for column, value in chunk.isnull().sum().items():
        missing_values[column] = (
            missing_values.get(column, 0) + int(value)
        )

    min_step = min(min_step, chunk["step"].min())
    max_step = max(max_step, chunk["step"].max())

print("\nDATASET PROFILE")
print("-" * 45)
print("Total transactions:", f"{total_rows:,}")
print("Time steps:", min_step, "to", max_step)

print("\nFRAUD DISTRIBUTION")
for label, count in fraud_counts.items():
    percentage = count / total_rows * 100
    print(f"Class {label}: {count:,} ({percentage:.5f}%)")

print("\nEXISTING FLAG DISTRIBUTION")
for label, count in flagged_counts.items():
    print(f"Flag {label}: {count:,}")

print("\nTRANSACTION TYPES")
for name, count in sorted(transaction_types.items()):
    print(f"{name}: {count:,}")

print("\nMISSING VALUES")
for column, count in missing_values.items():
    print(f"{column}: {count:,}")
