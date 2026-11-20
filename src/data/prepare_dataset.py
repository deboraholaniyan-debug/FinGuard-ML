
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_transaction_features import (
    build_transaction_features
)


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "paysim_transactions.csv"
)

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_INDEX_PATH = INTERIM_DIR / "train_indices.npy"
VALIDATION_INDEX_PATH = INTERIM_DIR / "validation_indices.npy"
TEST_INDEX_PATH = INTERIM_DIR / "test_indices.npy"

CHUNK_SIZE = 50_000

SPLIT_NAMES = {
    0: "train",
    1: "validation",
    2: "test"
}


# ==========================================
# 2. VALIDATE INPUT FILES
# ==========================================

required_files = [
    DATA_PATH,
    TRAIN_INDEX_PATH,
    VALIDATION_INDEX_PATH,
    TEST_INDEX_PATH
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

# Prevent accidental duplicate output files.
for split_name in SPLIT_NAMES.values():
    split_dir = OUTPUT_DIR / split_name

    if split_dir.exists() and any(split_dir.glob("*.parquet")):
        raise FileExistsError(
            f"Existing Parquet files found in {split_dir}. "
            "Move or remove them before running this pipeline."
        )

    split_dir.mkdir(parents=True, exist_ok=True)


# ==========================================
# 3. LOAD SPLIT INDICES
# ==========================================

train_indices = np.load(TRAIN_INDEX_PATH)
validation_indices = np.load(VALIDATION_INDEX_PATH)
test_indices = np.load(TEST_INDEX_PATH)

all_indices = np.concatenate([
    train_indices,
    validation_indices,
    test_indices
])

if len(np.unique(all_indices)) != len(all_indices):
    raise ValueError("Duplicate row indices detected across partitions.")

total_rows = len(all_indices)

if not np.array_equal(
    np.sort(all_indices),
    np.arange(total_rows)
):
    raise ValueError(
        "Split indices do not cover the dataset rows exactly."
    )

print("Split indices loaded successfully.")
print("Total rows:", total_rows)


# ==========================================
# 4. CREATE ROW-LEVEL SPLIT LOOKUP
# ==========================================

# 0 = train, 1 = validation, 2 = test
split_lookup = np.full(
    total_rows,
    -1,
    dtype=np.int8
)

split_lookup[train_indices] = 0
split_lookup[validation_indices] = 1
split_lookup[test_indices] = 2


# ==========================================
# 5. PROCESS CSV IN CHUNKS
# ==========================================

row_offset = 0
chunk_number = 0

partition_counts = {
    "train": 0,
    "validation": 0,
    "test": 0
}

partition_fraud_counts = {
    "train": 0,
    "validation": 0,
    "test": 0
}

print("\nStarting chunked data preparation...")
print("Chunk size:", CHUNK_SIZE)

for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=CHUNK_SIZE
):

    chunk_length = len(chunk)

    start = row_offset
    end = start + chunk_length

    if end > total_rows:
        raise ValueError(
            "CSV contains more rows than the saved split indices."
        )

    # Identify the original partition of each row.
    chunk_splits = split_lookup[start:end]

    # Generate features using the existing feature module.
    features = build_transaction_features(chunk)

    # Attach the target separately.
    features["target"] = (
        chunk["isFraud"]
        .to_numpy(dtype=np.int8)
    )

    # Save each partition independently.
    for split_id, split_name in SPLIT_NAMES.items():

        mask = chunk_splits == split_id

        if not mask.any():
            continue

        partition = features.loc[mask].copy()

        output_path = (
            OUTPUT_DIR
            / split_name
            / f"part_{chunk_number:05d}.parquet"
        )

        partition.to_parquet(
            output_path,
            index=False,
            engine="pyarrow"
        )

        partition_counts[split_name] += len(partition)

        partition_fraud_counts[split_name] += int(
            partition["target"].sum()
        )

    row_offset = end
    chunk_number += 1

    print(
        f"Processed chunk {chunk_number} | "
        f"Rows processed: {row_offset:,}/{total_rows:,}"
    )


# ==========================================
# 6. FINAL VALIDATION
# ==========================================

if row_offset != total_rows:
    raise ValueError(
        f"Row count mismatch: processed {row_offset:,}, "
        f"expected {total_rows:,}."
    )

print("\nDATA PREPARATION COMPLETED")
print("=" * 60)

for split_name in SPLIT_NAMES.values():

    total = partition_counts[split_name]
    fraud = partition_fraud_counts[split_name]
    legitimate = total - fraud

    print(f"\n{split_name.upper()}")
    print("Total transactions:", f"{total:,}")
    print("Legitimate:", f"{legitimate:,}")
    print("Fraud:", f"{fraud:,}")
    print(
        "Fraud percentage:",
        f"{(fraud / total * 100):.4f}%"
        if total else "N/A"
    )

print("\nProcessed files saved to:", OUTPUT_DIR)
