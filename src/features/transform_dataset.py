
from pathlib import Path

import json
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRANSFORMED_DIR = PROCESSED_DIR / "transformed"

MODEL_DIR = PROJECT_ROOT / "models" / "preprocessing"

BATCH_SIZE = 50_000

NUMERICAL_FEATURES = [
    "amount",
    "origin_balance_before",
    "destination_balance_before",
    "step",
    "hour",
    "log_amount"
]

CATEGORICAL_FEATURE = "transaction_type"

TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER"
]

TARGET_COLUMN = "target"

SPLITS = ["train", "validation", "test"]


# ==========================================
# 2. CREATE OUTPUT DIRECTORIES
# ==========================================

TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

for split in SPLITS:
    output_file = TRANSFORMED_DIR / f"{split}.parquet"

    if output_file.exists():
        raise FileExistsError(
            f"Output already exists: {output_file}. "
            "Move or remove existing transformed files first."
        )


# ==========================================
# 3. FIT SCALER ON TRAINING DATA ONLY
# ==========================================

train_files = sorted(
    (PROCESSED_DIR / "train").glob("*.parquet")
)

if not train_files:
    raise FileNotFoundError("Training Parquet files were not found.")

scaler = StandardScaler()

print("Fitting numerical scaler on training data...")

training_rows = 0

for file_path in train_files:

    parquet_file = pq.ParquetFile(file_path)

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=NUMERICAL_FEATURES
    ):

        batch_df = batch.to_pandas()

        scaler.partial_fit(batch_df[NUMERICAL_FEATURES])

        training_rows += len(batch_df)

        print(
            f"Scaler rows processed: {training_rows:,}",
            end="\r"
        )

print(f"\nScaler fitted on {training_rows:,} training rows.")

joblib.dump(
    scaler,
    MODEL_DIR / "standard_scaler.joblib"
)

print("Scaler saved successfully.")


# ==========================================
# 4. DEFINE TRANSFORMED FEATURE NAMES
# ==========================================

ONE_HOT_FEATURES = [
    f"transaction_type_{category}"
    for category in TRANSACTION_TYPES
]

FEATURE_NAMES = NUMERICAL_FEATURES + ONE_HOT_FEATURES

metadata = {
    "numerical_features": NUMERICAL_FEATURES,
    "categorical_feature": CATEGORICAL_FEATURE,
    "categories": TRANSACTION_TYPES,
    "transformed_features": FEATURE_NAMES,
    "target": TARGET_COLUMN,
    "dtype": "float32",
    "scaler_fitted_on": "training_partition_only"
}

with open(
    MODEL_DIR / "feature_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4)


# ==========================================
# 5. TRANSFORM EACH PARTITION IN BATCHES
# ==========================================

for split in SPLITS:

    print(f"\nTransforming {split.upper()} partition...")

    input_files = sorted(
        (PROCESSED_DIR / split).glob("*.parquet")
    )

    output_file = TRANSFORMED_DIR / f"{split}.parquet"

    writer = None
    rows_written = 0

    for file_path in input_files:

        parquet_file = pq.ParquetFile(file_path)

        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE
        ):

            batch_df = batch.to_pandas()

            # Scale numerical features using the training-fitted scaler.
            numerical_array = scaler.transform(
                batch_df[NUMERICAL_FEATURES]
            ).astype(np.float32)

            # Encode transaction type using a fixed category order.
            categorical_array = np.column_stack([
                (batch_df[CATEGORICAL_FEATURE].to_numpy() == category)
                for category in TRANSACTION_TYPES
            ]).astype(np.float32)

            # Combine numerical and categorical features.
            feature_array = np.column_stack([
                numerical_array,
                categorical_array
            ]).astype(np.float32)

            # Keep the target separate from the model features.
            target_array = batch_df[TARGET_COLUMN].to_numpy(
                dtype=np.int8
            )

            transformed_df = pd.DataFrame(
                feature_array,
                columns=FEATURE_NAMES
            )

            transformed_df[TARGET_COLUMN] = target_array

            table = pa.Table.from_pandas(
                transformed_df,
                preserve_index=False
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    output_file,
                    table.schema,
                    compression="snappy"
                )

            writer.write_table(table)

            rows_written += len(transformed_df)

    if writer is not None:
        writer.close()
    else:
        raise ValueError(
            f"No records were transformed for {split}."
        )

    print(
        f"{split.upper()} completed: "
        f"{rows_written:,} rows saved."
    )


# ==========================================
# 6. COMPLETION SUMMARY
# ==========================================

print("\nTRANSFORMATION COMPLETED")
print("=" * 55)

for split in SPLITS:

    output_file = TRANSFORMED_DIR / f"{split}.parquet"

    parquet_file = pq.ParquetFile(output_file)

    print(
        f"{split.upper()}: "
        f"{parquet_file.metadata.num_rows:,} rows"
    )

print("\nTransformed datasets:", TRANSFORMED_DIR)
print("Preprocessing artefacts:", MODEL_DIR)
