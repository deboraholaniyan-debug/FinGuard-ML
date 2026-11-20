
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformed"
)

EXPECTED_FEATURES = [
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

EXPECTED_FRAUD_COUNTS = {
    "train": 5749,
    "validation": 1232,
    "test": 1232
}

EXPECTED_ROWS = {
    "train": 4_453_834,
    "validation": 954_393,
    "test": 954_393
}


for split in ["train", "validation", "test"]:

    file_path = DATA_DIR / f"{split}.parquet"

    parquet_file = pq.ParquetFile(file_path)

    print(f"\n{split.upper()}")
    print("-" * 45)

    rows = 0
    fraud_count = 0
    invalid_values = 0
    schema = parquet_file.schema_arrow

    assert schema.names == EXPECTED_FEATURES + ["target"]

    for batch in parquet_file.iter_batches(
        batch_size=50_000
    ):

        batch_df = batch.to_pandas()

        rows += len(batch_df)
        fraud_count += int(batch_df["target"].sum())

        feature_values = batch_df[EXPECTED_FEATURES].to_numpy(
            dtype=np.float32
        )

        invalid_values += int(
            (~np.isfinite(feature_values)).sum()
        )

        assert batch_df["target"].isin([0, 1]).all()

        assert all(
            str(batch_df[column].dtype) == "float32"
            for column in EXPECTED_FEATURES
        )

    assert rows == EXPECTED_ROWS[split]
    assert fraud_count == EXPECTED_FRAUD_COUNTS[split]
    assert invalid_values == 0

    print("Rows:", f"{rows:,}")
    print("Fraud cases:", f"{fraud_count:,}")
    print("Invalid feature values:", invalid_values)
    print("Validation: PASSED")


print("\nALL TRANSFORMED DATASETS VALIDATED SUCCESSFULLY.")
