
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EXPECTED_FEATURES = [
    "amount",
    "transaction_type",
    "origin_balance_before",
    "destination_balance_before",
    "step",
    "hour",
    "log_amount"
]

EXPECTED_COLUMNS = EXPECTED_FEATURES + ["target"]

for split_name in ["train", "validation", "test"]:

    split_dir = PROCESSED_DIR / split_name
    parquet_files = sorted(split_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in {split_dir}"
        )

    dataset = pd.read_parquet(parquet_files)

    print(f"\n{split_name.upper()}")
    print("-" * 45)
    print("Shape:", dataset.shape)
    print("Columns:", dataset.columns.tolist())
    print("Fraud cases:", int(dataset["target"].sum()))
    print("Missing values:", int(dataset.isnull().sum().sum()))

    assert dataset.columns.tolist() == EXPECTED_COLUMNS
    assert dataset["target"].isin([0, 1]).all()
    assert dataset.isnull().sum().sum() == 0

    print("Validation: PASSED")
