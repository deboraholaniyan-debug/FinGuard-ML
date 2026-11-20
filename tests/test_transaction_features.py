
import pandas as pd

from src.features.build_transaction_features import (
    build_transaction_features
)

DATA_PATH = "data/raw/paysim_transactions.csv"

# Load a small sample
df = pd.read_csv(DATA_PATH, nrows=1000)

# Build features
features = build_transaction_features(df)

print("\nFEATURE ENGINEERING TEST")
print("=" * 55)

print("Original shape:", df.shape)
print("Feature shape:", features.shape)

print("\nGenerated features:")
print(features.columns.tolist())

print("\nSample features:")
print(features.head())

print("\nMissing values:")
print(features.isnull().sum())

print("\nFeature data types:")
print(features.dtypes)

assert len(features) == len(df)
assert features.shape[1] == 7
assert "transaction_type" in features.columns
assert features.isnull().sum().sum() == 0

print("\nFeature engineering test passed.")
