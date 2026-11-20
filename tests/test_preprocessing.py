
import pandas as pd

from src.features.build_transaction_features import (
    build_transaction_features
)
from src.features.preprocessing import create_preprocessor


DATA_PATH = "data/raw/paysim_transactions.csv"

# Load sample
df = pd.read_csv(DATA_PATH, nrows=1000)

# Generate features
features = build_transaction_features(df)

# Create and fit preprocessor on sample
preprocessor = create_preprocessor()

X_transformed = preprocessor.fit_transform(features)

print("\nPREPROCESSING TEST")
print("=" * 55)

print("Original feature shape:", features.shape)
print("Transformed shape:", X_transformed.shape)

print("\nTransformed feature names:")
print(preprocessor.get_feature_names_out())

print("\nMissing values:", pd.isna(X_transformed).sum())

assert X_transformed.shape[0] == len(df)
assert X_transformed.shape[1] == 11
assert pd.isna(X_transformed).sum() == 0

print("\nPreprocessing test passed.")
