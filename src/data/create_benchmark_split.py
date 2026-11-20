
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

DATA_PATH = "data/raw/paysim_transactions.csv"
OUTPUT_DIR = "data/interim"

SEED = 42

# Load the target column only
df = pd.read_csv(
    DATA_PATH,
    usecols=["isFraud"]
)

labels = df["isFraud"].to_numpy()
indices = np.arange(len(labels))

# First split: 70% training, 30% temporary
train_idx, temporary_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)

# Second split: divide temporary data into 15% validation and 15% test
validation_idx, test_idx = train_test_split(
    temporary_idx,
    test_size=0.50,
    random_state=SEED,
    stratify=labels[temporary_idx]
)

splits = {
    "train": train_idx,
    "validation": validation_idx,
    "test": test_idx
}

print("\nSTRATIFIED BENCHMARK SPLIT")
print("=" * 60)

for name, selected_idx in splits.items():

    split_labels = labels[selected_idx]

    total = len(selected_idx)
    fraud = int(split_labels.sum())
    legitimate = total - fraud

    print(f"\n{name.upper()}")
    print("-" * 40)
    print(f"Transactions: {total:,}")
    print(f"Legitimate: {legitimate:,}")
    print(f"Fraud: {fraud:,}")
    print(f"Fraud rate: {fraud / total * 100:.4f}%")

    np.save(
        f"{OUTPUT_DIR}/{name}_indices.npy",
        selected_idx
    )

print("\nSplit indices saved successfully.")
print("Random seed:", SEED)
