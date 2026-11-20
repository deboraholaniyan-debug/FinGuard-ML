
import pandas as pd

DATA_PATH = "data/raw/paysim_transactions.csv"

df = pd.read_csv(
    DATA_PATH,
    usecols=["isFraud"]
)

class_counts = df["isFraud"].value_counts().sort_index()

total = len(df)

print("\nCLASS IMBALANCE ANALYSIS")
print("=" * 50)

for label, count in class_counts.items():
    percentage = count / total * 100

    class_name = "Legitimate" if label == 0 else "Fraud"

    print(
        f"{class_name}: {count:,} "
        f"({percentage:.4f}%)"
    )

fraud_count = class_counts.get(1, 0)
legitimate_count = class_counts.get(0, 0)

print("\nImbalance ratio:")
print(f"1 : {legitimate_count / fraud_count:.2f}")

print("\nMajority-class accuracy baseline:")
print(f"{legitimate_count / total * 100:.4f}%")
