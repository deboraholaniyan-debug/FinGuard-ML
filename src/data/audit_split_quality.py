
import pandas as pd

DATA_PATH = "data/raw/paysim_transactions.csv"

df = pd.read_csv(
    DATA_PATH,
    usecols=["step", "isFraud"]
)

# Reproduce the chronological split
steps = sorted(df["step"].unique())

train_end = int(len(steps) * 0.70)
validation_end = int(len(steps) * 0.85)

splits = {
    "Train": steps[:train_end],
    "Validation": steps[train_end:validation_end],
    "Test": steps[validation_end:]
}

# Calculate transaction volume and fraud rate per time step
step_stats = df.groupby("step").agg(
    transactions=("isFraud", "size"),
    fraud_cases=("isFraud", "sum")
)

step_stats["fraud_rate"] = (
    step_stats["fraud_cases"]
    / step_stats["transactions"]
    * 100
)

print("\nSPLIT QUALITY AUDIT")
print("=" * 75)

for name, selected_steps in splits.items():

    subset = step_stats.loc[selected_steps]

    total_transactions = subset["transactions"].sum()
    total_fraud = subset["fraud_cases"].sum()

    sparse_steps = subset[subset["transactions"] < 100]
    sparse_transactions = sparse_steps["transactions"].sum()
    sparse_fraud = sparse_steps["fraud_cases"].sum()

    print(f"\n{name}")
    print("-" * 50)
    print(f"Time steps: {len(subset):,}")
    print(f"Transactions: {total_transactions:,}")
    print(f"Fraud cases: {total_fraud:,}")
    print(f"Overall fraud rate: {total_fraud / total_transactions * 100:.4f}%")
    print(f"Steps with fewer than 100 transactions: {len(sparse_steps):,}")
    print(f"Transactions in sparse steps: {sparse_transactions:,}")
    print(f"Fraud cases in sparse steps: {sparse_fraud:,}")

    if total_fraud:
        print(
            "Share of fraud cases from sparse steps: "
            f"{sparse_fraud / total_fraud * 100:.2f}%"
        )
