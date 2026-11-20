
import pandas as pd

DATA_PATH = "data/raw/paysim_transactions.csv"

# Read only the columns required for split analysis.
df = pd.read_csv(
    DATA_PATH,
    usecols=["step", "isFraud"]
)

# Determine boundaries using chronological time steps.
steps = sorted(df["step"].unique())

train_end = int(len(steps) * 0.70)
validation_end = int(len(steps) * 0.85)

train_steps = steps[:train_end]
validation_steps = steps[train_end:validation_end]
test_steps = steps[validation_end:]

splits = {
    "Train": train_steps,
    "Validation": validation_steps,
    "Test": test_steps
}

print("\nCHRONOLOGICAL SPLIT ANALYSIS")
print("-" * 65)

for name, selected_steps in splits.items():

    subset = df[df["step"].isin(selected_steps)]

    total = len(subset)
    fraud = int(subset["isFraud"].sum())
    rate = fraud / total * 100

    print(f"\n{name}")
    print("Step range:", min(selected_steps), "-", max(selected_steps))
    print("Transactions:", f"{total:,}")
    print("Fraud cases:", f"{fraud:,}")
    print("Fraud rate:", f"{rate:.4f}%")

print("\nTotal transactions:", f"{len(df):,}")
