
import pandas as pd

DATA_PATH = "data/raw/paysim_transactions.csv"

temporal = pd.read_csv(
    DATA_PATH,
    usecols=["step", "isFraud"]
).groupby("step").agg(
    transactions=("isFraud", "size"),
    fraud_cases=("isFraud", "sum")
)

temporal["fraud_rate"] = (
    temporal["fraud_cases"] / temporal["transactions"] * 100
)

print("\nTRANSACTION VOLUME BY PERIOD")
print("-" * 60)

for threshold in [10, 50, 100, 500, 1000]:
    sparse = temporal[temporal["transactions"] < threshold]

    print(
        f"Below {threshold:>4} transactions per step: "
        f"{len(sparse):>4} steps | "
        f"{sparse['transactions'].sum():>10,} transactions | "
        f"{sparse['fraud_cases'].sum():>6,} fraud cases"
    )

print("\nFIRST AND LAST STEPS WITH AT LEAST 100 TRANSACTIONS")

active = temporal[temporal["transactions"] >= 100]

print("First active step:", active.index.min())
print("Last active step:", active.index.max())
print("Active steps:", len(active))

print("\nSPARSE PERIOD SUMMARY")
print(temporal[temporal["transactions"] < 100].tail(20))
