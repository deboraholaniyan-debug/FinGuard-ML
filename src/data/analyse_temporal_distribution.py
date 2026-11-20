
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/paysim_transactions.csv")

records = []

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=["step", "isFraud"],
    chunksize=100_000
):
    grouped = chunk.groupby("step").agg(
        transactions=("isFraud", "size"),
        fraud_cases=("isFraud", "sum")
    )

    records.append(grouped)

temporal = (
    pd.concat(records)
    .groupby(level=0)
    .sum()
    .reset_index()
)

temporal["fraud_rate"] = (
    temporal["fraud_cases"] /
    temporal["transactions"] * 100
)

print("\nTEMPORAL DISTRIBUTION")
print(temporal.describe().round(4))

print("\nFIRST 10 TIME STEPS")
print(temporal.head(10).to_string(index=False))

print("\nLAST 10 TIME STEPS")
print(temporal.tail(10).to_string(index=False))

print("\nFRAUD COUNTS BY TIME QUARTILE")
print(
    temporal.assign(
        period=pd.qcut(
            temporal["step"],
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"]
        )
    )
    .groupby("period", observed=True)[
        ["transactions", "fraud_cases"]
    ]
    .sum()
)
