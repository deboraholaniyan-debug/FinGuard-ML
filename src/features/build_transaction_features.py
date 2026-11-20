
import numpy as np
import pandas as pd


def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build pre-transaction features from PaySim data.

    Assumption: risk scoring occurs before transaction execution.
    """

    features = pd.DataFrame(index=df.index)

    # Transaction characteristics
    features["amount"] = df["amount"]
    features["transaction_type"] = df["type"]

    # Pre-transaction account balances
    features["origin_balance_before"] = df["oldbalanceOrg"]
    features["destination_balance_before"] = df["oldbalanceDest"]

    # Temporal characteristics
    features["step"] = df["step"]
    features["hour"] = (df["step"] - 1) % 24

    # Log transformation
    features["log_amount"] = np.log1p(
        df["amount"].clip(lower=0)
    )

    return features
