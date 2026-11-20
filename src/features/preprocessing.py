
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERICAL_FEATURES = [
    "amount",
    "origin_balance_before",
    "destination_balance_before",
    "step",
    "hour",
    "log_amount"
]

CATEGORICAL_FEATURES = [
    "transaction_type"
]


def create_preprocessor():
    """Create a reusable feature preprocessing transformer."""

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            )
        ],
        remainder="drop"
    )
