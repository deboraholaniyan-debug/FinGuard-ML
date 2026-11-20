
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.build_transaction_features import (
    build_transaction_features
)


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "class_weight_experiments"
    / "unweighted.txt"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "models"
    / "preprocessing"
    / "standard_scaler.joblib"
)

NUMERICAL_COLUMNS = [
    "amount",
    "origin_balance_before",
    "destination_balance_before",
    "step",
    "hour",
    "log_amount"
]

CATEGORIES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER"
]

FEATURE_COLUMNS = NUMERICAL_COLUMNS + [
    f"transaction_type_{category}"
    for category in CATEGORIES
]


# ==========================================
# 2. LOAD MODEL AND SCALER
# ==========================================

model = lgb.Booster(model_file=str(MODEL_PATH))
scaler = joblib.load(SCALER_PATH)

app = FastAPI(
    title="FinGuard ML API",
    description="Financial transaction risk-scoring service",
    version="1.0.0"
)


# ==========================================
# 3. REQUEST SCHEMA
# ==========================================

class TransactionRequest(BaseModel):
    amount: float = Field(ge=0)
    transaction_type: str
    origin_balance_before: float = Field(ge=0)
    destination_balance_before: float = Field(ge=0)
    step: int = Field(ge=1)


# ==========================================
# 4. HEALTH CHECK
# ==========================================

@app.get("/")
def health_check():
    return {
        "service": "FinGuard ML",
        "status": "running",
        "model": "LightGBM unweighted baseline"
    }


# ==========================================
# 5. TRANSACTION RISK SCORING
# ==========================================

@app.post("/predict")
def predict_transaction(transaction: TransactionRequest):

    if transaction.transaction_type not in CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Transaction type must be one of {CATEGORIES}"
        )

    try:
        raw_transaction = pd.DataFrame([{
            "amount": transaction.amount,
            "type": transaction.transaction_type,
            "oldbalanceOrg": transaction.origin_balance_before,
            "oldbalanceDest": transaction.destination_balance_before,
            "step": transaction.step
        }])

        # Feature engineering.
        features = build_transaction_features(raw_transaction)

        # Numerical transformation.
        numerical = features[NUMERICAL_COLUMNS]

        scaled_values = scaler.transform(numerical)

        numerical_df = pd.DataFrame(
            scaled_values,
            columns=NUMERICAL_COLUMNS
        )

        # One-hot encoding.
        categorical_df = pd.DataFrame(
            [{
                f"transaction_type_{category}":
                    int(transaction.transaction_type == category)
                for category in CATEGORIES
            }]
        )

        # Combine features in training order.
        model_input = pd.concat(
            [numerical_df, categorical_df],
            axis=1
        )

        model_input = model_input[FEATURE_COLUMNS].astype(np.float32)

        # Predict.
        risk_score = float(model.predict(model_input)[0])

        return {
            "transaction_type": transaction.transaction_type,
            "risk_score": round(risk_score, 6),
            "score_interpretation": (
                "Model-generated fraud score; "
                "not a calibrated probability or an AML determination."
            )
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
