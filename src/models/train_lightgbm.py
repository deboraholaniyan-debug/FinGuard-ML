
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformed"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.parquet"
VALIDATION_PATH = DATA_DIR / "validation.parquet"

MODEL_PATH = MODEL_DIR / "lightgbm_baseline.txt"

FEATURE_COLUMNS = [
    "amount",
    "origin_balance_before",
    "destination_balance_before",
    "step",
    "hour",
    "log_amount",
    "transaction_type_CASH_IN",
    "transaction_type_CASH_OUT",
    "transaction_type_DEBIT",
    "transaction_type_PAYMENT",
    "transaction_type_TRANSFER"
]

TARGET_COLUMN = "target"


# ==========================================
# 2. LOAD TRAINING AND VALIDATION DATA
# ==========================================

print("Loading training data...")

train_df = pd.read_parquet(TRAIN_PATH)

print("Loading validation data...")

validation_df = pd.read_parquet(VALIDATION_PATH)

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMN]

X_validation = validation_df[FEATURE_COLUMNS]
y_validation = validation_df[TARGET_COLUMN]

print("\nDataset shapes")
print("Training:", X_train.shape)
print("Validation:", X_validation.shape)


# ==========================================
# 3. CALCULATE CLASS IMBALANCE WEIGHT
# ==========================================

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

scale_pos_weight = negative_count / positive_count

print("\nClass distribution")
print("Legitimate:", f"{negative_count:,}")
print("Fraud:", f"{positive_count:,}")
print("Scale positive weight:", round(scale_pos_weight, 2))


# ==========================================
# 4. CREATE LIGHTGBM DATASETS
# ==========================================

train_set = lgb.Dataset(
    X_train,
    label=y_train,
    feature_name=FEATURE_COLUMNS,
    free_raw_data=True
)

validation_set = lgb.Dataset(
    X_validation,
    label=y_validation,
    reference=train_set,
    feature_name=FEATURE_COLUMNS,
    free_raw_data=True
)


# ==========================================
# 5. DEFINE PR-AUC EVALUATION
# ==========================================

def pr_auc_metric(predictions, dataset):

    labels = dataset.get_label()

    score = average_precision_score(
        labels,
        predictions
    )

    return "pr_auc", score, True


# ==========================================
# 6. DEFINE MODEL PARAMETERS
# ==========================================

params = {
    "objective": "binary",
    "metric": "None",

    "boosting_type": "gbdt",
    "learning_rate": 0.05,

    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 100,

    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,

    "scale_pos_weight": scale_pos_weight,

    "max_bin": 127,

    "verbosity": -1,
    "num_threads": 4,
    "seed": 42
}


# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("\nStarting LightGBM training...")

model = lgb.train(
    params=params,
    train_set=train_set,
    num_boost_round=1000,
    valid_sets=[validation_set],
    valid_names=["validation"],
    feval=pr_auc_metric,
    callbacks=[
        lgb.early_stopping(
            stopping_rounds=50,
            first_metric_only=True
        ),
        lgb.log_evaluation(period=50)
    ]
)


# ==========================================
# 8. SAVE MODEL
# ==========================================

model.save_model(
    str(MODEL_PATH),
    num_iteration=model.best_iteration
)

print("\nModel training completed.")
print("Best iteration:", model.best_iteration)
print("Best validation PR-AUC:", model.best_score)
print("Model saved to:", MODEL_PATH)


# ==========================================
# 9. SAVE TRAINING METADATA
# ==========================================

metadata = {
    "model": "LightGBM",
    "objective": "binary",
    "random_seed": 42,
    "scale_pos_weight": scale_pos_weight,
    "best_iteration": model.best_iteration,
    "best_validation_score": model.best_score
}

joblib.dump(
    metadata,
    MODEL_DIR / "lightgbm_baseline_metadata.joblib"
)

print("Training metadata saved.")
