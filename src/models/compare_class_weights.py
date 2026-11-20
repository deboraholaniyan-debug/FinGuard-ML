
from pathlib import Path
import gc

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "transformed"
OUTPUT_DIR = PROJECT_ROOT / "models" / "class_weight_experiments"
REPORT_DIR = PROJECT_ROOT / "reports" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.parquet"
VALIDATION_PATH = DATA_DIR / "validation.parquet"

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

CLASS_WEIGHTS = {
    "unweighted": 1.0,
    "weight_25": 25.0,
    "weight_50": 50.0,
    "original_weight": 773.71
}

TOP_K = 500


# ==========================================
# 2. LOAD DATA
# ==========================================

print("Loading training data...")

train_df = pd.read_parquet(TRAIN_PATH)

print("Loading validation data...")

validation_df = pd.read_parquet(VALIDATION_PATH)

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMN]

X_validation = validation_df[FEATURE_COLUMNS]
y_validation = validation_df[TARGET_COLUMN].to_numpy()

del train_df, validation_df
gc.collect()

print("\nTraining shape:", X_train.shape)
print("Validation shape:", X_validation.shape)

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

print("Training fraud cases:", f"{positive_count:,}")


# ==========================================
# 3. DEFINE EVALUATION METRIC
# ==========================================

def pr_auc_metric(predictions, dataset):

    labels = dataset.get_label()

    score = average_precision_score(
        labels,
        predictions
    )

    return "pr_auc", score, True


# ==========================================
# 4. LIGHTGBM PARAMETERS
# ==========================================

BASE_PARAMS = {
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
    "max_bin": 127,
    "verbosity": -1,
    "num_threads": 4,
    "seed": 42
}


# ==========================================
# 5. CREATE LIGHTGBM DATASETS
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
# 6. RUN CONTROLLED EXPERIMENTS
# ==========================================

experiment_results = []

for experiment_name, weight in CLASS_WEIGHTS.items():

    print("\n" + "=" * 65)
    print("EXPERIMENT:", experiment_name)
    print("Positive-class weight:", weight)
    print("=" * 65)

    params = BASE_PARAMS.copy()
    params["scale_pos_weight"] = weight

    model = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=500,
        valid_sets=[validation_set],
        valid_names=["validation"],
        feval=pr_auc_metric,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                first_metric_only=True,
                verbose=False
            ),
            lgb.log_evaluation(period=50)
        ]
    )

    # Generate validation scores.
    predictions = model.predict(
        X_validation,
        num_iteration=model.best_iteration
    )

    # Ranking performance.
    pr_auc = average_precision_score(
        y_validation,
        predictions
    )

    # Top-K investigation performance.
    top_indices = np.argsort(-predictions)[:TOP_K]

    detected_fraud = int(y_validation[top_indices].sum())

    precision_at_k = detected_fraud / TOP_K

    recall_at_k = detected_fraud / positive_count if False else (
        detected_fraud / int(y_validation.sum())
    )

    # Prediction distribution.
    unique_scores = len(np.unique(predictions))

    # Save model.
    model_path = OUTPUT_DIR / f"{experiment_name}.txt"

    model.save_model(
        str(model_path),
        num_iteration=model.best_iteration
    )

    result = {
        "experiment": experiment_name,
        "scale_pos_weight": weight,
        "best_iteration": model.best_iteration,
        "validation_pr_auc": pr_auc,
        "precision_at_500": precision_at_k,
        "recall_at_500": recall_at_k,
        "unique_prediction_scores": unique_scores,
        "minimum_score": float(predictions.min()),
        "maximum_score": float(predictions.max())
    }

    experiment_results.append(result)

    print("\nExperiment results:")
    for key, value in result.items():
        print(f"{key}: {value}")

    del model, predictions
    gc.collect()


# ==========================================
# 7. COMPARE EXPERIMENTS
# ==========================================

results_df = pd.DataFrame(experiment_results)

results_df = results_df.sort_values(
    "validation_pr_auc",
    ascending=False
)

print("\n\nCLASS-WEIGHT EXPERIMENT COMPARISON")
print("=" * 100)

print(results_df.to_string(index=False))

results_path = REPORT_DIR / "class_weight_comparison.csv"

results_df.to_csv(
    results_path,
    index=False
)

print("\nResults saved to:", results_path)
print("Experiment models saved to:", OUTPUT_DIR)
