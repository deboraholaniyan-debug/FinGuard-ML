
from pathlib import Path

import lightgbm as lgb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lightgbm_baseline.txt"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. LOAD MODEL
# ==========================================

model = lgb.Booster(
    model_file=str(MODEL_PATH)
)

print("MODEL INFORMATION")
print("=" * 50)

print("Number of trees:", model.num_trees())
print("Number of features:", model.num_feature())
print("Best iteration:", model.best_iteration)


# ==========================================
# 2. FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "feature": model.feature_name(),
    "split_importance": model.feature_importance(
        importance_type="split"
    ),
    "gain_importance": model.feature_importance(
        importance_type="gain"
    )
})

importance = importance.sort_values(
    "gain_importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print("=" * 70)
print(importance.to_string(index=False))

importance.to_csv(
    OUTPUT_DIR / "baseline_feature_importance.csv",
    index=False
)


# ==========================================
# 3. TREE STRUCTURE
# ==========================================

model_dump = model.dump_model()

tree = model_dump["tree_info"][0]["tree_structure"]

print("\nFIRST TREE STRUCTURE")
print("=" * 50)

print("Tree leaves:", model_dump["tree_info"][0]["num_leaves"])
print("Tree depth:", model_dump["tree_info"][0]["tree_structure"].get("max_depth", "Not stored"))


# ==========================================
# 4. EXTRACT LEAF VALUES
# ==========================================

leaf_information = []


def extract_leaves(node):

    if "leaf_index" in node:

        leaf_information.append({
            "leaf_index": node["leaf_index"],
            "leaf_value": node["leaf_value"],
            "leaf_count": node["leaf_count"]
        })

        return

    extract_leaves(node["left_child"])
    extract_leaves(node["right_child"])


extract_leaves(tree)

leaf_df = pd.DataFrame(leaf_information)

print("\nLEAF DISTRIBUTION")
print("=" * 70)
print(
    leaf_df.sort_values(
        "leaf_value",
        ascending=False
    ).to_string(index=False)
)

leaf_df.to_csv(
    OUTPUT_DIR / "baseline_leaf_information.csv",
    index=False
)


print("\nModel inspection completed.")
