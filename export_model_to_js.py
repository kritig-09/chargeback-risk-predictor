"""
Step 8 — Export Trained Model to JS (for the dashboard's live custom-transaction predictor)
------------------------------------------------------------------------------------------
dashboard.html is standalone (no server), so the trained Random Forest can't be called
via joblib/predict_proba at view time. This script exports every tree's structure
(feature splits, thresholds, children, leaf probabilities) to a compact JSON blob that
gets pasted into dashboard.html as the `rfTrees` constant, and a matching JS function
walks the trees exactly the way sklearn does internally.

Run this any time rf_model.pkl is retrained, so the embedded JS model stays in sync
with the reported metrics / curated scenarios.
"""

import json
import joblib
import pandas as pd

MODEL_PATH = "/home/claude/chargeback_project/rf_model.pkl"
FEATURE_COLUMNS_SOURCE = "/home/claude/chargeback_project/X_test_rf.csv"
OUT_PATH = "/home/claude/chargeback_project/rf_trees_export.json"


def export_trees(model):
    trees = []
    for est in model.estimators_:
        t = est.tree_
        feature = t.feature.tolist()
        threshold = [round(x, 4) for x in t.threshold.tolist()]
        left = t.children_left.tolist()
        right = t.children_right.tolist()

        # Leaf probability of class 1 -- non-leaf nodes get -1 (unused, kept for
        # index alignment with feature/threshold/left/right arrays).
        prob1 = []
        for i in range(t.node_count):
            if left[i] == -1:  # leaf node
                counts = t.value[i][0]
                total = counts[0] + counts[1]
                p = counts[1] / total if total > 0 else 0.0
                prob1.append(round(float(p), 4))
            else:
                prob1.append(-1)

        trees.append({"f": feature, "t": threshold, "l": left, "r": right, "p": prob1})
    return trees


def main():
    model = joblib.load(MODEL_PATH)
    feature_columns = list(pd.read_csv(FEATURE_COLUMNS_SOURCE).columns)

    trees = export_trees(model)

    export = {"feature_columns": feature_columns, "trees": trees}
    with open(OUT_PATH, "w") as fp:
        json.dump(export, fp, separators=(",", ":"))

    print(f"Exported {len(trees)} trees -> {OUT_PATH}")
    print(f"Feature columns ({len(feature_columns)}): {feature_columns}")
    print(
        "\nPaste export['trees'] into dashboard.html as `const rfTrees = ...;`\n"
        "and export['feature_columns'] as `const FEATURE_COLUMNS = ...;` "
        "(order must match exactly -- it's the same column order the model was trained on)."
    )


if __name__ == "__main__":
    main()
