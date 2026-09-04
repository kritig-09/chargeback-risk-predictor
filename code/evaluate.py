"""
Step 4 — Model Evaluation
---------------------------
Now that we have predictions on the held-out test set, we measure how
good the model actually is -- not just "does it work" but "how well,
and what does it get wrong."
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

y_test = pd.read_csv("/home/claude/chargeback_project/y_test.csv")["is_chargeback"].values
y_pred = np.load("/home/claude/chargeback_project/y_pred.npy")
y_proba = np.load("/home/claude/chargeback_project/y_proba.npy")

print("=" * 70)
print("STEP 4: MODEL EVALUATION (Baseline — Logistic Regression)")
print("=" * 70)

# ---------- Confusion Matrix ----------
# This is the single most important table -- it shows exactly what
# the model got right and wrong, broken into 4 buckets.
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n[Confusion Matrix]")
print(f"                    Predicted: No CB   Predicted: CB")
print(f"  Actual: No CB          {tn:5d}              {fp:5d}   <- False Positives (FP)")
print(f"  Actual: CB             {fn:5d}              {tp:5d}   <- True Positives (TP)")
print(f"                          ^False Negatives (FN)")

print(f"\n  True Negatives  (TN): {tn}  -- correctly said 'safe'")
print(f"  False Positives (FP): {fp}  -- WRONGLY flagged a genuine customer")
print(f"  False Negatives (FN): {fn}  -- MISSED an actual chargeback")
print(f"  True Positives  (TP): {tp}  -- correctly caught a chargeback")

# ---------- Core metrics ----------
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n[Core Metrics]")
print(f"  Precision : {precision:.2%}  -- of everyone we flagged, how many were ACTUALLY risky")
print(f"  Recall    : {recall:.2%}  -- of all ACTUAL chargebacks, how many did we catch")
print(f"  F1 Score  : {f1:.2%}  -- balance between precision and recall")
print(f"  ROC-AUC   : {auc:.3f}  -- overall ability to rank risky vs safe (0.5=random, 1.0=perfect)")

print("\n[Full classification report]")
print(classification_report(y_test, y_pred, target_names=["No Chargeback", "Chargeback"]))

# Save for use in the false-positive cost analysis step
np.save("/home/claude/chargeback_project/confusion_matrix.npy", cm)
print("Saved confusion matrix for cost analysis step.")
