"""
Step 4b — Better Model: Random Forest
----------------------------------------
Random Forest builds many small decision trees and combines their votes.
It naturally handles "AND/OR" style rules (like our sale-period,
salary-week interactions) better than Logistic Regression's straight lines.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
)

df = pd.read_csv("/home/claude/chargeback_project/transactions.csv")

TARGET = "is_chargeback"
DROP_COLS = ["is_chargeback", "chargeback_reason", "chargeback_probability"]
X = df.drop(columns=DROP_COLS)
y = df[TARGET]

categorical_cols = ["payment_method", "delivery_confirmation_strength", "is_high_value_category"]
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Same split as before (same random_state=42) so we're comparing fairly
# on the exact same train/test rows as the Logistic Regression baseline.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Random Forest does NOT need feature scaling (trees don't care about
# scale the way straight-line models do) -- one less step.
rf_model = RandomForestClassifier(
    n_estimators=200,       # 200 trees, votes get averaged
    max_depth=8,            # keep trees shallow-ish to avoid overfitting
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# ---------- Metrics ----------
cm = confusion_matrix(y_test, y_pred_rf)
tn, fp, fn, tp = cm.ravel()
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)
auc = roc_auc_score(y_test, y_proba_rf)

print("=" * 70)
print("RANDOM FOREST — Results")
print("=" * 70)
print(f"\nConfusion Matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"Precision : {precision:.2%}")
print(f"Recall    : {recall:.2%}")
print(f"F1 Score  : {f1:.2%}")
print(f"ROC-AUC   : {auc:.3f}")

print("\n" + "=" * 70)
print("COMPARISON: Logistic Regression (baseline) vs Random Forest")
print("=" * 70)
print(f"{'Metric':<12}{'LogReg':>10}{'RandForest':>12}")
print(f"{'Precision':<12}{'20.39%':>10}{precision:>12.2%}")
print(f"{'Recall':<12}{'62.71%':>10}{recall:>12.2%}")
print(f"{'F1 Score':<12}{'30.77%':>10}{f1:>12.2%}")
print(f"{'ROC-AUC':<12}{'0.687':>10}{auc:>12.3f}")

# ---------- Feature importance (which signals matter most) ----------
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
print("\n[Top 10 most important features]")
print(importances.head(10).round(4))

# Save for next steps
import joblib
joblib.dump(rf_model, "/home/claude/chargeback_project/rf_model.pkl")
X_test.to_csv("/home/claude/chargeback_project/X_test_rf.csv", index=False)
np.save("/home/claude/chargeback_project/y_pred_rf.npy", y_pred_rf)
np.save("/home/claude/chargeback_project/y_proba_rf.npy", y_proba_rf)
print("\nSaved Random Forest model and predictions.")
