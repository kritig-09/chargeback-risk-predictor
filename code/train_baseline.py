"""
Step 3 — Train-Test Split + Baseline Model
--------------------------------------------
Goal: predict is_chargeback (0/1) from transaction features.
We use a simple, explainable model first (Logistic Regression) before
trying anything fancier -- this gives us a floor to compare against.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("/home/claude/chargeback_project/transactions.csv")

# ---------- 1. Choose which columns are inputs (X) vs target (y) ----------
# We drop chargeback_reason and chargeback_probability here:
#   - chargeback_reason is only known AFTER a chargeback happens -> can't use
#     it to predict whether one will happen (that would be "cheating")
#   - chargeback_probability is the internal score we used to GENERATE the
#     data -- a real system would never have this, so we must not train on it
TARGET = "is_chargeback"
DROP_COLS = ["is_chargeback", "chargeback_reason", "chargeback_probability"]

X = df.drop(columns=DROP_COLS)
y = df[TARGET]

# ---------- 2. Convert text categories into numbers (one-hot encoding) ----------
# e.g. payment_method: "card"/"upi"/"netbanking" becomes 3 separate
# 0/1 columns: payment_method_card, payment_method_upi, payment_method_netbanking
categorical_cols = ["payment_method", "delivery_confirmation_strength", "is_high_value_category"]
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"Total features after encoding: {X.shape[1]}")
print(f"Feature names: {list(X.columns)}\n")

# ---------- 3. Train-test split (80/20), held-out test set ----------
# stratify=y makes sure both train and test have the same ~12% chargeback
# rate (otherwise a random split could accidentally put most chargebacks
# in one side)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)}  (chargeback rate: {y_train.mean():.2%})")
print(f"Test size:  {len(X_test)}  (chargeback rate: {y_test.mean():.2%})\n")

# ---------- 4. Scale numeric features ----------
# Logistic Regression is sensitive to features being on very different
# scales (e.g. amount goes up to 100,000 while name_mismatch_score is 0-1).
# StandardScaler makes every feature have mean=0, std=1.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn scale from train only
X_test_scaled = scaler.transform(X_test)          # apply same scale to test

# ---------- 5. Train baseline Logistic Regression ----------
# class_weight='balanced' tells the model to pay extra attention to the
# minority class (chargebacks are only ~12% of data) without us having to
# manually resample anything -- one line, easy win.
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train_scaled, y_train)

# ---------- 6. Predict on the held-out test set ----------
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of chargeback

print("Baseline model trained. Sample predictions on test set:")
results_preview = pd.DataFrame({
    "actual": y_test.values[:10],
    "predicted": y_pred[:10],
    "risk_probability": np.round(y_proba[:10], 3)
})
print(results_preview)

# Save everything needed for the next step (evaluation)
import joblib
joblib.dump(model, "/home/claude/chargeback_project/baseline_model.pkl")
joblib.dump(scaler, "/home/claude/chargeback_project/scaler.pkl")
X_test.to_csv("/home/claude/chargeback_project/X_test.csv", index=False)
y_test.to_csv("/home/claude/chargeback_project/y_test.csv", index=False)
np.save("/home/claude/chargeback_project/y_pred.npy", y_pred)
np.save("/home/claude/chargeback_project/y_proba.npy", y_proba)

print("\nSaved model, scaler, and test predictions for evaluation step.")
