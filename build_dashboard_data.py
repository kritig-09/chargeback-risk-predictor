"""
Generate all data the dashboard needs, computed from the REAL trained
model -- no fabricated numbers. Produces dashboard_data.json.
"""
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

model = joblib.load("/home/claude/chargeback_project/rf_model.pkl")
FEATURE_COLUMNS = list(pd.read_csv("/home/claude/chargeback_project/X_test_rf.csv").columns)
THRESHOLD = 0.5


def classify_reason(txn):
    if txn.get("duplicate_transaction_flag", 0) == 1:
        return "Accidental-Duplicate"
    if (txn.get("refund_requested_before_chargeback", 0) == 1
            and txn.get("name_mismatch_score", 0) < 0.3):
        return "Service-Issue"
    if (txn.get("name_mismatch_score", 0) > 0.4
            or txn.get("device_sharing_flag", 0) == 1
            or txn.get("billing_shipping_mismatch", 0) == 1):
        return "Fraud-Suspected"
    return "Genuine-Dispute"


SUGGESTED_ACTION = {
    "Accidental-Duplicate": "Auto-refund the duplicate charge immediately. No investigation needed.",
    "Service-Issue": "Escalate to support -- check why the earlier refund request wasn't honored.",
    "Fraud-Suspected": "Freeze transaction, compile evidence (device/address/ID mismatch), route to fraud review.",
    "Genuine-Dispute": "Manual review -- collect delivery proof and respond to the dispute directly.",
}


def make_transaction(**overrides):
    base = {
        "amount": 1500, "hour_of_day": 14, "account_age_days": 400,
        "avg_order_value": 1400, "past_chargeback_count": 0,
        "billing_shipping_mismatch": 0, "name_mismatch_score": 0.05,
        "device_sharing_flag": 0, "transactions_last_24h": 1,
        "failed_attempts_before_success": 0, "checkout_speed_sec": 45,
        "secondary_verification_flag": 0, "refund_requested_before_chargeback": 0,
        "duplicate_transaction_flag": 0, "is_sale_period": 0,
        "is_salary_week": 0, "location_change_recent": 0,
        "payment_method_netbanking": 0, "payment_method_upi": 1,
        "delivery_confirmation_strength_not_applicable": 0,
        "delivery_confirmation_strength_otp_confirmed": 1,
        "delivery_confirmation_strength_signature": 0,
        "is_high_value_category_none": 1, "is_high_value_category_travel": 0,
    }
    base.update(overrides)
    return base


def predict(txn):
    row = pd.DataFrame([txn])[FEATURE_COLUMNS]
    proba = model.predict_proba(row)[0, 1]
    flagged = proba >= THRESHOLD
    result = {"risk_score": round(float(proba) * 100, 1), "flagged": bool(flagged)}
    if flagged:
        reason = classify_reason(txn)
        result["reason"] = reason
        result["action"] = SUGGESTED_ACTION[reason]
    else:
        result["reason"] = None
        result["action"] = None
    return result


def key_factors(txn):
    """Human-readable list of what's notable about this transaction."""
    factors = []
    if txn["amount"] > 20000:
        factors.append(f"High order value (Rs.{txn['amount']:,})")
    if txn["account_age_days"] < 15:
        factors.append(f"New account ({txn['account_age_days']} days old)")
    if txn["billing_shipping_mismatch"] == 1:
        factors.append("Billing/shipping address mismatch")
    if txn["name_mismatch_score"] > 0.4:
        factors.append("Cardholder name mismatch")
    if txn["device_sharing_flag"] == 1:
        factors.append("Device linked to multiple accounts")
    if txn["duplicate_transaction_flag"] == 1:
        factors.append("Duplicate of a recent payment")
    if txn["checkout_speed_sec"] < 10:
        factors.append(f"Very fast checkout ({txn['checkout_speed_sec']}s)")
    if txn["refund_requested_before_chargeback"] == 1:
        factors.append("Refund was requested and not resolved")
    if txn["is_sale_period"] == 1:
        factors.append("Occurred during a sale period")
    if txn["is_salary_week"] == 1:
        factors.append("Occurred during salary week")
    if txn["location_change_recent"] == 1:
        factors.append("Recent travel on file")
    if not factors:
        factors.append("No notable risk signals")
    return factors


scenarios = [
    {
        "id": "duplicate",
        "title": "Duplicate payment, loyal customer",
        "txn": make_transaction(amount=2500, account_age_days=900, avg_order_value=2400,
                                 duplicate_transaction_flag=1, transactions_last_24h=2),
    },
    {
        "id": "fraud_pattern",
        "title": "New account, stolen-card-like pattern",
        "txn": make_transaction(amount=22000, hour_of_day=3, account_age_days=2,
                                 avg_order_value=500, past_chargeback_count=2,
                                 billing_shipping_mismatch=1, name_mismatch_score=0.7,
                                 device_sharing_flag=1, transactions_last_24h=6,
                                 failed_attempts_before_success=3, checkout_speed_sec=4,
                                 payment_method_upi=0,
                                 delivery_confirmation_strength_otp_confirmed=0,
                                 delivery_confirmation_strength_not_applicable=1),
    },
    {
        "id": "refund_denied",
        "title": "Refund denied, escalated to bank",
        "txn": make_transaction(amount=3200, account_age_days=300, avg_order_value=2800,
                                 past_chargeback_count=1, refund_requested_before_chargeback=1,
                                 delivery_confirmation_strength_otp_confirmed=0,
                                 delivery_confirmation_strength_signature=1),
    },
    {
        "id": "sale_fast_checkout",
        "title": "Fast checkout during Diwali sale",
        "txn": make_transaction(checkout_speed_sec=5, is_sale_period=1),
    },
    {
        "id": "fast_checkout_normal",
        "title": "Fast checkout, ordinary day",
        "txn": make_transaction(checkout_speed_sec=5, is_sale_period=0),
    },
    {
        "id": "salary_high_amount",
        "title": "Big purchase, salary week",
        "txn": make_transaction(amount=25000, avg_order_value=1400, is_salary_week=1),
    },
    {
        "id": "medical_emergency",
        "title": "New customer, medical emergency",
        "txn": make_transaction(amount=18000, account_age_days=3, avg_order_value=18000,
                                 is_high_value_category_none=0),
    },
    {
        "id": "travel_mismatch",
        "title": "Address mismatch while traveling",
        "txn": make_transaction(billing_shipping_mismatch=1, location_change_recent=1),
    },
    {
        "id": "safe_regular",
        "title": "Regular, low-risk purchase",
        "txn": make_transaction(),
    },
]

for s in scenarios:
    pred = predict(s["txn"])
    s["risk_score"] = pred["risk_score"]
    s["flagged"] = pred["flagged"]
    s["reason"] = pred["reason"]
    s["action"] = pred["action"]
    s["factors"] = key_factors(s["txn"])
    s["amount"] = s["txn"]["amount"]

# ---------- Assemble everything ----------
y_test = pd.read_csv("/home/claude/chargeback_project/y_test.csv")["is_chargeback"].values
y_proba = np.load("/home/claude/chargeback_project/y_proba_rf.npy")
y_pred = (y_proba >= THRESHOLD).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
feat_imp = [{"name": n, "value": round(float(v) * 100, 1)} for n, v in importances.head(8).items()]

df = pd.read_csv("/home/claude/chargeback_project/transactions.csv")
fast = df["checkout_speed_sec"] < 10
high_amt = df["amount"] > 20000
mismatch = df["billing_shipping_mismatch"] == 1

context_tests = [
    {"name": "Fast checkout", "context_label": "During a sale",
     "without_context": round(df[fast & (df['is_sale_period'] == 0)]['is_chargeback'].mean() * 100, 1),
     "with_context": round(df[fast & (df['is_sale_period'] == 1)]['is_chargeback'].mean() * 100, 1)},
    {"name": "High amount (>Rs.20k)", "context_label": "During salary week",
     "without_context": round(df[high_amt & (df['is_salary_week'] == 0)]['is_chargeback'].mean() * 100, 1),
     "with_context": round(df[high_amt & (df['is_salary_week'] == 1)]['is_chargeback'].mean() * 100, 1)},
    {"name": "Address mismatch", "context_label": "Recent travel on file",
     "without_context": round(df[mismatch & (df['location_change_recent'] == 0)]['is_chargeback'].mean() * 100, 1),
     "with_context": round(df[mismatch & (df['location_change_recent'] == 1)]['is_chargeback'].mean() * 100, 1)},
]

avg_order_value = 3000
cost_per_fp = 150
value_caught = int(tp) * avg_order_value
cost_fp = int(fp) * cost_per_fp
net_value = value_caught - cost_fp

dashboard_data = {
    "metrics": {
        "precision": round(precision_score(y_test, y_pred) * 100, 1),
        "recall": round(recall_score(y_test, y_pred) * 100, 1),
        "f1": round(f1_score(y_test, y_pred) * 100, 1),
        "auc": round(roc_auc_score(y_test, y_proba), 3),
        "chargeback_rate": round(y_test.mean() * 100, 1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "test_size": len(y_test),
        "threshold": THRESHOLD,
    },
    "feature_importance": feat_imp,
    "context_tests": context_tests,
    "cost": {
        "avg_order_value": avg_order_value,
        "cost_per_fp": cost_per_fp,
        "value_caught": value_caught,
        "cost_fp": cost_fp,
        "net_value": net_value,
        "missed_loss": int(fn) * avg_order_value,
    },
    "scenarios": scenarios,
}

with open("/home/claude/chargeback_project/dashboard_data.json", "w") as f:
    json.dump(dashboard_data, f, indent=2)

print("Saved dashboard_data.json")
print(json.dumps(dashboard_data["metrics"], indent=2))
