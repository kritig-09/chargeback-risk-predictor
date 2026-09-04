"""
Step 7 — Reason-Classification Layer (Stage 2)
--------------------------------------------------
Stage 1 (already built): Random Forest predicts risk_probability (0-1)
Stage 2 (this file): IF risk is flagged, explain WHY and suggest an action.

This is intentionally rule-based, not a second ML model -- it's fast,
fully explainable (important for an audit trail), and doesn't suffer
from the small-sample-size problem a 4-class classifier would hit here.
"""

import pandas as pd
import numpy as np
import joblib

model = joblib.load("/home/claude/chargeback_project/rf_model.pkl")
FEATURE_COLUMNS = list(pd.read_csv("/home/claude/chargeback_project/X_test_rf.csv").columns)
THRESHOLD = 0.5


def classify_reason(txn: dict) -> str:
    """
    Same logic as data-generation's assign_reason(), but now used as a
    LIVE decision layer on new transactions, not just for labeling
    synthetic data. Order matters -- most specific/certain reason first.
    """
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
    "Service-Issue": "Escalate to customer support -- check why the earlier refund request was not honored.",
    "Fraud-Suspected": "Freeze/hold transaction, compile evidence (device, address, ID mismatch), route to fraud review.",
    "Genuine-Dispute": "Manual review -- collect delivery proof and order details, respond to the dispute directly."
}


def assess_transaction(txn: dict) -> dict:
    """
    Full pipeline for one transaction: Stage 1 (risk score) + Stage 2
    (reason + action), only computing reason/action if actually flagged.
    """
    row = pd.DataFrame([txn])[FEATURE_COLUMNS]
    risk_score = model.predict_proba(row)[0, 1]
    flagged = risk_score >= THRESHOLD

    result = {
        "risk_score": round(float(risk_score), 4),
        "flagged": bool(flagged),
        "reason": None,
        "suggested_action": None,
    }

    if flagged:
        reason = classify_reason(txn)
        result["reason"] = reason
        result["suggested_action"] = SUGGESTED_ACTION[reason]

    return result


if __name__ == "__main__":
    # Quick demo using a few of our edge-case scenarios from Step 6,
    # but forcing them to actually cross the threshold by combining
    # multiple risk signals -- so we get to see Stage 2 in action.
    demo_cases = [
        {
            "name": "Duplicate payment from a loyal customer",
            "txn": {
                "amount": 2500, "hour_of_day": 14, "account_age_days": 900,
                "avg_order_value": 2400, "past_chargeback_count": 0,
                "billing_shipping_mismatch": 0, "name_mismatch_score": 0.05,
                "device_sharing_flag": 0, "transactions_last_24h": 2,
                "failed_attempts_before_success": 0, "checkout_speed_sec": 40,
                "secondary_verification_flag": 0, "refund_requested_before_chargeback": 0,
                "duplicate_transaction_flag": 1, "is_sale_period": 0,
                "is_salary_week": 0, "location_change_recent": 0,
                "payment_method_netbanking": 0, "payment_method_upi": 1,
                "delivery_confirmation_strength_not_applicable": 0,
                "delivery_confirmation_strength_otp_confirmed": 1,
                "delivery_confirmation_strength_signature": 0,
                "is_high_value_category_none": 1, "is_high_value_category_travel": 0,
            }
        },
        {
            "name": "New account, stolen-card-like pattern",
            "txn": {
                "amount": 22000, "hour_of_day": 3, "account_age_days": 2,
                "avg_order_value": 500, "past_chargeback_count": 2,
                "billing_shipping_mismatch": 1, "name_mismatch_score": 0.7,
                "device_sharing_flag": 1, "transactions_last_24h": 6,
                "failed_attempts_before_success": 3, "checkout_speed_sec": 4,
                "secondary_verification_flag": 0, "refund_requested_before_chargeback": 0,
                "duplicate_transaction_flag": 0, "is_sale_period": 0,
                "is_salary_week": 0, "location_change_recent": 0,
                "payment_method_netbanking": 0, "payment_method_upi": 0,
                "delivery_confirmation_strength_not_applicable": 1,
                "delivery_confirmation_strength_otp_confirmed": 0,
                "delivery_confirmation_strength_signature": 0,
                "is_high_value_category_none": 1, "is_high_value_category_travel": 0,
            }
        },
        {
            "name": "Refund denied, escalated to bank",
            "txn": {
                "amount": 3200, "hour_of_day": 16, "account_age_days": 300,
                "avg_order_value": 2800, "past_chargeback_count": 1,
                "billing_shipping_mismatch": 0, "name_mismatch_score": 0.1,
                "device_sharing_flag": 0, "transactions_last_24h": 1,
                "failed_attempts_before_success": 0, "checkout_speed_sec": 50,
                "secondary_verification_flag": 0, "refund_requested_before_chargeback": 1,
                "duplicate_transaction_flag": 0, "is_sale_period": 0,
                "is_salary_week": 0, "location_change_recent": 0,
                "payment_method_netbanking": 0, "payment_method_upi": 1,
                "delivery_confirmation_strength_not_applicable": 0,
                "delivery_confirmation_strength_otp_confirmed": 0,
                "delivery_confirmation_strength_signature": 1,
                "is_high_value_category_none": 1, "is_high_value_category_travel": 0,
            }
        },
    ]

    print("=" * 90)
    print("STEP 7: RISK + REASON + SUGGESTED ACTION (full pipeline demo)")
    print("=" * 90)

    for case in demo_cases:
        result = assess_transaction(case["txn"])
        print(f"\n[{case['name']}]")
        print(f"  Risk score       : {result['risk_score']:.2%}")
        print(f"  Flagged          : {result['flagged']}")
        if result["flagged"]:
            print(f"  Reason           : {result['reason']}")
            print(f"  Suggested action : {result['suggested_action']}")
