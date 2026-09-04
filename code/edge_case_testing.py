"""
Step 6 — Edge Case Testing
-----------------------------
We hand-craft realistic examples for each context scenario we designed
(sale period, salary week, duplicate transaction, travel) and check
whether the FINAL model (Random Forest, threshold=0.5) reacts the way
a sensible risk system should.

This isn't about overall accuracy anymore -- it's about demonstrating
that specific, explainable behavior holds up on cases a judge might ask about.
"""

import pandas as pd
import numpy as np
import joblib

model = joblib.load("/home/claude/chargeback_project/rf_model.pkl")
FEATURE_COLUMNS = list(pd.read_csv("/home/claude/chargeback_project/X_test_rf.csv").columns)
THRESHOLD = 0.5


def make_transaction(**overrides):
    """Start from a 'typical safe' transaction, then override specific fields
    to build each test scenario. Any column not mentioned defaults to a
    common/neutral value."""
    base = {
        "amount": 1500,
        "hour_of_day": 14,
        "account_age_days": 400,
        "avg_order_value": 1400,
        "past_chargeback_count": 0,
        "billing_shipping_mismatch": 0,
        "name_mismatch_score": 0.05,
        "device_sharing_flag": 0,
        "transactions_last_24h": 1,
        "failed_attempts_before_success": 0,
        "checkout_speed_sec": 45,
        "secondary_verification_flag": 0,
        "refund_requested_before_chargeback": 0,
        "duplicate_transaction_flag": 0,
        "is_sale_period": 0,
        "is_salary_week": 0,
        "location_change_recent": 0,
        "payment_method_netbanking": 0,
        "payment_method_upi": 1,
        "delivery_confirmation_strength_not_applicable": 0,
        "delivery_confirmation_strength_otp_confirmed": 1,
        "delivery_confirmation_strength_signature": 0,
        "is_high_value_category_none": 1,
        "is_high_value_category_travel": 0,
    }
    base.update(overrides)
    return base


scenarios = []

# ---------- Scenario 1: Fast checkout during a sale (should NOT be flagged) ----------
scenarios.append((
    "Fast checkout during Diwali sale",
    make_transaction(checkout_speed_sec=5, is_sale_period=1),
    "Should be LOW risk -- fast checkout is normal during flash sales"
))
scenarios.append((
    "Fast checkout on a normal day (control)",
    make_transaction(checkout_speed_sec=5, is_sale_period=0),
    "Should be HIGHER risk -- same speed, but no sale to explain it"
))

# ---------- Scenario 2: High amount during salary week (should NOT be flagged) ----------
scenarios.append((
    "High amount during salary week",
    make_transaction(amount=25000, avg_order_value=1400, is_salary_week=1,
                      is_high_value_category_none=1),
    "Should be LOWER risk -- big purchase right after payday is common"
))
scenarios.append((
    "High amount, random day (control)",
    make_transaction(amount=25000, avg_order_value=1400, is_salary_week=0,
                      is_high_value_category_none=1),
    "Should be HIGHER risk -- same amount, no salary-week explanation"
))

# ---------- Scenario 3: Duplicate transaction (SHOULD be flagged, but as benign) ----------
scenarios.append((
    "Accidental duplicate payment",
    make_transaction(duplicate_transaction_flag=1, account_age_days=800,
                      past_chargeback_count=0),
    "Should be FLAGGED (real chargeback risk) but from a loyal, low-fraud-signal customer"
))

# ---------- Scenario 4: Address mismatch due to travel (should NOT be flagged) ----------
scenarios.append((
    "Address mismatch while traveling",
    make_transaction(billing_shipping_mismatch=1, location_change_recent=1),
    "Should be LOWER risk -- mismatch explained by recent travel"
))
scenarios.append((
    "Address mismatch, no travel (control)",
    make_transaction(billing_shipping_mismatch=1, location_change_recent=0),
    "Should be HIGHER risk -- same mismatch, no travel explanation"
))

# ---------- Scenario 5 (bonus): New customer, medical emergency, high amount ----------
scenarios.append((
    "New customer, medical emergency purchase",
    make_transaction(amount=18000, account_age_days=3, avg_order_value=18000,
                      is_high_value_category_none=0),
    "A naive model might flag 'new account + high amount' -- context should soften this"
))


print("=" * 90)
print("STEP 6: EDGE CASE TESTING (Random Forest, threshold=0.5)")
print("=" * 90)

for name, txn, expectation in scenarios:
    row = pd.DataFrame([txn])[FEATURE_COLUMNS]  # enforce exact column order
    proba = model.predict_proba(row)[0, 1]
    flagged = "FLAGGED" if proba >= THRESHOLD else "not flagged"
    print(f"\n[{name}]")
    print(f"  Expectation : {expectation}")
    print(f"  Risk score  : {proba:.2%}  -->  {flagged}")
