"""
EDA — Chargeback Risk Predictor
--------------------------------
Part B: Validate that our context-aware business rules actually worked
         in the generated data (sale period, salary week, duplicates, travel)
Part A: General exploratory analysis (distributions, correlations)
"""

import pandas as pd
import numpy as np

pd.set_option("display.width", 120)

df = pd.read_csv("/home/claude/chargeback_project/transactions.csv")

print("=" * 70)
print("PART B: EDGE CASE VALIDATION")
print("=" * 70)

# ---------- Test 1: Sale period should REDUCE risk of fast checkout ----------
fast = df["checkout_speed_sec"] < 10
print("\n[Test 1] Fast checkout (<10s) chargeback rate:")
print(f"  During SALE period     : {df[fast & (df['is_sale_period']==1)]['is_chargeback'].mean():.2%}")
print(f"  During NON-sale period : {df[fast & (df['is_sale_period']==0)]['is_chargeback'].mean():.2%}")
print("  --> Non-sale should be noticeably higher")

# ---------- Test 2: Salary week should REDUCE risk of high amount ----------
high_amt = df["amount"] > 20000
print("\n[Test 2] High amount (>20k) chargeback rate:")
print(f"  During SALARY week     : {df[high_amt & (df['is_salary_week']==1)]['is_chargeback'].mean():.2%}")
print(f"  During NON-salary week : {df[high_amt & (df['is_salary_week']==0)]['is_chargeback'].mean():.2%}")
print("  --> Non-salary-week should be noticeably higher")

# ---------- Test 3: Duplicate transaction should be a STRONG driver ----------
print("\n[Test 3] Duplicate transaction flag:")
print(f"  Chargeback rate WHEN duplicate=1 : {df[df['duplicate_transaction_flag']==1]['is_chargeback'].mean():.2%}")
print(f"  Chargeback rate WHEN duplicate=0 : {df[df['duplicate_transaction_flag']==0]['is_chargeback'].mean():.2%}")
dup_cb = df[(df["duplicate_transaction_flag"] == 1) & (df["is_chargeback"] == 1)]
print(f"  Reason breakdown for duplicate chargebacks:\n{dup_cb['chargeback_reason'].value_counts()}")

# ---------- Test 4: Recent travel should REDUCE risk of address mismatch ----------
mismatch = df["billing_shipping_mismatch"] == 1
print("\n[Test 4] Billing/shipping mismatch chargeback rate:")
print(f"  WITH recent travel    : {df[mismatch & (df['location_change_recent']==1)]['is_chargeback'].mean():.2%}")
print(f"  WITHOUT recent travel : {df[mismatch & (df['location_change_recent']==0)]['is_chargeback'].mean():.2%}")
print("  --> Without travel should be higher")

print("\n" + "=" * 70)
print("PART A: GENERAL EDA")
print("=" * 70)

print("\n[Overall] Chargeback rate:", f"{df['is_chargeback'].mean():.2%}")
print("\n[Missing values check]")
print(df.isnull().sum().sum(), "missing values total")

print("\n[Numeric feature summary]")
numeric_cols = ["amount", "account_age_days", "avg_order_value", "checkout_speed_sec",
                "transactions_last_24h", "name_mismatch_score"]
print(df[numeric_cols].describe().round(2))

print("\n[Correlation with is_chargeback] (numeric features only)")
numeric_for_corr = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_for_corr = [c for c in numeric_for_corr if c not in ["chargeback_probability"]]
corr = df[numeric_for_corr].corr()["is_chargeback"].sort_values(ascending=False)
print(corr)

print("\n[Categorical breakdowns]")
for col in ["payment_method", "delivery_confirmation_strength", "is_high_value_category"]:
    print(f"\n{col} vs chargeback rate:")
    print(df.groupby(col)["is_chargeback"].mean().round(3))
