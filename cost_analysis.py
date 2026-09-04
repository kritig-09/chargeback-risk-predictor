"""
Step 5 — False-Positive Cost Analysis
-----------------------------------------
Turn the confusion matrix into a business ₹ story:
what does this model actually cost/save Razorpay's merchants?

Assumptions are clearly labeled as ASSUMPTIONS -- in the real world these
numbers would come from Razorpay's actual historical data.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

y_test = pd.read_csv("/home/claude/chargeback_project/y_test.csv")["is_chargeback"].values
y_proba_rf = np.load("/home/claude/chargeback_project/y_proba_rf.npy")

THRESHOLD = 0.5
y_pred_final = (y_proba_rf >= THRESHOLD).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_final).ravel()

print("=" * 70)
print("FALSE-POSITIVE COST ANALYSIS (Random Forest, threshold=0.5)")
print("=" * 70)
print(f"\nOn {len(y_test)} test transactions:")
print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

# ---------- ASSUMPTIONS (clearly labeled -- would come from real data) ----------
# Average order value in our synthetic data, for realistic ₹ estimates
avg_order_value = 3000

COST_PER_MISSED_CHARGEBACK = avg_order_value       # money actually lost to fraud/dispute
COST_PER_FALSE_POSITIVE = 150                        # manual review cost + friction/trust cost (assumption)
BENEFIT_PER_CAUGHT_CHARGEBACK = avg_order_value      # money saved by catching it early

print(f"\n[Assumptions -- would be replaced with real merchant data in production]")
print(f"  Avg order value                    : Rs.{avg_order_value}")
print(f"  Cost per false positive (friction)  : Rs.{COST_PER_FALSE_POSITIVE}")
print(f"  Cost per missed chargeback (FN)      : Rs.{COST_PER_MISSED_CHARGEBACK}")

# ---------- Cost breakdown ----------
cost_of_false_positives = fp * COST_PER_FALSE_POSITIVE
cost_of_missed_chargebacks = fn * COST_PER_MISSED_CHARGEBACK
value_of_caught_chargebacks = tp * BENEFIT_PER_CAUGHT_CHARGEBACK

net_value = value_of_caught_chargebacks - cost_of_false_positives - cost_of_missed_chargebacks

print(f"\n[Cost/Value Breakdown]")
print(f"  Value from catching {tp} chargebacks       : +Rs.{value_of_caught_chargebacks:,}")
print(f"  Cost of {fp} false positives (friction)    : -Rs.{cost_of_false_positives:,}")
print(f"  {'-'*50}")
print(f"  VALUE ADDED by using this model (catches minus friction): Rs.{value_of_caught_chargebacks - cost_of_false_positives:,}")
print(f"\n  (Note: the {fn} missed chargebacks -Rs.{cost_of_missed_chargebacks:,} are a")
print(f"   cost that exists with or without the model -- they are NOT")
print(f"   caused by the model, so they don't belong in the model's 'cost'.")
print(f"   They just show how much room for improvement remains.)")

# ---------- Compare to "do nothing" baseline ----------
# If we used NO model at all, every chargeback (tp+fn) would be a pure loss
no_model_loss = (tp + fn) * COST_PER_MISSED_CHARGEBACK
print(f"\n[Baseline: no fraud-detection model at all]")
print(f"  Total loss from all {tp+fn} chargebacks (no detection): -Rs.{no_model_loss:,}")

improvement = no_model_loss - (cost_of_false_positives + cost_of_missed_chargebacks - value_of_caught_chargebacks)
savings = value_of_caught_chargebacks - cost_of_false_positives
print(f"\n[Bottom line]")
print(f"  Money saved by catching chargebacks         : Rs.{value_of_caught_chargebacks:,}")
print(f"  Money spent on false-positive friction       : Rs.{cost_of_false_positives:,}")
print(f"  Net savings vs having no model at all         : Rs.{savings:,}")
print(f"""
[Interpretation]
Even with imperfect precision, the model provides positive net value:
it prevents Rs.{value_of_caught_chargebacks:,} in chargeback losses at a
friction cost of Rs.{cost_of_false_positives:,} -- a worthwhile trade IF the
Rs.{COST_PER_FALSE_POSITIVE} assumed friction cost per false positive holds.
This assumption should be validated with real merchant/support-cost data.
""")
