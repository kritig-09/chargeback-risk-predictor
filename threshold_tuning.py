"""
Step 4c — Threshold Tuning
-----------------------------
Instead of the default 0.5 cutoff, we try several thresholds and see
how precision/recall trade off. This lets us pick a business-sensible
point rather than an arbitrary default.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

y_test = pd.read_csv("/home/claude/chargeback_project/y_test.csv")["is_chargeback"].values
y_proba_rf = np.load("/home/claude/chargeback_project/y_proba_rf.npy")

print("=" * 70)
print("THRESHOLD TUNING — Random Forest")
print("=" * 70)
print(f"\n{'Threshold':<12}{'Precision':>12}{'Recall':>12}{'F1':>10}{'Flagged':>10}")

thresholds = [0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]
results = []

for t in thresholds:
    y_pred_t = (y_proba_rf >= t).astype(int)
    p = precision_score(y_test, y_pred_t, zero_division=0)
    r = recall_score(y_test, y_pred_t, zero_division=0)
    f1 = f1_score(y_test, y_pred_t, zero_division=0)
    flagged = y_pred_t.sum()
    results.append((t, p, r, f1, flagged))
    print(f"{t:<12}{p:>12.2%}{r:>12.2%}{f1:>10.2%}{flagged:>10d}")

# Pick the threshold with best F1 as a reasonable "balanced" choice
best = max(results, key=lambda x: x[3])
print(f"\nBest F1 balance at threshold={best[0]}: "
      f"Precision={best[1]:.2%}, Recall={best[2]:.2%}, F1={best[3]:.2%}")

print("""
[How to read this]
- LOWER threshold (e.g. 0.2) -> flags more transactions -> catches more
  chargebacks (higher recall) but more false alarms (lower precision).
- HIGHER threshold (e.g. 0.7) -> only flags very confident cases ->
  fewer false alarms (higher precision) but misses more chargebacks.
- There's no single "correct" threshold -- it depends on which mistake
  costs the business more: annoying a genuine customer, or missing fraud.
""")
