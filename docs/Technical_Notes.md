# Technical Notes

Detailed experiment log behind the design decisions summarised in the main README. This is the "how we got here" record — every trade-off tested, with numbers.

---

## 1. Data generation: noise vs. signal trade-off

The chargeback rate was initially compressed from a natural ~16% (what the business rules produce on their own) down to ~12%, using a uniform scaling factor (`risk * 0.72`) plus added Gaussian noise (std 0.08), to look closer to a "realistic-looking" rate for a demo.

This was later diagnosed as the root cause of weak model performance: the added noise was comparable in size to several individual rule effects (e.g. +0.10 to +0.20 per rule), which blurred the learnable signal in the labels.

**Fix tested:** Removed the scaling factor and reduced noise std to 0.04.

| | Before (scaled + high noise) | After (unscaled + low noise) |
|---|---|---|
| Chargeback rate | ~16% (post-fix, from ~12%) | 16.18% |
| Logistic Regression precision | 20.4% | 28.65% |
| Logistic Regression ROC-AUC | 0.687 | 0.729 |
| Random Forest precision | 31.0% | 40.0% |
| Random Forest ROC-AUC | 0.688 | 0.749 |

**Second fix attempt (tested, rejected):** Instead of scaling, individual rule *trigger thresholds* were tightened (e.g. `amount > 20000` → `amount > 23000`) to reduce how often each rule fires, while keeping each rule's strength unchanged. This produced a 14.58% chargeback rate but *worse* model performance (RF precision 34.75%, ROC-AUC 0.706) than the unscaled version — some borderline cases that were previously learnable signal became invisible to every rule entirely, which is a worse form of information loss than scaling.

**Conclusion:** The original, unscaled ~16% rate was the best-performing configuration found. This is documented as a deliberate, tested choice — not an oversight.

---

## 2. Duplicate-transaction signal: rare-event under-weighting

The duplicate-transaction flag was designed as the single strongest chargeback driver in the rule set (+0.35 to the risk score, versus +0.10 to +0.20 for most other rules). But in the trained Random Forest, it initially ranked only **#10** in feature importance — far below its intended weight.

**Diagnosis:** Duplicates occur in only ~3% of transactions. With too few positive examples, the model could not reliably learn the pattern regardless of how strong the underlying rule was.

**Test performed:** Temporarily increased duplicate frequency from 3% to 7% and retrained.

| | 3% frequency (original) | 7% frequency (test) |
|---|---|---|
| Feature importance rank | #10 | **#1** |
| Duplicate-scenario risk score | 24.7% | 45.2% |
| Overall RF precision | 40.0% | ~34% (decreased) |
| Overall RF recall | 38.3% | ~30% (decreased) |

**Decision:** Rolled back to 3% frequency. The diagnosis (rare-event under-weighting, confirmed by the frequency test) is valid and documented, but the fix cost more in overall precision/recall than it gained in one feature's importance — not a favourable trade for the model actually shipped.

---

## 3. Threshold tuning — full table

Thresholds from 0.2 to 0.7 were swept on the final Random Forest model:

| Threshold | Precision | Recall | F1 | False positives |
|---|---|---|---|---|
| 0.20 | 16.6% | 100.0% | 28.5% | 974 |
| 0.30 | 20.2% | 95.1% | 33.3% | 764 |
| 0.35 | 23.5% | 87.7% | 37.1% | 604 |
| 0.40 | 27.4% | 72.8% | 39.9% | 430 |
| **0.50** | **40.0%** | **38.3%** | **39.1%** | **93** |
| 0.60 | 56.5% | 8.0% | 14.1% | 23 |
| 0.70 | 0.0% | 0.0% | 0.0% | 0 |

No threshold improves both precision and recall simultaneously — the curve is a strict trade-off. Threshold 0.50 was selected as the point where false positives drop sharply (93, down from 430 at 0.40) without recall collapsing (still 38.3%, versus a cliff to 8% at 0.60).

---

## 4. Why chargebacks over fraud or return abuse

Three loss types were considered:

- **Fraud** — highest public-data availability (e.g. anonymised credit-card fraud datasets), but those datasets use PCA-transformed, unlabelled columns that would prevent building the business-context features this project relies on.
- **Return abuse** — requires inventing subjective rules for what counts as "abusive" behaviour, with a real risk of circular logic (the rules define the data, and the data then confirms the rules).
- **Chargebacks** — risk signals are comparatively objective (amount, address mismatch, account age, delivery evidence), and the problem naturally extends into an explainable, actionable system rather than a plain classifier.

Chargebacks were chosen on this basis.
