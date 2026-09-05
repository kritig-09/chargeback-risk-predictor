# Chargeback Risk Predictor

An AI-based risk detector that predicts whether a transaction is likely to result in a chargeback, explains why, and recommends an appropriate action.

Built for the **AI Risk Manager** track — one class of loss (chargebacks), a working detector with measured precision and recall on a held-out test set, and honest reporting of false-positive cost.

---

## Table of Contents

- [Problem](#problem)
- [Approach](#approach)
- [Features Used](#features-used)
- [Results](#results)
- [Cost Impact](#cost-impact)
- [Known Limitations](#known-limitations)
- [Dashboard](#dashboard)
- [Visual Analysis (Power BI)](#visual-analysis-power-bi)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Full Report](#full-report)

---

## Problem

Merchants accepting online payments lose money to fraud, returns, and chargebacks. A chargeback occurs when a customer disputes a transaction with their bank, forcing a refund — often with an additional penalty to the merchant. Left undetected, a high chargeback rate can also get a merchant's payment account flagged or suspended.

Three loss types were considered for this problem: **fraud**, **return abuse**, and **chargebacks**. Chargebacks were chosen because the risk signals involved (transaction amount, address mismatch, account age, delivery evidence) are relatively objective, and the problem naturally extends into an explainable, actionable system rather than a plain yes/no classifier.

## Approach

```
Synthetic data generation (business-rule based)
        │
        ▼
Exploratory analysis + edge-case validation
        │
        ▼
Model training — Logistic Regression → Random Forest
        │
        ▼
Threshold tuning
        │
        ▼
Cost-impact analysis (₹ value)
        │
        ▼
Edge-case scenario testing
        │
        ▼
Reason-classification layer (why + suggested action)
        │
        ▼
Interactive dashboard
```

**Why synthetic data:** Real chargeback-labelled datasets are not publicly available. Public fraud datasets that do exist use anonymised, PCA-transformed columns, which rules out building the business-context features (sale-period effects, salary-week effects, delivery evidence strength) this project relies on. A synthetic dataset was generated instead, from explicit, documented business rules — with random noise added so it isn't perfectly separable.

## Features Used

20 features across six categories:

| Category              | Features                                                                                                                     |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Core transaction      | amount, hour of day, payment method                                                                                          |
| Customer history      | account age, average order value, past chargeback count                                                                      |
| Mismatch signals      | billing/shipping mismatch, name mismatch score, device-sharing flag                                                          |
| Velocity              | transactions in last 24h, failed attempts, checkout speed                                                                    |
| Verification/evidence | secondary (OTP) verification, delivery confirmation strength, refund-requested-before-chargeback, duplicate-transaction flag |
| **Context flags**     | sale-period, salary-week, high-value category (medical/travel), recent location change                                       |

The context flags are the core idea of this project: they prevent a naive model from flagging normal consumer behaviour as fraud — e.g. a large purchase right after payday, a fast checkout during a flash sale, or an address mismatch caused by genuine travel.

## Results

Final model: **Random Forest**, threshold = 0.5

| Metric                     | Value |
| -------------------------- | ----- |
| Precision                  | 40.0% |
| Recall                     | 38.3% |
| F1 Score                   | 39.1% |
| ROC-AUC                    | 0.749 |
| Chargeback rate (test set) | 16.2% |

Random Forest was chosen over Logistic Regression (which had higher recall but much lower precision, 20–29%) because minimising false positives — avoiding friction for genuine customers — was treated as the higher priority for a payment platform.

![Random Forest results](screenshot/random_forest_results.png) ![Confusion matrix](screenshot/model_evaluation_confusion_matrix.png)

**Context-awareness was explicitly validated** — the same raw signal produces a different risk level depending on circumstance:

![Edge case validation](screenshot/eda_edge_case_validation.png)

## Cost Impact

Confusion-matrix outcomes were converted into an illustrative ₹ estimate (assumed: ₹3,000 average order value, ₹150 friction cost per false positive — both estimates, not real merchant figures):

| Item                                         | Amount        |
| -------------------------------------------- | ------------- |
| Value from 62 chargebacks caught early       | + ₹1,86,000   |
| Cost of 93 false positives (review friction) | − ₹13,950     |
| **Net value added by the model**             | **₹1,72,050** |

The 100 still-missed chargebacks (₹3,00,000) are not counted as a model cost — that loss occurs with or without any detection system, and represents room for future improvement rather than a weakness introduced by the model.

## Known Limitations

Stated openly, as the evaluation bar for this track requires honest metrics including false-positive cost:

- **Precision/recall trade-off:** Roughly 6 in 10 flagged transactions turn out safe, and about 6 in 10 real chargebacks are missed. Threshold tuning found no setting that improves both simultaneously — this is inherent to the current feature set.
- **Noise vs. signal:** An earlier version compressed the chargeback rate from ~16% to ~12% using scaling + noise, which was later diagnosed as the cause of weak model performance. Removing the scaling improved Random Forest precision from 31% to 40% and ROC-AUC from 0.688 to 0.749. A second fix attempt (tightening trigger thresholds) was also tested and performed worse — confirming the original ~16% rate as the best-tested configuration.
- **Duplicate-transaction signal is under-weighted** relative to its true strength, because duplicates are naturally rare (~3% of transactions). Confirmed to be a data-volume issue, not a logic issue: temporarily raising duplicate frequency to 7% moved this feature from importance rank #10 to #1. Rolled back because it slightly reduced overall precision/recall.
- All results are based on synthetic, rule-designed data, not real transaction history. A production version would need to learn these relationships from real, unlabelled data.

## Dashboard

`dashboard.html` is a standalone, self-contained interactive dashboard — open it directly in any browser, no server required. It includes:

- Headline business impact and model metrics
- Feature importance and confusion matrix
- A **live "Try a Transaction"** explorer with two modes:
  - 9 curated real scenarios, with real model predictions
  - A custom-transaction form — enter all 20 raw features and get a live prediction from the actual trained Random Forest (200 trees), exported to JS and run entirely client-side, no server required. Reason/action logic reuses the exact rules from `reason_classifier.py`.
- Chargeback-reason breakdown
- Context-awareness comparisons

## Visual Analysis (Power BI)

Supplementary charts built directly from the generated dataset:

![Chargeback reasons](screenshot/pbi_chargeback_reasons.png) ![Payment method](screenshot/pbi_payment_method.png) ![Delivery confirmation](screenshot/pbi_delivery_confirmation.png)

![Sale period effect](screenshot/pbi_sale_period_effect.png) ![Salary week effect](screenshot/pbi_salary_week_effect.png)

Findings:

- Chargebacks are **not primarily a fraud problem** — Service-Issue (43.9%) and Genuine-Dispute (30.4%) together account for nearly three-quarters of cases.
- Card and UPI carry marginally higher risk than netbanking.
- Fast-checkout risk is lower during a sale period, confirming the context-awareness design.
- Among high-amount transactions, risk drops sharply during salary week.
- OTP-confirmed delivery has the lowest chargeback rate of all delivery-confirmation types.

## Repository Structure

```
chargeback-risk-predictor/
├── README.md
├── requirements.txt
├── dashboard.html
├── code/
│   ├── generate_data.py           # Synthetic data generation
│   ├── eda.py                     # Exploratory analysis + edge-case validation
│   ├── train_baseline.py          # Logistic Regression baseline
│   ├── train_random_forest.py     # Final model
│   ├── evaluate.py                # Confusion matrix, precision/recall/F1/AUC
│   ├── threshold_tuning.py        # Precision-recall trade-off across thresholds
│   ├── cost_analysis.py           # ₹ business-impact estimate
│   ├── edge_case_testing.py       # Scenario-based behavioural testing
│   ├── reason_classifier.py       # Stage 2: reason + suggested action
│   ├── build_dashboard_data.py    # Generates data embedded in dashboard.html
│   └── export_model_to_js.py      # Exports trained RF model to JS for the live custom-transaction predictor
├── screenshot/                   # Output evidence for every step above
└── Project_Report.docx            # Full detailed report
```

## How to Run

```
pip install -r requirements.txt

cd code
python generate_data.py            # generates transactions.csv (seeded, deterministic)
python eda.py
python train_baseline.py
python train_random_forest.py
python evaluate.py
python threshold_tuning.py
python cost_analysis.py
python edge_case_testing.py
python reason_classifier.py
python export_model_to_js.py       # Exports the model for dashboard.html's custom-transaction feature
```

Data generation uses a fixed random seed (42), so re-running `generate_data.py` reproduces the exact same 5,000 transactions and downstream results shown in this README.

## Full Report

See [`Project_Report.docx`](Project_Report.docx) for the complete write-up — problem framing, full feature list, all validation steps, and every design decision with its reasoning.
