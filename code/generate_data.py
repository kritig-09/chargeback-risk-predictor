"""
Synthetic Data Generator — Chargeback Risk Predictor
------------------------------------------------------
Generates fake but realistic e-commerce transactions with 20 features.
Each transaction gets a chargeback probability based on business rules,
then we roll a dice (with randomness) to decide if it ACTUALLY became
a chargeback. Context flags (sale period, salary week, etc.) reduce
risk for cases that would otherwise look suspicious.
"""

import numpy as np
import pandas as pd

# Reproducibility — same random data every time we run this
np.random.seed(42)

N = 5000  # number of transactions to generate


def generate_transactions(n=N):
    data = {}

    # ---------- 1. CORE TRANSACTION SIGNALS ----------
    # Most orders are small, a few are large (real spending looks like this).
    # sigma raised to 1.4 so we get a meaningful tail of high-value (>20k) txns
    # to actually test the "high amount" rules against.
    data["amount"] = np.round(np.random.lognormal(mean=6.5, sigma=1.4, size=n), 2)
    data["amount"] = np.clip(data["amount"], 100, 100000)

    data["hour_of_day"] = np.random.randint(0, 24, size=n)

    data["payment_method"] = np.random.choice(
        ["card", "upi", "netbanking"], size=n, p=[0.45, 0.45, 0.10]
    )

    # ---------- 2. CUSTOMER HISTORY SIGNALS ----------
    data["account_age_days"] = np.random.exponential(scale=300, size=n).astype(int)
    data["account_age_days"] = np.clip(data["account_age_days"], 0, 3000)

    data["avg_order_value"] = np.round(
        np.random.lognormal(mean=6.2, sigma=0.8, size=n), 2
    )

    data["past_chargeback_count"] = np.random.choice(
        [0, 1, 2, 3], size=n, p=[0.90, 0.07, 0.02, 0.01]
    )

    # ---------- 3. MISMATCH SIGNALS ----------
    data["billing_shipping_mismatch"] = np.random.choice(
        [0, 1], size=n, p=[0.85, 0.15]
    )
    data["name_mismatch_score"] = np.round(np.random.beta(1, 8, size=n), 2)  # mostly low
    data["device_sharing_flag"] = np.random.choice([0, 1], size=n, p=[0.95, 0.05])

    # ---------- 4. VELOCITY SIGNALS ----------
    data["transactions_last_24h"] = np.random.poisson(lam=1.2, size=n)
    data["failed_attempts_before_success"] = np.random.choice(
        [0, 1, 2, 3, 4], size=n, p=[0.75, 0.15, 0.06, 0.03, 0.01]
    )
    # checkout speed in seconds (fast = suspicious UNLESS sale period)
    data["checkout_speed_sec"] = np.round(np.random.exponential(scale=45, size=n), 1)

    # ---------- 5. VERIFICATION / EVIDENCE SIGNALS ----------
    # Large txns are more likely to trigger bank verification call/OTP
    data["secondary_verification_flag"] = (
        (data["amount"] > 15000) & (np.random.rand(n) > 0.15)
    ).astype(int)

    data["delivery_confirmation_strength"] = np.random.choice(
        ["otp_confirmed", "signature", "left_at_door", "not_applicable"],
        size=n, p=[0.35, 0.25, 0.25, 0.15]
    )

    data["refund_requested_before_chargeback"] = np.random.choice(
        [0, 1], size=n, p=[0.6, 0.4]
    )

    # Duplicate transaction: same-ish amount, very close in time, rare event.
    # NOTE: We tested bumping this to 7% to help the model learn the signal
    # better (it worked -- feature importance rank went from #10 to #1,
    # and the scenario risk-score roughly doubled). But that change also
    # shifted overall precision/recall on the main metrics we report
    # everywhere else, so we rolled back to the original 3% and documented
    # this as a known limitation instead (see README / pitch notes).
    data["duplicate_transaction_flag"] = np.random.choice(
        [0, 1], size=n, p=[0.97, 0.03]
    )

    # ---------- 6. CONTEXT FLAGS (explain-away anomalies) ----------
    data["is_sale_period"] = np.random.choice([0, 1], size=n, p=[0.80, 0.20])
    data["is_salary_week"] = np.random.choice([0, 1], size=n, p=[0.83, 0.17])  # ~5/30 days
    data["is_high_value_category"] = np.random.choice(
        ["none", "medical", "travel"], size=n, p=[0.80, 0.10, 0.10]
    )
    data["location_change_recent"] = np.random.choice([0, 1], size=n, p=[0.88, 0.12])

    df = pd.DataFrame(data)
    return df


def compute_chargeback_risk(df):
    """
    Rule-based risk score (0 to 1ish before noise).
    Each rule ADDS or SUBTRACTS risk. Context flags subtract risk
    from signals that would otherwise look suspicious.
    """
    risk = np.zeros(len(df))

    # --- Base risk factors ---
    risk += (df["amount"] > 20000) * 0.15
    risk += (df["account_age_days"] < 15) * 0.20
    risk += (df["amount"] > 3 * df["avg_order_value"]) * 0.15
    risk += df["past_chargeback_count"] * 0.12
    risk += df["billing_shipping_mismatch"] * 0.10
    risk += (df["name_mismatch_score"] > 0.4) * 0.20
    risk += df["device_sharing_flag"] * 0.15
    risk += (df["transactions_last_24h"] > 4) * 0.15
    risk += (df["failed_attempts_before_success"] >= 2) * 0.10
    risk += ((df["hour_of_day"] >= 1) & (df["hour_of_day"] <= 4)) * 0.05

    # --- Checkout speed: suspicious ONLY if not sale period ---
    fast_checkout = df["checkout_speed_sec"] < 10
    risk += np.where(fast_checkout & (df["is_sale_period"] == 0), 0.15, 0)
    # during sale period, fast checkout is NOT penalized at all

    # --- High amount: suspicious LESS if salary week or high value category ---
    high_amount = df["amount"] > 20000
    explained = (df["is_salary_week"] == 1) | (df["is_high_value_category"] != "none")
    risk += np.where(high_amount & ~explained, 0.10, 0)  # extra penalty only if unexplained
    risk -= np.where(high_amount & explained, 0.15, 0)   # reduce risk if explained

    # --- Location mismatch: suspicious LESS if recent travel ---
    # (bumped from -0.08 to -0.18 so this signal survives the later
    # risk-scaling step and shows up consistently, not just sometimes)
    risk += np.where(
        (df["billing_shipping_mismatch"] == 1) & (df["location_change_recent"] == 1),
        -0.18, 0
    )

    # --- Verification passed but still disputes: bump risk (friendly fraud signal) ---
    risk += df["secondary_verification_flag"] * 0.05  # slight bump, real signal is in combo w/ dispute

    # --- Strong delivery evidence: protects merchant, lowers "fraud-type" chargeback odds ---
    risk -= (df["delivery_confirmation_strength"] == "otp_confirmed") * 0.10

    # --- Refund denied before → genuine escalation, not fraud, but IS a chargeback driver ---
    risk += df["refund_requested_before_chargeback"] * 0.10

    # --- Duplicate transaction: strong independent driver, NOT fraud-related ---
    risk += df["duplicate_transaction_flag"] * 0.35

    # Add noise so it's not a perfectly learnable rule (real world is messy).
    # NOTE: std was originally 0.08, combined with a 0.72 scale-down factor
    # to hit a ~12% chargeback rate. That combination blurred the signal too
    # much (noise was comparable in size to many individual rule effects),
    # which is why the trained model's precision/recall were weak.
    # Fix: drop the scale-down (base rules already average ~15.7% risk,
    # which is an acceptable rate on its own) and reduce noise so real
    # signal comes through more clearly.
    risk += np.random.normal(0, 0.04, size=len(df))

    # NOTE: We tried a light 0.88 scale-down here to bring the rate from
    # ~16% closer to 12%, but it actually hurt model performance (AUC
    # dropped from 0.749 to 0.697). Reason: scaling down risk before the
    # Bernoulli draw makes the resulting labels noisier relative to signal
    # (variance-to-mean ratio rises as probability drops). So we accept
    # the slightly higher ~16% rate in exchange for a model that actually
    # learns the patterns well -- a better trade for this project's goals.

    # Clip into 0-1 range as a rough probability
    risk = np.clip(risk, 0.01, 0.95)
    return risk


def assign_reason(row):
    """Assign a chargeback reason label — only meaningful when is_chargeback == 1."""
    if row["duplicate_transaction_flag"] == 1:
        return "Accidental-Duplicate"
    if row["refund_requested_before_chargeback"] == 1 and row["name_mismatch_score"] < 0.3:
        return "Service-Issue"
    if row["name_mismatch_score"] > 0.4 or row["device_sharing_flag"] == 1 or row["billing_shipping_mismatch"] == 1:
        return "Fraud-Suspected"
    return "Genuine-Dispute"


def main():
    df = generate_transactions(N)
    risk = compute_chargeback_risk(df)
    df["chargeback_probability"] = np.round(risk, 3)

    # Roll the dice: actual outcome based on probability (adds realism/noise)
    df["is_chargeback"] = (np.random.rand(len(df)) < risk).astype(int)

    # Assign reason only for actual chargebacks.
    # NOTE: use "Not-Applicable" instead of "N/A" -- pandas reads "N/A" as a
    # missing value by default when loading the CSV back, which corrupted EDA.
    df["chargeback_reason"] = np.where(
        df["is_chargeback"] == 1,
        df.apply(assign_reason, axis=1),
        "Not-Applicable"
    )

    # Save
    out_path = "/home/claude/chargeback_project/transactions.csv"
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} transactions -> {out_path}")
    print(f"\nChargeback rate: {df['is_chargeback'].mean():.2%}")
    print(f"\nReason breakdown (among chargebacks):")
    print(df[df["is_chargeback"] == 1]["chargeback_reason"].value_counts())
    print(f"\nSample rows:")
    print(df.head(3).T)


if __name__ == "__main__":
    main()
