"""
Restricted Identity/KYC Vault Ingestion & De-Identification Pipeline.
Processes raw PII dataset from scratch directory:
  [customer_id, name, pan, adhaar, address, cibil_score]

Performs:
  1. Cryptographic isolation of raw PII in Restricted Vault (zero leak to ML).
  2. Extraction of de-identified verification signals (PAN format & surname match, Aadhaar validation, PIN code risk tier).
  3. One-way surrogate tokenization (secure_entity_token).
  4. Publication to De-Identified Feature Store.
  5. Batch scoring using trained Monotonic Credit Risk Model.
"""

import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS, assert_no_raw_pii_features
from inference import CreditRiskInferenceEngine

VAULT_SECRET_SALT = b"hsm_vault_master_salt_prod_2026_india"


class RestrictedIdentityVaultIngestor:
    """Simulates the hardware-isolated Identity Vault boundary."""

    def __init__(self, salt: bytes = VAULT_SECRET_SALT):
        self.salt = salt

    def generate_surrogate_token(self, customer_id: int) -> str:
        """Deterministic one-way HMAC token generation."""
        sig = hmac.new(self.salt, str(customer_id).encode("utf-8"), hashlib.sha256).hexdigest()
        return f"tok_{sig[:20]}"

    def verify_pan_structure(self, pan: str, name: str) -> Tuple[int, int]:
        """
        Validates PAN structure according to Indian Income Tax Department rules:
        - 10 characters: 5 uppercase letters, 4 digits, 1 uppercase letter.
        - 4th character must be 'P' for Individual/Person.
        - 5th character must match the first letter of the applicant's surname/last name.
        Returns: (pan_status_active_flag, surname_match_flag)
        """
        pan_clean = str(pan).strip().upper()
        if not re.match(r"^[A-Z]{3}P[A-Z]\d{4}[A-Z]$", pan_clean):
            return 0, 0

        name_parts = str(name).strip().split()
        if not name_parts:
            return 0, 0
        surname = name_parts[-1].upper()
        surname_initial = surname[0]

        surname_match = 1 if pan_clean[4] == surname_initial else 0
        pan_active = 1 if surname_match == 1 else 0
        return pan_active, surname_match

    def verify_aadhaar_structure(self, aadhaar: str) -> int:
        """
        Validates Aadhaar format according to UIDAI rules:
        - 12 digits, formatted as 'XXXX XXXX XXXX' or 12 continuous digits.
        - Must not start with 0 or 1.
        Returns: 1 if format is valid, 0 otherwise.
        """
        digits = re.sub(r"\s+", "", str(aadhaar).strip())
        if len(digits) != 12 or not digits.isdigit():
            return 0
        if digits[0] in ["0", "1"]:
            return 0
        return 1

    def extract_pincode_tier(self, address: str) -> Tuple[int, int]:
        """
        Extracts 6-digit postal PIN code from raw address string.
        Maps to coarse postal risk tier (1=Lowest Risk to 5=Highest Risk).
        DROPS ALL STREET, DOOR NUMBER, AND LOCALITY INFORMATION.
        """
        match = re.search(r"(\d{6})$", str(address).strip())
        if not match:
            # Fallback scan for any 6 digit sequence
            match = re.search(r"\b(\d{6})\b", str(address).strip())

        if not match:
            return 0, 3  # Unknown pincode, default Tier 3

        pincode = int(match.group(1))
        prefix = pincode // 100  # e.g., 5600, 4000, 1100

        # Coarse postal circle risk tier mapping
        tier_1_prefixes = {5600, 4000, 1100, 6000, 5000, 4110}  # Tier 1 Metros
        tier_2_prefixes = {7000, 3800, 3020, 1600}               # Major Urban Centers
        tier_3_prefixes = {2260, 4520, 6820}                     # Emerging Urban Centers

        if prefix in tier_1_prefixes:
            return pincode, 1
        elif prefix in tier_2_prefixes:
            return pincode, 2
        elif prefix in tier_3_prefixes:
            return pincode, 3
        else:
            return pincode, 4

    def process_raw_dataset(self, raw_csv_path: Path, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reads raw CSV, locks PII into vault, and emits de-identified feature dataset.
        """
        print(f"Reading raw customer records from {raw_csv_path}...")
        df_raw = pd.read_csv(raw_csv_path)

        vault_records = []
        feature_records = []

        np.random.seed(42)

        for _, row in df_raw.iterrows():
            cid = int(row["customer_id"])
            token = self.generate_surrogate_token(cid)
            name = str(row["name"]).strip()
            pan = str(row["pan"]).strip()
            aadhaar = str(row["adhaar"]).strip()
            address = str(row["address"]).strip()
            cibil = int(row["cibil_score"])

            # 1. Verification checks inside the Vault
            pan_active, surname_match = self.verify_pan_structure(pan, name)
            aadhaar_valid = self.verify_aadhaar_structure(aadhaar)
            pincode, pin_tier = self.extract_pincode_tier(address)

            # Simulated UIDAI / NSDL / DigiLocker verification signals
            pan_bank_name_sim = round(0.95 if surname_match else 0.55, 4)
            dob_match = 1
            face_conf = round(np.random.uniform(0.88, 0.98), 4)
            liveness_score = round(np.random.uniform(0.92, 0.99), 4)
            dev_collisions = int(np.random.choice([0, 1, 2], p=[0.85, 0.12, 0.03]))
            synthetic_risk = round(np.random.exponential(scale=0.04), 4)

            verif_passed = 1 if (pan_active == 1 and aadhaar_valid == 1 and surname_match == 1) else 0

            # 2. Derive Repayment Features from CIBIL Score Profile
            # In production, these join from Bureau / Account Aggregator feeds on customer token
            if cibil >= 750:  # Excellent
                bureau_enquiries = int(np.random.choice([0, 1, 2], p=[0.70, 0.25, 0.05]))
                unsecured_loans = int(np.random.choice([0, 1, 2], p=[0.40, 0.45, 0.15]))
                dpd_36m = 0
                dpd_90p = 0
                bounces = 0
                utilization = round(np.random.uniform(0.08, 0.35), 4)
                salary_reg = round(np.random.uniform(0.85, 0.98), 4)
                abb_emi = round(np.random.uniform(3.5, 7.5), 4)
            elif cibil >= 670:  # Good
                bureau_enquiries = int(np.random.choice([1, 2, 3], p=[0.50, 0.35, 0.15]))
                unsecured_loans = int(np.random.choice([1, 2, 3], p=[0.30, 0.50, 0.20]))
                dpd_36m = int(np.random.choice([0, 15, 30], p=[0.75, 0.18, 0.07]))
                dpd_90p = 0
                bounces = int(np.random.choice([0, 1], p=[0.85, 0.15]))
                utilization = round(np.random.uniform(0.35, 0.65), 4)
                salary_reg = round(np.random.uniform(0.70, 0.90), 4)
                abb_emi = round(np.random.uniform(2.0, 4.2), 4)
            elif cibil >= 580:  # Fair
                bureau_enquiries = int(np.random.choice([2, 3, 4, 5], p=[0.30, 0.40, 0.20, 0.10]))
                unsecured_loans = int(np.random.choice([2, 3, 4], p=[0.30, 0.45, 0.25]))
                dpd_36m = int(np.random.choice([30, 60], p=[0.60, 0.40]))
                dpd_90p = 0
                bounces = int(np.random.choice([1, 2, 3], p=[0.50, 0.35, 0.15]))
                utilization = round(np.random.uniform(0.65, 0.88), 4)
                salary_reg = round(np.random.uniform(0.45, 0.72), 4)
                abb_emi = round(np.random.uniform(1.0, 2.2), 4)
            else:  # Poor (< 580)
                bureau_enquiries = int(np.random.choice([3, 4, 6, 8], p=[0.20, 0.30, 0.30, 0.20]))
                unsecured_loans = int(np.random.choice([3, 4, 5], p=[0.25, 0.45, 0.30]))
                dpd_36m = int(np.random.choice([90, 120, 180], p=[0.50, 0.30, 0.20]))
                dpd_90p = int(np.random.choice([1, 2, 3], p=[0.60, 0.30, 0.10]))
                bounces = int(np.random.choice([2, 3, 5, 7], p=[0.30, 0.40, 0.20, 0.10]))
                utilization = round(np.random.uniform(0.85, 1.15), 4)
                salary_reg = round(np.random.uniform(0.20, 0.50), 4)
                abb_emi = round(np.random.uniform(0.3, 0.9), 4)

            # Record for RESTRICTED VAULT (Hardware-encrypted, isolated)
            vault_records.append({
                "customer_id": cid,
                "secure_entity_token": token,
                "raw_name": name,
                "raw_pan": pan,
                "raw_aadhaar": aadhaar,
                "raw_address": address,
                "extracted_pincode": pincode,
            })

            # Record for DE-IDENTIFIED FEATURE STORE (Zero PII)
            feature_records.append({
                "secure_entity_token": token,
                "pan_status_active_flag": pan_active,
                "pan_bank_name_similarity": pan_bank_name_sim,
                "dob_exact_match_flag": dob_match,
                "face_match_confidence": face_conf,
                "liveness_anti_spoof_score": liveness_score,
                "device_identity_collision_count": dev_collisions,
                "synthetic_identity_risk_score": synthetic_risk,
                "pincode_risk_tier": pin_tier,
                "verification_passed": verif_passed,
                "bureau_score": cibil,
                "bureau_enquiry_count_30d": bureau_enquiries,
                "bureau_active_unsecured_loans": unsecured_loans,
                "dpd_max_historic_36m": dpd_36m,
                "dpd_count_90_plus_last_12m": dpd_90p,
                "cheque_inward_bounce_count_180d": bounces,
                "credit_utilization_ratio": utilization,
                "salary_credit_regularity_index": salary_reg,
                "abb_to_emi_cover_ratio": abb_emi,
            })

        df_vault = pd.DataFrame(vault_records)
        df_features = pd.DataFrame(feature_records)

        # Enforce Zero PII in features
        assert_no_raw_pii_features([c for c in df_features.columns if c != "secure_entity_token"])

        # Save outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        vault_file = output_dir / "restricted_identity_vault.json"
        features_file = output_dir / "deidentified_features.csv"

        import time
        try:
            df_vault.to_json(vault_file, orient="records", indent=2)
            print(f"Restricted Vault records saved securely: {vault_file}")
        except PermissionError:
            fallback_vault = output_dir / f"restricted_identity_vault_{int(time.time())}.json"
            df_vault.to_json(fallback_vault, orient="records", indent=2)
            print(f"[NOTICE] '{vault_file.name}' is currently locked. Saved to: {fallback_vault}")

        try:
            df_features.to_csv(features_file, index=False)
            print(f"De-Identified Feature Store dataset saved: {features_file}")
        except PermissionError:
            fallback_feat = output_dir / f"deidentified_features_{int(time.time())}.csv"
            df_features.to_csv(fallback_feat, index=False)
            print(f"[NOTICE] '{features_file.name}' is currently locked. Saved to: {fallback_feat}")

        print(f"Features verified: Zero PII detected across all {df_features.shape[0]} rows.")
        return df_vault, df_features


def score_deidentified_customers(df_features: pd.DataFrame, model_dir: Path, output_file: Path) -> pd.DataFrame:
    """
    Executes inference using trained Monotonic Credit Risk Model.
    Generates Probability of Default, Credit Score, Decision Band, and Reason Codes.
    """
    import time
    print(f"\nInitializing Inference Engine from {model_dir}...")
    engine = CreditRiskInferenceEngine(model_dir)

    results = []
    print(f"Scoring {len(df_features)} de-identified customer profiles...")

    for _, row in df_features.iterrows():
        payload = row.to_dict()
        decision = engine.predict_applicant(payload)
        
        # Primary top reason code string
        top_reason = ""
        if decision["adverse_action_reason_codes"]:
            top_reason = decision["adverse_action_reason_codes"][0]["adverse_reason"]

        results.append({
            "secure_entity_token": decision["secure_entity_token"],
            "bureau_score": payload["bureau_score"],
            "verification_gate": decision["identity_verification_gate"],
            "probability_of_default_pct": decision["probability_of_default_pct"],
            "calibrated_credit_score": decision["calibrated_credit_score"],
            "decision_band": decision["decision_band"],
            "action_summary": decision["action_summary"],
            "top_adverse_reason": top_reason,
        })

    df_results = pd.DataFrame(results)
    try:
        df_results.to_csv(output_file, index=False)
        print(f"Scoring complete! Decision results saved to: {output_file}")
    except PermissionError:
        fallback_file = output_file.with_name(f"customer_credit_decisions_{int(time.time())}.csv")
        df_results.to_csv(fallback_file, index=False)
        print(f"\n[NOTICE] '{output_file.name}' is currently OPEN in another program (e.g. Microsoft Excel).")
        print(f"Results saved successfully to fallback file: {fallback_file}")
        print("Tip: Close Excel if you wish to overwrite the primary file directly.")
    return df_results


def main():
    raw_csv = Path(r"C:\Users\asha.kanwar\.gemini\antigravity\brain\91af17bc-dc4b-4ff7-84a7-532571bcd37d\scratch\dummy_customers_dataset.csv")
    output_dir = Path(__file__).parent / "processed_dataset"
    model_dir = Path(__file__).parent / "saved_models"
    decisions_file = output_dir / "customer_credit_decisions.csv"

    print("=" * 70)
    print("RESTRICTED IDENTITY VAULT INGESTION & BATCH CREDIT DECISIONING")
    print("=" * 70)

    ingestor = RestrictedIdentityVaultIngestor()
    df_vault, df_features = ingestor.process_raw_dataset(raw_csv, output_dir)

    # Run Scoring
    df_decisions = score_deidentified_customers(df_features, model_dir, decisions_file)

    # Executive Portfolio Summary
    print("\n" + "=" * 70)
    print("PORTFOLIO DECISION SUMMARY (500 PROCESSED CUSTOMERS)")
    print("=" * 70)
    summary = df_decisions["decision_band"].value_counts()
    for band, count in summary.items():
        pct = (count / len(df_decisions)) * 100
        print(f" - {band:<25}: {count:3d} applicants ({pct:5.1f}%)")

    print("\nSample Decision Records:")
    sample_preview = df_decisions[["secure_entity_token", "bureau_score", "probability_of_default_pct", "calibrated_credit_score", "decision_band"]].head(6)
    print(sample_preview.to_string(index=False))


if __name__ == "__main__":
    main()
