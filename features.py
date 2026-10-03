"""
Feature Definitions, Monotonic Constraints, and Adverse Action Mapping
Strictly enforces zero raw PII in feature definitions.
"""

from typing import Dict, List

# Verification & Repayment Feature Definitions
FEATURE_DEFINITIONS: Dict[str, Dict] = {
    # ----------------------------------------------------
    # 1. De-Identified KYC & Identity Verification Outputs
    # ----------------------------------------------------
    "pan_status_active_flag": {
        "type": "int",
        "description": "NSDL/ITD PAN active and operative status (1=Active, 0=Inactive/Issues)",
        "monotone": -1,  # 0 (inactive) increases risk
        "reason_adverse": "Tax identity status is unverified or marked inoperative",
    },
    "pan_bank_name_similarity": {
        "type": "float",
        "description": "Fuzzy name match similarity between identity record and bank account [0.0 - 1.0]",
        "monotone": -1,  # Low similarity increases risk
        "reason_adverse": "Name discrepancy observed between bank account and government identity record",
    },
    "dob_exact_match_flag": {
        "type": "int",
        "description": "Consistency of Date of Birth across verified identity documents (1=Match, 0=Mismatch)",
        "monotone": -1,  # Mismatch increases risk
        "reason_adverse": "Date of birth discrepancy identified across verified government records",
    },
    "face_match_confidence": {
        "type": "float",
        "description": "Facial biometric cosine similarity between live selfie and government photo [0.0 - 1.0]",
        "monotone": -1,  # Low confidence increases risk
        "reason_adverse": "Facial biometric verification similarity score below acceptable threshold",
    },
    "liveness_anti_spoof_score": {
        "type": "float",
        "description": "Computer vision liveness anti-spoofing confidence [0.0 - 1.0]",
        "monotone": -1,  # Low liveness increases risk
        "reason_adverse": "Biometric liveness verification confidence did not meet anti-spoofing criteria",
    },
    "device_identity_collision_count": {
        "type": "int",
        "description": "Distinct application tokens linked to identical device hardware fingerprint",
        "monotone": 1,  # Higher collision increases fraud risk
        "reason_adverse": "High frequency of loan applications detected from the same device hardware",
    },
    "synthetic_identity_risk_score": {
        "type": "float",
        "description": "Graph anomaly and synthetic identity risk estimator [0.0 - 1.0]",
        "monotone": 1,  # Higher score increases risk
        "reason_adverse": "Identity network pattern flagged elevated synthetic identity risk attributes",
    },
    "pincode_risk_tier": {
        "type": "int",
        "description": "Coarse geographic risk tier aggregated at postal level (1=Low Risk to 5=High Risk)",
        "monotone": 1,  # Higher tier increases risk
        "reason_adverse": "Elevated geographic portfolio default concentration in postal circle",
    },
    "verification_passed": {
        "type": "int",
        "description": "Deterministic pass/fail gate based on statutory verification rules (1=Pass, 0=Fail)",
        "monotone": -1,  # Fail increases risk
        "reason_adverse": "Overall identity and statutory KYC gate requirements not satisfied",
    },

    # ----------------------------------------------------
    # 2. Repayment-Related Financial & Bureau Features
    # ----------------------------------------------------
    "bureau_score": {
        "type": "int",
        "description": "Credit bureau score (e.g. CIBIL/Experian 300 - 900)",
        "monotone": -1,  # Lower score increases risk
        "reason_adverse": "Credit bureau score is below the institutional benchmark",
    },
    "bureau_enquiry_count_30d": {
        "type": "int",
        "description": "Number of credit inquiries across all lenders in previous 30 days",
        "monotone": 1,  # High credit hunger increases risk
        "reason_adverse": "Excessive credit-seeking inquiries recorded across lenders within the last 30 days",
    },
    "bureau_active_unsecured_loans": {
        "type": "int",
        "description": "Number of currently active unsecured personal or consumer loans",
        "monotone": 1,  # Higher unsecured leverage increases risk
        "reason_adverse": "High count of outstanding active unsecured credit facilities",
    },
    "dpd_max_historic_36m": {
        "type": "int",
        "description": "Maximum Days Past Due (delinquency) recorded in past 36 months",
        "monotone": 1,  # Past default increases risk
        "reason_adverse": "Historical payment delinquency (Days Past Due) observed in credit tradelines",
    },
    "dpd_count_90_plus_last_12m": {
        "type": "int",
        "description": "Number of 90+ DPD default events recorded in the last 12 months",
        "monotone": 1,  # Recent severe default increases risk
        "reason_adverse": "Severe loan payment delinquency (90+ DPD) recorded within the past 12 months",
    },
    "cheque_inward_bounce_count_180d": {
        "type": "int",
        "description": "Total inward NACH/mandate/cheque dishonor events in the past 180 days",
        "monotone": 1,  # Bounces increase risk
        "reason_adverse": "Frequent banking mandate/NACH or cheque dishonor events in the last 180 days",
    },
    "credit_utilization_ratio": {
        "type": "float",
        "description": "Revolving credit line utilization ratio (Outstanding balance / Total sanctioned limit)",
        "monotone": 1,  # High utilization increases risk
        "reason_adverse": "Revolving credit line utilization ratio exceeds recommended thresholds",
    },
    "salary_credit_regularity_index": {
        "type": "float",
        "description": "Stability of monthly banking income deposits via Account Aggregator [0.0 - 1.0]",
        "monotone": -1,  # Low regularity increases risk
        "reason_adverse": "Inconsistent banking salary or operating revenue deposit regularity",
    },
    "abb_to_emi_cover_ratio": {
        "type": "float",
        "description": "Average Monthly Bank Balance (ABB) divided by proposed loan monthly EMI",
        "monotone": -1,  # Low liquidity coverage increases risk
        "reason_adverse": "Average monthly bank balance provides insufficient debt-servicing coverage for the proposed EMI",
    },
}

FEATURE_COLUMNS: List[str] = list(FEATURE_DEFINITIONS.keys())
MONOTONIC_CONSTRAINTS: List[int] = [FEATURE_DEFINITIONS[f]["monotone"] for f in FEATURE_COLUMNS]

def assert_no_raw_pii_features(columns: List[str]) -> None:
    """
    Validates that no raw PII field has accidentally been passed into the feature list.
    Only approved derived verification metrics (e.g. pan_status_active_flag, pan_bank_name_similarity)
    are permitted.
    """
    allowed_derived_exceptions = {
        "pan_status_active_flag",
        "pan_bank_name_similarity",
        "dob_exact_match_flag",
    }
    for col in columns:
        lower_col = col.lower()
        if lower_col in allowed_derived_exceptions:
            continue
        if any(term == lower_col for term in ["pan", "aadhaar", "name", "phone", "mobile", "email", "address"]):
            raise ValueError(f"CRITICAL VIOLATION: Raw PII column '{col}' detected in ML feature vector!")
