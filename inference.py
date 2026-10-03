"""
Credit Risk Inference & Decisioning Engine
Consumes de-identified feature payloads keyed by 'secure_entity_token'.
Outputs:
  - Probability of Default (PD)
  - 300 - 900 Credit Score
  - Decision Band (Auto-Approve / Standard Review / EDD / Decline)
  - TreeSHAP-grounded Adverse Action Reason Codes (Zero PII)
"""

from pathlib import Path
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
import shap

from features import (
    FEATURE_COLUMNS,
    FEATURE_DEFINITIONS,
    assert_no_raw_pii_features,
)


class CreditRiskInferenceEngine:
    """Production inference and decisioning engine."""

    def __init__(self, artifact_dir: Path):
        self.calibrated_model = joblib.load(artifact_dir / "calibrated_credit_model.joblib")
        self.raw_model = joblib.load(artifact_dir / "raw_booster_model.joblib")
        self.feature_columns = FEATURE_COLUMNS
        self.explainer = shap.TreeExplainer(self.raw_model)

    def predict_applicant(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single applicant's de-identified feature payload.
        Ensures zero PII is present.
        """
        token = payload.get("secure_entity_token", "UNKNOWN_TOKEN")

        # 1. Assert zero PII in payload keys
        payload_keys = list(payload.keys())
        assert_no_raw_pii_features([k for k in payload_keys if k != "secure_entity_token"])

        # 2. Extract feature vector in exact order
        try:
            row_data = [payload[col] for col in self.feature_columns]
        except KeyError as e:
            raise ValueError(f"Missing required feature for scoring: {e}")

        df_input = pd.DataFrame([row_data], columns=self.feature_columns)

        # 3. Model Scoring & Calibrated Probability of Default (PD)
        pd_prob = float(self.calibrated_model.predict_proba(df_input)[0, 1])

        # 4. Calibrated 300 - 900 Credit Score (Higher = Better Creditworthiness)
        # Scaled non-linearly: Score = 900 - 600 * PD
        credit_score = int(np.clip(np.round(900 - 600 * pd_prob), 300, 900))

        # 5. Deterministic Statutory Check & Verification Overrides
        verification_passed = payload.get("verification_passed", 0)
        pan_active = payload.get("pan_status_active_flag", 1)
        collision_count = payload.get("device_identity_collision_count", 0)
        synthetic_risk = payload.get("synthetic_identity_risk_score", 0.0)

        # 6. Decision Band Allocation
        if verification_passed == 0 or pan_active == 0 or collision_count >= 5 or synthetic_risk >= 0.70:
            decision_band = "DECLINE"
            action_summary = "Application declined due to statutory identity verification failure or severe fraud indicators."
        elif pd_prob < 0.06 and credit_score >= 750:
            decision_band = "AUTO_APPROVE"
            action_summary = "Eligible for straight-through processing (STP) and immediate disbursement."
        elif pd_prob < 0.18 and credit_score >= 650:
            decision_band = "STANDARD_REVIEW"
            action_summary = "Eligible under standard credit criteria. Proceed with standard customer due diligence."
        elif pd_prob < 0.35 and credit_score >= 550:
            decision_band = "ENHANCED_DUE_DILIGENCE"
            action_summary = "Elevated risk profile. Requires manual underwriter review and additional bank statement analysis."
        else:
            decision_band = "DECLINE"
            action_summary = "Application declined based on composite credit risk score and repayment default probability."

        # 7. TreeSHAP Adverse Action Reason Codes
        # Explains the specific drivers pushing the risk score upward
        sv = self.explainer.shap_values(df_input)
        if isinstance(sv, list):
            sample_shap = sv[1][0] if len(sv) > 1 else sv[0][0]
        elif len(sv.shape) == 2:
            sample_shap = sv[0]
        else:
            sample_shap = sv[0, :, 1] if sv.shape[-1] == 2 else sv[0, :, 0]

        # Features with positive SHAP contribution (increasing default probability)
        risk_increasing_indices = np.where(sample_shap > 0)[0]
        # Sort descending by magnitude of risk impact
        sorted_risk_indices = risk_increasing_indices[np.argsort(-sample_shap[risk_increasing_indices])]

        reason_codes: List[Dict[str, Any]] = []
        for idx in sorted_risk_indices[:4]:  # Top 4 drivers
            feat_name = self.feature_columns[idx]
            reason_codes.append({
                "feature": feat_name,
                "adverse_reason": FEATURE_DEFINITIONS[feat_name]["reason_adverse"],
                "shap_impact": round(float(sample_shap[idx]), 4),
                "applicant_value": payload[feat_name],
            })

        return {
            "secure_entity_token": token,
            "probability_of_default_pct": round(pd_prob * 100, 2),
            "calibrated_credit_score": credit_score,
            "decision_band": decision_band,
            "action_summary": action_summary,
            "adverse_action_reason_codes": reason_codes,
            "identity_verification_gate": "PASSED" if verification_passed == 1 else "FAILED",
        }
