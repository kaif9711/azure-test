"""Supervisor Pipeline Agent

Combines DocumentValidatorAgent + CrossCheckerAgent outputs with business rules
and produces a final decision object.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any

from .document_validator_agent import DocumentValidatorAgent
from .cross_checker_agent import CrossCheckerAgent

logger = logging.getLogger(__name__)

@dataclass
class SupervisorDecision:
    doc_risk: float
    model_risk: float
    final_decision: str
    explanation: list
    combined_score: float
    details: Dict[str, Any]

class SupervisorAgentPipeline:
    def __init__(self, doc_agent: DocumentValidatorAgent | None = None, cross_agent: CrossCheckerAgent | None = None):
        self.doc_agent = doc_agent or DocumentValidatorAgent()
        self.cross_agent = cross_agent or CrossCheckerAgent()

    def process_claim(self, file_path: str, claim_meta: Dict[str, Any]) -> SupervisorDecision:
        doc_res = self.doc_agent.validate(file_path)
        # Convert fields for cross checker
        minimal_fields = [
            {"name": f.name, "value": f.value, "confidence": f.confidence} for f in doc_res.extracted_fields
        ]
        cross_res = self.cross_agent.cross_check(minimal_fields, claim_meta)

        combined = self._combine(doc_res.doc_risk, cross_res.model_risk)
        decision, explanation = self._decide(doc_res.doc_risk, cross_res.model_risk, claim_meta, doc_res.issues)

        return SupervisorDecision(
            doc_risk=doc_res.doc_risk,
            model_risk=cross_res.model_risk,
            final_decision=decision,
            explanation=explanation,
            combined_score=combined,
            details={
                "document": {
                    "model_used": doc_res.model_used,
                    "issues": doc_res.issues,
                    "fields": minimal_fields,
                    "timing_seconds": doc_res.timing_seconds,
                },
                "cross_checker": {
                    "fraud_probability": cross_res.fraud_probability,
                    "anomaly_score": cross_res.anomaly_score,
                    "features": cross_res.features,
                    "feature_importance": cross_res.feature_importance,
                }
            }
        )

    def _combine(self, doc_risk: float, model_risk: float) -> float:
        # Weighted average: give doc analysis slightly more initial trust (can tune later)
        return round(0.55 * doc_risk + 0.45 * model_risk, 4)

    def _decide(self, doc_risk: float, model_risk: float, meta: Dict[str, Any], issues):
        explanation = []
        # Basic rules (extend as needed)
        if doc_risk > 0.8:
            explanation.append("High document risk")
        if model_risk > 0.7:
            explanation.append("Model risk above threshold")
        if meta.get('policy_duration', meta.get('policy_duration_months', 12)) < 2:
            explanation.append("Policy age < 2 months")
        if meta.get('claim_amount', 0) and meta.get('claim_amount', 0) > 50000:
            explanation.append("High claim amount")
        explanation.extend([f"Issue: {i}" for i in issues[:3]])  # cap verbosity

        combined = self._combine(doc_risk, model_risk)
        if combined >= 0.75:
            decision = "investigate"
        elif combined >= 0.55:
            decision = "review"
        elif combined >= 0.35:
            decision = "hold"
            explanation.append("Needs additional verification")
        else:
            decision = "approve"
        return decision, explanation
