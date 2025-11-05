"""High-level pipeline convenience wrapper."""
from __future__ import annotations
from typing import Dict, Any
from .supervisor_agent import SupervisorAgentPipeline

class FraudClaimPipeline:
    def __init__(self, pipeline: SupervisorAgentPipeline | None = None):
        self.pipeline = pipeline or SupervisorAgentPipeline()

    def run(self, file_path: str, claim_meta: Dict[str, Any]) -> Dict[str, Any]:
        decision = self.pipeline.process_claim(file_path, claim_meta)
        # Harmonize naming with API expectations.
        return {
            "doc_risk": decision.doc_risk,
            "model_risk": decision.model_risk,
            "final_decision": decision.final_decision,
            "explanation": decision.explanation,
            "combined_score": decision.combined_score,
            "combined_risk_score": decision.combined_score,  # alias
            "doc_findings": decision.details.get("document", {}).get("fields", []),
            "model_features": decision.details.get("cross_checker", {}).get("features", {}),
            "weights": {"doc": 0.55, "model": 0.45},
            "details": decision.details,
        }
