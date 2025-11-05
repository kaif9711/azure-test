import json
from pathlib import Path
from fastapi.testclient import TestClient
import backend.api as api_module

# We will stub the multi-agent pipeline to avoid loading heavy models during tests.

def test_document_eval_endpoint(monkeypatch, tmp_path):
    app = api_module.app

    class DummyPipeline:
        def run(self, doc_text, claim_meta):
            return {
                "doc_risk": 0.4,
                "model_risk": 0.6,
                "final_decision": "review",
                "combined_score": 0.49,
                "combined_risk_score": 0.49,
                "doc_findings": [{"name": "policy_number", "value": "POL123", "confidence": 0.9}],
                "model_features": {"age": 45},
                "weights": {"doc": 0.55, "model": 0.45},
                "details": {}
            }

    # Monkeypatch the pipeline factory
    def fake_ensure():
        return DummyPipeline()

    monkeypatch.setattr(api_module, '_ensure_multi_agent_pipeline', fake_ensure)

    client = TestClient(app)

    payload = {
        "claim_id": "CLM-TST-1",
        "claim_amount": 1234.56,
        "incident_type": "injury",
        "incident_description": "Unit test incident",
        "document_text": "Claimant Name: John Doe Policy Number: POL123 Incident Date: 2025-01-01 Claim Amount: $1234.56"
    }

    headers = {"Authorization": "Bearer demo-token"}
    resp = client.post("/claims/document-eval", json=payload, headers=headers)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["claim_id"] == payload["claim_id"]
    assert "combined_risk_score" in data
    assert data["final_decision"] == "review"
