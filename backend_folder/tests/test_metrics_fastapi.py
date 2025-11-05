"""Tests for metrics FastAPI endpoints using real DB."""
import pytest

pytestmark = pytest.mark.usefixtures("client")


def test_metrics_fraud_statistics_requires_auth(client):
    r = client.get("/metrics/fraud-statistics")
    assert r.status_code in (401, 403)


def test_metrics_fraud_statistics_ok(client, auth_header):
    r = client.get("/metrics/fraud-statistics", headers=auth_header)
    assert r.status_code == 200
    body = r.json()
    for k in ["total_claims", "approved_claims", "pending_claims", "rejected_claims", "claims_trend", "risk_distribution"]:
        assert k in body


def test_metrics_system_health_ok(client, auth_header):
    r = client.get("/metrics/system-health", headers=auth_header)
    assert r.status_code == 200
    body = r.json()
    assert "database" in body and "resources" in body


def test_metrics_user_activity_ok(client, auth_header):
    r = client.get("/metrics/user-activity", headers=auth_header)
    assert r.status_code == 200
    body = r.json()
    assert "overview" in body and "role_distribution" in body


def test_metrics_performance_report_ok(client, auth_header):
    r = client.get("/metrics/performance-report", headers=auth_header)
    assert r.status_code == 200
    body = r.json()
    assert "summary" in body and "kpis" in body
