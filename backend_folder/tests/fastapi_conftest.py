"""FastAPI test fixtures using the real MySQL database.

These tests assume:
- A running MySQL instance reachable via the env var DATABASE_URL (mysql://user:pass@host:port/db)
- The schema will be initialized by the FastAPI startup event (init_db in utils.db)

If the database is unreachable the tests will be skipped gracefully.
"""
import os
import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app
# We import lazily so that environment variables are already set before app startup
try:
    from api import app  # triggers startup events when TestClient is created
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"Failed to import FastAPI application: {e}")

from utils.db import get_db_connection


def _db_available():
    try:
        conn = get_db_connection()
        conn.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def db_available():
    if not _db_available():
        pytest.skip("Real database not available (DATABASE_URL not reachable)")
    return True


@pytest.fixture(scope="session")
def client(db_available):  # type: ignore
    """Provide a TestClient bound to the real DB (do not recreate per test)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def admin_token(client):  # pragma: no cover - simple helper
    """Obtain an admin JWT via /auth/login (seeded admin@example.com)."""
    payload = {"email": "admin@example.com", "password": "admin123"}
    resp = client.post("/auth/login", json=payload)
    if resp.status_code != 200:
        pytest.skip(f"Cannot login as admin user: {resp.status_code} {resp.text}")
    return resp.json()["access_token"]


@pytest.fixture
def auth_header(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture
def db_conn(db_available):
    conn = get_db_connection()
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


def ensure_policy_for_user(conn, user_email: str) -> int:
    cur = conn.cursor()
    # fetch user id
    cur.execute("SELECT id FROM users WHERE email=%s", (user_email,))
    row = cur.fetchone()
    if not row:
        raise RuntimeError("User not found for policy creation")
    user_id = row[0]
    cur.execute("SELECT id FROM policies WHERE user_id=%s LIMIT 1", (user_id,))
    p = cur.fetchone()
    if p:
        return p[0]
    # create a policy
    import uuid, datetime
    policy_number = f"TEST-POL-{uuid.uuid4().hex[:8]}"
    cur.execute(
        """INSERT INTO policies (user_id, policy_number, policy_type, coverage_amount, premium, start_date, is_active)
        VALUES (%s,%s,%s,%s,%s,%s,true)""",
        (user_id, policy_number, "comprehensive", 100000.00, 1200.00, datetime.date.today())
    )
    conn.commit()
    return cur.lastrowid


@pytest.fixture
def sample_claim(db_conn):
    """Insert a sample claim directly (bypassing API) for metrics & risk recalculation tests."""
    policy_id = ensure_policy_for_user(db_conn, "admin@example.com")
    cur = db_conn.cursor()
    import uuid, datetime
    claim_id = str(uuid.uuid4())
    cur.execute(
        """INSERT INTO claims (id, user_id, policy_id, claim_amount, incident_date, incident_description, location, status, risk_score)
        VALUES (%s, (SELECT id FROM users WHERE email=%s), %s, %s, %s, %s, %s, 'submitted', %s)""",
        (claim_id, "admin@example.com", policy_id, 5000.00, datetime.date.today(), "Test incident", "Test City", 0.2)
    )
    db_conn.commit()
    return claim_id
