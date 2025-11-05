"""FastAPI Admin Routes (port of essential functionality)"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from utils.db import get_db_connection
from auth_fastapi import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


def require_admin(user: Dict[str, Any]):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")


@router.get("/dashboard")
def admin_dashboard(current_user: Dict[str, Any] = Depends(get_current_user)):
    require_admin(current_user)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), SUM(CASE WHEN is_active THEN 1 ELSE 0 END) FROM users")
        total_users, active_users = cur.fetchone()
        cur.execute("SELECT COUNT(*), SUM(CASE WHEN status='submitted' THEN 1 ELSE 0 END), SUM(CASE WHEN status='approved' THEN 1 ELSE 0 END) FROM claims")
        claim_totals = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM claims WHERE risk_score>0.7")
        high_risk = cur.fetchone()[0]
        return {
            "timestamp": datetime.now().isoformat(),
            "system_overview": {
                "total_users": total_users or 0,
                "active_users": active_users or 0,
                "inactive_users": (total_users or 0) - (active_users or 0),
                "total_claims": claim_totals[0] or 0,
                "pending_claims": claim_totals[1] or 0,
                "approved_claims": claim_totals[2] or 0,
                "high_risk_claims": high_risk or 0
            }
        }
    except Exception:
        logger.exception("Dashboard failed")
        raise HTTPException(status_code=500, detail="Failed to load admin dashboard")
    finally:
        if 'conn' in locals():
            conn.close()


@router.get("/users")
def list_users(
    page: int = 1,
    per_page: int = 50,
    role: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    require_admin(current_user)
    per_page = min(per_page, 100)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        base = "SELECT u.id,u.email,u.first_name,u.last_name,u.role,u.is_active,u.created_at,u.last_login,u.updated_at FROM users u"
        where = []
        params = []
        if role:
            where.append("u.role=%s")
            params.append(role)
        if status == 'active':
            where.append("u.is_active=true")
        elif status == 'inactive':
            where.append("u.is_active=false")
        if search:
            where.append("(u.first_name LIKE %s OR u.last_name LIKE %s OR u.email LIKE %s)")
            like = f"%{search}%"
            params.extend([like, like, like])
        if where:
            base += " WHERE " + " AND ".join(where)
        count_q = "SELECT COUNT(*) FROM (" + base + ") t"
        cur.execute(count_q, params)
        total = cur.fetchone()[0]
        offset = (page - 1) * per_page
        q = base + " ORDER BY u.created_at DESC LIMIT %s OFFSET %s"
        cur.execute(q, params + [per_page, offset])
        rows = cur.fetchall()
        users = [
            {
                "id": r[0],
                "email": r[1],
                "first_name": r[2],
                "last_name": r[3],
                "role": r[4],
                "is_active": bool(r[5]),
                "created_at": r[6].isoformat() if r[6] else None,
                "last_login": r[7].isoformat() if r[7] else None,
                "updated_at": r[8].isoformat() if r[8] else None
            } for r in rows
        ]
        return {
            "users": users,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "pages": (total + per_page - 1) // per_page
            }
        }
    except Exception:
        logger.exception("List users failed")
        raise HTTPException(status_code=500, detail="Failed to retrieve users")
    finally:
        if 'conn' in locals():
            conn.close()


@router.get("/audit-logs")
def audit_logs(
    page: int = 1,
    per_page: int = 50,
    sort: str = "-timestamp",
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Return audit logs placeholder so frontend doesn't 404.

    If an audit_log table exists later, wire it here. For now return empty list with pagination.
    """
    require_admin(current_user)
    return {
        "logs": [],
        "pagination": {"page": page, "per_page": per_page, "total": 0, "pages": 0},
        "sorting": sort,
        "message": "Audit logging not yet implemented"
    }
