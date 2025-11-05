"""FastAPI Metrics Routes

Provides simplified fraud statistics compatible with frontend expectations.
Returns flat keys (total_claims, approved_claims, pending_claims, rejected_claims, claims_trend, risk_distribution).
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from utils.db import get_db_connection
from auth_fastapi import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])  # /metrics/fraud-statistics etc.

ALLOWED_ROLES = {"admin", "supervisor", "investigator"}

def _role_check(user: Dict[str, Any]):
    if user.get("role") not in ALLOWED_ROLES:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

@router.get("/fraud-statistics")
def fraud_statistics(days: int = 30, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Return fraud statistics summary consumed by dashboard JS.

    Shape expected by frontend (main.js):
    {
      total_claims, approved_claims, pending_claims, rejected_claims,
      claims_trend: [{date, count}],
      risk_distribution: { low: n, medium_low: n, medium: n, high: n, very_high: n }
    }
    """
    _role_check(current_user)
    if days <= 0 or days > 365:
        days = 30
    start_date = datetime.utcnow() - timedelta(days=days)
    try:
        conn = get_db_connection(); cur = conn.cursor()
        # Aggregate status counts & averages
        cur.execute(
            """SELECT 
                COUNT(*) as total_claims,
                SUM(CASE WHEN status='approved' THEN 1 ELSE 0 END) AS approved,
                SUM(CASE WHEN status='submitted' THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END) AS rejected,
                SUM(CASE WHEN risk_score>0.7 THEN 1 ELSE 0 END) AS high_risk,
                AVG(risk_score) as avg_risk
            FROM claims WHERE created_at >= %s""",
            (start_date,)
        )
        row = cur.fetchone() or (0,0,0,0,0, None)
        total, approved, pending, rejected, high_risk, avg_risk = row

        # Trend (daily claim count)
        cur.execute(
            """SELECT DATE(created_at) as d, COUNT(*) FROM claims
                WHERE created_at >= %s GROUP BY DATE(created_at) ORDER BY d ASC""",
            (start_date,)
        )
        trend_rows = cur.fetchall()
        claims_trend = [{"date": r[0].isoformat(), "count": r[1]} for r in trend_rows]

        # Risk distribution buckets
        cur.execute(
            """SELECT 
                SUM(CASE WHEN risk_score IS NOT NULL AND risk_score < 0.3 THEN 1 ELSE 0 END) AS low,
                SUM(CASE WHEN risk_score >=0.3 AND risk_score <0.5 THEN 1 ELSE 0 END) AS medium_low,
                SUM(CASE WHEN risk_score >=0.5 AND risk_score <0.7 THEN 1 ELSE 0 END) AS medium,
                SUM(CASE WHEN risk_score >=0.7 AND risk_score <0.9 THEN 1 ELSE 0 END) AS high,
                SUM(CASE WHEN risk_score >=0.9 THEN 1 ELSE 0 END) AS very_high
            FROM claims WHERE created_at >= %s""",
            (start_date,)
        )
        rd = cur.fetchone() or (0,0,0,0,0)
        risk_distribution = {
            "low": rd[0],
            "medium_low": rd[1],
            "medium": rd[2],
            "high": rd[3],
            "very_high": rd[4]
        }
        return {
            "period_days": days,
            "total_claims": total or 0,
            "approved_claims": approved or 0,
            "pending_claims": pending or 0,
            "rejected_claims": rejected or 0,
            "high_risk_claims": high_risk or 0,
            "average_risk_score": float(avg_risk) if avg_risk is not None else 0.0,
            "claims_trend": claims_trend,
            "risk_distribution": risk_distribution,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to compute fraud statistics")
        raise HTTPException(status_code=500, detail="Failed to compute fraud statistics")
    finally:
        if 'conn' in locals():
            conn.close()


@router.get("/system-health")
def system_health(current_user: Dict[str, Any] = Depends(get_current_user)):
    _role_check(current_user)
    import psutil, os
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM claims"); total_claims = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM users WHERE is_active=true"); active_users = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM claim_documents"); total_docs = cur.fetchone()[0]
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        cpu = process.cpu_percent(interval=0.05)
        upload_dir = "uploads"
        disk_mb = 0
        if os.path.exists(upload_dir):
            for root, dirs, files in os.walk(upload_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        disk_mb += os.path.getsize(fp) / 1024 / 1024
                    except OSError:
                        pass
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "database": {
                "total_claims": total_claims,
                "active_users": active_users,
                "total_documents": total_docs
            },
            "resources": {
                "memory_usage_mb": round(mem_mb, 2),
                "cpu_usage_percent": round(cpu, 2),
                "upload_disk_usage_mb": round(disk_mb, 2)
            }
        }
    except Exception:
        logger.exception("system health failed")
        raise HTTPException(status_code=500, detail="Failed to retrieve system health")
    finally:
        if 'conn' in locals():
            conn.close()


@router.get("/user-activity")
def user_activity(days: int = 7, current_user: Dict[str, Any] = Depends(get_current_user)):
    _role_check(current_user)
    if days <= 0 or days > 90:
        days = 7
    start_date = datetime.utcnow() - timedelta(days=days)
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("""SELECT COUNT(DISTINCT user_id), COUNT(*) FROM claims WHERE created_at >= %s""", (start_date,))
        active_users, total_activities = cur.fetchone() or (0,0)
        cur.execute("""SELECT DATE(created_at), COUNT(*) FROM users WHERE created_at >= %s GROUP BY DATE(created_at) ORDER BY 1 ASC""", (start_date,))
        reg_rows = cur.fetchall()
        cur.execute("""SELECT u.first_name,u.last_name,u.email,u.role, COUNT(c.id) as claim_count
                       FROM users u LEFT JOIN claims c ON u.id=c.user_id AND c.created_at >= %s
                       WHERE u.is_active=true GROUP BY u.id ORDER BY claim_count DESC LIMIT 10""", (start_date,))
        act_rows = cur.fetchall()
        cur.execute("SELECT role, COUNT(*), SUM(CASE WHEN is_active THEN 1 ELSE 0 END) FROM users GROUP BY role")
        role_rows = cur.fetchall()
        return {
            "period_days": days,
            "overview": {
                "active_users": active_users or 0,
                "total_activities": total_activities or 0,
                "average_activities_per_user": (total_activities/active_users) if active_users else 0
            },
            "registration_trends": [ {"date": r[0].isoformat(), "new_users": r[1]} for r in reg_rows ],
            "most_active_users": [ {"name": f"{r[0]} {r[1]}", "email": r[2], "role": r[3], "claim_count": r[4]} for r in act_rows ],
            "role_distribution": [ {"role": r[0], "total_users": r[1], "active_users": r[2], "activity_rate": (r[2]/r[1]*100) if r[1] else 0} for r in role_rows ]
        }
    except Exception:
        logger.exception("user activity failed")
        raise HTTPException(status_code=500, detail="Failed to retrieve user activity")
    finally:
        if 'conn' in locals():
            conn.close()


@router.get("/performance-report")
def performance_report(current_user: Dict[str, Any] = Depends(get_current_user)):
    _role_check(current_user)
    # Compose from existing endpoints internally (avoid double queries for brevity we re-query simpler summary)
    try:
        stats = fraud_statistics(current_user=current_user)
        health = system_health(current_user=current_user)
        activity = user_activity(current_user=current_user)
        # Derived KPIs (simplified)
        high_risk_rate = (stats["high_risk_claims"] / stats["total_claims"] * 100) if stats["total_claims"] else 0
        recommendations = []
        if high_risk_rate < 5:
            recommendations.append("Consider lowering detection thresholds; few high-risk claims detected")
        if high_risk_rate > 30:
            recommendations.append("High high-risk ratio – review potential false positives")
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "summary": {
                "total_claims": stats["total_claims"],
                "high_risk_claims": stats["high_risk_claims"],
                "average_risk_score": stats["average_risk_score"],
                "active_users": activity["overview"]["active_users"],
            },
            "kpis": {
                "high_risk_rate": high_risk_rate,
                "average_claims_per_active_user": activity["overview"]["average_activities_per_user"],
            },
            "recommendations": recommendations,
            "components": {
                "fraud_statistics": stats,
                "system_health": health,
                "user_activity": activity
            }
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("performance report failed")
        raise HTTPException(status_code=500, detail="Failed to generate performance report")


@router.get("/ml-performance")
def ml_performance(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Placeholder ML performance endpoint expected by frontend.

    Derives simple aggregates from claims.risk_score until real model eval is integrated.
    """
    _role_check(current_user)
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT COUNT(*), AVG(risk_score), SUM(CASE WHEN risk_score>=0.7 THEN 1 ELSE 0 END) FROM claims WHERE risk_score IS NOT NULL")
        row = cur.fetchone() or (0, None, 0)
        total_scored, avg_risk, high_risk = row
        high_risk_rate = (high_risk / total_scored * 100) if total_scored else 0
        cur.execute("""SELECT DATE(created_at), AVG(risk_score) FROM claims
                       WHERE created_at >= (CURRENT_DATE - INTERVAL 7 DAY) AND risk_score IS NOT NULL
                       GROUP BY DATE(created_at) ORDER BY 1 ASC""")
        trend_rows = cur.fetchall()
        trend = [{"date": r[0].isoformat(), "avg_risk_score": float(r[1]) if r[1] is not None else 0.0} for r in trend_rows]
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "total_scored_claims": total_scored,
            "average_risk_score": float(avg_risk) if avg_risk is not None else 0.0,
            "high_risk_rate_percent": high_risk_rate,
            "risk_score_trend": trend,
            "metrics": {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None
            },
            "notes": "Placeholder performance derived from risk_score distribution"
        }
    except Exception:
        logger.exception("ml performance failed")
        raise HTTPException(status_code=500, detail="Failed to retrieve ML performance")
    finally:
        if 'conn' in locals():
            conn.close()
