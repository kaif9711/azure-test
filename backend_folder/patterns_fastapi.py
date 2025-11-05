"""FastAPI router for managing fraud patterns and recalculating claim risk.

Endpoints:
- GET /patterns : list patterns with optional filters
- POST /patterns : create new pattern (admin only)
- PUT /patterns/{pattern_id} : update existing pattern (admin only)
- POST /patterns/recalculate-claim-risk/{claim_id} : recompute risk score for a claim based on active patterns
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from utils.db import get_db_connection
from auth_fastapi import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/patterns", tags=["fraud-patterns"]) 

class PatternCreate(BaseModel):
    pattern_name: str = Field(..., max_length=100)
    pattern_type: str = Field(..., max_length=50)
    description: Optional[str] = None
    risk_weight: float = Field(..., ge=0.0, le=5.0, description="Relative weight / multiplier for risk contribution")
    is_active: Optional[bool] = True

class PatternUpdate(BaseModel):
    pattern_name: Optional[str] = Field(None, max_length=100)
    pattern_type: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = None
    risk_weight: Optional[float] = Field(None, ge=0.0, le=5.0)
    is_active: Optional[bool] = None

class PatternOut(BaseModel):
    id: int
    pattern_name: str
    pattern_type: str
    description: Optional[str]
    risk_weight: float
    is_active: bool
    created_at: datetime
    updated_at: datetime

class ClaimRiskRecalcResult(BaseModel):
    claim_id: str
    previous_risk_score: Optional[float]
    recalculated_risk_score: float
    pattern_contributions: Dict[str, float]
    applied_patterns: List[int]
    updated_at: datetime
    stored_matches: int


def _require_admin(user: Dict[str, Any]):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")

@router.get("", response_model=List[PatternOut])
def list_patterns(include_inactive: bool = False, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        if include_inactive and current_user.get("role") == "admin":
            cur.execute("SELECT id,pattern_name,pattern_type,description,risk_weight,is_active,created_at,updated_at FROM fraud_patterns ORDER BY id ASC")
        else:
            cur.execute("SELECT id,pattern_name,pattern_type,description,risk_weight,is_active,created_at,updated_at FROM fraud_patterns WHERE is_active=true ORDER BY id ASC")
        rows = cur.fetchall()
        return [
            PatternOut(
                id=r[0], pattern_name=r[1], pattern_type=r[2], description=r[3], risk_weight=float(r[4]),
                is_active=bool(r[5]), created_at=r[6], updated_at=r[7]
            ) for r in rows
        ]
    except Exception:
        logger.exception("Failed to list patterns")
        raise HTTPException(status_code=500, detail="Failed to list patterns")
    finally:
        if 'conn' in locals():
            conn.close()

@router.post("", response_model=PatternOut, status_code=201)
def create_pattern(payload: PatternCreate, current_user: Dict[str, Any] = Depends(get_current_user)):
    _require_admin(current_user)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO fraud_patterns (pattern_name,pattern_type,description,risk_weight,is_active) VALUES (%s,%s,%s,%s,%s)",
            (payload.pattern_name, payload.pattern_type, payload.description, payload.risk_weight, payload.is_active)
        )
        conn.commit()
        pattern_id = cur.lastrowid
        cur.execute("SELECT id,pattern_name,pattern_type,description,risk_weight,is_active,created_at,updated_at FROM fraud_patterns WHERE id=%s", (pattern_id,))
        r = cur.fetchone()
        return PatternOut(
            id=r[0], pattern_name=r[1], pattern_type=r[2], description=r[3], risk_weight=float(r[4]),
            is_active=bool(r[5]), created_at=r[6], updated_at=r[7]
        )
    except Exception as e:
        if 'Duplicate' in str(e):
            raise HTTPException(status_code=400, detail="Pattern name already exists")
        logger.exception("Failed to create pattern")
        raise HTTPException(status_code=500, detail="Failed to create pattern")
    finally:
        if 'conn' in locals():
            conn.close()

@router.put("/{pattern_id}", response_model=PatternOut)
def update_pattern(pattern_id: int, payload: PatternUpdate, current_user: Dict[str, Any] = Depends(get_current_user)):
    _require_admin(current_user)
    if not any([payload.pattern_name, payload.pattern_type, payload.description is not None, payload.risk_weight is not None, payload.is_active is not None]):
        raise HTTPException(status_code=400, detail="No fields to update provided")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        sets = []
        params = []
        if payload.pattern_name is not None:
            sets.append("pattern_name=%s"); params.append(payload.pattern_name)
        if payload.pattern_type is not None:
            sets.append("pattern_type=%s"); params.append(payload.pattern_type)
        if payload.description is not None:
            sets.append("description=%s"); params.append(payload.description)
        if payload.risk_weight is not None:
            sets.append("risk_weight=%s"); params.append(payload.risk_weight)
        if payload.is_active is not None:
            sets.append("is_active=%s"); params.append(payload.is_active)
        sql = "UPDATE fraud_patterns SET " + ",".join(sets) + ", updated_at=NOW() WHERE id=%s"
        params.append(pattern_id)
        cur.execute(sql, params)
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Pattern not found")
        conn.commit()
        cur.execute("SELECT id,pattern_name,pattern_type,description,risk_weight,is_active,created_at,updated_at FROM fraud_patterns WHERE id=%s", (pattern_id,))
        r = cur.fetchone()
        return PatternOut(
            id=r[0], pattern_name=r[1], pattern_type=r[2], description=r[3], risk_weight=float(r[4]),
            is_active=bool(r[5]), created_at=r[6], updated_at=r[7]
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to update pattern")
        raise HTTPException(status_code=500, detail="Failed to update pattern")
    finally:
        if 'conn' in locals():
            conn.close()

# --- Claim Risk Recalculation ---
# Simplified heuristic: current risk_score = base(existing or 0) + sum(active_pattern_weights * factor)
# Pattern match logic would normally inspect claim + related artifacts; here we just apply all active patterns as demonstration.
@router.post("/recalculate-claim-risk/{claim_id}", response_model=ClaimRiskRecalcResult)
def recalc_claim_risk(claim_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    if current_user.get("role") not in {"admin", "supervisor", "investigator"}:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT risk_score, claim_amount, incident_date FROM claims WHERE id=%s", (claim_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Claim not found")
        previous_risk = float(row[0]) if row[0] is not None else None
        cur.execute("SELECT id, pattern_name, risk_weight FROM fraud_patterns WHERE is_active=true")
        patterns = cur.fetchall()
        contributions = {}
        total_add = 0.0
        applied_ids = []
        for p in patterns:
            pid, pname, weight = p[0], p[1], float(p[2])
            contribution = min(weight * 0.1, 1.0)  # arbitrary scaling factor
            contributions[pname] = contribution
            total_add += contribution
            applied_ids.append(pid)
        new_score = min((previous_risk or 0.0) + total_add, 1.0)
        # Update claim risk
        cur.execute("UPDATE claims SET risk_score=%s, updated_at=NOW() WHERE id=%s", (new_score, claim_id))

        # Persist pattern matches snapshot: clear existing then insert new
        stored_matches = 0
        try:
            cur.execute("DELETE FROM claim_pattern_matches WHERE claim_id=%s", (claim_id,))
            if applied_ids:
                import json
                insert_sql = ("""INSERT INTO claim_pattern_matches (claim_id, pattern_id, match_score, match_details) 
                                VALUES (%s,%s,%s,%s)""")
                for pid in applied_ids:
                    pname = next((k for k,v in contributions.items() if k == next((pp[1] for pp in patterns if pp[0]==pid), None)), None)
                    # simpler mapping using loop; build details
                    details = {
                        "pattern_name": next((pp[1] for pp in patterns if pp[0]==pid), "unknown"),
                        "applied_weight": float(next((pp[2] for pp in patterns if pp[0]==pid), 0.0)),
                        "contribution": contributions.get(next((pp[1] for pp in patterns if pp[0]==pid), ''), 0.0)
                    }
                    cur.execute(insert_sql, (claim_id, pid, details["contribution"], json.dumps(details)))
                    stored_matches += 1
        except Exception:
            # Do not fail whole request if match persistence fails; log and continue
            logger.exception("Failed to persist claim pattern matches")
        conn.commit()
        return ClaimRiskRecalcResult(
            claim_id=claim_id,
            previous_risk_score=previous_risk,
            recalculated_risk_score=new_score,
            pattern_contributions=contributions,
            applied_patterns=applied_ids,
            updated_at=datetime.utcnow(),
            stored_matches=stored_matches
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to recalc claim risk")
        raise HTTPException(status_code=500, detail="Failed to recalc claim risk")
    finally:
        if 'conn' in locals():
            conn.close()
