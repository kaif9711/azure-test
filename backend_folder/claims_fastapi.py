"""FastAPI Claims Routes

Ported from legacy Flask implementation to FastAPI, adapted for MySQL.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import date
from datetime import datetime
import uuid
import os
import logging

from utils.db import get_db_connection
from auth_fastapi import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/claims", tags=["claims"])


def require_role(user: Dict[str, Any], roles: List[str]):
    if user.get("role") not in roles:
        raise HTTPException(status_code=403, detail="Insufficient permissions")


class ClaimCreate(BaseModel):
    claim_amount: float = Field(..., gt=0)
    incident_date: date
    incident_description: str
    policy_number: str
    location: Optional[str] = None

    def to_values(self):
        return (
            self.claim_amount,
            self.incident_date.isoformat(),
            self.incident_description,
            self.policy_number,
            self.location or ""
        )

def _insert_claim(cur, current_user, payload: ClaimCreate):
    # Verify policy
    cur.execute(
        "SELECT id, coverage_amount FROM policies WHERE policy_number=%s AND user_id=%s AND is_active=true",
        (payload.policy_number, current_user['user_id'])
    )
    policy = cur.fetchone()
    if not policy:
        raise HTTPException(status_code=400, detail="Invalid or inactive policy")
    if payload.claim_amount > float(policy[1]):
        raise HTTPException(status_code=400, detail="Claim amount exceeds coverage")
    claim_id = str(uuid.uuid4())
    cur.execute(
        """INSERT INTO claims (id, user_id, policy_id, claim_amount, incident_date, incident_description, location, status)
        VALUES (%s,%s,%s,%s,%s,%s,%s,'submitted')""",
        (claim_id, current_user['user_id'], policy[0], payload.claim_amount, payload.incident_date, payload.incident_description, payload.location or "")
    )
    return claim_id


class PolicyCreate(BaseModel):
    policy_type: str = Field(..., description="Type of the policy (e.g., basic, comprehensive, premium, standard)")
    coverage_amount: float = Field(..., gt=0, description="Coverage amount")
    premium: float = Field(..., gt=0, description="Premium amount")
    policy_number: Optional[str] = Field(None, description="Optional policy number; if omitted it will be generated")
    user_id: Optional[int] = Field(None, description="(Admin only) Create policy for a specific user id")

@router.post("/create", status_code=201)
def submit_claim_json(payload: ClaimCreate, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Submit a new claim using JSON body (preferred)."""
    try:
        if payload.incident_date > date.today():
            raise HTTPException(status_code=400, detail="Incident date cannot be in the future")
        conn = get_db_connection(); cur = conn.cursor()
        claim_id = _insert_claim(cur, current_user, payload)
        conn.commit()
        return {
            "message": "Claim submitted successfully",
            "claim_id": claim_id,
            "status": "submitted",
            "created_at": datetime.now().isoformat(),
            "next_steps": [
                "Upload supporting documents",
                "Await initial review",
                "Possible follow-up contact"
            ]
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to submit claim (JSON)")
        raise HTTPException(status_code=500, detail="Failed to submit claim")
    finally:
        if 'conn' in locals():
            conn.close()


@router.post("", status_code=201)
async def submit_claim(request: Request,
    claim_amount: Optional[float] = None,
    incident_date: Optional[str] = None,
    incident_description: Optional[str] = None,
    policy_number: Optional[str] = None,
    policy_id: Optional[str] = None,
    location: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)):
    """Submit a new claim.

    Supports both legacy form/query params and JSON body.
    If Content-Type is application/json, we parse JSON into ClaimCreate.
    """
    try:
        if request.headers.get('content-type','').startswith('application/json'):
            data = await request.json()
            payload = ClaimCreate(**data)
        else:
            # Legacy mode using query/form style
            # Allow either policy_number (preferred) or policy_id (convert to number)
            if not policy_number and policy_id:
                # lookup policy_number by id for this user
                tmp_conn = get_db_connection(); tmp_cur = tmp_conn.cursor()
                tmp_cur.execute("SELECT policy_number FROM policies WHERE id=%s AND user_id=%s AND is_active=true", (policy_id, current_user['user_id']))
                row = tmp_cur.fetchone(); tmp_conn.close()
                if row:
                    policy_number = row[0]
            missing = [k for k,v in {"claim_amount":claim_amount, "incident_date":incident_date, "incident_description":incident_description, "policy_number":policy_number}.items() if v is None]
            if missing:
                raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing)}")
            try:
                inc_dt = datetime.fromisoformat(incident_date)  # type: ignore
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid incident date format")
            payload = ClaimCreate(
                claim_amount=claim_amount,  # type: ignore
                incident_date=inc_dt.date(),
                incident_description=incident_description,  # type: ignore
                policy_number=policy_number,  # type: ignore
                location=location
            )
        if payload.incident_date > datetime.utcnow().date():
            raise HTTPException(status_code=400, detail="Incident date cannot be in the future")
        conn = get_db_connection(); cur = conn.cursor()
        claim_id = _insert_claim(cur, current_user, payload)
        conn.commit()
        return {
            "message": "Claim submitted successfully",
            "claim_id": claim_id,
            "status": "submitted",
            "created_at": datetime.utcnow().isoformat(),
            "next_steps": [
                "Upload supporting documents",
                "Await initial review",
                "Possible follow-up contact"
            ]
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to submit claim (unified)")
        raise HTTPException(status_code=500, detail="Failed to submit claim")
    finally:
        if 'conn' in locals():
            conn.close()


@router.get("")
def list_claims(
    page: int = 1,
    per_page: int = 20,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    per_page = min(per_page, 100)
    admin_view = current_user.get("role") in {"admin", "supervisor", "investigator"}
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        base = (
            "SELECT c.id,c.user_id,c.claim_amount,c.incident_date,c.incident_description,c.location,c.status,c.created_at,c.updated_at,c.risk_score,"
            "u.first_name,u.last_name,u.email,p.policy_number,p.policy_type FROM claims c "
            "JOIN users u ON c.user_id=u.id JOIN policies p ON c.policy_id=p.id"
        )
        where = []
        params = []
        if not admin_view:
            where.append("c.user_id=%s")
            params.append(current_user['user_id'])
        if status:
            where.append("c.status=%s")
            params.append(status)
        if date_from:
            where.append("c.created_at >= %s")
            params.append(date_from)
        if date_to:
            where.append("c.created_at <= %s")
            params.append(date_to)
        if where:
            base += " WHERE " + " AND ".join(where)
        count_query = "SELECT COUNT(*) FROM (" + base + ") t"
        cur.execute(count_query, params)
        total = cur.fetchone()[0]
        offset = (page - 1) * per_page
        query = base + " ORDER BY c.created_at DESC LIMIT %s OFFSET %s"
        cur.execute(query, params + [per_page, offset])
        rows = cur.fetchall()
        claims = []
        for r in rows:
            item = {
                "id": r[0],
                "user_id": r[1],
                "claim_amount": float(r[2]),
                "incident_date": r[3].isoformat() if r[3] else None,
                "incident_description": r[4],
                "location": r[5],
                "status": r[6],
                "created_at": r[7].isoformat() if r[7] else None,
                "updated_at": r[8].isoformat() if r[8] else None,
                "risk_score": float(r[9]) if r[9] is not None else None,
                "policy_number": r[13],
                "policy_type": r[14]
            }
            if admin_view:
                item["claimant"] = {"first_name": r[10], "last_name": r[11], "email": r[12]}
            claims.append(item)
        return {
            "claims": claims,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "pages": (total + per_page - 1) // per_page
            }
        }
    except Exception:
        logger.exception("Failed to list claims")
        raise HTTPException(status_code=500, detail="Failed to retrieve claims")
    finally:
        if 'conn' in locals():
            conn.close()


@router.get("/{claim_id}")
def claim_details(claim_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    admin_view = current_user.get("role") in {"admin", "supervisor", "investigator"}
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        where = "c.id=%s" if admin_view else "c.id=%s AND c.user_id=%s"
        params = [claim_id] if admin_view else [claim_id, current_user['user_id']]
        cur.execute(
            f"""SELECT c.id,c.user_id,c.policy_id,c.claim_amount,c.incident_date,c.incident_description,c.location,c.status,c.created_at,c.updated_at,\n"
            f"c.risk_score,c.document_score,c.supervisor_decision,c.rejection_reason,\n"
            f"u.first_name,u.last_name,u.email,p.policy_number,p.policy_type,p.coverage_amount\n"
            f"FROM claims c JOIN users u ON c.user_id=u.id JOIN policies p ON c.policy_id=p.id WHERE {where}""",
            params
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Claim not found")
        # documents
        cur.execute("SELECT id,filename,file_type,file_size,upload_date,validation_status,validation_score FROM claim_documents WHERE claim_id=%s ORDER BY upload_date DESC", (claim_id,))
        docs = cur.fetchall()
        # status history
        cur.execute("SELECT status,changed_at,changed_by_user_id,notes FROM claim_status_history WHERE claim_id=%s ORDER BY changed_at DESC", (claim_id,))
        hist = cur.fetchall()
        # pattern matches (joined with pattern meta)
        cur.execute(
            """SELECT cpm.pattern_id, fp.name, fp.severity, cpm.match_score, cpm.match_details, cpm.matched_at
                FROM claim_pattern_matches cpm
                JOIN fraud_patterns fp ON cpm.pattern_id = fp.id
                WHERE cpm.claim_id=%s
                ORDER BY cpm.matched_at DESC""",
            (claim_id,)
        )
        pm_rows = cur.fetchall()
        return {
            "id": row[0],
            "user_id": row[1],
            "policy_id": row[2],
            "claim_amount": float(row[3]),
            "incident_date": row[4].isoformat() if row[4] else None,
            "incident_description": row[5],
            "location": row[6],
            "status": row[7],
            "created_at": row[8].isoformat() if row[8] else None,
            "updated_at": row[9].isoformat() if row[9] else None,
            "risk_score": float(row[10]) if row[10] else None,
            "document_score": float(row[11]) if row[11] else None,
            "supervisor_decision": row[12],
            "rejection_reason": row[13],
            "claimant": {"first_name": row[14], "last_name": row[15], "email": row[16]},
            "policy": {"policy_number": row[17], "policy_type": row[18], "coverage_amount": float(row[19])},
            "documents": [
                {
                    "id": d[0],
                    "filename": d[1],
                    "file_type": d[2],
                    "file_size": d[3],
                    "upload_date": d[4].isoformat() if d[4] else None,
                    "validation_status": d[5],
                    "validation_score": float(d[6]) if d[6] else None
                } for d in docs
            ],
            "status_history": [
                {
                    "status": h[0],
                    "changed_at": h[1].isoformat() if h[1] else None,
                    "changed_by_user_id": h[2],
                    "notes": h[3]
                } for h in hist
            ],
            "pattern_matches": [
                {
                    "pattern_id": pm[0],
                    "pattern_name": pm[1],
                    "severity": pm[2],
                    "match_score": float(pm[3]) if pm[3] is not None else None,
                    "match_details": pm[4],
                    "matched_at": pm[5].isoformat() if pm[5] else None
                } for pm in pm_rows
            ]
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to get claim details")
        raise HTTPException(status_code=500, detail="Failed to retrieve claim details")
    finally:
        if 'conn' in locals():
            conn.close()


@router.put("/{claim_id}/status")
def update_claim_status(claim_id: str, status: str, notes: Optional[str] = None, current_user: Dict[str, Any] = Depends(get_current_user)):
    require_role(current_user, ["admin", "supervisor"])
    valid_status = {"submitted", "under_review", "approved", "rejected", "investigation", "closed"}
    if status not in valid_status:
        raise HTTPException(status_code=400, detail="Invalid status")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE claims SET status=%s, updated_at=NOW() WHERE id=%s", (status, claim_id))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Claim not found")
        cur.execute(
            "INSERT INTO claim_status_history (claim_id,status,changed_by_user_id,notes) VALUES (%s,%s,%s,%s)",
            (claim_id, status, current_user['user_id'], notes or "")
        )
        conn.commit()
        return {"message": "Claim status updated", "claim_id": claim_id, "new_status": status}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to update claim status")
        raise HTTPException(status_code=500, detail="Failed to update claim status")
    finally:
        if 'conn' in locals():
            conn.close()


# Placeholder for document upload (simplified ingestion)
@router.post("/{claim_id}/upload")
def upload_claim_document(
    claim_id: str,
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    try:
        # Check claim ownership or admin
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT user_id,status FROM claims WHERE id=%s", (claim_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Claim not found")
        if current_user['role'] not in {"admin", "supervisor"} and str(row[0]) != str(current_user['user_id']):
            raise HTTPException(status_code=403, detail="Forbidden")
        if row[1] in {"approved", "rejected", "closed"}:
            raise HTTPException(status_code=400, detail="Cannot upload for this claim status")
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        doc_id = str(uuid.uuid4())
        filename = f"{doc_id}_{file.filename}"
        path = os.path.join(upload_dir, filename)
        with open(path, "wb") as f:
            f.write(file.file.read())
        cur.execute(
            """INSERT INTO claim_documents (id,claim_id,filename,original_filename,file_type,file_size,file_path,uploaded_by_user_id,validation_status)\n"
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'pending')""",
            (doc_id, claim_id, filename, file.filename, os.path.splitext(file.filename)[1].lstrip('.'), file.size or 0, path, current_user['user_id'])
        )
        conn.commit()
        return {"message": "Document uploaded", "document_id": doc_id, "filename": file.filename}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail="Failed to upload document")
    finally:
        if 'conn' in locals():
            conn.close()

"""Additional document management endpoints (FastAPI-native) to align with frontend expectations.

The frontend JS currently uses:
  POST   /claims/{claim_id}/documents          (FormData with key 'documents' repeated for multi-file upload)
  GET    /claims/{claim_id}/documents          (List existing documents)
  DELETE /claims/{claim_id}/documents/{doc_id} (Delete a specific document)

Historically we only exposed a single-file endpoint at /claims/{claim_id}/upload expecting field name 'file'.
We retain that for backward compatibility and add the richer set below.
"""

from pathlib import Path

@router.post("/{claim_id}/documents")
async def upload_claim_documents(
    claim_id: str,
    documents: List[UploadFile] = File(..., description="One or more files to attach"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    try:
        if not documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        conn = get_db_connection(); cur = conn.cursor()
        # Validate claim & permissions
        cur.execute("SELECT user_id,status FROM claims WHERE id=%s", (claim_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Claim not found")
        if current_user['role'] not in {"admin", "supervisor"} and str(row[0]) != str(current_user['user_id']):
            raise HTTPException(status_code=403, detail="Forbidden")
        if row[1] in {"approved", "rejected", "closed"}:
            raise HTTPException(status_code=400, detail="Cannot upload for this claim status")
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        saved = []
        for up in documents:
            doc_id = str(uuid.uuid4())
            safe_name = f"{doc_id}_{up.filename}"
            path = upload_dir / safe_name
            content = await up.read()
            with open(path, "wb") as f:
                f.write(content)
            file_type = os.path.splitext(up.filename)[1].lstrip('.')
            file_size = len(content)
            cur.execute(
                """INSERT INTO claim_documents (id,claim_id,filename,original_filename,file_type,file_size,file_path,uploaded_by_user_id,validation_status)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'pending')""",
                (doc_id, claim_id, safe_name, up.filename, file_type, file_size, str(path), current_user['user_id'])
            )
            saved.append({
                "document_id": doc_id,
                "filename": up.filename,
                "file_type": file_type,
                "file_size": file_size
            })
        conn.commit()
        return {"message": f"Uploaded {len(saved)} document(s)", "documents": saved}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Multi-document upload failed")
        raise HTTPException(status_code=500, detail="Failed to upload documents")
    finally:
        if 'conn' in locals():
            conn.close()


@router.get("/{claim_id}/documents")
def list_claim_documents(claim_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        conn = get_db_connection(); cur = conn.cursor()
        # Permission / ownership check
        cur.execute("SELECT user_id FROM claims WHERE id=%s", (claim_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Claim not found")
        if current_user['role'] not in {"admin", "supervisor"} and str(row[0]) != str(current_user['user_id']):
            raise HTTPException(status_code=403, detail="Forbidden")
        cur.execute("""SELECT id,filename,original_filename,file_type,file_size,upload_date,validation_status,validation_score
                       FROM claim_documents WHERE claim_id=%s ORDER BY upload_date DESC""", (claim_id,))
        docs = cur.fetchall()
        return {"documents": [
            {
                "id": d[0],
                "filename": d[1],
                "original_filename": d[2],
                "file_type": d[3],
                "file_size": d[4],
                "upload_date": d[5].isoformat() if d[5] else None,
                "validation_status": d[6],
                "validation_score": float(d[7]) if d[7] is not None else None
            } for d in docs
        ]}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to list claim documents")
        raise HTTPException(status_code=500, detail="Failed to list claim documents")
    finally:
        if 'conn' in locals():
            conn.close()


@router.delete("/{claim_id}/documents/{document_id}")
def delete_claim_document(
    claim_id: str,
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT user_id FROM claims WHERE id=%s", (claim_id,))
        claim_row = cur.fetchone()
        if not claim_row:
            raise HTTPException(status_code=404, detail="Claim not found")
        if current_user['role'] not in {"admin", "supervisor"} and str(claim_row[0]) != str(current_user['user_id']):
            raise HTTPException(status_code=403, detail="Forbidden")
        cur.execute("SELECT file_path FROM claim_documents WHERE id=%s AND claim_id=%s", (document_id, claim_id))
        doc = cur.fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        # Delete file from disk
        try:
            if doc[0] and os.path.exists(doc[0]):
                os.remove(doc[0])
        except Exception as fs_err:
            logger.warning(f"Failed to remove file from disk for document {document_id}: {fs_err}")
        cur.execute("DELETE FROM claim_documents WHERE id=%s", (document_id,))
        conn.commit()
        return {"message": "Document deleted", "document_id": document_id}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to delete claim document")
        raise HTTPException(status_code=500, detail="Failed to delete claim document")
    finally:
        if 'conn' in locals():
            conn.close()


# ---- Policies (supporting claim submission) ----
from fastapi import Query
from typing import Optional

@router.get("/policies")
def list_policies(
    all_policies: bool = Query(False, alias="all", description="Admin: return all active policies"),
    user_id: Optional[str] = Query(None, description="Admin: filter by user_id"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List active policies.

    Default: current user's active policies.
    Admin/Supervisor: can pass all=true to list all active policies or user_id to filter by a specific user.
    """
    try:
        conn = get_db_connection(); cur = conn.cursor()

        is_admin = current_user.get('role') in {"admin", "supervisor"}
        if is_admin and all_policies:
            cur.execute(
                """SELECT id, policy_number, policy_type, coverage_amount, start_date, end_date, is_active
                       FROM policies WHERE is_active=true ORDER BY created_at DESC"""
            )
        elif is_admin and user_id is not None:
            cur.execute(
                """SELECT id, policy_number, policy_type, coverage_amount, start_date, end_date, is_active
                       FROM policies WHERE user_id=%s AND is_active=true ORDER BY created_at DESC""",
                (user_id,)
            )
        else:
            cur.execute(
                """SELECT id, policy_number, policy_type, coverage_amount, start_date, end_date, is_active
                       FROM policies WHERE user_id=%s AND is_active=true ORDER BY created_at DESC""",
                (current_user['user_id'],)
            )

        rows = cur.fetchall()
        return {"policies": [
            {
                "id": r[0],
                "policy_number": r[1],
                "policy_type": r[2],
                "coverage_amount": float(r[3]) if r[3] is not None else 0.0,
                "start_date": r[4].isoformat() if r[4] else None,
                "end_date": r[5].isoformat() if r[5] else None,
                "is_active": bool(r[6])
            } for r in rows
        ]}
    except Exception:
        logger.exception("Failed to list policies")
        raise HTTPException(status_code=500, detail="Failed to retrieve policies")
    finally:
        if 'conn' in locals():
            conn.close()


@router.get("/insured-policies", summary="List all insured policies (canonical 170)")
def list_insured_policies(
    q: Optional[str] = None,
    limit: int = 200,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Return the canonical list of policies sourced from insured_tb joined to policies.

    This is intended for claim submission UI to let a (properly authenticated) insured user
    pick their policy number. A normal user will still only be able to successfully submit a claim
    if their user_id matches the policy's user_id (enforced in claim insertion).

    Filters:
      q: optional substring filter applied to policy_number or last_name.
    """
    limit = max(1, min(limit, 500))
    try:
        conn = get_db_connection(); cur = conn.cursor()
        base = ("""
            SELECT p.policy_number, p.policy_type, p.coverage_amount, p.user_id,
                   i.first_name, i.last_name, i.policy_start_date, i.policy_status, i.credit_score_segment
            FROM policies p
            JOIN insured_tb i ON p.policy_number = i.policy_number
            WHERE p.is_active=true
        """)
        params = []
        if q:
            base += " AND (p.policy_number LIKE %s OR i.last_name LIKE %s OR i.first_name LIKE %s)"
            like = f"%{q}%"
            params.extend([like, like, like])
        base += " ORDER BY p.policy_number LIMIT %s"
        params.append(limit)
        cur.execute(base, params)
        rows = cur.fetchall()
        return {
            "policies": [
                {
                    "policy_number": r[0],
                    "policy_type": r[1],
                    "coverage_amount": float(r[2]) if r[2] is not None else None,
                    "user_id": r[3],
                    "first_name": r[4],
                    "last_name": r[5],
                    "policy_start_date": r[6].isoformat() if r[6] else None,
                    "policy_status": r[7],
                    "credit_score_segment": r[8]
                } for r in rows
            ],
            "count": len(rows)
        }
    except Exception:
        logger.exception("Failed to list insured policies")
        raise HTTPException(status_code=500, detail="Failed to retrieve insured policies")
    finally:
        if 'conn' in locals():
            conn.close()


@router.post("/policies", status_code=201)
def create_policy(payload: PolicyCreate, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Create a new policy for the current user (or another user if admin/supervisor)."""
    try:
        target_user_id = current_user['user_id']
        if payload.user_id is not None:
            if current_user.get('role') not in {"admin", "supervisor"}:
                raise HTTPException(status_code=403, detail="Insufficient permissions to assign user_id")
            target_user_id = payload.user_id
        # Basic validation
        policy_type = payload.policy_type.strip().lower()
        allowed_types = {"basic", "comprehensive", "premium", "standard"}
        if policy_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"Unsupported policy_type. Allowed: {', '.join(sorted(allowed_types))}")
        conn = get_db_connection(); cur = conn.cursor()
        # Confirm target user exists & active
        cur.execute("SELECT id,is_active FROM users WHERE id=%s", (target_user_id,))
        urow = cur.fetchone()
        if not urow or not urow[1]:
            raise HTTPException(status_code=400, detail="Target user not found or inactive")
        # Policy number generation if missing
        policy_number = payload.policy_number.strip() if payload.policy_number else None
        if not policy_number:
            # Generate e.g. POL-U<user>-YYYYMMDD-RND4
            from datetime import datetime as _dt
            import random,string
            rnd = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            policy_number = f"POL-U{target_user_id}-{_dt.utcnow().strftime('%Y%m%d')}-{rnd}"
        # Ensure uniqueness
        cur.execute("SELECT 1 FROM policies WHERE policy_number=%s", (policy_number,))
        if cur.fetchone():
            raise HTTPException(status_code=409, detail="Policy number already exists")
        cur.execute(
            """INSERT INTO policies (user_id, policy_number, policy_type, coverage_amount, premium, start_date, is_active)
                   VALUES (%s,%s,%s,%s,%s,CURDATE(),true)""",
            (target_user_id, policy_number, policy_type, payload.coverage_amount, payload.premium)
        )
        pid = cur.lastrowid
        conn.commit()
        return {
            "policy": {
                "id": pid,
                "user_id": target_user_id,
                "policy_number": policy_number,
                "policy_type": policy_type,
                "coverage_amount": float(payload.coverage_amount),
                "premium": float(payload.premium),
                "start_date": None,  # will show as maybe today if fetched via list endpoint
                "is_active": True
            }
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to create policy")
        raise HTTPException(status_code=500, detail="Failed to create policy")
    finally:
        if 'conn' in locals():
            conn.close()


@router.get("/policies/{policy_id}")
def policy_details(policy_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get details for a specific policy (must belong to current user unless admin)."""
    try:
        admin_view = current_user.get("role") in {"admin", "supervisor"}
        conn = get_db_connection(); cur = conn.cursor()
        if admin_view:
            cur.execute("""SELECT id, user_id, policy_number, policy_type, coverage_amount, start_date, end_date, is_active
                           FROM policies WHERE id=%s""", (policy_id,))
        else:
            cur.execute("""SELECT id, user_id, policy_number, policy_type, coverage_amount, start_date, end_date, is_active
                           FROM policies WHERE id=%s AND user_id=%s""", (policy_id, current_user['user_id']))
        r = cur.fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="Policy not found")
        return {"policy": {
            "id": r[0],
            "user_id": r[1],
            "policy_number": r[2],
            "policy_type": r[3],
            "coverage_amount": float(r[4]) if r[4] is not None else 0.0,
            "start_date": r[5].isoformat() if r[5] else None,
            "end_date": r[6].isoformat() if r[6] else None,
            "is_active": bool(r[7])
        }}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to get policy details")
        raise HTTPException(status_code=500, detail="Failed to retrieve policy details")
    finally:
        if 'conn' in locals():
            conn.close()
