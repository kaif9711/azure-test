"""FastAPI Authentication Routes (MySQL + JWT)

Provides /auth endpoints for registration, login, profile management and password change.
Uses existing password utilities in utils.auth_utils and MySQL connector in utils.db.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
import time
import logging
import jwt  # PyJWT

from utils.db import get_db_connection
from utils.auth_utils import (
    validate_password, hash_password, verify_password,
    validate_email as legacy_validate_email
)

logger = logging.getLogger(__name__)

JWT_SECRET = os.getenv("JWT_SECRET", "dev-change-me")
JWT_ALG = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXP_SECONDS = int(os.getenv("JWT_EXP_SECONDS", "3600"))

router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer()


# ----------------------------- Pydantic Models ----------------------------- #

class RegisterRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=8)
    first_name: str
    last_name: str
    # Incoming role will be ignored unless it's 'user'; elevation now requires admin endpoint
    role: Optional[str] = Field("user")

class LoginRequest(BaseModel):
    email: str
    password: str

class ProfileUpdateRequest(BaseModel):
    first_name: Optional[str]
    last_name: Optional[str]
    email: Optional[str]

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

class RoleUpdateRequest(BaseModel):
    user_id: int
    new_role: str


# ----------------------------- Helper Functions ---------------------------- #

def create_access_token(user_id: int, email: str, role: str) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "iat": now,
        "exp": now + JWT_EXP_SECONDS
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    # PyJWT returns str in recent versions
    return token


def decode_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    token = credentials.credentials
    # Development fallback
    if token == "demo-token":
        return {"user_id": "demo_user", "email": "demo@example.com", "role": "admin"}
    data = decode_token(token)
    return {"user_id": data["sub"], "email": data.get("email"), "role": data.get("role", "user")}


def fetch_user_by_email(cursor, email: str):
    cursor.execute("SELECT id, email, password_hash, first_name, last_name, role, is_active, created_at, last_login FROM users WHERE email=%s", (email,))
    return cursor.fetchone()


def user_row_to_dict(row):
    # Align with frontend expectations
    return {
        "id": row[0],
        "email": row[1],
        "first_name": row[3],
        "last_name": row[4],
        "role": row[5],
        "is_active": bool(row[6]),
        "created_at": row[7].isoformat() if row[7] else None,
        "last_login": row[8].isoformat() if row[8] else None
    }


# ------------------------------- Endpoints --------------------------------- #

@router.post("/register")
def register(payload: RegisterRequest):
    try:
        # Normalize & basic email format validation using legacy helper
        payload.email = payload.email.strip().lower()
        email_check = legacy_validate_email(payload.email)
        if not email_check['valid']:
            raise HTTPException(status_code=400, detail=email_check['message'])

        # Password strength
        pwd_check = validate_password(payload.password)
        if not pwd_check["valid"]:
            raise HTTPException(status_code=400, detail=pwd_check["message"])

        # Enforce that self-registration cannot escalate beyond 'user'
        valid_roles = ['user', 'admin', 'investigator', 'supervisor']
        if payload.role != 'user':
            logger.info("Ignoring elevated role request on self-registration; defaulting to 'user'")
        role = 'user'

        conn = get_db_connection()
        cursor = conn.cursor()

        # Existing user
        if fetch_user_by_email(cursor, payload.email.lower()):
            raise HTTPException(status_code=409, detail="User with this email already exists")

        pwd_hash = hash_password(payload.password)
        cursor.execute(
            """INSERT INTO users (email, password_hash, first_name, last_name, role, is_active)
            VALUES (%s,%s,%s,%s,%s,true)""",
            (payload.email.lower(), pwd_hash, payload.first_name.strip(), payload.last_name.strip(), role)
        )
        user_id = cursor.lastrowid
        conn.commit()

        # Auto-provision a default active policy for the new user (dev usability)
        try:
            cursor.execute("SELECT COUNT(*) FROM policies WHERE user_id=%s", (user_id,))
            if cursor.fetchone()[0] == 0:
                # Simple unique-ish policy number
                import datetime
                pn = f"POL-U-{user_id}-{int(time.time())}"  # time imported above
                cursor.execute(
                    """INSERT INTO policies (user_id, policy_number, policy_type, coverage_amount, premium, start_date, is_active)
                        VALUES (%s,%s,%s,%s,%s,CURDATE(),true)""",
                    (user_id, pn, 'standard', 75000.00, 500.00)
                )
                conn.commit()
                logger.info(f"Created default policy {pn} for new user {user_id}")
        except Exception as pol_err:
            logger.warning(f"Failed to auto-create default policy for user {user_id}: {pol_err}")

        # Fetch inserted user for timestamps
        cursor.execute("SELECT id, email, password_hash, first_name, last_name, role, is_active, created_at, last_login FROM users WHERE id=%s", (user_id,))
        row = cursor.fetchone()
        user_dict = user_row_to_dict(row)

        token = create_access_token(user_dict['id'], user_dict['email'], user_dict['role'])

        return {
            "message": "User registered successfully",
            "access_token": token,
            "user": user_dict
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Registration failed")
        raise HTTPException(status_code=500, detail="Registration failed")
    finally:
        if 'conn' in locals():
            conn.close()


@router.get("/check-email")
def check_email_availability(email: str):
    """Check if an email is available for registration.

    Returns: { email: str, available: bool }
    """
    try:
        norm = email.strip().lower()
        conn = get_db_connection(); cur = conn.cursor()
        row = fetch_user_by_email(cur, norm)
        return {"email": norm, "available": row is None}
    except Exception:
        logger.exception("Email availability check failed")
        raise HTTPException(status_code=500, detail="Failed to check email")
    finally:
        if 'conn' in locals():
            conn.close()


@router.post("/login")
def login(payload: LoginRequest):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        email_norm = payload.email.strip().lower()
        row = fetch_user_by_email(cursor, email_norm)
        if not row:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if not row[6]:  # is_active
            raise HTTPException(status_code=401, detail="Account is deactivated")
        if not verify_password(payload.password, row[2]):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # update last_login
        cursor.execute("UPDATE users SET last_login = NOW() WHERE id=%s", (row[0],))
        conn.commit()

        user_dict = user_row_to_dict(row)
        token = create_access_token(user_dict['id'], user_dict['email'], user_dict['role'])
        return {
            "message": "Login successful",
            "access_token": token,
            "user": user_dict
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Login failed")
        raise HTTPException(status_code=500, detail="Login failed")
    finally:
        if 'conn' in locals():
            conn.close()


@router.get("/profile")
def get_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, email, password_hash, first_name, last_name, role, is_active, created_at, last_login FROM users WHERE id=%s", (current_user['user_id'],))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        return {"user": user_row_to_dict(row)}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Get profile failed")
        raise HTTPException(status_code=500, detail="Failed to get profile")
    finally:
        if 'conn' in locals():
            conn.close()


@router.put("/profile")
def update_profile(payload: ProfileUpdateRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        updates = {}
        if payload.first_name is not None:
            updates['first_name'] = payload.first_name.strip()
        if payload.last_name is not None:
            updates['last_name'] = payload.last_name.strip()
        if payload.email is not None:
            # Additional validation
            email_check = legacy_validate_email(payload.email)
            if not email_check['valid']:
                raise HTTPException(status_code=400, detail=email_check['message'])
            updates['email'] = payload.email.lower()

        if not updates:
            raise HTTPException(status_code=400, detail="No valid fields to update")

        conn = get_db_connection()
        cursor = conn.cursor()

        if 'email' in updates:
            cursor.execute("SELECT id FROM users WHERE email=%s AND id!=%s", (updates['email'], current_user['user_id']))
            if cursor.fetchone():
                raise HTTPException(status_code=409, detail="Email is already taken")

        set_clause = ', '.join(f"{k}=%s" for k in updates.keys())
        params = list(updates.values()) + [current_user['user_id']]
        cursor.execute(f"UPDATE users SET {set_clause}, updated_at=NOW() WHERE id=%s", params)
        conn.commit()

        cursor.execute("SELECT id, email, password_hash, first_name, last_name, role, is_active, created_at, last_login FROM users WHERE id=%s", (current_user['user_id'],))
        row = cursor.fetchone()
        return {"message": "Profile updated successfully", "user": user_row_to_dict(row)}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Update profile failed")
        raise HTTPException(status_code=500, detail="Failed to update profile")
    finally:
        if 'conn' in locals():
            conn.close()


@router.post("/change-password")
def change_password(payload: ChangePasswordRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        # Validate new password
        pwd_check = validate_password(payload.new_password)
        if not pwd_check['valid']:
            raise HTTPException(status_code=400, detail=pwd_check['message'])

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE id=%s", (current_user['user_id'],))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        if not verify_password(payload.current_password, row[0]):
            raise HTTPException(status_code=401, detail="Current password is incorrect")

        new_hash = hash_password(payload.new_password)
        cursor.execute("UPDATE users SET password_hash=%s, updated_at=NOW() WHERE id=%s", (new_hash, current_user['user_id']))
        conn.commit()
        return {"message": "Password changed successfully"}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Change password failed")
        raise HTTPException(status_code=500, detail="Failed to change password")
    finally:
        if 'conn' in locals():
            conn.close()


@router.get("/verify-token")
def verify_token(current_user: Dict[str, Any] = Depends(get_current_user)):
    return {"valid": True, "user_id": current_user['user_id'], "email": current_user.get('email'), "role": current_user.get('role')}


@router.post("/logout")
def logout():
    # Stateless JWT – client just discards token.
    return {"message": "Logout successful"}


@router.post("/set-role")
def set_role(payload: RoleUpdateRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Admin-only endpoint to change a user's role.

    This provides controlled elevation instead of arbitrary role selection at registration.
    """
    try:
        if current_user.get('role') != 'admin':
            raise HTTPException(status_code=403, detail="Admin access required")
        valid_roles = ['user', 'admin', 'investigator', 'supervisor']
        if payload.new_role not in valid_roles:
            raise HTTPException(status_code=400, detail="Invalid role")
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE id=%s", (payload.user_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="User not found")
        cur.execute("UPDATE users SET role=%s, updated_at=NOW() WHERE id=%s", (payload.new_role, payload.user_id))
        conn.commit()
        cur.execute("SELECT id, email, password_hash, first_name, last_name, role, is_active, created_at, last_login FROM users WHERE id=%s", (payload.user_id,))
        row = cur.fetchone()
        return {"message": "Role updated", "user": user_row_to_dict(row)}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Set role failed")
        raise HTTPException(status_code=500, detail="Failed to update role")
    finally:
        if 'conn' in locals():
            conn.close()


# Expose dependency for other modules
__all__ = ["router", "get_current_user"]
