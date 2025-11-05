"""User management utility.

Allows creating or updating a user directly in the database for development / admin tasks
without going through the /auth/register endpoint (useful when frontend registration
is blocked or when seeding a temporary account).

Example:
    python backend/manage_users.py \
        --email test@test.com \
        --password Test2025 \
        --first-name Test \
        --last-name User \
        --role user

Requires the DATABASE_URL environment variable to be set, e.g.:
    export DATABASE_URL=mysql://fraud_user:fraud_password@localhost:3306/fraud_claims

Supported roles: user, admin, investigator, supervisor
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from utils.db import get_db_connection
from utils.auth_utils import hash_password

VALID_ROLES = {"user", "admin", "investigator", "supervisor"}


def ensure_db_url():
    if not os.getenv("DATABASE_URL"):
        print("[ERROR] DATABASE_URL environment variable is not set.")
        print("Set it first, e.g. mysql://fraud_user:fraud_password@localhost:3306/fraud_claims")
        sys.exit(1)


def create_or_update_user(
    email: str,
    password: str,
    first_name: str,
    last_name: str,
    role: str,
    reset_password: bool = False,
) -> str:
    """Create a user if not exists; optionally reset password if exists.

    Returns a status message.
    """
    ensure_db_url()
    email_norm = email.strip().lower()
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role '{role}'. Must be one of: {', '.join(VALID_ROLES)}")

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email = %s", (email_norm,))
        row = cur.fetchone()

        if row:
            user_id = row["id"] if isinstance(row, dict) else row[0]
            if reset_password:
                pw_hash = hash_password(password)
                cur.execute(
                    "UPDATE users SET password_hash = %s, first_name = %s, last_name = %s, role = %s, updated_at = NOW() WHERE id = %s",
                    (pw_hash, first_name, last_name, role, user_id),
                )
                conn.commit()
                return f"Updated existing user {email_norm} (id={user_id}) and reset password."
            return f"User {email_norm} already exists (id={user_id}). Use --reset-password to update."
        else:
            pw_hash = hash_password(password)
            cur.execute(
                "INSERT INTO users (email, password_hash, first_name, last_name, role, is_active) VALUES (%s, %s, %s, %s, %s, true)",
                (email_norm, pw_hash, first_name, last_name, role),
            )
            new_id = cur.lastrowid
            conn.commit()
            return f"Created user {email_norm} (id={new_id})."
    finally:
        conn.close()


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Manage users (create/update) for Fraud Detection app")
    parser.add_argument("--email", required=True, help="User email")
    parser.add_argument("--password", required=True, help="Plaintext password to set")
    parser.add_argument("--first-name", default="Temp", help="First name")
    parser.add_argument("--last-name", default="User", help="Last name")
    parser.add_argument("--role", default="user", choices=sorted(VALID_ROLES), help="User role")
    parser.add_argument(
        "--reset-password",
        action="store_true",
        help="If user exists, reset password & update name/role",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None):
    args = parse_args(argv)
    try:
        msg = create_or_update_user(
            email=args.email,
            password=args.password,
            first_name=args.first_name,
            last_name=args.last_name,
            role=args.role,
            reset_password=args.reset_password,
        )
        print(msg)
        print("Done.")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
