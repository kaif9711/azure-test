"""Synchronize user accounts from insured_tb and associate policies.

Creates (if missing) a user row for every insured_tb entry using the insured's email,
first_name, and last_name, assigning a default password (hashed) and role 'user'.
Then updates policies.user_id to point to that user's id based on matching policy_number.

Default password for all created users: insured123  (advise immediate reset in production)

Usage:
  DATABASE_URL=... python backend/sync_users_from_insured.py

Environment Flags:
  DRY_RUN=1 -> Do not commit changes, only log intended operations.
"""
from __future__ import annotations
import os, sys, logging
from typing import Dict

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('sync_users_from_insured')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils.db import get_db_connection  # type: ignore
from utils.auth_utils import hash_password  # type: ignore

DEFAULT_PWD = 'insured123'

def sync(dry_run: bool = False) -> Dict[str,int]:
    conn = get_db_connection()
    try:
        cur = conn.cursor(dictionary=True)
        # Fetch insured rows
        cur.execute("""
            SELECT policy_number, email, first_name, last_name
            FROM insured_tb
            ORDER BY policy_number
        """)
        insured_rows = cur.fetchall()
        if not insured_rows:
            logger.warning("No insured rows found; aborting.")
            return {"insured":0, "users_created":0, "policies_updated":0}

        # Build email -> user id map
        cur.execute("SELECT id, email FROM users")
        existing_users = {r['email'].lower(): r['id'] for r in cur.fetchall()}

        pwd_hash = hash_password(DEFAULT_PWD)
        users_created = 0
        policies_updated = 0

        for ins in insured_rows:
            email = (ins['email'] or '').strip()
            if not email:
                # Skip if no email (could extend with synthetic emails)
                logger.warning(f"Skipping insured with policy {ins['policy_number']} due to missing email")
                continue
            key = email.lower()
            user_id = existing_users.get(key)
            if not user_id:
                if dry_run:
                    logger.info(f"[DRY] Would create user {email}")
                else:
                    cur.execute(
                        "INSERT INTO users (email,password_hash,first_name,last_name,role,is_active) VALUES (%s,%s,%s,%s,'user',true)",
                        (email, pwd_hash, ins['first_name'] or 'First', ins['last_name'] or 'Last')
                    )
                    user_id = cur.lastrowid
                    existing_users[key] = user_id
                    users_created += 1
                    logger.info(f"Created user {email} (id={user_id})")
            if user_id:
                # Update policies to reference this user for the matching policy_number
                if dry_run:
                    logger.info(f"[DRY] Would assign policy {ins['policy_number']} to user {user_id}")
                else:
                    cur.execute(
                        "UPDATE policies SET user_id=%s WHERE policy_number=%s",
                        (user_id, ins['policy_number'])
                    )
                    if cur.rowcount:
                        policies_updated += cur.rowcount

        if dry_run:
            logger.info("Dry run complete; rolling back.")
            conn.rollback()
        else:
            conn.commit()
        return {"insured": len(insured_rows), "users_created": users_created, "policies_updated": policies_updated}
    finally:
        conn.close()


def main():  # pragma: no cover
    dry = os.getenv('DRY_RUN','0') == '1'
    stats = sync(dry_run=dry)
    logger.info(f"Sync finished: insured={stats['insured']} users_created={stats['users_created']} policies_updated={stats['policies_updated']}")
    if dry:
        logger.info("No changes committed (dry run).")

if __name__ == '__main__':  # pragma: no cover
    main()
