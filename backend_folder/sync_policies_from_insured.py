"""Utility script to synchronize the policies table with policy_numbers from insured_tb.

Removes all existing rows from policies and recreates entries derived from insured_tb so that
policy_number sets match going forward.

ASSUMPTIONS / NOTES:
1. All generated policies are attached to the admin user (email=admin@example.com) if present; otherwise user_id is NULL.
2. coverage_amount & premium are heuristically derived from annual_income & credit_score_segment for demo purposes:
      coverage_amount = annual_income * factor(credit_score_segment)
      premium = coverage_amount * rate(credit_score_segment)
   Factors: High=2.5, Medium=2.0, Low=1.5 (default=2.0)
   Rates:   High=0.006, Medium=0.008, Low=0.010 (default=0.008)
   NULL annual_income is treated as 50_000 baseline.
3. start_date uses insured_tb.policy_start_date if present, else CURRENT_DATE - 90 days.
4. is_active mapped from (policy_status='Active').
5. Idempotent in the sense it fully truncates (DELETE) policies before re-inserting; run with caution in prod.

USAGE:
    Set DATABASE_URL env var then run:
        python backend/sync_policies_from_insured.py

EXIT CODES:
  0 success, non-zero on error.
"""

from __future__ import annotations
import os
import sys
import logging
from datetime import date, timedelta

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("sync_policies")

# Ensure we can import utils.db (backend/utils)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from utils.db import get_db_connection
except Exception as e:  # pragma: no cover
    logger.error(f"Failed to import DB utilities: {e}")
    raise


FACTORS = {"High": 2.5, "Medium": 2.0, "Low": 1.5}
RATES = {"High": 0.006, "Medium": 0.008, "Low": 0.010}
ANNUAL_INCOME_FALLBACK = 50_000


def derive_coverage_and_premium(annual_income, credit_segment):
    if annual_income is None:
        annual_income = ANNUAL_INCOME_FALLBACK
    factor = FACTORS.get(credit_segment, 2.0)
    rate = RATES.get(credit_segment, 0.008)
    coverage = float(annual_income) * factor
    premium = coverage * rate
    return round(coverage, 2), round(premium, 2)


def sync_policies(dry_run: bool = False) -> int:
    """Synchronize policies from insured_tb.

    Returns number of inserted policy rows.
    """
    conn = get_db_connection()
    try:
        cur = conn.cursor(dictionary=True)

        # Fetch admin user id if exists
        cur.execute("SELECT id FROM users WHERE email=%s", ('admin@example.com',))
        admin_row = cur.fetchone()
        admin_id = admin_row['id'] if admin_row else None
        if admin_id:
            logger.info(f"Using admin user id={admin_id} for all policies.")
        else:
            logger.warning("Admin user not found; policies will have NULL user_id (claims endpoint requiring user match may fail).")

        # Pull insured data
        cur.execute("""
            SELECT policy_number, annual_income, credit_score_segment, policy_start_date, policy_status
            FROM insured_tb
            ORDER BY policy_number
        """)
        rows = cur.fetchall()
        if not rows:
            logger.warning("No rows found in insured_tb; aborting.")
            return 0

        logger.info(f"Found {len(rows)} insured records to transform into policies.")

        # Prepare insert payload
        policy_records = []
        for r in rows:
            coverage, premium = derive_coverage_and_premium(r['annual_income'], r['credit_score_segment'])
            start_date = r['policy_start_date'] or (date.today() - timedelta(days=90))
            is_active = (r['policy_status'] or '').lower() == 'active'
            # Map credit segment to policy_type
            credit_segment = (r['credit_score_segment'] or 'Medium')
            if credit_segment == 'High':
                policy_type = 'premium'
            elif credit_segment == 'Low':
                policy_type = 'basic'
            else:
                policy_type = 'standard'
            policy_records.append((admin_id, r['policy_number'], policy_type, coverage, premium, start_date, is_active))

        logger.info("Preview first 3 transformed rows:\n" + "\n".join(str(p) for p in policy_records[:3]))

        if dry_run:
            logger.info("Dry run enabled; not modifying database.")
            return len(policy_records)

        # Delete existing policies and insert new
        cur.execute("DELETE FROM policies")
        cur.executemany(
            """
            INSERT INTO policies
                (user_id, policy_number, policy_type, coverage_amount, premium, start_date, is_active)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            """,
            policy_records
        )
        conn.commit()
        logger.info(f"Inserted {cur.rowcount} policies (replaced prior contents).")
        return len(policy_records)
    finally:
        conn.close()


def main():  # pragma: no cover
    dry_run = os.getenv("DRY_RUN", "0") == "1"
    count = sync_policies(dry_run=dry_run)
    logger.info(f"Sync completed. Policies now reflect insured_tb set (rows: {count}).")


if __name__ == "__main__":  # pragma: no cover
    main()
