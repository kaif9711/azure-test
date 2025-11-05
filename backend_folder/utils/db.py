"""MySQL Database utilities"""

import logging
import os
from contextlib import contextmanager
import mysql.connector

logger = logging.getLogger(__name__)

def parse_mysql_url(url: str):
    # mysql://user:pass@host:3306/db
    if not url.startswith("mysql://"):
        raise ValueError("Invalid MySQL URL")
    without_scheme = url[len("mysql://"):]
    creds, hostpart = without_scheme.split('@', 1)
    user, password = creds.split(':', 1)
    hostport, db = hostpart.split('/', 1)
    if ':' in hostport:
        host, port = hostport.split(':', 1)
        port = int(port)
    else:
        host, port = hostport, 3306
    return {
        'user': user,
        'password': password,
        'host': host,
        'port': port,
        'db': db
    }

def get_db_connection():
    try:
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError("DATABASE_URL not set")
        cfg = parse_mysql_url(database_url)
        conn = mysql.connector.connect(
            host=cfg['host'], user=cfg['user'], password=cfg['password'], database=cfg['db'], port=cfg['port']
        )
        return conn
    except Exception as e:
        logger.error(f"MySQL connection error: {e}")
        raise

@contextmanager
def get_db_cursor():
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"DB operation error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def init_db(app=None):
    force = os.getenv('FORCE_SCHEMA_INIT', '0') == '1'
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        lock_acquired = True
        if not force:
            # Acquire advisory lock to prevent race between multiple workers
            try:
                cursor.execute("SELECT GET_LOCK('fraud_schema_init', 15)")
                lock_acquired = cursor.fetchone()[0] == 1
            except Exception as lock_err:
                logger.warning(f"Could not acquire schema init lock: {lock_err}")
                lock_acquired = False

        if not lock_acquired and not force:
            # Another worker likely doing initialization; wait briefly to ensure tables appear
            import time
            logger.info("Schema init lock not acquired; waiting for existing initializer...")
            for _ in range(10):  # wait up to ~5s
                time.sleep(0.5)
                try:
                    cursor.execute("SHOW TABLES LIKE 'users'")
                    if cursor.fetchone():
                        logger.info("Detected initialized schema by another worker.")
                        return
                except Exception:
                    pass
            logger.warning("Schema tables not detected after waiting; proceeding without lock to initialize.")
            # Re-attempt to acquire forcefully (set force flag locally)
            force = True
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            first_name VARCHAR(100) NOT NULL,
            last_name VARCHAR(100) NOT NULL,
            role VARCHAR(50) DEFAULT 'user',
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            last_login TIMESTAMP NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS policies (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            policy_number VARCHAR(50) UNIQUE NOT NULL,
            policy_type VARCHAR(50) NOT NULL,
            coverage_amount DECIMAL(15,2) NOT NULL,
            premium DECIMAL(10,2) NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NULL,
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            CONSTRAINT fk_policy_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS claims (
            id CHAR(36) PRIMARY KEY,
            user_id INT,
            policy_id INT,
            claim_amount DECIMAL(15,2) NOT NULL,
            incident_date DATE NOT NULL,
            incident_description TEXT NOT NULL,
            location VARCHAR(255),
            status VARCHAR(50) DEFAULT 'submitted',
            risk_score DECIMAL(5,4),
            document_score DECIMAL(5,4),
            supervisor_decision VARCHAR(50),
            rejection_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            CONSTRAINT fk_claim_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            CONSTRAINT fk_claim_policy FOREIGN KEY (policy_id) REFERENCES policies(id) ON DELETE SET NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS claim_documents (
            id CHAR(36) PRIMARY KEY,
            claim_id CHAR(36),
            filename VARCHAR(255) NOT NULL,
            original_filename VARCHAR(255) NOT NULL,
            file_type VARCHAR(10) NOT NULL,
            file_size INT NOT NULL,
            file_path VARCHAR(500) NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            uploaded_by_user_id INT,
            validation_status VARCHAR(20) DEFAULT 'pending',
            validation_score DECIMAL(5,4),
            validation_details TEXT,
            CONSTRAINT fk_doc_claim FOREIGN KEY (claim_id) REFERENCES claims(id) ON DELETE CASCADE,
            CONSTRAINT fk_doc_user FOREIGN KEY (uploaded_by_user_id) REFERENCES users(id) ON DELETE SET NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS claim_status_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            claim_id CHAR(36),
            status VARCHAR(50) NOT NULL,
            changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            changed_by_user_id INT,
            notes TEXT,
            CONSTRAINT fk_hist_claim FOREIGN KEY (claim_id) REFERENCES claims(id) ON DELETE CASCADE,
            CONSTRAINT fk_hist_user FOREIGN KEY (changed_by_user_id) REFERENCES users(id) ON DELETE SET NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        # Additional tables migrated from legacy Postgres schema (MySQL-adapted)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            action VARCHAR(100) NOT NULL,
            resource_type VARCHAR(50) NOT NULL,
            resource_id VARCHAR(100),
            user_id INT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ip_address VARCHAR(45),
            user_agent TEXT,
            details JSON,
            CONSTRAINT fk_audit_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_settings (
            id INT AUTO_INCREMENT PRIMARY KEY,
            setting_key VARCHAR(100) UNIQUE NOT NULL,
            setting_value TEXT NOT NULL,
            description TEXT,
            updated_by INT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            CONSTRAINT fk_setting_user FOREIGN KEY (updated_by) REFERENCES users(id) ON DELETE SET NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ml_model_metrics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            accuracy DECIMAL(5,4),
            precision_score DECIMAL(5,4),
            recall_score DECIMAL(5,4),
            f1_score DECIMAL(5,4),
            training_date TIMESTAMP NOT NULL,
            is_active BOOLEAN DEFAULT false,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS fraud_patterns (
            id INT AUTO_INCREMENT PRIMARY KEY,
            pattern_name VARCHAR(100) NOT NULL,
            pattern_type VARCHAR(50) NOT NULL,
            description TEXT,
            risk_weight DECIMAL(3,2) NOT NULL DEFAULT 0.50,
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uniq_pattern_name (pattern_name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS claim_pattern_matches (
            id INT AUTO_INCREMENT PRIMARY KEY,
            claim_id CHAR(36) NOT NULL,
            pattern_id INT NOT NULL,
            match_score DECIMAL(5,4) NOT NULL,
            match_details JSON,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_cpm_claim FOREIGN KEY (claim_id) REFERENCES claims(id) ON DELETE CASCADE,
            CONSTRAINT fk_cpm_pattern FOREIGN KEY (pattern_id) REFERENCES fraud_patterns(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        # Schema versioning table (simple single-row implementation for future migrations)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            id INT PRIMARY KEY AUTO_INCREMENT,
            version INT NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        # Ensure at least one version row exists
        cursor.execute("SELECT COUNT(*) FROM schema_version")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO schema_version (version) VALUES (1)")

        # Indexes - tolerant to duplicates
        def create_index(sql: str):
            try:
                cursor.execute(sql)
            except mysql.connector.Error as e:  # type: ignore
                # 1061 duplicate key name
                if getattr(e, 'errno', None) == 1061:
                    logger.debug(f"Index already exists: {sql}")
                else:
                    raise

        create_index("CREATE INDEX idx_claims_user_id ON claims(user_id)")
        create_index("CREATE INDEX idx_claims_status ON claims(status)")
        create_index("CREATE INDEX idx_claims_created_at ON claims(created_at)")
        create_index("CREATE INDEX idx_users_email ON users(email)")
        # Additional indexes for new tables
        create_index("CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id)")
        create_index("CREATE INDEX idx_audit_logs_action ON audit_logs(action)")
        create_index("CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp)")
        create_index("CREATE INDEX idx_system_settings_key ON system_settings(setting_key)")
        create_index("CREATE INDEX idx_ml_model_metrics_active ON ml_model_metrics(is_active)")
        create_index("CREATE INDEX idx_fraud_patterns_active ON fraud_patterns(is_active)")
        create_index("CREATE INDEX idx_claim_pattern_matches_claim_id ON claim_pattern_matches(claim_id)")
        create_index("CREATE INDEX idx_claim_pattern_matches_pattern_id ON claim_pattern_matches(pattern_id)")

        # Default admin
        cursor.execute("SELECT COUNT(*) FROM users WHERE role='admin'")
        if cursor.fetchone()[0] == 0:
            from utils.auth_utils import hash_password
            pwd = hash_password('admin123')
            cursor.execute("INSERT INTO users (email, password_hash, first_name, last_name, role, is_active) VALUES (%s,%s,%s,%s,%s, true)",
                           ('admin@example.com', pwd, 'Admin', 'User', 'admin'))

        # Seed default system settings (idempotent)
        cursor.executemany(
            "INSERT IGNORE INTO system_settings (setting_key, setting_value, description) VALUES (%s,%s,%s)",
            [
                ('fraud_threshold', '0.7', 'Threshold for flagging claims as high-risk'),
                ('auto_approve_threshold', '0.3', 'Threshold for auto-approving low-risk claims'),
                ('max_file_size_mb', '16', 'Maximum file upload size in MB'),
                ('session_timeout_minutes', '30', 'User session timeout in minutes'),
                ('email_notifications_enabled', 'true', 'Enable email notifications for claim updates'),
                ('ml_confidence_threshold', '0.8', 'Minimum ML confidence score for automated decisions'),
                ('pattern_match_threshold', '0.6', 'Threshold for flagging pattern matches'),
                ('document_auth_threshold', '0.7', 'Minimum document authenticity score'),
                ('auto_investigate_threshold', '0.85', 'Risk score threshold for automatic investigation'),
                ('ml_model_retrain_days', '30', 'Days between ML model retraining')
            ]
        )

        # Seed fraud patterns (idempotent)
        cursor.executemany(
            "INSERT IGNORE INTO fraud_patterns (pattern_name, pattern_type, description, risk_weight, is_active) VALUES (%s,%s,%s,%s,true)",
            [
                ('Multiple Claims Short Period', 'temporal', 'User submitted multiple claims within 30 days', 0.80),
                ('High Value Claim', 'monetary', 'Claim amount exceeds policy coverage by significant margin', 0.70),
                ('Weekend Incident', 'temporal', 'Incident occurred on weekend or holiday', 0.30),
                ('Rushed Documentation', 'behavioral', 'Documents uploaded very quickly after incident', 0.60),
                ('Location Inconsistency', 'geographical', 'Incident location inconsistent with user profile', 0.80),
                ('Previous Rejection History', 'historical', 'User has previous rejected claims', 0.70),
                ('Large Claim Small Policy', 'monetary', 'Claim amount disproportionate to policy premium', 0.90),
                ('Missing Documentation', 'procedural', 'Required documents not provided within timeframe', 0.50),
                ('Suspicious Contact Pattern', 'behavioral', 'Multiple contacts regarding claim status', 0.40),
                ('Policy Recent Purchase', 'temporal', 'Policy purchased recently before incident', 0.80)
            ]
        )

        # Seed sample policies if none exist (dev convenience)
        cursor.execute("SELECT COUNT(*) FROM policies")
        if cursor.fetchone()[0] == 0:
            cursor.execute("SELECT id FROM users WHERE email=%s", ('admin@example.com',))
            admin_row = cursor.fetchone()
            admin_id = admin_row[0] if admin_row else None
            # If no admin yet seeded (edge case), pick any user
            if not admin_id:
                cursor.execute("SELECT id FROM users ORDER BY id LIMIT 1")
                row = cursor.fetchone()
                admin_id = row[0] if row else None
            if admin_id:
                cursor.executemany(
                    "INSERT IGNORE INTO policies (user_id, policy_number, policy_type, coverage_amount, premium, start_date, is_active) VALUES (%s,%s,%s,%s,%s,CURDATE() - INTERVAL 90 DAY,true)",
                    [
                        (admin_id, 'POL-ADMIN-001', 'comprehensive', 150000.00, 1200.00),
                        (admin_id, 'POL-ADMIN-002', 'basic', 50000.00, 600.00),
                        (admin_id, 'POL-ADMIN-003', 'premium', 250000.00, 1800.00)
                    ]
                )
                logger.info("Seeded sample policies for admin user")
            else:
                logger.warning("No users found to attach sample policies during seeding")
        conn.commit()
        logger.info("MySQL schema ensured")
        # Release advisory lock if we acquired it and not forced bypass
        if lock_acquired and not force:
            try:
                cursor.execute("SELECT RELEASE_LOCK('fraud_schema_init')")
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Init DB failed: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

def test_connection():
    try:
        with get_db_cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone()[0] == 1
    except Exception:
        return False

def get_table_info():
    info = {}
    try:
        with get_db_cursor() as cur:
            cur.execute("SHOW TABLES")
            tables = [row[0] for row in cur.fetchall()]
            for t in tables:
                cur.execute(f"SHOW COLUMNS FROM {t}")
                cols = cur.fetchall()
                info[t] = [c[0] for c in cols]
    except Exception as e:
        logger.error(f"get_table_info error: {e}")
    return info