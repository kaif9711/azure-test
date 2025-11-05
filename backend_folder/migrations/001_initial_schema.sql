-- Initial database schema for Fraudulent Claim Detection Agent
-- Run this script to set up the initial database structure

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    role VARCHAR(50) DEFAULT 'user' CHECK (role IN ('user', 'admin', 'investigator', 'supervisor')),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Policies table
CREATE TABLE IF NOT EXISTS policies (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    policy_number VARCHAR(50) UNIQUE NOT NULL,
    policy_type VARCHAR(50) NOT NULL,
    coverage_amount DECIMAL(15,2) NOT NULL,
    premium DECIMAL(10,2) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Claims table
CREATE TABLE IF NOT EXISTS claims (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    policy_id INTEGER REFERENCES policies(id),
    claim_amount DECIMAL(15,2) NOT NULL,
    incident_date DATE NOT NULL,
    incident_description TEXT NOT NULL,
    location VARCHAR(255),
    status VARCHAR(50) DEFAULT 'submitted' CHECK (status IN (
        'submitted', 'under_review', 'approved', 'rejected', 
        'investigation', 'closed'
    )),
    risk_score DECIMAL(5,4),
    document_score DECIMAL(5,4),
    supervisor_decision VARCHAR(50),
    rejection_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Claim documents table
CREATE TABLE IF NOT EXISTS claim_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    claim_id UUID REFERENCES claims(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(10) NOT NULL,
    file_size INTEGER NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    uploaded_by_user_id INTEGER REFERENCES users(id),
    validation_status VARCHAR(20) DEFAULT 'pending',
    validation_score DECIMAL(5,4),
    validation_details TEXT
);

-- Claim status history table
CREATE TABLE IF NOT EXISTS claim_status_history (
    id SERIAL PRIMARY KEY,
    claim_id UUID REFERENCES claims(id) ON DELETE CASCADE,
    status VARCHAR(50) NOT NULL,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    changed_by_user_id INTEGER REFERENCES users(id),
    notes TEXT
);

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(100),
    user_id INTEGER REFERENCES users(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT,
    details JSONB
);

-- System settings table
CREATE TABLE IF NOT EXISTS system_settings (
    id SERIAL PRIMARY KEY,
    setting_key VARCHAR(100) UNIQUE NOT NULL,
    setting_value TEXT NOT NULL,
    description TEXT,
    updated_by INTEGER REFERENCES users(id),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_claims_user_id ON claims(user_id);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);
CREATE INDEX IF NOT EXISTS idx_claims_risk_score ON claims(risk_score);
CREATE INDEX IF NOT EXISTS idx_claims_created_at ON claims(created_at);
CREATE INDEX IF NOT EXISTS idx_claim_documents_claim_id ON claim_documents(claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_status_history_claim_id ON claim_status_history(claim_id);
CREATE INDEX IF NOT EXISTS idx_policies_user_id ON policies(user_id);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_claims_description_fts ON claims USING gin(to_tsvector('english', incident_description));
CREATE INDEX IF NOT EXISTS idx_claims_location_fts ON claims USING gin(to_tsvector('english', location));

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_claims_user_status ON claims(user_id, status);
CREATE INDEX IF NOT EXISTS idx_claims_status_created ON claims(status, created_at);
CREATE INDEX IF NOT EXISTS idx_policies_user_active ON policies(user_id, is_active);

-- Triggers for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to tables
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_policies_updated_at ON policies;
CREATE TRIGGER update_policies_updated_at 
    BEFORE UPDATE ON policies 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_claims_updated_at ON claims;
CREATE TRIGGER update_claims_updated_at 
    BEFORE UPDATE ON claims 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default system settings
INSERT INTO system_settings (setting_key, setting_value, description) VALUES
    ('fraud_threshold', '0.7', 'Threshold for flagging claims as high-risk'),
    ('auto_approve_threshold', '0.3', 'Threshold for auto-approving low-risk claims'),
    ('max_file_size_mb', '16', 'Maximum file upload size in MB'),
    ('session_timeout_minutes', '30', 'User session timeout in minutes'),
    ('email_notifications_enabled', 'true', 'Enable email notifications for claim updates')
ON CONFLICT (setting_key) DO NOTHING;

-- Create default admin user (password: admin123)
-- Note: Change this password in production!
INSERT INTO users (email, password_hash, first_name, last_name, role, is_active) VALUES
    ('admin@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj1EZ5fW0O7e', 'Admin', 'User', 'admin', true)
ON CONFLICT (email) DO NOTHING;

-- Create sample data (for development only)
-- Insert sample users
INSERT INTO users (email, password_hash, first_name, last_name, role, is_active) VALUES
    ('john.doe@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj1EZ5fW0O7e', 'John', 'Doe', 'user', true),
    ('jane.smith@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj1EZ5fW0O7e', 'Jane', 'Smith', 'user', true),
    ('investigator@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj1EZ5fW0O7e', 'Investigation', 'Agent', 'investigator', true),
    ('supervisor@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj1EZ5fW0O7e', 'Super', 'Visor', 'supervisor', true)
ON CONFLICT (email) DO NOTHING;

-- Insert sample policies
INSERT INTO policies (user_id, policy_number, policy_type, coverage_amount, premium, start_date, is_active) VALUES
    (2, 'POL-001', 'comprehensive', 100000.00, 1200.00, CURRENT_DATE - INTERVAL '12 months', true),
    (3, 'POL-002', 'basic', 50000.00, 600.00, CURRENT_DATE - INTERVAL '8 months', true),
    (2, 'POL-003', 'premium', 200000.00, 2400.00, CURRENT_DATE - INTERVAL '6 months', true),
    (3, 'POL-004', 'comprehensive', 75000.00, 900.00, CURRENT_DATE - INTERVAL '18 months', true)
ON CONFLICT (policy_number) DO NOTHING;

-- Create views for common queries
CREATE OR REPLACE VIEW claim_summary AS
SELECT 
    c.id,
    c.claim_amount,
    c.status,
    c.risk_score,
    c.created_at,
    u.first_name || ' ' || u.last_name as claimant_name,
    u.email as claimant_email,
    p.policy_number,
    p.policy_type,
    COUNT(cd.id) as document_count
FROM claims c
JOIN users u ON c.user_id = u.id
JOIN policies p ON c.policy_id = p.id
LEFT JOIN claim_documents cd ON c.id = cd.claim_id
GROUP BY c.id, u.first_name, u.last_name, u.email, p.policy_number, p.policy_type;

CREATE OR REPLACE VIEW high_risk_claims AS
SELECT *
FROM claim_summary
WHERE risk_score > 0.7
ORDER BY risk_score DESC, created_at DESC;

CREATE OR REPLACE VIEW user_claim_stats AS
SELECT 
    u.id as user_id,
    u.first_name || ' ' || u.last_name as user_name,
    u.email,
    COUNT(c.id) as total_claims,
    SUM(CASE WHEN c.status = 'approved' THEN c.claim_amount ELSE 0 END) as total_approved_amount,
    AVG(c.risk_score) as average_risk_score,
    MAX(c.created_at) as last_claim_date
FROM users u
LEFT JOIN claims c ON u.id = c.user_id
WHERE u.role = 'user'
GROUP BY u.id, u.first_name, u.last_name, u.email
ORDER BY total_claims DESC;

-- Create function to calculate claim processing time
CREATE OR REPLACE FUNCTION calculate_processing_time(claim_id UUID)
RETURNS INTERVAL AS $$
DECLARE
    created_time TIMESTAMP;
    completed_time TIMESTAMP;
BEGIN
    SELECT created_at INTO created_time FROM claims WHERE id = claim_id;
    
    SELECT MAX(changed_at) INTO completed_time 
    FROM claim_status_history 
    WHERE claim_id = claim_id 
    AND status IN ('approved', 'rejected', 'closed');
    
    IF completed_time IS NULL THEN
        RETURN NULL;
    END IF;
    
    RETURN completed_time - created_time;
END;
$$ LANGUAGE plpgsql;