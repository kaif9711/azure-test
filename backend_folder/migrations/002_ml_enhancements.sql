-- Migration: Add machine learning confidence scores
-- Run this after the initial schema has been created

-- Add ML confidence columns to claims table
ALTER TABLE claims 
ADD COLUMN IF NOT EXISTS ml_confidence DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS document_authenticity_score DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS text_analysis_score DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS pattern_match_score DECIMAL(5,4);

-- Create ML model metrics table
CREATE TABLE IF NOT EXISTS ml_model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    training_date TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create fraud patterns table
CREATE TABLE IF NOT EXISTS fraud_patterns (
    id SERIAL PRIMARY KEY,
    pattern_name VARCHAR(100) NOT NULL,
    pattern_type VARCHAR(50) NOT NULL,
    description TEXT,
    risk_weight DECIMAL(3,2) NOT NULL DEFAULT 0.5,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create claim pattern matches table
CREATE TABLE IF NOT EXISTS claim_pattern_matches (
    id SERIAL PRIMARY KEY,
    claim_id UUID REFERENCES claims(id) ON DELETE CASCADE,
    pattern_id INTEGER REFERENCES fraud_patterns(id),
    match_score DECIMAL(5,4) NOT NULL,
    match_details JSONB,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add indexes for new columns
CREATE INDEX IF NOT EXISTS idx_claims_ml_confidence ON claims(ml_confidence);
CREATE INDEX IF NOT EXISTS idx_claims_doc_auth_score ON claims(document_authenticity_score);
CREATE INDEX IF NOT EXISTS idx_claim_pattern_matches_claim_id ON claim_pattern_matches(claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_pattern_matches_pattern_id ON claim_pattern_matches(pattern_id);
CREATE INDEX IF NOT EXISTS idx_fraud_patterns_type ON fraud_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_fraud_patterns_active ON fraud_patterns(is_active);

-- Add trigger for fraud_patterns updated_at
DROP TRIGGER IF EXISTS update_fraud_patterns_updated_at ON fraud_patterns;
CREATE TRIGGER update_fraud_patterns_updated_at 
    BEFORE UPDATE ON fraud_patterns 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default fraud patterns
INSERT INTO fraud_patterns (pattern_name, pattern_type, description, risk_weight, is_active) VALUES
    ('Multiple Claims Short Period', 'temporal', 'User submitted multiple claims within 30 days', 0.8, true),
    ('High Value Claim', 'monetary', 'Claim amount exceeds policy coverage by significant margin', 0.7, true),
    ('Weekend Incident', 'temporal', 'Incident occurred on weekend or holiday', 0.3, true),
    ('Rushed Documentation', 'behavioral', 'Documents uploaded very quickly after incident', 0.6, true),
    ('Location Inconsistency', 'geographical', 'Incident location inconsistent with user profile', 0.8, true),
    ('Previous Rejection History', 'historical', 'User has previous rejected claims', 0.7, true),
    ('Large Claim Small Policy', 'monetary', 'Claim amount disproportionate to policy premium', 0.9, true),
    ('Missing Documentation', 'procedural', 'Required documents not provided within timeframe', 0.5, true),
    ('Suspicious Contact Pattern', 'behavioral', 'Multiple contacts regarding claim status', 0.4, true),
    ('Policy Recent Purchase', 'temporal', 'Policy purchased recently before incident', 0.8, true)
ON CONFLICT (pattern_name) DO NOTHING;

-- Insert sample ML model metrics
INSERT INTO ml_model_metrics (model_name, model_version, accuracy, precision_score, recall_score, f1_score, training_date, is_active) VALUES
    ('DocumentValidator', 'v1.0', 0.8945, 0.8721, 0.9123, 0.8916, CURRENT_TIMESTAMP - INTERVAL '30 days', true),
    ('RiskChecker', 'v1.2', 0.9234, 0.9012, 0.9445, 0.9225, CURRENT_TIMESTAMP - INTERVAL '15 days', true),
    ('SupervisorAgent', 'v1.1', 0.9567, 0.9345, 0.9789, 0.9564, CURRENT_TIMESTAMP - INTERVAL '7 days', true),
    ('DocumentValidator', 'v0.9', 0.8456, 0.8234, 0.8678, 0.8453, CURRENT_TIMESTAMP - INTERVAL '60 days', false),
    ('RiskChecker', 'v1.1', 0.9012, 0.8789, 0.9234, 0.9009, CURRENT_TIMESTAMP - INTERVAL '45 days', false)
ON CONFLICT DO NOTHING;

-- Create view for ML model performance tracking
CREATE OR REPLACE VIEW ml_model_performance AS
SELECT 
    model_name,
    model_version,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    training_date,
    is_active,
    ROW_NUMBER() OVER (PARTITION BY model_name ORDER BY training_date DESC) as version_rank
FROM ml_model_metrics
ORDER BY model_name, training_date DESC;

-- Create view for active fraud patterns
CREATE OR REPLACE VIEW active_fraud_patterns AS
SELECT 
    pattern_name,
    pattern_type,
    description,
    risk_weight,
    created_at,
    (SELECT COUNT(*) FROM claim_pattern_matches cpm WHERE cpm.pattern_id = fp.id) as match_count
FROM fraud_patterns fp
WHERE is_active = true
ORDER BY risk_weight DESC;

-- Create function to calculate total risk score
CREATE OR REPLACE FUNCTION calculate_total_risk_score(claim_id UUID)
RETURNS DECIMAL(5,4) AS $$
DECLARE
    base_risk DECIMAL(5,4) := 0.0;
    pattern_risk DECIMAL(5,4) := 0.0;
    total_risk DECIMAL(5,4) := 0.0;
BEGIN
    -- Get base risk score
    SELECT COALESCE(risk_score, 0.0) INTO base_risk 
    FROM claims WHERE id = claim_id;
    
    -- Calculate pattern-based risk
    SELECT COALESCE(SUM(cpm.match_score * fp.risk_weight), 0.0) INTO pattern_risk
    FROM claim_pattern_matches cpm
    JOIN fraud_patterns fp ON cpm.pattern_id = fp.id
    WHERE cpm.claim_id = claim_id AND fp.is_active = true;
    
    -- Combine risks (weighted average)
    total_risk := (base_risk * 0.6) + (pattern_risk * 0.4);
    
    -- Cap at 1.0
    IF total_risk > 1.0 THEN
        total_risk := 1.0;
    END IF;
    
    RETURN total_risk;
END;
$$ LANGUAGE plpgsql;

-- Update existing system settings for ML thresholds
INSERT INTO system_settings (setting_key, setting_value, description) VALUES
    ('ml_confidence_threshold', '0.8', 'Minimum ML confidence score for automated decisions'),
    ('pattern_match_threshold', '0.6', 'Threshold for flagging pattern matches'),
    ('document_auth_threshold', '0.7', 'Minimum document authenticity score'),
    ('auto_investigate_threshold', '0.85', 'Risk score threshold for automatic investigation'),
    ('ml_model_retrain_days', '30', 'Days between ML model retraining')
ON CONFLICT (setting_key) DO UPDATE SET 
    setting_value = EXCLUDED.setting_value,
    updated_at = CURRENT_TIMESTAMP;