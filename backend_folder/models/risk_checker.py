"""
Risk Checker Module
Machine learning-based fraud risk assessment for insurance claims
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from typing import Dict, Any, List, Tuple
import os
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class RiskChecker:
    """ML-powered fraud risk assessment system"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.getenv('MODEL_PATH', './models/')
        self.fraud_threshold = float(os.getenv('FRAUD_THRESHOLD', 0.7))
        
        # Initialize models
        self.fraud_classifier = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Risk factors and weights
        self.risk_factors = {
            'claim_amount': 0.25,
            'claim_frequency': 0.20,
            'document_quality': 0.15,
            'claimant_history': 0.15,
            'temporal_patterns': 0.10,
            'location_risk': 0.10,
            'policy_details': 0.05
        }
        
        # Load pre-trained models if available
        self._load_models()
    
    def assess_claim_risk(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main risk assessment function
        
        Args:
            claim_data: Dictionary containing claim information
            
        Returns:
            Dictionary with risk assessment results
        """
        try:
            # Extract and engineer features
            features = self._extract_features(claim_data)
            
            # Get risk scores from different models
            fraud_probability = self._predict_fraud_probability(features)
            anomaly_score = self._detect_anomalies(features)
            rule_based_score = self._apply_rule_based_checks(claim_data)
            
            # Combine scores
            final_risk_score = self._combine_risk_scores(
                fraud_probability, 
                anomaly_score, 
                rule_based_score
            )
            
            # Determine risk level and recommendations
            risk_level = self._determine_risk_level(final_risk_score)
            risk_factors = self._identify_risk_factors(claim_data, features)
            recommendations = self._generate_recommendations(risk_level, risk_factors)
            
            results = {
                'risk_score': round(final_risk_score, 4),
                'risk_level': risk_level,
                'fraud_probability': round(fraud_probability, 4),
                'anomaly_score': round(anomaly_score, 4),
                'rule_based_score': round(rule_based_score, 4),
                'risk_factors': risk_factors,
                'recommendations': recommendations,
                'requires_manual_review': final_risk_score > self.fraud_threshold,
                'assessment_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Risk assessment completed: {risk_level} risk ({final_risk_score:.4f})")
            return results
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {str(e)}")
            return self._create_error_response(str(e))
    
    def _extract_features(self, claim_data: Dict[str, Any]) -> np.ndarray:
        """Extract and engineer features for ML models"""
        features = []
        
        try:
            # Claim amount features
            claim_amount = float(claim_data.get('claim_amount', 0))
            features.extend([
                claim_amount,
                np.log1p(claim_amount),  # Log transformation
                1 if claim_amount > 50000 else 0,  # High amount flag
                1 if claim_amount > 100000 else 0  # Very high amount flag
            ])
            
            # Temporal features
            claim_date = pd.to_datetime(claim_data.get('claim_date', datetime.now()))
            features.extend([
                claim_date.weekday(),  # Day of week
                claim_date.hour if hasattr(claim_date, 'hour') else 12,  # Hour of day
                claim_date.month,  # Month
                1 if claim_date.weekday() >= 5 else 0,  # Weekend flag
                1 if claim_date.hour < 6 or claim_date.hour > 22 else 0  # Off-hours flag
            ])
            
            # Claimant features
            claimant_age = int(claim_data.get('claimant_age', 35))
            features.extend([
                claimant_age,
                1 if claimant_age < 25 or claimant_age > 65 else 0,  # Age risk flag
                int(claim_data.get('previous_claims', 0)),
                int(claim_data.get('policy_duration_months', 12))
            ])
            
            # Location features
            location_risk = self._calculate_location_risk(claim_data.get('location'))
            features.append(location_risk)
            
            # Policy features
            policy_type = claim_data.get('policy_type', 'standard')
            policy_premium = float(claim_data.get('policy_premium', 1000))
            features.extend([
                1 if policy_type == 'premium' else 0,
                policy_premium,
                claim_amount / max(policy_premium, 1)  # Claim to premium ratio
            ])
            
            # Document quality features
            doc_score = float(claim_data.get('document_confidence_score', 1.0))
            features.extend([
                doc_score,
                1 if doc_score < 0.7 else 0,  # Low quality flag
                int(claim_data.get('documents_count', 1))
            ])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            # Return zero features if extraction fails
            return np.zeros((1, 20))
    
    def _predict_fraud_probability(self, features: np.ndarray) -> float:
        """Predict fraud probability using trained classifier"""
        try:
            if self.fraud_classifier is None:
                # Use rule-based approach if no trained model
                return self._rule_based_fraud_probability(features)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get probability
            prob = self.fraud_classifier.predict_proba(features_scaled)[0][1]
            return float(prob)
            
        except Exception as e:
            logger.warning(f"Error predicting fraud probability: {str(e)}")
            return 0.5  # Default neutral probability
    
    def _detect_anomalies(self, features: np.ndarray) -> float:
        """Detect anomalies in claim features"""
        try:
            if self.anomaly_detector is None:
                return self._simple_anomaly_detection(features)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get anomaly score (convert to 0-1 range)
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            normalized_score = max(0, min(1, (1 - anomaly_score) / 2))
            return float(normalized_score)
            
        except Exception as e:
            logger.warning(f"Error detecting anomalies: {str(e)}")
            return 0.3  # Default low anomaly score
    
    def _apply_rule_based_checks(self, claim_data: Dict[str, Any]) -> float:
        """Apply rule-based fraud detection logic"""
        risk_score = 0.0
        
        try:
            # High amount claims
            claim_amount = float(claim_data.get('claim_amount', 0))
            if claim_amount > 100000:
                risk_score += 0.3
            elif claim_amount > 50000:
                risk_score += 0.2
            
            # Frequent claims
            previous_claims = int(claim_data.get('previous_claims', 0))
            if previous_claims > 5:
                risk_score += 0.25
            elif previous_claims > 2:
                risk_score += 0.15
            
            # Recent policy
            policy_duration = int(claim_data.get('policy_duration_months', 12))
            if policy_duration < 3:
                risk_score += 0.2
            elif policy_duration < 6:
                risk_score += 0.1
            
            # Weekend/holiday claims
            claim_date = pd.to_datetime(claim_data.get('claim_date', datetime.now()))
            if claim_date.weekday() >= 5:  # Weekend
                risk_score += 0.1
            
            # Document quality issues
            doc_score = float(claim_data.get('document_confidence_score', 1.0))
            if doc_score < 0.5:
                risk_score += 0.3
            elif doc_score < 0.7:
                risk_score += 0.15
            
            # Multiple claims in short period
            if claim_data.get('claims_last_30_days', 0) > 1:
                risk_score += 0.2
            
            return min(1.0, risk_score)
            
        except Exception as e:
            logger.warning(f"Error in rule-based checks: {str(e)}")
            return 0.3
    
    def _combine_risk_scores(self, fraud_prob: float, anomaly_score: float, rule_score: float) -> float:
        """Combine different risk scores into final score"""
        # Weighted combination
        weights = {
            'fraud_probability': 0.5,
            'anomaly_score': 0.3,
            'rule_based': 0.2
        }
        
        combined_score = (
            weights['fraud_probability'] * fraud_prob +
            weights['anomaly_score'] * anomaly_score +
            weights['rule_based'] * rule_score
        )
        
        return min(1.0, max(0.0, combined_score))
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score"""
        if risk_score >= 0.8:
            return 'very_high'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        elif risk_score >= 0.2:
            return 'low'
        else:
            return 'very_low'
    
    def _identify_risk_factors(self, claim_data: Dict[str, Any], features: np.ndarray) -> List[str]:
        """Identify specific risk factors in the claim"""
        risk_factors = []
        
        # High claim amount
        claim_amount = float(claim_data.get('claim_amount', 0))
        if claim_amount > 50000:
            risk_factors.append('High claim amount')
        
        # Frequent claims
        if int(claim_data.get('previous_claims', 0)) > 2:
            risk_factors.append('Multiple previous claims')
        
        # New policy
        if int(claim_data.get('policy_duration_months', 12)) < 6:
            risk_factors.append('Recent policy inception')
        
        # Document quality
        if float(claim_data.get('document_confidence_score', 1.0)) < 0.7:
            risk_factors.append('Poor document quality')
        
        # Temporal patterns
        claim_date = pd.to_datetime(claim_data.get('claim_date', datetime.now()))
        if claim_date.weekday() >= 5:
            risk_factors.append('Weekend claim')
        
        # Young or elderly claimant
        age = int(claim_data.get('claimant_age', 35))
        if age < 25 or age > 65:
            risk_factors.append('Age-related risk factor')
        
        return risk_factors
    
    def _generate_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Generate recommendations based on risk assessment"""
        recommendations = []
        
        if risk_level in ['very_high', 'high']:
            recommendations.append('Requires immediate manual review')
            recommendations.append('Consider special investigation unit (SIU) referral')
            recommendations.append('Verify all supporting documentation')
            recommendations.append('Contact claimant for additional information')
        elif risk_level == 'medium':
            recommendations.append('Enhanced review recommended')
            recommendations.append('Verify key documents and information')
            recommendations.append('Consider additional documentation requests')
        else:
            recommendations.append('Standard processing can proceed')
            recommendations.append('Routine verification sufficient')
        
        # Specific recommendations based on risk factors
        if 'Poor document quality' in risk_factors:
            recommendations.append('Request higher quality documentation')
        
        if 'Multiple previous claims' in risk_factors:
            recommendations.append('Review claim history pattern')
        
        if 'High claim amount' in risk_factors:
            recommendations.append('Verify claim amount justification')
        
        return list(set(recommendations))  # Remove duplicates
    
    def _calculate_location_risk(self, location: str) -> float:
        """Calculate risk based on location (simplified implementation)"""
        if not location:
            return 0.3
        
        # High-risk locations (this would be based on historical data)
        high_risk_areas = ['downtown', 'industrial', 'high-crime']
        location_lower = location.lower()
        
        for area in high_risk_areas:
            if area in location_lower:
                return 0.8
        
        return 0.2  # Default low risk
    
    def _rule_based_fraud_probability(self, features: np.ndarray) -> float:
        """Simple rule-based fraud probability when no ML model is available"""
        try:
            # Extract key features
            claim_amount = features[0][0] if features.shape[1] > 0 else 0
            previous_claims = features[0][10] if features.shape[1] > 10 else 0
            
            prob = 0.1  # Base probability
            
            # High amount increases probability
            if claim_amount > 50000:
                prob += 0.3
            elif claim_amount > 25000:
                prob += 0.2
            
            # Multiple claims increase probability
            if previous_claims > 3:
                prob += 0.3
            elif previous_claims > 1:
                prob += 0.15
            
            return min(1.0, prob)
            
        except Exception:
            return 0.3
    
    def _simple_anomaly_detection(self, features: np.ndarray) -> float:
        """Simple anomaly detection when no trained model is available"""
        try:
            # Calculate z-scores for key features
            if features.shape[1] < 3:
                return 0.2
            
            claim_amount = features[0][0]
            age = features[0][5]
            previous_claims = features[0][10]
            
            # Simple thresholds
            anomaly_score = 0.0
            
            if claim_amount > 100000 or claim_amount < 100:
                anomaly_score += 0.3
            
            if age < 20 or age > 80:
                anomaly_score += 0.2
            
            if previous_claims > 5:
                anomaly_score += 0.3
            
            return min(1.0, anomaly_score)
            
        except Exception:
            return 0.3
    
    def _load_models(self):
        """Load pre-trained ML models"""
        try:
            model_files = {
                'fraud_classifier': 'fraud_classifier.joblib',
                'anomaly_detector': 'anomaly_detector.joblib',
                'scaler': 'scaler.joblib'
            }
            
            for model_name, filename in model_files.items():
                filepath = os.path.join(self.model_path, filename)
                if os.path.exists(filepath):
                    setattr(self, model_name, joblib.load(filepath))
                    logger.info(f"Loaded {model_name} from {filepath}")
                else:
                    logger.info(f"Model file {filename} not found, using fallback methods")
                    
        except Exception as e:
            logger.warning(f"Error loading models: {str(e)}")
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response for failed risk assessment"""
        return {
            'risk_score': 0.5,
            'risk_level': 'medium',
            'fraud_probability': 0.5,
            'anomaly_score': 0.5,
            'rule_based_score': 0.5,
            'risk_factors': ['Assessment error occurred'],
            'recommendations': ['Manual review required due to assessment error'],
            'requires_manual_review': True,
            'error': error_message,
            'assessment_timestamp': datetime.now().isoformat()
        }
    
    def train_model(self, training_data: pd.DataFrame):
        """Train the fraud detection models (for future implementation)"""
        # This would be implemented for model training
        # Currently using rule-based and simple heuristics
        pass