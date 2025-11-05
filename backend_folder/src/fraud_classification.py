"""
Fraud Classification Module
Real-time fraud detection and risk assessment
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import joblib
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class FraudClassifier:
    """
    Production fraud detection classifier with real-time prediction capabilities
    """
    
    def __init__(self, model_path: str = None, threshold: float = 0.5):
        self.model = None
        self.model_metadata = {}
        self.threshold = threshold
        self.feature_columns = []
        self.scaler = None
        self.prediction_history = []
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load trained model and metadata
        
        Args:
            model_path: Path to saved model file
        """
        try:
            model_path = Path(model_path)
            
            # Load model
            self.model = joblib.load(model_path)
            
            # Load metadata if available
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                    self.feature_columns = self.model_metadata.get('feature_columns', [])
            
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict_fraud_probability(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud probability for a single claim
        
        Args:
            claim_data: Dictionary containing claim features
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if not self.model:
                raise ValueError("No model loaded. Please load a model first.")
            
            # Convert to DataFrame
            df = pd.DataFrame([claim_data])
            
            # Preprocess features
            X = self._preprocess_features(df)
            
            # Make prediction
            fraud_probability = self.model.predict_proba(X)[0, 1]
            is_fraud = fraud_probability > self.threshold
            risk_level = self._get_risk_level(fraud_probability)
            
            # Generate explanation
            explanation = self._generate_explanation(claim_data, fraud_probability, X)
            
            result = {
                'claim_id': claim_data.get('claim_id', 'unknown'),
                'fraud_probability': float(fraud_probability),
                'is_fraud_predicted': bool(is_fraud),
                'risk_level': risk_level,
                'threshold_used': self.threshold,
                'prediction_timestamp': datetime.now().isoformat(),
                'model_version': self.model_metadata.get('model_version', 'unknown'),
                'explanation': explanation
            }
            
            # Store prediction history
            self.prediction_history.append(result)
            
            logger.info(f"Fraud prediction completed for claim {result['claim_id']}: {fraud_probability:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in fraud prediction: {str(e)}")
            raise
    
    def batch_predict(self, claims_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict fraud for multiple claims
        
        Args:
            claims_data: List of claim dictionaries
            
        Returns:
            List of prediction results
        """
        try:
            logger.info(f"Starting batch prediction for {len(claims_data)} claims")
            results = []
            
            for claim_data in claims_data:
                try:
                    result = self.predict_fraud_probability(claim_data)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error predicting claim {claim_data.get('claim_id', 'unknown')}: {str(e)}")
                    # Add error result
                    results.append({
                        'claim_id': claim_data.get('claim_id', 'unknown'),
                        'error': str(e),
                        'prediction_timestamp': datetime.now().isoformat()
                    })
            
            logger.info(f"Batch prediction completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise
    
    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess claim features for model input
        
        Args:
            df: DataFrame with claim data
            
        Returns:
            Preprocessed features DataFrame
        """
        try:
            # Create a copy
            df_processed = df.copy()
            
            # Feature engineering (similar to training pipeline)
            if 'claim_amount' in df_processed.columns:
                df_processed['claim_amount_log'] = np.log1p(df_processed['claim_amount'])
                df_processed['is_high_value_claim'] = (
                    df_processed['claim_amount'] > 50000
                ).astype(int)
                df_processed['is_round_amount'] = (
                    df_processed['claim_amount'] % 1000 == 0
                ).astype(int)
            
            if 'incident_date' in df_processed.columns:
                df_processed['incident_date'] = pd.to_datetime(df_processed['incident_date'])
                df_processed['days_since_incident'] = (
                    datetime.now() - df_processed['incident_date']
                ).dt.days
                df_processed['incident_month'] = df_processed['incident_date'].dt.month
                df_processed['is_weekend_incident'] = (
                    df_processed['incident_date'].dt.dayofweek.isin([5, 6])
                ).astype(int)
            
            if 'incident_description' in df_processed.columns:
                df_processed['description_length'] = df_processed['incident_description'].str.len()
                df_processed['description_word_count'] = df_processed['incident_description'].str.split().str.len()
                
                # Suspicious keywords
                suspicious_keywords = ['emergency', 'urgent', 'immediate', 'critical', 'severe']
                df_processed['has_suspicious_keywords'] = df_processed['incident_description'].str.lower().str.contains(
                    '|'.join(suspicious_keywords), na=False
                ).astype(int)
            
            # Select only model features
            if self.feature_columns:
                available_features = [col for col in self.feature_columns if col in df_processed.columns]
                if len(available_features) < len(self.feature_columns):
                    logger.warning(f"Missing features: {set(self.feature_columns) - set(available_features)}")
                df_processed = df_processed[available_features]
            
            # Handle missing values
            df_processed = df_processed.fillna(0)
            
            return df_processed
            
        except Exception as e:
            logger.error(f"Error in feature preprocessing: {str(e)}")
            raise
    
    def _get_risk_level(self, probability: float) -> str:
        """
        Convert probability to risk level category
        
        Args:
            probability: Fraud probability (0-1)
            
        Returns:
            Risk level string
        """
        if probability >= 0.8:
            return 'HIGH'
        elif probability >= 0.5:
            return 'MEDIUM'
        elif probability >= 0.2:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    def _generate_explanation(self, claim_data: Dict[str, Any], probability: float, 
                            features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate explanation for fraud prediction
        
        Args:
            claim_data: Original claim data
            probability: Predicted probability
            features_df: Preprocessed features
            
        Returns:
            Dictionary with explanation details
        """
        try:
            explanation = {
                'risk_factors': [],
                'protective_factors': [],
                'key_indicators': {}
            }
            
            # High claim amount
            if claim_data.get('claim_amount', 0) > 50000:
                explanation['risk_factors'].append('High claim amount (>$50,000)')
            
            # Recent incident
            if 'incident_date' in claim_data:
                incident_date = pd.to_datetime(claim_data['incident_date'])
                days_ago = (datetime.now() - incident_date).days
                if days_ago <= 7:
                    explanation['risk_factors'].append('Very recent incident (within 7 days)')
                elif days_ago <= 30:
                    explanation['risk_factors'].append('Recent incident (within 30 days)')
            
            # Weekend incident
            if features_df.get('is_weekend_incident', [0])[0] == 1:
                explanation['risk_factors'].append('Incident occurred on weekend')
            
            # Round amount
            if features_df.get('is_round_amount', [0])[0] == 1:
                explanation['risk_factors'].append('Claim amount is suspiciously round')
            
            # Suspicious keywords
            if features_df.get('has_suspicious_keywords', [0])[0] == 1:
                explanation['risk_factors'].append('Description contains suspicious keywords')
            
            # Key indicators
            explanation['key_indicators'] = {
                'claim_amount': claim_data.get('claim_amount', 0),
                'days_since_incident': features_df.get('days_since_incident', [0])[0],
                'description_length': features_df.get('description_length', [0])[0]
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return {'error': 'Could not generate explanation'}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_loaded': self.model is not None,
            'model_type': type(self.model).__name__ if self.model else None,
            'threshold': self.threshold,
            'feature_count': len(self.feature_columns),
            'predictions_made': len(self.prediction_history),
            'metadata': self.model_metadata
        }
    
    def update_threshold(self, new_threshold: float) -> None:
        """
        Update fraud detection threshold
        
        Args:
            new_threshold: New threshold value (0-1)
        """
        if not 0 <= new_threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        self.threshold = new_threshold
        logger.info(f"Fraud detection threshold updated to {new_threshold}")
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about recent predictions
        
        Returns:
            Dictionary with prediction statistics
        """
        try:
            if not self.prediction_history:
                return {'message': 'No predictions made yet'}
            
            recent_predictions = [p for p in self.prediction_history if 'fraud_probability' in p]
            
            probabilities = [p['fraud_probability'] for p in recent_predictions]
            fraud_predictions = [p['is_fraud_predicted'] for p in recent_predictions]
            
            stats = {
                'total_predictions': len(recent_predictions),
                'fraud_predictions': sum(fraud_predictions),
                'non_fraud_predictions': len(fraud_predictions) - sum(fraud_predictions),
                'fraud_rate': sum(fraud_predictions) / len(fraud_predictions) if fraud_predictions else 0,
                'average_probability': np.mean(probabilities) if probabilities else 0,
                'probability_std': np.std(probabilities) if probabilities else 0,
                'risk_level_distribution': {}
            }
            
            # Risk level distribution
            risk_levels = [p.get('risk_level', 'UNKNOWN') for p in recent_predictions]
            for level in ['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH']:
                stats['risk_level_distribution'][level] = risk_levels.count(level)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating prediction statistics: {str(e)}")
            return {'error': str(e)}


class RealTimeFraudDetector:
    """
    Real-time fraud detection system with streaming capabilities
    """
    
    def __init__(self, classifier: FraudClassifier):
        self.classifier = classifier
        self.alert_thresholds = {
            'HIGH': 0.8,
            'MEDIUM': 0.6,
            'LOW': 0.4
        }
        self.alerts = []
    
    def process_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a claim in real-time and generate alerts if necessary
        
        Args:
            claim_data: Claim data dictionary
            
        Returns:
            Processing result with alerts
        """
        try:
            # Get fraud prediction
            prediction = self.classifier.predict_fraud_probability(claim_data)
            
            # Check for alerts
            alerts = self._check_alerts(prediction)
            
            # Add alerts to result
            prediction['alerts'] = alerts
            
            # Store alerts
            self.alerts.extend(alerts)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in real-time processing: {str(e)}")
            raise
    
    def _check_alerts(self, prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check if prediction triggers any alerts
        
        Args:
            prediction: Prediction result
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        probability = prediction['fraud_probability']
        
        # High probability alert
        if probability >= self.alert_thresholds['HIGH']:
            alerts.append({
                'type': 'HIGH_FRAUD_RISK',
                'severity': 'CRITICAL',
                'message': f'Very high fraud probability: {probability:.2%}',
                'timestamp': datetime.now().isoformat(),
                'claim_id': prediction['claim_id']
            })
        
        # Medium probability alert
        elif probability >= self.alert_thresholds['MEDIUM']:
            alerts.append({
                'type': 'MEDIUM_FRAUD_RISK',
                'severity': 'WARNING',
                'message': f'Elevated fraud probability: {probability:.2%}',
                'timestamp': datetime.now().isoformat(),
                'claim_id': prediction['claim_id']
            })
        
        return alerts
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get alerts from the last N hours
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent alerts
        """
        try:
            cutoff_time = datetime.now() - pd.Timedelta(hours=hours)
            
            recent_alerts = []
            for alert in self.alerts:
                alert_time = datetime.fromisoformat(alert['timestamp'])
                if alert_time >= cutoff_time:
                    recent_alerts.append(alert)
            
            return recent_alerts
            
        except Exception as e:
            logger.error(f"Error getting recent alerts: {str(e)}")
            return []


# Utility functions for production deployment
def create_sample_claim() -> Dict[str, Any]:
    """Create a sample claim for testing"""
    return {
        'claim_id': 'CLM-TEST-001',
        'claimant_name': 'John Doe',
        'policy_number': 'POL-123456',
        'claim_amount': 15000.00,
        'incident_date': '2024-01-15',
        'incident_type': 'Auto Accident',
        'incident_description': 'Vehicle collision on highway during rush hour',
        'incident_location': 'Highway 101, San Francisco',
        'claimant_age': 35,
        'policy_duration': 24
    }


def validate_claim_data(claim_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate claim data for required fields
    
    Args:
        claim_data: Claim data dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    required_fields = ['claim_id', 'claim_amount', 'incident_date']
    errors = []
    
    for field in required_fields:
        if field not in claim_data or claim_data[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Validate data types
    if 'claim_amount' in claim_data:
        try:
            float(claim_data['claim_amount'])
        except (ValueError, TypeError):
            errors.append("claim_amount must be a valid number")
    
    if 'incident_date' in claim_data:
        try:
            pd.to_datetime(claim_data['incident_date'])
        except:
            errors.append("incident_date must be a valid date")
    
    return len(errors) == 0, errors


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_claim = create_sample_claim()
    
    # Validate claim
    is_valid, errors = validate_claim_data(sample_claim)
    print(f"Claim validation: {is_valid}")
    if errors:
        print("Errors:", errors)
    
    # Note: In production, you would load a real trained model
    print("\nSample claim data:")
    print(json.dumps(sample_claim, indent=2))
    
    print(f"\nFraud classification module loaded successfully")
    print("To use: Initialize FraudClassifier with a trained model path")