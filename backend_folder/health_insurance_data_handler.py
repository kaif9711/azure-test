"""
Health Insurance Dataset Integration Module
==========================================

This module handles the integration of health insurance datasets
for fraud detection training and prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthInsuranceDataHandler:
    """
    Handles health insurance dataset operations including:
    - Data loading and validation
    - Data preprocessing for fraud detection
    - Feature engineering specific to health insurance
    - Integration with existing ML pipeline
    """
    
    def __init__(self, data_path: str = "data/health_insurance_dataset.csv"):
        self.data_path = Path(data_path)
        self.raw_data = None
        self.processed_data = None
        self.feature_columns = []
        self.target_column = "is_fraud"
        
    def load_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load health insurance dataset from various formats
        
        Args:
            file_path: Optional custom path to dataset
            
        Returns:
            DataFrame containing the loaded dataset
        """
        try:
            path = Path(file_path) if file_path else self.data_path
            
            if not path.exists():
                logger.error(f"Dataset file not found: {path}")
                raise FileNotFoundError(f"Dataset file not found: {path}")
            
            # Determine file type and load accordingly
            if path.suffix.lower() == '.csv':
                self.raw_data = pd.read_csv(path)
            elif path.suffix.lower() == '.json':
                self.raw_data = pd.read_json(path)
            elif path.suffix.lower() in ['.xlsx', '.xls']:
                self.raw_data = pd.read_excel(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
            
            logger.info(f"Dataset loaded successfully: {self.raw_data.shape[0]} rows, {self.raw_data.shape[1]} columns")
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def validate_dataset(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Validate the dataset structure and content
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            "is_valid": True,
            "issues": [],
            "statistics": {},
            "recommendations": []
        }
        
        try:
            # Basic structure validation
            if data.empty:
                validation_results["is_valid"] = False
                validation_results["issues"].append("Dataset is empty")
                return validation_results
            
            # Check for required columns (common health insurance fields)
            recommended_columns = [
                'claim_id', 'patient_age', 'claim_amount', 'diagnosis_code',
                'provider_id', 'incident_date', 'is_fraud'
            ]
            
            missing_columns = [col for col in recommended_columns if col not in data.columns]
            if missing_columns:
                validation_results["recommendations"].append(
                    f"Consider adding these columns if available: {missing_columns}"
                )
            
            # Check data quality
            validation_results["statistics"] = {
                "total_rows": len(data),
                "total_columns": len(data.columns),
                "missing_values": data.isnull().sum().to_dict(),
                "duplicate_rows": data.duplicated().sum(),
                "data_types": data.dtypes.to_dict()
            }
            
            # Check for fraud label
            fraud_columns = [col for col in data.columns if 'fraud' in col.lower()]
            if not fraud_columns:
                validation_results["issues"].append("No fraud label column found")
                validation_results["recommendations"].append(
                    "Please ensure you have a column indicating fraudulent claims"
                )
            
            # Check data balance
            if fraud_columns:
                fraud_col = fraud_columns[0]
                fraud_distribution = data[fraud_col].value_counts()
                validation_results["statistics"]["fraud_distribution"] = fraud_distribution.to_dict()
                
                # Check for severe imbalance
                fraud_rate = fraud_distribution.get(True, 0) / len(data)
                if fraud_rate < 0.01 or fraud_rate > 0.9:
                    validation_results["recommendations"].append(
                        f"Dataset appears imbalanced (fraud rate: {fraud_rate:.2%}). "
                        "Consider using techniques like SMOTE for balancing."
                    )
            
            logger.info("Dataset validation completed successfully")
            return validation_results
            
        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["issues"].append(f"Validation error: {str(e)}")
            logger.error(f"Dataset validation failed: {str(e)}")
            return validation_results
    
    def preprocess_for_fraud_detection(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess health insurance data for fraud detection
        
        Args:
            data: Raw dataset DataFrame
            
        Returns:
            Preprocessed DataFrame ready for ML training
        """
        try:
            processed_data = data.copy()
            
            # Handle missing values
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            categorical_columns = processed_data.select_dtypes(include=['object']).columns
            
            # Fill numeric missing values with median
            for col in numeric_columns:
                if processed_data[col].isnull().any():
                    processed_data[col].fillna(processed_data[col].median(), inplace=True)
            
            # Fill categorical missing values with mode
            for col in categorical_columns:
                if processed_data[col].isnull().any():
                    processed_data[col].fillna(processed_data[col].mode()[0], inplace=True)
            
            # Feature engineering for health insurance fraud detection
            processed_data = self._create_health_insurance_features(processed_data)
            
            # Remove or encode categorical variables
            processed_data = self._encode_categorical_variables(processed_data)
            
            # Remove duplicates
            processed_data = processed_data.drop_duplicates()
            
            self.processed_data = processed_data
            logger.info(f"Data preprocessing completed: {processed_data.shape[0]} rows, {processed_data.shape[1]} columns")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def _create_health_insurance_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create health insurance specific features for fraud detection
        """
        try:
            # Age-based features
            if 'patient_age' in data.columns:
                data['age_group'] = pd.cut(data['patient_age'], 
                                         bins=[0, 25, 45, 65, 100], 
                                         labels=['young', 'adult', 'middle_aged', 'senior'])
                data['is_senior'] = (data['patient_age'] >= 65).astype(int)
            
            # Claim amount features
            if 'claim_amount' in data.columns:
                data['claim_amount_log'] = np.log1p(data['claim_amount'])
                data['is_high_value_claim'] = (data['claim_amount'] > data['claim_amount'].quantile(0.9)).astype(int)
            
            # Date-based features
            date_columns = [col for col in data.columns if 'date' in col.lower()]
            for date_col in date_columns:
                try:
                    data[date_col] = pd.to_datetime(data[date_col])
                    data[f'{date_col}_month'] = data[date_col].dt.month
                    data[f'{date_col}_dayofweek'] = data[date_col].dt.dayofweek
                    data[f'{date_col}_is_weekend'] = (data[date_col].dt.dayofweek >= 5).astype(int)
                except:
                    logger.warning(f"Could not parse date column: {date_col}")
            
            # Provider-based features (if available)
            if 'provider_id' in data.columns:
                provider_stats = data.groupby('provider_id').agg({
                    'claim_amount': ['count', 'mean', 'sum']
                }).round(2)
                provider_stats.columns = ['provider_claim_count', 'provider_avg_claim', 'provider_total_claims']
                data = data.merge(provider_stats, left_on='provider_id', right_index=True, how='left')
            
            # Diagnosis-based features (if available)
            if 'diagnosis_code' in data.columns:
                # Create binary features for common diagnosis categories
                common_diagnoses = data['diagnosis_code'].value_counts().head(10).index
                for diagnosis in common_diagnoses:
                    data[f'diagnosis_{diagnosis}'] = (data['diagnosis_code'] == diagnosis).astype(int)
            
            return data
            
        except Exception as e:
            logger.error(f"Error creating health insurance features: {str(e)}")
            return data
    
    def _encode_categorical_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables for machine learning
        """
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        
        try:
            categorical_columns = data.select_dtypes(include=['object']).columns
            categorical_columns = [col for col in categorical_columns if col != self.target_column]
            
            for col in categorical_columns:
                if data[col].nunique() <= 10:  # One-hot encode low cardinality
                    dummies = pd.get_dummies(data[col], prefix=col)
                    data = pd.concat([data, dummies], axis=1)
                    data.drop(col, axis=1, inplace=True)
                else:  # Label encode high cardinality
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col].astype(str))
            
            return data
            
        except Exception as e:
            logger.error(f"Error encoding categorical variables: {str(e)}")
            return data
    
    def generate_sample_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """
        Generate sample health insurance data for testing
        
        Args:
            num_samples: Number of sample records to generate
            
        Returns:
            DataFrame with sample health insurance claims data
        """
        np.random.seed(42)
        
        # Generate sample data
        data = {
            'claim_id': [f'CLM{str(i).zfill(6)}' for i in range(1, num_samples + 1)],
            'patient_age': np.random.randint(18, 90, num_samples),
            'claim_amount': np.random.lognormal(7, 1, num_samples).round(2),
            'diagnosis_code': np.random.choice(['A123', 'B456', 'C789', 'D012', 'E345', 'F678'], num_samples),
            'provider_id': np.random.choice([f'PROV{i:03d}' for i in range(1, 51)], num_samples),
            'incident_date': pd.date_range(start='2023-01-01', end='2024-12-31', periods=num_samples),
        }
        
        # Create fraud labels (10% fraud rate)
        fraud_probability = 0.1
        data['is_fraud'] = np.random.choice([True, False], num_samples, p=[fraud_probability, 1-fraud_probability])
        
        # Make fraud cases more suspicious
        fraud_mask = data['is_fraud']
        data['claim_amount'] = np.where(fraud_mask, 
                                      data['claim_amount'] * np.random.uniform(1.5, 3.0, num_samples),
                                      data['claim_amount'])
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {num_samples} sample health insurance records")
        
        return df
    
    def save_processed_data(self, data: pd.DataFrame, output_path: str = "data/processed_health_insurance_data.csv"):
        """
        Save processed data to file
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            data.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of the dataset
        """
        if self.processed_data is None:
            return {"error": "No processed data available"}
        
        return {
            "shape": self.processed_data.shape,
            "columns": list(self.processed_data.columns),
            "missing_values": self.processed_data.isnull().sum().to_dict(),
            "fraud_distribution": self.processed_data[self.target_column].value_counts().to_dict() if self.target_column in self.processed_data.columns else {},
            "numeric_summary": self.processed_data.describe().to_dict()
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize the data handler
    handler = HealthInsuranceDataHandler()
    
    # If you don't have data yet, generate sample data for testing
    print("Generating sample data for demonstration...")
    sample_data = handler.generate_sample_data(1000)
    
    # Save sample data
    handler.save_processed_data(sample_data, "data/sample_health_insurance_data.csv")
    
    # Load and process the data
    data = handler.load_dataset("data/sample_health_insurance_data.csv")
    
    # Validate the dataset
    validation = handler.validate_dataset(data)
    print("\nDataset Validation Results:")
    print(json.dumps(validation, indent=2, default=str))
    
    # Preprocess the data
    processed_data = handler.preprocess_for_fraud_detection(data)
    
    # Get summary
    summary = handler.get_data_summary()
    print("\nData Summary:")
    print(json.dumps(summary, indent=2, default=str))