"""
Data Preprocessing Module
Handles cleaning, transformation, and feature engineering for insurance claims
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

logger = logging.getLogger(__name__)

class ClaimDataPreprocessor:
    """
    Comprehensive preprocessing pipeline for insurance claim data
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        self.feature_columns = []
        self.target_column = 'is_fraud'
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_structured_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize structured claim data
        
        Args:
            df: Raw claims DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        try:
            logger.info(f"Starting data cleaning for {len(df)} records")
            df_clean = df.copy()
            
            # Standardize column names
            df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
            
            # Handle missing values in key columns
            if 'claim_amount' in df_clean.columns:
                # Remove negative claim amounts
                df_clean = df_clean[df_clean['claim_amount'] >= 0]
                # Fill missing amounts with median
                df_clean['claim_amount'].fillna(df_clean['claim_amount'].median(), inplace=True)
            
            # Clean and standardize date columns
            date_columns = [col for col in df_clean.columns if 'date' in col]
            for date_col in date_columns:
                df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
            
            # Clean text columns
            text_columns = ['claimant_name', 'incident_description', 'incident_location']
            for col in text_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].astype(str).str.strip()
                    df_clean[col] = df_clean[col].replace('nan', np.nan)
            
            # Remove duplicate records
            if 'claim_id' in df_clean.columns:
                df_clean = df_clean.drop_duplicates(subset=['claim_id'])
            
            logger.info(f"Data cleaning completed. {len(df_clean)} records remaining")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error in data cleaning: {str(e)}")
            raise
    
    def create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for fraud detection
        
        Args:
            df: Cleaned claims DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        try:
            logger.info("Creating engineered features")
            df_features = df.copy()
            
            # Time-based features
            if 'incident_date' in df_features.columns:
                df_features['incident_date'] = pd.to_datetime(df_features['incident_date'])
                df_features['days_since_incident'] = (
                    datetime.now() - df_features['incident_date']
                ).dt.days
                
                df_features['incident_month'] = df_features['incident_date'].dt.month
                df_features['incident_day_of_week'] = df_features['incident_date'].dt.dayofweek
                df_features['is_weekend_incident'] = df_features['incident_day_of_week'].isin([5, 6]).astype(int)
            
            # Claim amount features
            if 'claim_amount' in df_features.columns:
                df_features['claim_amount_log'] = np.log1p(df_features['claim_amount'])
                df_features['is_high_value_claim'] = (
                    df_features['claim_amount'] > df_features['claim_amount'].quantile(0.9)
                ).astype(int)
                
                # Round amount flag (suspicious if ends in 000)
                df_features['is_round_amount'] = (
                    df_features['claim_amount'] % 1000 == 0
                ).astype(int)
            
            # Policy features
            if 'policy_number' in df_features.columns:
                # Count claims per policy
                policy_counts = df_features['policy_number'].value_counts()
                df_features['claims_per_policy'] = df_features['policy_number'].map(policy_counts)
                df_features['multiple_claims_same_policy'] = (
                    df_features['claims_per_policy'] > 1
                ).astype(int)
            
            # Claimant features
            if 'claimant_name' in df_features.columns:
                # Count claims per claimant
                claimant_counts = df_features['claimant_name'].value_counts()
                df_features['claims_per_claimant'] = df_features['claimant_name'].map(claimant_counts)
                df_features['repeat_claimant'] = (
                    df_features['claims_per_claimant'] > 1
                ).astype(int)
            
            # Text-based features
            if 'incident_description' in df_features.columns:
                df_features['description_length'] = df_features['incident_description'].str.len()
                df_features['description_word_count'] = df_features['incident_description'].str.split().str.len()
                
                # Suspicious keywords
                suspicious_keywords = ['emergency', 'urgent', 'immediate', 'critical', 'severe']
                df_features['has_suspicious_keywords'] = df_features['incident_description'].str.lower().str.contains(
                    '|'.join(suspicious_keywords), na=False
                ).astype(int)
            
            logger.info(f"Created {len(df_features.columns) - len(df.columns)} new features")
            return df_features
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def preprocess_text_data(self, text: str) -> str:
        """
        Clean and preprocess text data
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """
        try:
            if pd.isna(text) or text == '':
                return ''
            
            # Convert to lowercase
            text = str(text).lower()
            
            # Remove punctuation and digits
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            processed_tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token not in self.stop_words and len(token) > 2
            ]
            
            return ' '.join(processed_tokens)
            
        except Exception as e:
            logger.error(f"Error in text preprocessing: {str(e)}")
            return ''
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """
        Encode categorical features for ML models
        
        Args:
            df: DataFrame with categorical columns
            categorical_columns: List of column names to encode
            
        Returns:
            DataFrame with encoded features
        """
        try:
            logger.info(f"Encoding categorical features: {categorical_columns}")
            df_encoded = df.copy()
            
            for column in categorical_columns:
                if column in df_encoded.columns:
                    # Fill missing values
                    df_encoded[column] = df_encoded[column].fillna('Unknown')
                    
                    # Use label encoding for binary categories, one-hot for others
                    unique_values = df_encoded[column].nunique()
                    
                    if unique_values <= 2:
                        # Binary encoding
                        if column not in self.label_encoders:
                            self.label_encoders[column] = LabelEncoder()
                            df_encoded[f'{column}_encoded'] = self.label_encoders[column].fit_transform(
                                df_encoded[column]
                            )
                        else:
                            df_encoded[f'{column}_encoded'] = self.label_encoders[column].transform(
                                df_encoded[column]
                            )
                    else:
                        # One-hot encoding for multi-class
                        dummies = pd.get_dummies(df_encoded[column], prefix=column, dummy_na=True)
                        df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    
                    # Drop original column
                    df_encoded = df_encoded.drop(columns=[column])
            
            logger.info("Categorical encoding completed")
            return df_encoded
            
        except Exception as e:
            logger.error(f"Error in categorical encoding: {str(e)}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using appropriate imputation strategies
        
        Args:
            df: DataFrame with missing values
            
        Returns:
            DataFrame with imputed values
        """
        try:
            logger.info("Handling missing values")
            df_imputed = df.copy()
            
            # Separate numeric and categorical columns
            numeric_columns = df_imputed.select_dtypes(include=[np.number]).columns
            categorical_columns = df_imputed.select_dtypes(include=['object']).columns
            
            # Impute numeric columns with median
            if len(numeric_columns) > 0:
                if 'numeric' not in self.imputers:
                    self.imputers['numeric'] = SimpleImputer(strategy='median')
                    df_imputed[numeric_columns] = self.imputers['numeric'].fit_transform(
                        df_imputed[numeric_columns]
                    )
                else:
                    df_imputed[numeric_columns] = self.imputers['numeric'].transform(
                        df_imputed[numeric_columns]
                    )
            
            # Impute categorical columns with mode
            if len(categorical_columns) > 0:
                if 'categorical' not in self.imputers:
                    self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
                    df_imputed[categorical_columns] = self.imputers['categorical'].fit_transform(
                        df_imputed[categorical_columns]
                    )
                else:
                    df_imputed[categorical_columns] = self.imputers['categorical'].transform(
                        df_imputed[categorical_columns]
                    )
            
            logger.info("Missing value imputation completed")
            return df_imputed
            
        except Exception as e:
            logger.error(f"Error in missing value handling: {str(e)}")
            raise
    
    def scale_features(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Scale numerical features for ML models
        
        Args:
            df: DataFrame with features to scale
            feature_columns: List of columns to scale
            
        Returns:
            DataFrame with scaled features
        """
        try:
            logger.info(f"Scaling features: {feature_columns}")
            df_scaled = df.copy()
            
            # Only scale numeric columns
            numeric_features = [
                col for col in feature_columns 
                if col in df_scaled.columns and df_scaled[col].dtype in ['int64', 'float64']
            ]
            
            if len(numeric_features) > 0:
                df_scaled[numeric_features] = self.scaler.fit_transform(df_scaled[numeric_features])
            
            logger.info("Feature scaling completed")
            return df_scaled
            
        except Exception as e:
            logger.error(f"Error in feature scaling: {str(e)}")
            raise
    
    def prepare_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets
        
        Args:
            df: Complete preprocessed DataFrame
            test_size: Proportion of test data
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        try:
            from sklearn.model_selection import train_test_split
            
            logger.info(f"Splitting data: train={1-test_size:.1%}, test={test_size:.1%}")
            
            # Stratify by target if it exists
            stratify_column = None
            if self.target_column in df.columns:
                stratify_column = df[self.target_column]
            
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_column
            )
            
            logger.info(f"Train set: {len(train_df)} records, Test set: {len(test_df)} records")
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error in train-test split: {str(e)}")
            raise
    
    def get_preprocessing_summary(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary of preprocessing steps
        
        Args:
            original_df: Original DataFrame before preprocessing
            processed_df: DataFrame after preprocessing
            
        Returns:
            Dictionary containing preprocessing summary
        """
        summary = {
            'original_shape': original_df.shape,
            'processed_shape': processed_df.shape,
            'features_created': processed_df.shape[1] - original_df.shape[1],
            'missing_values_original': original_df.isnull().sum().sum(),
            'missing_values_processed': processed_df.isnull().sum().sum(),
            'duplicate_records_removed': len(original_df) - len(processed_df),
            'feature_columns': list(processed_df.columns)
        }
        
        return summary


# Utility functions
def create_fraud_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create specific fraud indicator features
    """
    df_indicators = df.copy()
    
    # Suspicious timing patterns
    if 'incident_date' in df_indicators.columns:
        df_indicators['is_recent_policy_claim'] = 0  # Would need policy start date
        df_indicators['is_holiday_claim'] = 0  # Would need holiday calendar
    
    # Amount-based indicators
    if 'claim_amount' in df_indicators.columns:
        # Statistical outliers
        q25 = df_indicators['claim_amount'].quantile(0.25)
        q75 = df_indicators['claim_amount'].quantile(0.75)
        iqr = q75 - q25
        outlier_threshold = q75 + 1.5 * iqr
        df_indicators['is_amount_outlier'] = (
            df_indicators['claim_amount'] > outlier_threshold
        ).astype(int)
    
    return df_indicators


if __name__ == "__main__":
    # Example usage
    from data_ingestion import create_sample_data
    
    # Create sample data
    sample_df = create_sample_data()
    
    # Initialize preprocessor
    preprocessor = ClaimDataPreprocessor()
    
    # Preprocessing pipeline
    print("Original data shape:", sample_df.shape)
    
    # Clean data
    cleaned_df = preprocessor.clean_structured_data(sample_df)
    print("After cleaning:", cleaned_df.shape)
    
    # Create features
    featured_df = preprocessor.create_engineered_features(cleaned_df)
    print("After feature engineering:", featured_df.shape)
    
    # Handle categorical variables
    categorical_cols = ['incident_type']
    encoded_df = preprocessor.encode_categorical_features(featured_df, categorical_cols)
    print("After encoding:", encoded_df.shape)
    
    # Get summary
    summary = preprocessor.get_preprocessing_summary(sample_df, encoded_df)
    print("Preprocessing summary:", summary)