"""
Unit Tests for Fraud Detection System
Comprehensive test suite covering all modules
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import os
import json
from pathlib import Path

# Import modules to test
try:
    from backend.src.data_ingestion import DataIngestionPipeline, ClaimDocumentProcessor
    from backend.src.data_preprocessing import ClaimDataPreprocessor
    from backend.src.ml_training import FraudDetectionModelTrainer
    from backend.src.fraud_classification import FraudClassifier, RealTimeFraudDetector, validate_claim_data
    from backend.src.ai_summarization import DocumentSummarizer
except ImportError:
    # For testing without full installation
    pass


class TestDataIngestion:
    """Tests for data ingestion pipeline"""
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing"""
        return pd.DataFrame({
            'claim_id': ['C001', 'C002', 'C003'],
            'claimant_name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'claim_amount': [1500.0, 2500.0, 3000.0],
            'incident_type': ['Auto', 'Home', 'Auto'],
            'incident_date': ['2023-01-15', '2023-02-20', '2023-03-10']
        })
    
    @pytest.fixture
    def data_pipeline(self):
        """Create data ingestion pipeline instance"""
        return DataIngestionPipeline()
    
    def test_load_csv_data(self, data_pipeline, sample_csv_data, tmp_path):
        """Test CSV data loading"""
        # Create temporary CSV file
        csv_file = tmp_path / "test_claims.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        # Test loading
        result = data_pipeline.load_csv_data(str(csv_file))
        
        assert result is not None
        assert len(result) == 3
        assert 'claim_id' in result.columns
        assert result['claim_id'].iloc[0] == 'C001'
    
    def test_load_json_data(self, data_pipeline, tmp_path):
        """Test JSON data loading"""
        # Create sample JSON data
        json_data = {
            "claims": [
                {
                    "claim_id": "J001",
                    "claimant_name": "Test User",
                    "claim_amount": 1000.0
                }
            ]
        }
        
        json_file = tmp_path / "test_claims.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f)
        
        result = data_pipeline.load_json_data(str(json_file))
        
        assert result is not None
        assert len(result) == 1
        assert result[0]['claim_id'] == 'J001'
    
    def test_validate_data_structure(self, data_pipeline, sample_csv_data):
        """Test data structure validation"""
        # Test valid data
        is_valid, errors = data_pipeline.validate_data_structure(sample_csv_data)
        assert is_valid
        assert len(errors) == 0
        
        # Test invalid data (missing required column)
        invalid_data = sample_csv_data.drop('claim_id', axis=1)
        is_valid, errors = data_pipeline.validate_data_structure(invalid_data)
        assert not is_valid
        assert len(errors) > 0
    
    def test_batch_process_files(self, data_pipeline, sample_csv_data, tmp_path):
        """Test batch file processing"""
        # Create multiple test files
        for i in range(3):
            csv_file = tmp_path / f"claims_{i}.csv"
            sample_csv_data.to_csv(csv_file, index=False)
        
        results = data_pipeline.batch_process_files(str(tmp_path), "*.csv")
        
        assert len(results) == 3
        for result in results:
            assert result['status'] == 'success'
            assert result['data'] is not None


class TestDataPreprocessing:
    """Tests for data preprocessing"""
    
    @pytest.fixture
    def sample_claims_data(self):
        """Create sample claims data"""
        np.random.seed(42)
        return pd.DataFrame({
            'claim_id': [f'C{i:03d}' for i in range(100)],
            'claim_amount': np.random.uniform(100, 10000, 100),
            'claimant_age': np.random.randint(18, 80, 100),
            'policy_duration': np.random.randint(1, 120, 100),
            'incident_type': np.random.choice(['Auto', 'Home', 'Health'], 100),
            'incident_location': np.random.choice(['Urban', 'Rural', 'Suburban'], 100),
            'is_fraud': np.random.choice([0, 1], 100, p=[0.8, 0.2])  # 20% fraud
        })
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance"""
        return ClaimDataPreprocessor()
    
    def test_clean_data(self, preprocessor, sample_claims_data):
        """Test data cleaning"""
        # Add some missing values
        dirty_data = sample_claims_data.copy()
        dirty_data.loc[0:5, 'claim_amount'] = None
        dirty_data.loc[10:15, 'claimant_age'] = -1  # Invalid age
        
        cleaned_data = preprocessor.clean_data(dirty_data)
        
        # Check that missing values are handled
        assert cleaned_data['claim_amount'].isnull().sum() < dirty_data['claim_amount'].isnull().sum()
        # Check that invalid ages are fixed
        assert (cleaned_data['claimant_age'] < 0).sum() == 0
    
    def test_engineer_features(self, preprocessor, sample_claims_data):
        """Test feature engineering"""
        featured_data = preprocessor.engineer_features(sample_claims_data)
        
        # Check for new features
        expected_features = [
            'claim_amount_per_age', 'claim_frequency_score',
            'amount_category', 'risk_score'
        ]
        
        for feature in expected_features:
            assert feature in featured_data.columns
    
    def test_encode_categorical_features(self, preprocessor, sample_claims_data):
        """Test categorical encoding"""
        encoded_data = preprocessor.encode_categorical_features(sample_claims_data)
        
        # Check that categorical columns are encoded
        assert 'incident_type_Auto' in encoded_data.columns or \
               'incident_type' not in encoded_data.columns or \
               encoded_data['incident_type'].dtype in ['int64', 'uint8']
    
    def test_split_data(self, preprocessor, sample_claims_data):
        """Test train-test split"""
        X = sample_claims_data.drop(['is_fraud', 'claim_id'], axis=1)
        y = sample_claims_data['is_fraud']
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)
        
        assert len(X_train) == int(0.8 * len(X))
        assert len(X_test) == len(X) - len(X_train)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)


class TestMLTraining:
    """Tests for ML training module"""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 1000
        
        X = pd.DataFrame({
            'claim_amount': np.random.uniform(100, 10000, n_samples),
            'claimant_age': np.random.randint(18, 80, n_samples),
            'policy_duration': np.random.randint(1, 120, n_samples),
            'incident_type_Auto': np.random.choice([0, 1], n_samples),
            'incident_type_Home': np.random.choice([0, 1], n_samples),
            'risk_score': np.random.uniform(0, 1, n_samples)
        })
        
        # Create target with some correlation to features
        y = ((X['claim_amount'] > 8000) & 
             (X['risk_score'] > 0.7) & 
             (X['claimant_age'] < 30)).astype(int)
        
        return X, y
    
    @pytest.fixture
    def trainer(self):
        """Create trainer instance"""
        return FraudDetectionModelTrainer()
    
    def test_prepare_baseline_models(self, trainer):
        """Test baseline model preparation"""
        models = trainer.prepare_baseline_models()
        
        assert len(models) >= 3  # Should have at least 3 baseline models
        assert 'LogisticRegression' in models
        assert 'RandomForest' in models
        assert 'XGBoost' in models
    
    def test_train_baseline_models(self, trainer, sample_training_data):
        """Test baseline model training"""
        X, y = sample_training_data
        
        results = trainer.train_baseline_models(X, y)
        
        assert 'model_results' in results
        assert len(results['model_results']) >= 3
        
        # Check that each model has required metrics
        for model_name, model_result in results['model_results'].items():
            assert 'accuracy' in model_result
            assert 'precision' in model_result
            assert 'recall' in model_result
            assert 'f1_score' in model_result
            assert 'auc_roc' in model_result
    
    def test_hyperparameter_tuning(self, trainer, sample_training_data):
        """Test hyperparameter tuning"""
        X, y = sample_training_data
        
        # Test with a simple model for faster execution
        param_grid = {
            'C': [0.1, 1.0],
            'penalty': ['l1', 'l2']
        }
        
        best_model, best_params, cv_scores = trainer.hyperparameter_tuning(
            X, y, 'LogisticRegression', param_grid, cv=3
        )
        
        assert best_model is not None
        assert best_params is not None
        assert len(cv_scores) == 3  # 3-fold CV
    
    def test_evaluate_model(self, trainer, sample_training_data):
        """Test model evaluation"""
        X, y = sample_training_data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.3)
        
        # Train a simple model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        metrics = trainer.evaluate_model(model, X_test, y_test)
        
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        for metric in required_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1


class TestFraudClassification:
    """Tests for fraud classification module"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock trained model"""
        model = Mock()
        model.predict_proba.return_value = np.array([[0.7, 0.3], [0.2, 0.8]])
        model.predict.return_value = np.array([0, 1])
        return model
    
    @pytest.fixture
    def sample_claim(self):
        """Create sample claim data"""
        return {
            'claim_id': 'TEST001',
            'claimant_name': 'Test User',
            'policy_number': 'POL123',
            'claim_amount': 2500.0,
            'incident_date': '2023-06-15',
            'incident_type': 'Auto',
            'incident_description': 'Minor collision',
            'claimant_age': 35,
            'policy_duration': 24
        }
    
    def test_validate_claim_data(self, sample_claim):
        """Test claim data validation"""
        # Test valid claim
        is_valid, errors = validate_claim_data(sample_claim)
        assert is_valid
        assert len(errors) == 0
        
        # Test invalid claim (missing required field)
        invalid_claim = sample_claim.copy()
        del invalid_claim['claim_id']
        
        is_valid, errors = validate_claim_data(invalid_claim)
        assert not is_valid
        assert len(errors) > 0
    
    @patch('joblib.load')
    def test_fraud_classifier_initialization(self, mock_joblib_load, mock_model):
        """Test fraud classifier initialization"""
        mock_joblib_load.return_value = mock_model
        
        classifier = FraudClassifier('fake_model_path.joblib')
        
        assert classifier.model is not None
        assert classifier.model_path == 'fake_model_path.joblib'
        mock_joblib_load.assert_called_once_with('fake_model_path.joblib')
    
    @patch('joblib.load')
    def test_predict_fraud_probability(self, mock_joblib_load, mock_model, sample_claim):
        """Test fraud probability prediction"""
        mock_joblib_load.return_value = mock_model
        classifier = FraudClassifier('fake_model_path.joblib')
        
        with patch.object(classifier, '_preprocess_claim_data') as mock_preprocess:
            mock_preprocess.return_value = np.array([[1, 2, 3, 4, 5]])
            
            result = classifier.predict_fraud_probability(sample_claim)
            
            assert 'claim_id' in result
            assert 'fraud_probability' in result
            assert 'is_fraud_predicted' in result
            assert 'risk_level' in result
            assert result['claim_id'] == 'TEST001'
            assert 0 <= result['fraud_probability'] <= 1
    
    def test_determine_risk_level(self):
        """Test risk level determination"""
        # Test low risk
        assert FraudClassifier._determine_risk_level(0.1) == 'LOW'
        
        # Test medium risk
        assert FraudClassifier._determine_risk_level(0.4) == 'MEDIUM'
        
        # Test high risk
        assert FraudClassifier._determine_risk_level(0.8) == 'HIGH'


class TestAISummarization:
    """Tests for AI summarization module"""
    
    @pytest.fixture
    def sample_document(self):
        """Create sample document text"""
        return """
        This is a comprehensive insurance claim report regarding the incident that occurred on March 15, 2023.
        The claimant, John Smith, was involved in a vehicular accident at the intersection of Main Street and First Avenue.
        The total estimated damage amounts to $3,500 for vehicle repairs and additional medical expenses.
        
        According to the police report, the accident occurred during heavy rain conditions at approximately 2:30 PM.
        The claimant was traveling northbound on Main Street when another vehicle ran a red light.
        There were two witnesses present at the scene who corroborated the claimant's version of events.
        
        Medical examination revealed minor injuries including bruising and whiplash.
        The claimant received treatment at City General Hospital emergency room.
        All medical documentation has been properly submitted with this claim.
        
        The investigating officer determined that the other driver was at fault for the accident.
        No citations were issued to our insured party.
        """
    
    @pytest.fixture
    def summarizer(self):
        """Create document summarizer instance"""
        return DocumentSummarizer()
    
    def test_extractive_summarization(self, summarizer, sample_document):
        """Test extractive summarization"""
        summary = summarizer.extractive_summarization(sample_document, max_sentences=2)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert len(summary) < len(sample_document)  # Should be shorter
    
    def test_keyword_based_summarization(self, summarizer, sample_document):
        """Test keyword-based summarization"""
        summary = summarizer.keyword_based_summarization(sample_document, top_k=3)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        # Should contain key terms
        assert any(keyword in summary.lower() for keyword in ['accident', 'claim', 'damage'])
    
    def test_extract_structured_data(self, summarizer, sample_document):
        """Test structured data extraction"""
        structured_data = summarizer.extract_structured_data(sample_document)
        
        assert isinstance(structured_data, dict)
        
        # Check for expected extracted information
        if 'names' in structured_data:
            assert 'John Smith' in str(structured_data['names'])
        
        if 'amounts' in structured_data:
            assert any('3500' in str(amount) or '3,500' in str(amount) 
                      for amount in structured_data['amounts'])
        
        if 'dates' in structured_data:
            assert any('March 15, 2023' in str(date) or '2023-03-15' in str(date)
                      for date in structured_data['dates'])
    
    def test_summarize_document(self, summarizer, sample_document):
        """Test complete document summarization"""
        result = summarizer.summarize_document(sample_document)
        
        assert isinstance(result, dict)
        assert 'summaries' in result
        assert 'structured_data' in result
        assert 'key_insights' in result
        assert 'quality_score' in result
        assert 'summary_timestamp' in result
        
        # Check summaries
        assert 'extractive' in result['summaries']
        assert 'keyword_based' in result['summaries']
        
        # Check quality score
        assert 0 <= result['quality_score'] <= 1


class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.fixture
    def complete_pipeline_data(self):
        """Create data for end-to-end pipeline testing"""
        return {
            'claims_data': pd.DataFrame({
                'claim_id': ['INT001', 'INT002'],
                'claimant_name': ['Integration Test 1', 'Integration Test 2'],
                'claim_amount': [1500.0, 8500.0],
                'incident_type': ['Auto', 'Home'],
                'claimant_age': [25, 55],
                'policy_duration': [12, 48],
                'is_fraud': [0, 1]
            }),
            'document_text': "This is a test insurance document for integration testing."
        }
    
    def test_end_to_end_pipeline(self, complete_pipeline_data):
        """Test complete pipeline from data ingestion to prediction"""
        # This would test the integration of all components
        # In practice, you'd run this with real components
        
        claims_data = complete_pipeline_data['claims_data']
        
        # Test that we can process the data through each stage
        assert len(claims_data) > 0
        assert 'claim_id' in claims_data.columns
        assert 'is_fraud' in claims_data.columns
        
        # Mock the pipeline stages
        preprocessor = ClaimDataPreprocessor()
        
        # Test preprocessing works
        cleaned_data = preprocessor.clean_data(claims_data)
        assert len(cleaned_data) <= len(claims_data)  # Should not increase size
        
        # Test feature engineering
        featured_data = preprocessor.engineer_features(cleaned_data)
        assert len(featured_data.columns) >= len(cleaned_data.columns)
    
    def test_api_data_flow(self):
        """Test data flow through API endpoints (mock)"""
        # Test the data structures expected by the API
        
        # Sample claim data for API
        api_claim = {
            'claim_id': 'API001',
            'claimant_name': 'API Test User',
            'policy_number': 'POL_API_123',
            'claim_amount': 2500.0,
            'incident_date': '2023-06-15',
            'incident_type': 'Auto',
            'incident_description': 'API test incident',
            'claimant_age': 30,
            'policy_duration': 18
        }
        
        # Validate claim data structure
        is_valid, errors = validate_claim_data(api_claim)
        assert is_valid, f"API claim data should be valid, errors: {errors}"


# Test configuration and fixtures
@pytest.fixture(scope="session")
def test_config():
    """Test configuration"""
    return {
        'test_data_dir': 'test_data',
        'model_dir': 'test_models',
        'log_level': 'DEBUG'
    }

@pytest.fixture(autouse=True)
def setup_test_environment(test_config):
    """Setup test environment before each test"""
    # Create test directories
    os.makedirs(test_config['test_data_dir'], exist_ok=True)
    os.makedirs(test_config['model_dir'], exist_ok=True)
    
    yield
    
    # Cleanup after tests (optional)
    # You might want to keep test artifacts for debugging


# Performance tests
class TestPerformance:
    """Performance tests for the system"""
    
    def test_data_processing_performance(self):
        """Test data processing performance with large dataset"""
        # Create large dataset
        np.random.seed(42)
        large_data = pd.DataFrame({
            'claim_id': [f'PERF{i:06d}' for i in range(10000)],
            'claim_amount': np.random.uniform(100, 50000, 10000),
            'claimant_age': np.random.randint(18, 80, 10000),
            'incident_type': np.random.choice(['Auto', 'Home', 'Health'], 10000)
        })
        
        preprocessor = ClaimDataPreprocessor()
        
        import time
        start_time = time.time()
        cleaned_data = preprocessor.clean_data(large_data)
        processing_time = time.time() - start_time
        
        # Should process 10k records in reasonable time (< 10 seconds)
        assert processing_time < 10.0, f"Processing took too long: {processing_time} seconds"
        assert len(cleaned_data) == len(large_data)
    
    def test_batch_prediction_performance(self):
        """Test batch prediction performance"""
        # This would test the performance of batch predictions
        # Mock test since we don't have a real model loaded
        
        batch_size = 1000
        claims = []
        
        for i in range(batch_size):
            claim = {
                'claim_id': f'BATCH{i:04d}',
                'claimant_name': f'User {i}',
                'claim_amount': float(np.random.uniform(100, 10000)),
                'incident_type': np.random.choice(['Auto', 'Home', 'Health']),
                'claimant_age': int(np.random.randint(18, 80))
            }
            claims.append(claim)
        
        # Test that we can validate all claims quickly
        start_time = time.time()
        valid_claims = 0
        for claim in claims:
            is_valid, _ = validate_claim_data(claim)
            if is_valid:
                valid_claims += 1
        
        validation_time = time.time() - start_time
        
        assert validation_time < 5.0, f"Validation took too long: {validation_time} seconds"
        assert valid_claims > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])