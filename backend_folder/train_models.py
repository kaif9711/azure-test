#!/usr/bin/env python3
"""
Training Script for Fraud Detection Models
Comprehensive script to train, evaluate, and deploy fraud detection models
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.data_ingestion import DataIngestionPipeline
    from src.data_preprocessing import ClaimDataPreprocessor
    from src.ml_training import FraudDetectionModelTrainer
    from src.fraud_classification import FraudClassifier
except ImportError as e:
    logging.error(f"Failed to import modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class FraudModelTrainingPipeline:
    """Complete training pipeline for fraud detection models"""
    
    def __init__(self, config):
        self.config = config
        self.data_pipeline = DataIngestionPipeline()
        self.preprocessor = ClaimDataPreprocessor()
        self.trainer = FraudDetectionModelTrainer()
        
        # Create necessary directories
        os.makedirs(config['model_output_dir'], exist_ok=True)
        os.makedirs(config['data_dir'], exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    
    def load_training_data(self):
        """Load and combine training data from multiple sources"""
        logger.info("Loading training data...")
        
        training_data = []
        
        # Load CSV files
        csv_files = Path(self.config['data_dir']).glob('*.csv')
        for csv_file in csv_files:
            logger.info(f"Loading CSV file: {csv_file}")
            data = self.data_pipeline.load_csv_data(str(csv_file))
            if data is not None and not data.empty:
                training_data.append(data)
        
        # Load JSON files
        json_files = Path(self.config['data_dir']).glob('*.json')
        for json_file in json_files:
            logger.info(f"Loading JSON file: {json_file}")
            data = self.data_pipeline.load_json_data(str(json_file))
            if data:
                df = pd.DataFrame(data)
                training_data.append(df)
        
        if not training_data:
            logger.error("No training data found!")
            return None
        
        # Combine all data
        combined_data = pd.concat(training_data, ignore_index=True)
        logger.info(f"Combined training data shape: {combined_data.shape}")
        
        return combined_data
    
    def prepare_training_data(self, raw_data):
        """Prepare data for training"""
        logger.info("Preprocessing training data...")
        
        # Clean data
        cleaned_data = self.preprocessor.clean_data(raw_data)
        
        # Engineer features
        featured_data = self.preprocessor.engineer_features(cleaned_data)
        
        # Encode categorical variables
        encoded_data = self.preprocessor.encode_categorical_features(featured_data)
        
        # Prepare features and target
        if 'is_fraud' not in encoded_data.columns:
            logger.error("Target column 'is_fraud' not found in data!")
            return None, None
        
        # Separate features and target
        feature_cols = [col for col in encoded_data.columns 
                       if col not in ['is_fraud', 'claim_id', 'claimant_name']]
        
        X = encoded_data[feature_cols]
        y = encoded_data['is_fraud']
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple fraud detection models"""
        logger.info("Training baseline models...")
        
        # Train baseline models
        baseline_results = self.trainer.train_baseline_models(X, y)
        
        # Find best performing model
        best_model_name = None
        best_score = 0
        
        for model_name, results in baseline_results['model_results'].items():
            f1_score = results.get('f1_score', 0)
            if f1_score > best_score:
                best_score = f1_score
                best_model_name = model_name
        
        logger.info(f"Best baseline model: {best_model_name} (F1: {best_score:.4f})")
        
        # Hyperparameter tuning for best model
        if self.config.get('hyperparameter_tuning', True):
            logger.info(f"Performing hyperparameter tuning for {best_model_name}...")
            
            param_grids = self.trainer.get_hyperparameter_grids()
            if best_model_name in param_grids:
                best_model, best_params, cv_scores = self.trainer.hyperparameter_tuning(
                    X, y, best_model_name, param_grids[best_model_name]
                )
                
                logger.info(f"Best parameters: {best_params}")
                logger.info(f"CV scores: {cv_scores}")
                
                return best_model, best_model_name, best_params, baseline_results
        
        # Return baseline model if no tuning
        best_model = baseline_results['trained_models'][best_model_name]
        return best_model, best_model_name, {}, baseline_results
    
    def evaluate_final_model(self, model, X, y, model_name):
        """Evaluate final model performance"""
        logger.info("Evaluating final model...")
        
        # Split data for final evaluation
        X_train, X_test, y_train, y_test = self.trainer.split_data(X, y, test_size=0.2)
        
        # Train on full training set
        model.fit(X_train, y_train)
        
        # Evaluate
        final_metrics = self.trainer.evaluate_model(model, X_test, y_test)
        
        logger.info("Final Model Performance:")
        for metric, value in final_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return final_metrics
    
    def save_model_artifacts(self, model, model_name, params, metrics, feature_names):
        """Save model and associated artifacts"""
        logger.info("Saving model artifacts...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(self.config['model_output_dir'])
        
        # Save model
        model_path = model_dir / f"fraud_detection_model_{timestamp}.joblib"
        self.trainer.save_model(model, str(model_path))
        
        # Save current model (for production use)
        current_model_path = model_dir / "fraud_detection_model.joblib"
        self.trainer.save_model(model, str(current_model_path))
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'timestamp': timestamp,
            'hyperparameters': params,
            'performance_metrics': metrics,
            'feature_names': feature_names,
            'model_path': str(model_path),
            'training_config': self.config
        }
        
        metadata_path = model_dir / f"model_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save current metadata
        current_metadata_path = model_dir / "current_model_metadata.json"
        with open(current_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return str(model_path), str(metadata_path)
    
    def generate_training_report(self, baseline_results, final_metrics, model_name):
        """Generate comprehensive training report"""
        logger.info("Generating training report...")
        
        report = {
            'training_timestamp': datetime.now().isoformat(),
            'configuration': self.config,
            'baseline_results': baseline_results,
            'final_model': {
                'name': model_name,
                'metrics': final_metrics
            },
            'data_summary': {
                'total_records': len(self.raw_data) if hasattr(self, 'raw_data') else 'N/A',
                'feature_count': len(self.feature_names) if hasattr(self, 'feature_names') else 'N/A'
            }
        }
        
        report_path = Path(self.config['model_output_dir']) / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Training report saved to: {report_path}")
        return str(report_path)
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        try:
            logger.info("Starting fraud detection model training pipeline...")
            
            # Load data
            self.raw_data = self.load_training_data()
            if self.raw_data is None:
                logger.error("Failed to load training data")
                return False
            
            # Prepare data
            X, y = self.prepare_training_data(self.raw_data)
            if X is None or y is None:
                logger.error("Failed to prepare training data")
                return False
            
            self.feature_names = list(X.columns)
            
            # Train models
            best_model, model_name, best_params, baseline_results = self.train_models(X, y)
            
            # Final evaluation
            final_metrics = self.evaluate_final_model(best_model, X, y, model_name)
            
            # Save artifacts
            model_path, metadata_path = self.save_model_artifacts(
                best_model, model_name, best_params, final_metrics, self.feature_names
            )
            
            # Generate report
            report_path = self.generate_training_report(baseline_results, final_metrics, model_name)
            
            logger.info("Training pipeline completed successfully!")
            logger.info(f"Model: {model_path}")
            logger.info(f"Report: {report_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            return False


def create_sample_training_data(output_dir, num_samples=10000):
    """Create sample training data for testing"""
    logger.info(f"Creating sample training data with {num_samples} samples...")
    
    np.random.seed(42)
    
    # Generate synthetic fraud detection data
    data = {
        'claim_id': [f'CLM{i:06d}' for i in range(num_samples)],
        'claimant_name': [f'Claimant_{i}' for i in range(num_samples)],
        'policy_number': [f'POL{np.random.randint(1000, 9999)}' for _ in range(num_samples)],
        'claim_amount': np.random.lognormal(7, 1.5, num_samples),  # Log-normal distribution
        'claimant_age': np.random.randint(18, 80, num_samples),
        'policy_duration': np.random.randint(1, 120, num_samples),
        'incident_type': np.random.choice(['Auto', 'Home', 'Health', 'Life'], num_samples),
        'incident_location': np.random.choice(['Urban', 'Rural', 'Suburban'], num_samples),
        'incident_date': pd.date_range('2020-01-01', '2023-12-31', periods=num_samples).strftime('%Y-%m-%d')
    }
    
    # Create fraud labels with realistic patterns
    fraud_probability = (
        (data['claim_amount'] > 15000).astype(float) * 0.3 +
        (np.array(data['claimant_age']) < 25).astype(float) * 0.2 +
        (np.array(data['policy_duration']) < 6).astype(float) * 0.3 +
        np.random.random(num_samples) * 0.2
    )
    
    data['is_fraud'] = (fraud_probability > 0.7).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = Path(output_dir) / 'sample_training_data.csv'
    df.to_csv(output_path, index=False)
    
    logger.info(f"Sample data created: {output_path}")
    logger.info(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    
    return str(output_path)


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train fraud detection models')
    parser.add_argument('--data-dir', default='data', help='Directory containing training data')
    parser.add_argument('--model-dir', default='models', help='Directory to save trained models')
    parser.add_argument('--create-sample-data', action='store_true', help='Create sample training data')
    parser.add_argument('--sample-size', type=int, default=10000, help='Size of sample data to create')
    parser.add_argument('--no-tuning', action='store_true', help='Skip hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'data_dir': args.data_dir,
        'model_output_dir': args.model_dir,
        'hyperparameter_tuning': not args.no_tuning,
        'random_state': 42
    }
    
    # Create sample data if requested
    if args.create_sample_data:
        create_sample_training_data(args.data_dir, args.sample_size)
    
    # Run training pipeline
    pipeline = FraudModelTrainingPipeline(config)
    success = pipeline.run_training_pipeline()
    
    if success:
        logger.info("Training completed successfully!")
        sys.exit(0)
    else:
        logger.error("Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()