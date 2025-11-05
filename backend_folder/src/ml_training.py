"""
ML Model Training Module
Handles training, evaluation, and selection of fraud detection models
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import joblib
import os
from datetime import datetime
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class FraudDetectionModelTrainer:
    """
    Comprehensive ML model trainer for fraud detection
    """
    
    def __init__(self, model_save_path: str = "models/"):
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.feature_importance = {}
        
        # Define model configurations
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [1000]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', None]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            }
        }
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'is_fraud') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training
        
        Args:
            df: Preprocessed DataFrame
            target_column: Name of target column
            
        Returns:
            Tuple of (features, target)
        """
        try:
            logger.info("Preparing data for model training")
            
            # Separate features and target
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in DataFrame")
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Remove any remaining non-numeric columns
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            X = X[numeric_columns]
            
            logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
            logger.info(f"Target distribution: {y.value_counts().to_dict()}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise
    
    def train_baseline_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train baseline models without hyperparameter tuning
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Dictionary of model results
        """
        try:
            logger.info("Training baseline models")
            baseline_results = {}
            
            for model_name, config in self.model_configs.items():
                logger.info(f"Training baseline {model_name}")
                
                # Train model
                model = config['model']
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                
                baseline_results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logger.info(f"{model_name} - AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
            
            return baseline_results
            
        except Exception as e:
            logger.error(f"Error in baseline training: {str(e)}")
            raise
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            model_name: str, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model
        
        Args:
            X_train, y_train: Training data
            model_name: Name of model to tune
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with best model and parameters
        """
        try:
            logger.info(f"Starting hyperparameter tuning for {model_name}")
            
            if model_name not in self.model_configs:
                raise ValueError(f"Model {model_name} not configured")
            
            config = self.model_configs[model_name]
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                cv=cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            results = {
                'best_model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            logger.info(f"Best parameters for {model_name}: {results['best_params']}")
            logger.info(f"Best cross-validation score: {results['best_score']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            raise
    
    def train_optimized_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train models with hyperparameter optimization
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Dictionary of optimized model results
        """
        try:
            logger.info("Training optimized models with hyperparameter tuning")
            optimized_results = {}
            
            for model_name in self.model_configs.keys():
                # Perform hyperparameter tuning
                tuning_results = self.hyperparameter_tuning(X_train, y_train, model_name)
                
                # Get best model
                best_model = tuning_results['best_model']
                
                # Make predictions on test set
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                
                optimized_results[model_name] = {
                    'model': best_model,
                    'best_params': tuning_results['best_params'],
                    'cv_score': tuning_results['best_score'],
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                # Store feature importance if available
                if hasattr(best_model, 'feature_importances_'):
                    self.feature_importance[model_name] = dict(zip(
                        X_train.columns, 
                        best_model.feature_importances_
                    ))
                elif hasattr(best_model, 'coef_'):
                    self.feature_importance[model_name] = dict(zip(
                        X_train.columns,
                        abs(best_model.coef_[0])
                    ))
                
                logger.info(f"Optimized {model_name} - AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
            
            self.models = optimized_results
            return optimized_results
            
        except Exception as e:
            logger.error(f"Error in optimized training: {str(e)}")
            raise
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'auc': roc_auc_score(y_true, y_pred_proba),
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def compare_models(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Compare model performance across different metrics
        
        Args:
            results: Dictionary of model results
            
        Returns:
            DataFrame with comparison metrics
        """
        try:
            comparison_data = []
            
            for model_name, result in results.items():
                metrics = result['metrics']
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1-Score': metrics.get('f1', 0),
                    'AUC': metrics.get('auc', 0)
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('AUC', ascending=False)
            
            logger.info("Model comparison completed")
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error in model comparison: {str(e)}")
            raise
    
    def select_best_model(self, results: Dict[str, Any], primary_metric: str = 'auc') -> str:
        """
        Select the best model based on specified metric
        
        Args:
            results: Dictionary of model results
            primary_metric: Metric to optimize for
            
        Returns:
            Name of best model
        """
        try:
            best_score = 0
            best_model_name = None
            
            for model_name, result in results.items():
                score = result['metrics'].get(primary_metric, 0)
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
            
            if best_model_name:
                self.best_model = results[best_model_name]['model']
                logger.info(f"Best model selected: {best_model_name} ({primary_metric}: {best_score:.4f})")
            
            return best_model_name
            
        except Exception as e:
            logger.error(f"Error selecting best model: {str(e)}")
            raise
    
    def save_model(self, model, model_name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Save trained model to disk
        
        Args:
            model: Trained model object
            model_name: Name for saved model
            metadata: Additional metadata to save
            
        Returns:
            Path to saved model
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{model_name}_{timestamp}.joblib"
            model_path = self.model_save_path / model_filename
            
            # Save model
            joblib.dump(model, model_path)
            
            # Save metadata
            if metadata:
                metadata_path = self.model_save_path / f"{model_name}_{timestamp}_metadata.json"
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Model saved: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path: str):
        """
        Load saved model from disk
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded model object
        """
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded from: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_model_report(self, model_name: str, results: Dict[str, Any], 
                            X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Generate comprehensive model performance report
        
        Args:
            model_name: Name of the model
            results: Model results dictionary
            X_test, y_test: Test data
            
        Returns:
            Complete model report
        """
        try:
            model_result = results[model_name]
            model = model_result['model']
            metrics = model_result['metrics']
            
            # Classification report
            y_pred = model_result['predictions']
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Feature importance (if available)
            feature_importance = self.feature_importance.get(model_name, {})
            top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
            
            report = {
                'model_name': model_name,
                'model_type': type(model).__name__,
                'training_date': datetime.now().isoformat(),
                'metrics': metrics,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'feature_importance': feature_importance,
                'top_features': top_features,
                'model_parameters': model.get_params() if hasattr(model, 'get_params') else {},
                'test_set_size': len(y_test),
                'feature_count': X_test.shape[1]
            }
            
            logger.info(f"Generated comprehensive report for {model_name}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating model report: {str(e)}")
            raise


class ModelEvaluator:
    """
    Advanced model evaluation and visualization utilities
    """
    
    def __init__(self):
        self.evaluation_results = {}
    
    def plot_roc_curves(self, results: Dict[str, Any], y_test: pd.Series, 
                       save_path: str = None) -> None:
        """Plot ROC curves for all models"""
        try:
            plt.figure(figsize=(10, 8))
            
            for model_name, result in results.items():
                y_pred_proba = result['probabilities']
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc = result['metrics']['auc']
                
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves Comparison')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting ROC curves: {str(e)}")
    
    def plot_precision_recall_curves(self, results: Dict[str, Any], y_test: pd.Series,
                                   save_path: str = None) -> None:
        """Plot Precision-Recall curves for all models"""
        try:
            plt.figure(figsize=(10, 8))
            
            for model_name, result in results.items():
                y_pred_proba = result['probabilities']
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                
                plt.plot(recall, precision, label=f'{model_name}')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves Comparison')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting PR curves: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic fraud detection data
    X = pd.DataFrame({
        'claim_amount': np.random.lognormal(8, 1, n_samples),
        'claimant_age': np.random.randint(18, 80, n_samples),
        'policy_duration': np.random.randint(1, 120, n_samples),
        'claim_amount_log': np.random.normal(0, 1, n_samples),
        'is_weekend_incident': np.random.binomial(1, 0.3, n_samples),
        'description_length': np.random.randint(10, 500, n_samples)
    })
    
    # Create synthetic target with some correlation to features
    fraud_probability = (
        0.3 * (X['claim_amount'] > X['claim_amount'].quantile(0.8)) +
        0.2 * (X['claimant_age'] < 30) +
        0.1 * X['is_weekend_incident'] +
        0.1 * (X['policy_duration'] < 30)
    )
    y = np.random.binomial(1, fraud_probability, n_samples)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize trainer
    trainer = FraudDetectionModelTrainer()
    
    # Train baseline models
    baseline_results = trainer.train_baseline_models(X_train, y_train, X_test, y_test)
    
    # Compare models
    comparison = trainer.compare_models(baseline_results)
    print("\nModel Comparison:")
    print(comparison)
    
    # Select best model
    best_model_name = trainer.select_best_model(baseline_results)
    print(f"\nBest model: {best_model_name}")
    
    # Generate report
    report = trainer.generate_model_report(best_model_name, baseline_results, X_test, y_test)
    print(f"\nTop features for {best_model_name}:")
    for feature, importance in list(report['top_features'].items())[:5]:
        print(f"  {feature}: {importance:.4f}")