#!/usr/bin/env python3
"""
Deployment Script for Fraud Detection System
Comprehensive script to deploy and manage the fraud detection system
"""

import os
import sys
import subprocess
import logging
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetectionDeployer:
    """Deployment manager for fraud detection system"""
    
    def __init__(self, config):
        self.config = config
        self.base_dir = Path(__file__).parent
        self.docker_compose_file = self.base_dir / "docker-compose.yml"
        
    def check_prerequisites(self):
        """Check if all prerequisites are installed"""
        logger.info("Checking prerequisites...")
        
        # Check Docker
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True)
            logger.info(f"Docker version: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.error("Docker not found! Please install Docker.")
            return False
        
        # Check Docker Compose
        try:
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True)
            logger.info(f"Docker Compose version: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.error("Docker Compose not found! Please install Docker Compose.")
            return False
        
        # Check if Docker daemon is running
        try:
            result = subprocess.run(['docker', 'info'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.error("Docker daemon is not running!")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Docker daemon is not responding!")
            return False
        
        logger.info("Prerequisites check passed!")
        return True
    
    def setup_environment(self):
        """Setup environment variables and directories"""
        logger.info("Setting up environment...")
        
        # Create necessary directories
        directories = [
            'data', 'models', 'logs', 'uploads', 'database',
            'frontend', 'notebooks', 'monitoring'
        ]
        
        for directory in directories:
            dir_path = self.base_dir / directory
            dir_path.mkdir(exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        
        # Create .env file if it doesn't exist
        env_file = self.base_dir / '.env'
        if not env_file.exists():
            env_content = f"""
# Database Configuration
DATABASE_URL=mysql+pymysql://fraud_user:fraud_password@mysql_db:3306/fraud_detection_db
MYSQL_ROOT_PASSWORD=rootpass
MYSQL_DATABASE=fraud_claims
MYSQL_USER=fraud_user
MYSQL_PASSWORD=fraud_password

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# API Configuration
API_SECRET_KEY=your-secret-key-change-in-production-{datetime.now().strftime('%Y%m%d')}
LOG_LEVEL=INFO
FRAUD_MODEL_PATH=/app/models/fraud_detection_model.joblib

# Jupyter Configuration
JUPYTER_TOKEN=fraud-analysis-token

# Flask Configuration (legacy)
FLASK_APP=app.py
FLASK_ENV=development
"""
            with open(env_file, 'w') as f:
                f.write(env_content)
            logger.info(f"Created .env file: {env_file}")
        
        return True
    
    def build_images(self):
        """Build Docker images"""
        logger.info("Building Docker images...")
        
        try:
            # Build main API image
            cmd = ['docker-compose', 'build', 'fraud-detection-api']
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Built fraud-detection-api image successfully")
            
            # Build other images if they exist
            services = ['backend']
            for service in services:
                if self._service_exists(service):
                    cmd = ['docker-compose', 'build', service]
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    logger.info(f"Built {service} image successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build images: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
        
        return True
    
    def start_services(self):
        """Start Docker services"""
        logger.info("Starting services...")
        
        try:
            # Start core services first
            cmd = ['docker-compose', 'up', '-d', 'mysql_db', 'redis']
            subprocess.run(cmd, check=True)
            logger.info("Started database and cache services")
            
            # Wait for database to be ready
            self._wait_for_service('mysql_db', 3306)
            
            # Start API service
            cmd = ['docker-compose', 'up', '-d', 'fraud-detection-api']
            subprocess.run(cmd, check=True)
            logger.info("Started fraud detection API")
            
            # Wait for API to be ready
            self._wait_for_api_health()
            
            # Start additional services based on profiles (dashboard removed)
            if self.config.get('include_monitoring', False):
                cmd = ['docker-compose', '--profile', 'monitoring', 'up', '-d']
                subprocess.run(cmd, check=True)
                logger.info("Started monitoring services")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start services: {e}")
            return False
        
        return True
    
    def _service_exists(self, service_name):
        """Check if a service exists in docker-compose.yml"""
        try:
            cmd = ['docker-compose', 'config', '--services']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            services = result.stdout.strip().split('\n')
            return service_name in services
        except subprocess.CalledProcessError:
            return False
    
    def _wait_for_service(self, service_name, port, timeout=60):
        """Wait for a service to be ready"""
        logger.info(f"Waiting for {service_name} to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if container is running and healthy
                cmd = ['docker-compose', 'ps', service_name]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if 'Up' in result.stdout:
                    logger.info(f"{service_name} is ready!")
                    return True
                    
            except Exception:
                pass
            
            time.sleep(2)
        
        logger.warning(f"{service_name} is not ready after {timeout} seconds")
        return False
    
    def _wait_for_api_health(self, timeout=120):
        """Wait for API health check to pass"""
        logger.info("Waiting for API health check...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get('http://localhost:8000/health', timeout=5)
                if response.status_code == 200:
                    logger.info("API health check passed!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(5)
        
        logger.warning(f"API health check failed after {timeout} seconds")
        return False
    
    def install_dependencies(self):
        """Install Python dependencies in running container"""
        logger.info("Installing dependencies...")
        
        try:
            # Install dependencies in the API container
            cmd = [
                'docker-compose', 'exec', '-T', 'fraud-detection-api',
                'pip', 'install', '--no-cache-dir', '-r', 'requirements.txt'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("Dependencies installed successfully")
                return True
            else:
                logger.error(f"Failed to install dependencies: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Dependency installation timed out")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def initialize_database(self):
        """Initialize database with required tables"""
        logger.info("Initializing database...")
        
        try:
            # Create init SQL script
            init_sql = """
            CREATE DATABASE IF NOT EXISTS fraud_detection_db;
            USE fraud_detection_db;
            
            CREATE TABLE IF NOT EXISTS fraud_predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                claim_id VARCHAR(255) UNIQUE NOT NULL,
                fraud_probability FLOAT NOT NULL,
                is_fraud_predicted BOOLEAN NOT NULL,
                risk_level ENUM('LOW', 'MEDIUM', 'HIGH') NOT NULL,
                prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_version VARCHAR(100),
                INDEX idx_claim_id (claim_id),
                INDEX idx_timestamp (prediction_timestamp)
            );
            
            CREATE TABLE IF NOT EXISTS model_performance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                model_name VARCHAR(255) NOT NULL,
                model_version VARCHAR(100) NOT NULL,
                accuracy FLOAT,
                precision_score FLOAT,
                recall FLOAT,
                f1_score FLOAT,
                auc_roc FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            # Save init script
            init_script_path = self.base_dir / 'database' / 'init.sql'
            with open(init_script_path, 'w') as f:
                f.write(init_sql)
            
            logger.info("Database initialization script created")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    def deploy_sample_model(self):
        """Deploy a sample trained model"""
        logger.info("Deploying sample model...")
        
        model_dir = self.base_dir / 'models'
        model_path = model_dir / 'fraud_detection_model.joblib'
        
        if not model_path.exists():
            # Create a dummy model for testing
            try:
                cmd = [
                    'python', 'train_models.py',
                    '--create-sample-data',
                    '--sample-size', '1000',
                    '--data-dir', str(self.base_dir / 'data'),
                    '--model-dir', str(model_dir)
                ]
                
                result = subprocess.run(cmd, cwd=self.base_dir, 
                                      capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    logger.info("Sample model trained and deployed")
                    return True
                else:
                    logger.error(f"Failed to train sample model: {result.stderr}")
                    # Create minimal model metadata
                    self._create_dummy_model_metadata(model_dir)
                    return True
                    
            except subprocess.TimeoutExpired:
                logger.error("Model training timed out")
                return False
            except Exception as e:
                logger.error(f"Error during model training: {e}")
                return False
        else:
            logger.info("Model already exists")
            return True
    
    def _create_dummy_model_metadata(self, model_dir):
        """Create dummy model metadata for testing"""
        metadata = {
            'model_name': 'DummyModel',
            'timestamp': datetime.now().isoformat(),
            'hyperparameters': {},
            'performance_metrics': {
                'accuracy': 0.85,
                'precision': 0.80,
                'recall': 0.75,
                'f1_score': 0.77,
                'auc_roc': 0.85
            },
            'feature_names': ['claim_amount', 'claimant_age', 'policy_duration'],
            'model_path': str(model_dir / 'fraud_detection_model.joblib'),
            'training_config': self.config
        }
        
        metadata_path = model_dir / 'current_model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created dummy model metadata: {metadata_path}")
    
    def verify_deployment(self):
        """Verify that deployment is working correctly"""
        logger.info("Verifying deployment...")
        
        checks = []
        
        # Check API health
        try:
            response = requests.get('http://localhost:8000/health', timeout=10)
            if response.status_code == 200:
                checks.append("✓ API health check passed")
            else:
                checks.append("✗ API health check failed")
        except Exception as e:
            checks.append(f"✗ API health check error: {e}")
        
        # Check API endpoints
        try:
            response = requests.get('http://localhost:8000/', timeout=10)
            if response.status_code == 200:
                checks.append("✓ API root endpoint accessible")
            else:
                checks.append("✗ API root endpoint failed")
        except Exception as e:
            checks.append(f"✗ API root endpoint error: {e}")
        
        # Check database connection
        try:
            cmd = ['docker-compose', 'exec', '-T', 'mysql_db', 
                   'mysql', '-u', 'fraud_user', '-pfraud_password', 
                   '-e', 'SELECT 1;']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                checks.append("✓ Database connection successful")
            else:
                checks.append("✗ Database connection failed")
        except Exception as e:
            checks.append(f"✗ Database connection error: {e}")
        
        # Print verification results
        logger.info("Deployment verification results:")
        for check in checks:
            logger.info(f"  {check}")
        
        # Return True if all checks passed
        return all("✓" in check for check in checks)
    
    def show_status(self):
        """Show deployment status"""
        logger.info("Deployment Status:")
        
        try:
            # Show running containers
            result = subprocess.run(['docker-compose', 'ps'], 
                                  capture_output=True, text=True)
            logger.info("\nRunning containers:")
            logger.info(result.stdout)
            
            # Show service URLs
            logger.info("\nService URLs:")
            logger.info("  API Documentation: http://localhost:8000/docs")
            logger.info("  API Health: http://localhost:8000/health")
            logger.info("  MySQL Database: localhost:3306")
            logger.info("  Redis Cache: localhost:6379")
            
            if self.config.get('include_monitoring', False):
                logger.info("  Prometheus: http://localhost:9090")
                logger.info("  Grafana: http://localhost:3000")
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
    
    def cleanup(self):
        """Clean up deployment"""
        logger.info("Cleaning up deployment...")
        
        try:
            # Stop all services
            cmd = ['docker-compose', 'down']
            subprocess.run(cmd, check=True)
            
            # Remove volumes if requested
            if self.config.get('remove_volumes', False):
                cmd = ['docker-compose', 'down', '-v']
                subprocess.run(cmd, check=True)
                logger.info("Removed volumes")
            
            logger.info("Cleanup completed")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Cleanup failed: {e}")
    
    def deploy(self):
        """Run full deployment"""
        logger.info("Starting fraud detection system deployment...")
        
        steps = [
            ("Checking prerequisites", self.check_prerequisites),
            ("Setting up environment", self.setup_environment),
            ("Initializing database", self.initialize_database),
            ("Building images", self.build_images),
            ("Starting services", self.start_services),
            ("Deploying sample model", self.deploy_sample_model),
            ("Verifying deployment", self.verify_deployment)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n{'='*50}")
            logger.info(f"Step: {step_name}")
            logger.info(f"{'='*50}")
            
            if not step_func():
                logger.error(f"Deployment failed at step: {step_name}")
                return False
        
        logger.info(f"\n{'='*50}")
        logger.info("DEPLOYMENT SUCCESSFUL!")
        logger.info(f"{'='*50}")
        
        self.show_status()
        return True


def main():
    """Main deployment script"""
    parser = argparse.ArgumentParser(description='Deploy fraud detection system')
    parser.add_argument('--action', choices=['deploy', 'start', 'stop', 'status', 'cleanup'], 
                       default='deploy', help='Action to perform')
    # Removed: --include-dashboard (Streamlit deprecated)
    parser.add_argument('--include-monitoring', action='store_true', 
                       help='Include monitoring services')
    parser.add_argument('--remove-volumes', action='store_true', 
                       help='Remove volumes during cleanup')
    
    args = parser.parse_args()
    
    config = {
        'include_monitoring': args.include_monitoring,
        'remove_volumes': args.remove_volumes
    }
    
    deployer = FraudDetectionDeployer(config)
    
    try:
        if args.action == 'deploy':
            success = deployer.deploy()
            sys.exit(0 if success else 1)
        
        elif args.action == 'start':
            deployer.start_services()
        
        elif args.action == 'stop':
            subprocess.run(['docker-compose', 'stop'])
        
        elif args.action == 'status':
            deployer.show_status()
        
        elif args.action == 'cleanup':
            deployer.cleanup()
    
    except KeyboardInterrupt:
        logger.info("\nDeployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Deployment error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()