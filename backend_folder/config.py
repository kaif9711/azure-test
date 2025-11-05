import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    
    # Database Configuration
    # Switched default to MySQL; adjust if using SQLAlchemy with specific driver (e.g. mysql+mysqlconnector://)
    DATABASE_URL = os.environ.get('DATABASE_URL') or \
        'mysql://fraud_user:fraud_password@localhost:3306/fraud_claims'
    
    # JWT Configuration
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-change-in-production'
    JWT_ACCESS_TOKEN_EXPIRES = int(os.environ.get('JWT_ACCESS_TOKEN_EXPIRES', 3600))
    
    # CORS Configuration
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',')
    
    # File Upload Configuration
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', './uploads')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'doc', 'docx'}
    
    # Redis Configuration
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    # ML Model Configuration
    MODEL_PATH = os.environ.get('MODEL_PATH', './models/')
    FRAUD_THRESHOLD = float(os.environ.get('FRAUD_THRESHOLD', 0.7))
    DOCUMENT_VALIDATION_ENDPOINT = os.environ.get(
        'DOCUMENT_VALIDATION_ENDPOINT', 
        'http://localhost:8001'
    )
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', './logs/app.log')
    
    # Email Configuration
    EMAIL_SMTP_SERVER = os.environ.get('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
    EMAIL_SMTP_PORT = int(os.environ.get('EMAIL_SMTP_PORT', 587))
    EMAIL_USERNAME = os.environ.get('EMAIL_USERNAME')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')
    
    # Pagination
    CLAIMS_PER_PAGE = 20
    USERS_PER_PAGE = 50
    
    # Security
    PASSWORD_MIN_LENGTH = 8
    SESSION_TIMEOUT = 30 * 60  # 30 minutes in seconds
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        # Ensure upload directory exists
        upload_dir = app.config.get('UPLOAD_FOLDER')
        if upload_dir and not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)
        
        # Ensure logs directory exists
        log_file = app.config.get('LOG_FILE')
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Log to syslog in production
        import logging
        from logging.handlers import SysLogHandler
        syslog_handler = SysLogHandler()
        syslog_handler.setLevel(logging.WARNING)
        app.logger.addHandler(syslog_handler)


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}