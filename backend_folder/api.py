"""
FastAPI Application for Fraud Detection System
RESTful API endpoints for claim processing, summarization, and fraud detection
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json
import os
from pathlib import Path
import uuid

# Import our custom modules
try:
    from src.data_ingestion import DataIngestionPipeline, ClaimDocumentProcessor
    from src.data_preprocessing import ClaimDataPreprocessor
    from src.ml_training import FraudDetectionModelTrainer
    from src.fraud_classification import FraudClassifier, RealTimeFraudDetector, validate_claim_data
    from src.ai_summarization import DocumentSummarizer
except ImportError as e:
    logging.warning(f"Some modules not available: {e}")

# Multi-agent pipeline (lazy import to avoid heavy startup if not needed)
try:
    from agents.pipeline import FraudClaimPipeline
except Exception as e:  # broad to avoid startup crash if agents missing
    logging.warning(f"Multi-agent pipeline not available yet: {e}")

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraudulent Claim Detection Agent",
    description="AI-powered insurance fraud detection system with document summarization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from auth_fastapi import router as auth_router, get_current_user
from claims_fastapi import router as claims_router
from admin_fastapi import router as admin_router
from patterns_fastapi import router as patterns_router
from metrics_fastapi import router as metrics_router

# Security (legacy bearer kept for backward compatibility with demo-token paths)
security = HTTPBearer()

# Global instances
data_pipeline = None
summarizer = None
fraud_classifier = None
real_time_detector = None
multi_agent_pipeline = None

# Data models
class ClaimData(BaseModel):
    """Claim data model for API requests"""
    claim_id: str = Field(..., description="Unique claim identifier")
    claimant_name: str = Field(..., description="Name of the claimant")
    policy_number: str = Field(..., description="Policy number")
    claim_amount: float = Field(..., gt=0, description="Claim amount in USD")
    incident_date: str = Field(..., description="Date of incident (YYYY-MM-DD)")
    incident_type: str = Field(..., description="Type of incident")
    incident_description: str = Field(..., description="Description of incident")
    incident_location: Optional[str] = Field(None, description="Location of incident")
    claimant_age: Optional[int] = Field(None, ge=0, le=120, description="Age of claimant")
    policy_duration: Optional[int] = Field(None, ge=0, description="Policy duration in months")

class PredictionResponse(BaseModel):
    """Response model for fraud predictions"""
    claim_id: str
    fraud_probability: float
    is_fraud_predicted: bool
    risk_level: str
    explanation: Dict[str, Any]
    prediction_timestamp: str
    model_version: str

class SummaryResponse(BaseModel):
    """Response model for document summaries"""
    document_id: str
    summary_timestamp: str
    extractive_summary: str
    keyword_summary: str
    structured_data: Dict[str, Any]
    key_insights: List[str]
    quality_score: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup (models + database schema)."""
    global data_pipeline, summarizer, fraud_classifier, real_time_detector

    # Initialize DB (MySQL flavor)
    from utils import db as db_utils
    try:
        db_utils.init_db(app)
        temp_email = os.getenv("DEV_TEMP_USER_EMAIL")
        temp_password = os.getenv("DEV_TEMP_USER_PASSWORD")
        if temp_email and temp_password:
            try:
                from utils.auth_utils import hash_password
                conn2 = db_utils.get_db_connection()
                cur2 = conn2.cursor()
                cur2.execute("SELECT id FROM users WHERE email=%s", (temp_email.lower(),))
                if not cur2.fetchone():
                    cur2.execute(
                        "INSERT INTO users (email, password_hash, first_name, last_name, role, is_active) VALUES (%s,%s,%s,%s,%s,true)",
                        (temp_email.lower(), hash_password(temp_password), "Temp", "User", "user"),
                    )
                    conn2.commit()
                    logger.info(f"Created dev temp user {temp_email}")
                conn2.close()
            except Exception as tmp_err:
                logger.warning(f"Failed to create DEV temp user: {tmp_err}")
    except Exception as e:
        logger.warning(f"Database init failed: {e}")

    # Initialize ML / pipeline components
    try:
        data_pipeline = DataIngestionPipeline()
    except Exception as e:
        logger.warning(f"Data pipeline init failed: {e}")
    try:
        summarizer = DocumentSummarizer()
    except Exception as e:
        logger.warning(f"Summarizer init failed: {e}")
    try:
        model_path = os.getenv('FRAUD_MODEL_PATH', 'models/fraud_detection_model.joblib')
        if os.path.exists(model_path):
            fraud_classifier = FraudClassifier(model_path)
            real_time_detector = RealTimeFraudDetector(fraud_classifier)
            logger.info("Fraud detection model loaded successfully")
        else:
            logger.warning("No fraud detection model found. Predictions will not be available.")
    except Exception as e:
        logger.error(f"Model init error: {e}")
    logger.info("Startup sequence completed")

app.include_router(auth_router)
app.include_router(claims_router)
app.include_router(admin_router)
app.include_router(patterns_router)
app.include_router(metrics_router)

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services = {
        "data_pipeline": "available" if data_pipeline else "unavailable",
        "summarizer": "available" if summarizer else "unavailable",
        "fraud_classifier": "available" if fraud_classifier else "unavailable",
        "real_time_detector": "available" if real_time_detector else "unavailable",
        "multi_agent_pipeline": "available" if multi_agent_pipeline else "unavailable"
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        services=services
    )

# Document upload and summarization
@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    current_user: dict = Depends(get_current_user)
):
    """Upload and process insurance claim document"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.txt'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / f"{doc_id}_{file.filename}"
        
        # Save file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Process document in background if requested
        if background_tasks and data_pipeline:
            background_tasks.add_task(process_document_background, str(file_path), doc_id)
        
        return {
            "document_id": doc_id,
            "filename": file.filename,
            "file_size": len(content),
            "upload_timestamp": datetime.now().isoformat(),
            "status": "uploaded",
            "file_path": str(file_path)
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/{document_id}/summarize", response_model=SummaryResponse)
async def summarize_document(
    document_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Generate AI summary of uploaded document"""
    try:
        if not summarizer:
            raise HTTPException(status_code=503, detail="Summarization service not available")
        
        # Find document file
        upload_dir = Path("uploads")
        doc_files = list(upload_dir.glob(f"{document_id}_*"))
        
        if not doc_files:
            raise HTTPException(status_code=404, detail="Document not found")
        
        file_path = doc_files[0]
        
        # Extract document content
        if data_pipeline:
            doc_data = data_pipeline.ingest_document(str(file_path))
            document_text = doc_data['text_content']
        else:
            # Fallback: read as text
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                document_text = f.read()
        
        if not document_text:
            raise HTTPException(status_code=400, detail="Could not extract text from document")
        
        # Generate summary
        summary_result = summarizer.summarize_document(document_text)
        
        return SummaryResponse(
            document_id=document_id,
            summary_timestamp=summary_result['summary_timestamp'],
            extractive_summary=summary_result['summaries']['extractive']['summary'],
            keyword_summary=summary_result['summaries']['keyword_based']['summary'],
            structured_data=summary_result['structured_data'],
            key_insights=summary_result['key_insights'],
            quality_score=summary_result['quality_score']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Fraud detection endpoints
@app.post("/claims/predict", response_model=PredictionResponse)
async def predict_fraud(
    claim: ClaimData,
    current_user: dict = Depends(get_current_user)
):
    """Predict fraud probability for a claim"""
    try:
        if not fraud_classifier:
            raise HTTPException(status_code=503, detail="Fraud detection service not available")
        
        # Validate claim data
        claim_dict = claim.dict()
        is_valid, errors = validate_claim_data(claim_dict)
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid claim data: {', '.join(errors)}")
        
        # Get prediction
        prediction = fraud_classifier.predict_fraud_probability(claim_dict)
        
        return PredictionResponse(
            claim_id=prediction['claim_id'],
            fraud_probability=prediction['fraud_probability'],
            is_fraud_predicted=prediction['is_fraud_predicted'],
            risk_level=prediction['risk_level'],
            explanation=prediction['explanation'],
            prediction_timestamp=prediction['prediction_timestamp'],
            model_version=prediction['model_version']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting fraud: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/claims/batch-predict")
async def batch_predict_fraud(
    claims: List[ClaimData],
    current_user: dict = Depends(get_current_user)
):
    """Predict fraud for multiple claims"""
    try:
        if not fraud_classifier:
            raise HTTPException(status_code=503, detail="Fraud detection service not available")
        
        if len(claims) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 claims per batch")
        
        # Convert to dictionaries
        claims_data = [claim.dict() for claim in claims]
        
        # Get batch predictions
        predictions = fraud_classifier.batch_predict(claims_data)
        
        return {
            "batch_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "total_claims": len(claims),
            "predictions": predictions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Real-time monitoring endpoints
@app.get("/monitoring/alerts")
async def get_recent_alerts(
    hours: int = 24,
    current_user: dict = Depends(get_current_user)
):
    """Get recent fraud alerts"""
    try:
        if not real_time_detector:
            raise HTTPException(status_code=503, detail="Real-time detection service not available")
        
        alerts = real_time_detector.get_recent_alerts(hours)
        
        return {
            "timeframe_hours": hours,
            "alert_count": len(alerts),
            "alerts": alerts,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/statistics")
async def get_prediction_statistics(
    current_user: dict = Depends(get_current_user)
):
    """Get fraud prediction statistics"""
    try:
        if not fraud_classifier:
            raise HTTPException(status_code=503, detail="Fraud detection service not available")
        
        stats = fraud_classifier.get_prediction_statistics()
        model_info = fraud_classifier.get_model_info()
        
        return {
            "model_info": model_info,
            "prediction_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Model management endpoints
@app.post("/model/update-threshold")
async def update_fraud_threshold(
    threshold: float = Query(..., ge=0.0, le=1.0, description="New fraud detection threshold (0-1)"),
    current_user: dict = Depends(get_current_user)
):
    """Update fraud detection threshold"""
    try:
        if not fraud_classifier:
            raise HTTPException(status_code=503, detail="Fraud detection service not available")
        
        # Check user permissions (simplified)
        if current_user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        fraud_classifier.update_threshold(threshold)
        
        return {
            "message": f"Fraud detection threshold updated to {threshold}",
            "new_threshold": threshold,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating threshold: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def get_model_information(current_user: dict = Depends(get_current_user)):
    """Get information about loaded models"""
    try:
        info = {
            "fraud_classifier": fraud_classifier.get_model_info() if fraud_classifier else "Not loaded",
            "services": {
                "data_pipeline": data_pipeline is not None,
                "summarizer": summarizer is not None,
                "fraud_classifier": fraud_classifier is not None,
                "real_time_detector": real_time_detector is not None
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Fraudulent Claim Detection Agent",
        "version": "1.0.0",
        "description": "AI-powered insurance fraud detection system",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "auth_register": "/auth/register",
            "auth_login": "/auth/login",
            "upload_document": "/documents/upload",
            "summarize": "/documents/{document_id}/summarize",
            "predict_fraud": "/claims/predict",
            "batch_predict": "/claims/batch-predict",
            "pipeline_claim_eval": "/claims/document-eval",
            "alerts": "/monitoring/alerts",
            "statistics": "/monitoring/statistics"
        },
        "timestamp": datetime.now().isoformat()
    }

# ---------------- Multi-Agent Pipeline Endpoint ---------------- #

class ClaimDocumentEvalRequest(BaseModel):
    """Request body for unified document + claim evaluation via multi-agent pipeline"""
    claim_id: str
    claimant_name: Optional[str] = None
    policy_number: Optional[str] = None
    claim_amount: Optional[float] = None
    incident_date: Optional[str] = None
    incident_type: Optional[str] = None
    incident_description: Optional[str] = None
    document_path: Optional[str] = Field(None, description="Path to an already uploaded claim document")
    document_text: Optional[str] = Field(None, description="Raw text if no file has been uploaded")

class ClaimDocumentEvalResponse(BaseModel):
    claim_id: str
    final_decision: str
    combined_risk_score: float
    doc_risk: float
    model_risk: float
    explanation: Dict[str, Any]
    processing_time_ms: float
    timestamp: str

def _ensure_multi_agent_pipeline():
    """Create the multi-agent pipeline singleton if artifacts are present."""
    global multi_agent_pipeline
    if multi_agent_pipeline is not None:
        return multi_agent_pipeline
    # Check model artifacts quickly
    expected_files = [
        Path("models/lgbm_fraud_model.txt"),
        Path("models/iso_forest.joblib"),
        Path("models/scaler.joblib")
    ]
    if not all(p.exists() for p in expected_files):
        missing = [p.name for p in expected_files if not p.exists()]
        raise HTTPException(status_code=503, detail=f"Model artifacts missing for multi-agent pipeline: {missing}")
    try:
        multi_agent_pipeline = FraudClaimPipeline()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize pipeline: {e}")
    return multi_agent_pipeline

@app.post("/claims/document-eval", response_model=ClaimDocumentEvalResponse)
async def evaluate_claim_with_document(payload: ClaimDocumentEvalRequest, current_user: dict = Depends(get_current_user)):
    """Run the full multi-agent pipeline: document validation -> cross-checker -> supervisor decision.

    Provide either document_path (previously uploaded) or raw document_text.
    """
    import time
    start = time.time()
    try:
        pipeline = _ensure_multi_agent_pipeline()
        # Acquire document text
        doc_text = payload.document_text
        if not doc_text and payload.document_path:
            candidate = Path(payload.document_path)
            if not candidate.exists():
                raise HTTPException(status_code=404, detail=f"document_path not found: {payload.document_path}")
            try:
                # naive read (real impl: detect type, parse accordingly)
                doc_text = candidate.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Unable to read document: {e}")
        if not doc_text:
            raise HTTPException(status_code=400, detail="Must supply either document_text or document_path")

        claim_meta = {k: v for k, v in payload.dict().items() if k not in {"document_text", "document_path"} and v is not None}
        result = pipeline.run(doc_text, claim_meta)
        elapsed = (time.time() - start) * 1000.0
        return ClaimDocumentEvalResponse(
            claim_id=payload.claim_id,
            final_decision=result.get("final_decision"),
            combined_risk_score=result.get("combined_risk_score"),
            doc_risk=result.get("doc_risk"),
            model_risk=result.get("model_risk"),
            explanation={
                "doc_findings": result.get("doc_findings"),
                "model_features": result.get("model_features"),
                "weights": result.get("weights")
            },
            processing_time_ms=elapsed,
            timestamp=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Pipeline evaluation failed")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def process_document_background(file_path: str, document_id: str):
    """Background task for document processing"""
    try:
        logger.info(f"Processing document {document_id} in background")
        
        if data_pipeline:
            doc_data = data_pipeline.ingest_document(file_path)
            
            # You could store results in database here
            logger.info(f"Document {document_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler returning JSONResponse"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler returning JSONResponse"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    import uvicorn
    
    # Create necessary directories
    Path("uploads").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Run the API
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )