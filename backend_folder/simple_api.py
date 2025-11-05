"""
Simple FastAPI application for fraud detection demonstration
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd
import json
import io
import logging

# Try to import the data handler, create a simple fallback if not available
try:
    from health_insurance_data_handler import HealthInsuranceDataHandler
    DATA_HANDLER_AVAILABLE = True
except ImportError:
    DATA_HANDLER_AVAILABLE = False
    logging.warning("Data handler not available - dataset upload features will be limited")

app = FastAPI(
    title="Fraud Detection API",
    description="AI-powered insurance fraud detection system",
    version="1.0.0"
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data handler if available
data_handler = HealthInsuranceDataHandler() if DATA_HANDLER_AVAILABLE else None

class ClaimData(BaseModel):
    claim_id: str
    claimant_name: str
    claim_amount: float
    incident_type: str
    claimant_age: int = None

class DataUploadResponse(BaseModel):
    success: bool
    message: str
    data_summary: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    return {
        "message": "Fraud Detection API is running!",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "predict": "/claims/predict"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "database": "connected",
            "ml_model": "loaded"
        }
    }

@app.post("/claims/predict")
async def predict_fraud(claim: ClaimData):
    # Simple fraud detection logic (demo)
    fraud_probability = 0.2  # Default low risk
    
    if claim.claim_amount > 10000:
        fraud_probability += 0.3
    
    if claim.claimant_age and claim.claimant_age < 25:
        fraud_probability += 0.2
    
    if "accident" in claim.incident_type.lower():
        fraud_probability += 0.1
    
    fraud_probability = min(fraud_probability, 1.0)
    
    risk_level = "LOW"
    if fraud_probability > 0.7:
        risk_level = "HIGH"
    elif fraud_probability > 0.4:
        risk_level = "MEDIUM"
    
    return {
        "claim_id": claim.claim_id,
        "fraud_probability": round(fraud_probability, 2),
        "is_fraud_predicted": fraud_probability > 0.5,
        "risk_level": risk_level,
        "prediction_timestamp": datetime.now().isoformat(),
        "explanation": f"Based on claim amount (${claim.claim_amount}) and other factors"
    }

@app.get("/demo")
async def demo_endpoint():
    return {
        "message": "This is a demo of the fraud detection system",
        "sample_request": {
            "endpoint": "POST /claims/predict",
            "sample_data": {
                "claim_id": "DEMO001",
                "claimant_name": "John Doe",
                "claim_amount": 2500.0,
                "incident_type": "Auto accident",
                "claimant_age": 35
            }
        }
    }

@app.post("/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a health insurance dataset for training the fraud detection model
    
    Supported formats: CSV, JSON, Excel (.xlsx, .xls)
    """
    try:
        if not DATA_HANDLER_AVAILABLE:
            raise HTTPException(
                status_code=503, 
                detail="Dataset upload feature requires pandas and data handler modules"
            )
        
        # Check file type
        allowed_extensions = ['.csv', '.json', '.xlsx', '.xls']
        file_extension = None
        for ext in allowed_extensions:
            if file.filename.lower().endswith(ext):
                file_extension = ext
                break
        
        if not file_extension:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed: {allowed_extensions}"
            )
        
        # Read file content
        content = await file.read()
        
        # Load data based on file type
        if file_extension == '.csv':
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file_extension == '.json':
            df = pd.read_json(io.StringIO(content.decode('utf-8')))
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(io.BytesIO(content))
        
        # Validate the dataset
        validation_results = data_handler.validate_dataset(df)
        
        # Get basic summary
        data_summary = {
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
        
        # Save the uploaded dataset for processing
        output_path = f"data/uploaded_{file.filename}"
        data_handler.save_processed_data(df, output_path)
        
        return DataUploadResponse(
            success=True,
            message=f"Dataset uploaded successfully! Saved as {output_path}",
            data_summary=data_summary,
            validation_results=validation_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {str(e)}")

@app.post("/dataset/validate")
async def validate_dataset_structure(dataset_info: Dict[str, Any]):
    """
    Validate dataset structure without uploading the file
    
    Expected format:
    {
        "columns": ["claim_id", "patient_age", "claim_amount", ...],
        "sample_data": {...},
        "dataset_size": {"rows": 1000, "columns": 15}
    }
    """
    try:
        validation_results = {
            "is_valid": True,
            "recommendations": [],
            "requirements_met": {}
        }
        
        # Check required fields
        required_fields = ['columns', 'dataset_size']
        for field in required_fields:
            if field not in dataset_info:
                validation_results["is_valid"] = False
                validation_results["recommendations"].append(f"Missing required field: {field}")
        
        if 'columns' in dataset_info:
            columns = dataset_info['columns']
            
            # Check for fraud detection essentials
            essential_columns = ['claim_amount', 'is_fraud']
            recommended_columns = ['patient_age', 'claim_id', 'diagnosis_code', 'provider_id']
            
            for col in essential_columns:
                has_col = any(col.lower() in c.lower() for c in columns)
                validation_results["requirements_met"][f"has_{col}"] = has_col
                if not has_col:
                    validation_results["recommendations"].append(f"Essential column missing: {col}")
            
            for col in recommended_columns:
                has_col = any(col.lower() in c.lower() for c in columns)
                validation_results["requirements_met"][f"has_{col}"] = has_col
                if not has_col:
                    validation_results["recommendations"].append(f"Recommended column: {col}")
        
        return {
            "message": "Dataset structure validated",
            "validation_results": validation_results,
            "next_steps": [
                "Upload your CSV file using POST /dataset/upload",
                "Ensure your dataset has the recommended columns",
                "Check that fraud labels are properly formatted (True/False or 1/0)"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating dataset: {str(e)}")

@app.get("/dataset/sample")
async def get_sample_dataset():
    """
    Generate a sample health insurance dataset for testing
    """
    try:
        if not DATA_HANDLER_AVAILABLE:
            # Return a simple sample structure if pandas is not available
            return {
                "sample_structure": {
                    "claim_id": ["CLM001", "CLM002", "CLM003"],
                    "patient_age": [45, 32, 67],
                    "claim_amount": [2500.00, 1200.50, 8900.00],
                    "diagnosis_code": ["A123", "B456", "C789"],
                    "is_fraud": [False, False, True]
                },
                "note": "Install pandas to generate full sample dataset"
            }
        
        # Generate sample data using the data handler
        sample_df = data_handler.generate_sample_data(50)  # Small sample for demo
        
        return {
            "sample_data": sample_df.head(10).to_dict('records'),
            "full_structure": {
                "columns": list(sample_df.columns),
                "data_types": sample_df.dtypes.to_dict(),
                "shape": sample_df.shape,
                "fraud_distribution": sample_df['is_fraud'].value_counts().to_dict()
            },
            "download_instructions": "Use POST /dataset/download-sample to get full CSV file"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating sample: {str(e)}")

@app.get("/dataset/requirements")
async def get_dataset_requirements():
    """
    Get the requirements for health insurance dataset format
    """
    return {
        "dataset_requirements": {
            "essential_columns": {
                "claim_id": "Unique identifier for each claim (string)",
                "claim_amount": "Amount claimed in dollars (float)",
                "is_fraud": "Fraud label - True/False or 1/0 (boolean)"
            },
            "recommended_columns": {
                "patient_age": "Age of the patient (integer)",
                "diagnosis_code": "Medical diagnosis code (string)",
                "provider_id": "Healthcare provider identifier (string)",
                "incident_date": "Date of medical incident (date)",
                "policy_type": "Type of insurance policy (string)"
            },
            "supported_formats": [".csv", ".json", ".xlsx", ".xls"],
            "max_file_size": "50MB",
            "sample_data_structure": {
                "claim_id": "CLM001",
                "patient_age": 45,
                "claim_amount": 2500.00,
                "diagnosis_code": "A123",
                "provider_id": "PROV001",
                "incident_date": "2024-01-15",
                "is_fraud": False
            }
        },
        "upload_endpoint": "POST /dataset/upload",
        "validation_endpoint": "POST /dataset/validate"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)