"""
Data Ingestion Module
Handles loading and initial processing of insurance claim documents
"""

import pandas as pd
import json
import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import PyPDF2
import docx
from datetime import datetime

logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    """
    Pipeline for ingesting various types of insurance claim data
    Supports: CSV, JSON, PDF, DOCX files
    """
    
    def __init__(self, data_directory: str = "data/raw"):
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.supported_formats = ['.csv', '.json', '.pdf', '.docx', '.txt']
        
    def ingest_structured_data(self, file_path: str) -> pd.DataFrame:
        """
        Ingest structured data from CSV or JSON files
        
        Args:
            file_path: Path to the data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
                logger.info(f"Loaded CSV file: {file_path} with {len(df)} records")
                
            elif file_path.suffix.lower() == '.json':
                df = pd.read_json(file_path)
                logger.info(f"Loaded JSON file: {file_path} with {len(df)} records")
                
            else:
                raise ValueError(f"Unsupported structured data format: {file_path.suffix}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error ingesting structured data from {file_path}: {str(e)}")
            raise
    
    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest unstructured document data
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dict containing document metadata and text content
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            document_data = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_size': file_path.stat().st_size,
                'file_type': file_path.suffix.lower(),
                'ingestion_timestamp': datetime.now().isoformat(),
                'text_content': '',
                'metadata': {}
            }
            
            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                document_data['text_content'] = self._extract_pdf_text(file_path)
                
            elif file_path.suffix.lower() == '.docx':
                document_data['text_content'] = self._extract_docx_text(file_path)
                
            elif file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    document_data['text_content'] = f.read()
                    
            else:
                logger.warning(f"Unsupported document format: {file_path.suffix}")
                
            logger.info(f"Ingested document: {file_path}")
            return document_data
            
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {str(e)}")
            raise
    
    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            return ""
    
    def _extract_docx_text(self, docx_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {str(e)}")
            return ""
    
    def batch_ingest_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Batch ingest all supported files from a directory
        
        Args:
            directory_path: Path to directory containing files
            
        Returns:
            List of ingested document data
        """
        try:
            directory_path = Path(directory_path)
            ingested_data = []
            
            for file_path in directory_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                    try:
                        if file_path.suffix.lower() in ['.csv', '.json']:
                            # Handle structured data
                            df = self.ingest_structured_data(file_path)
                            ingested_data.append({
                                'file_path': str(file_path),
                                'data_type': 'structured',
                                'records_count': len(df),
                                'columns': list(df.columns),
                                'data': df
                            })
                        else:
                            # Handle documents
                            doc_data = self.ingest_document(file_path)
                            doc_data['data_type'] = 'document'
                            ingested_data.append(doc_data)
                            
                    except Exception as e:
                        logger.error(f"Failed to ingest {file_path}: {str(e)}")
                        continue
            
            logger.info(f"Batch ingested {len(ingested_data)} files from {directory_path}")
            return ingested_data
            
        except Exception as e:
            logger.error(f"Error in batch ingestion: {str(e)}")
            raise
    
    def validate_claim_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate ingested claim data for required fields and quality
        
        Args:
            data: Ingested claim DataFrame
            
        Returns:
            Validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Required fields for insurance claims
            required_fields = [
                'claim_id', 'claimant_name', 'claim_amount', 
                'incident_date', 'policy_number'
            ]
            
            # Check for required fields
            missing_fields = [field for field in required_fields if field not in data.columns]
            if missing_fields:
                validation_results['errors'].append(f"Missing required fields: {missing_fields}")
                validation_results['is_valid'] = False
            
            # Data quality checks
            if 'claim_amount' in data.columns:
                negative_amounts = data['claim_amount'] < 0
                if negative_amounts.any():
                    validation_results['warnings'].append(
                        f"Found {negative_amounts.sum()} negative claim amounts"
                    )
            
            # Duplicate check
            if 'claim_id' in data.columns:
                duplicates = data.duplicated(subset=['claim_id'])
                if duplicates.any():
                    validation_results['warnings'].append(
                        f"Found {duplicates.sum()} duplicate claim IDs"
                    )
            
            # Statistics
            validation_results['statistics'] = {
                'total_records': len(data),
                'columns_count': len(data.columns),
                'missing_values_per_column': data.isnull().sum().to_dict(),
                'data_types': data.dtypes.astype(str).to_dict()
            }
            
            logger.info(f"Data validation completed. Valid: {validation_results['is_valid']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in data validation: {str(e)}")
            validation_results['errors'].append(f"Validation error: {str(e)}")
            validation_results['is_valid'] = False
            return validation_results


class ClaimDocumentProcessor:
    """
    Specialized processor for insurance claim documents
    """
    
    def __init__(self):
        self.claim_patterns = {
            'claim_number': r'(?:Claim\s+(?:Number|#|ID):\s*)([A-Z0-9-]+)',
            'policy_number': r'(?:Policy\s+(?:Number|#|ID):\s*)([A-Z0-9-]+)',
            'incident_date': r'(?:Incident\s+Date:\s*)(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            'claim_amount': r'(?:Claim\s+Amount:\s*)\$?([\d,]+\.?\d*)',
            'claimant_name': r'(?:Claimant:\s*)([A-Za-z\s]+)'
        }
    
    def extract_claim_attributes(self, document_text: str) -> Dict[str, Any]:
        """
        Extract key claim attributes from document text using regex patterns
        
        Args:
            document_text: Raw text from claim document
            
        Returns:
            Dictionary of extracted attributes
        """
        import re
        
        extracted_data = {}
        
        for field, pattern in self.claim_patterns.items():
            match = re.search(pattern, document_text, re.IGNORECASE)
            if match:
                extracted_data[field] = match.group(1).strip()
            else:
                extracted_data[field] = None
        
        # Additional processing
        if extracted_data.get('claim_amount'):
            # Clean amount field
            amount_str = extracted_data['claim_amount'].replace(',', '')
            try:
                extracted_data['claim_amount'] = float(amount_str)
            except ValueError:
                extracted_data['claim_amount'] = None
        
        return extracted_data
    
    def detect_missing_information(self, extracted_data: Dict[str, Any]) -> List[str]:
        """
        Identify missing critical information in claim data
        
        Args:
            extracted_data: Extracted claim attributes
            
        Returns:
            List of missing fields
        """
        critical_fields = [
            'claim_number', 'policy_number', 'incident_date', 
            'claim_amount', 'claimant_name'
        ]
        
        missing_fields = []
        for field in critical_fields:
            if not extracted_data.get(field):
                missing_fields.append(field)
        
        return missing_fields


# Example usage and testing functions
def create_sample_data():
    """Create sample data for testing"""
    sample_claims = pd.DataFrame({
        'claim_id': ['CLM001', 'CLM002', 'CLM003', 'CLM004', 'CLM005'],
        'claimant_name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Mike Wilson'],
        'policy_number': ['POL123456', 'POL123457', 'POL123458', 'POL123459', 'POL123460'],
        'claim_amount': [5000.00, 15000.00, 2500.00, 50000.00, 3000.00],
        'incident_date': ['2024-01-15', '2024-01-20', '2024-01-25', '2024-02-01', '2024-02-05'],
        'incident_type': ['Auto Accident', 'Property Damage', 'Theft', 'Auto Accident', 'Medical'],
        'is_fraud': [0, 1, 0, 1, 0]  # Target variable for ML
    })
    
    return sample_claims


if __name__ == "__main__":
    # Example usage
    pipeline = DataIngestionPipeline()
    
    # Create sample data for testing
    sample_data = create_sample_data()
    
    # Validate data
    validation_results = pipeline.validate_claim_data(sample_data)
    print("Validation Results:", validation_results)
    
    # Test document processor
    processor = ClaimDocumentProcessor()
    
    sample_document = """
    Claim Number: CLM-2024-001
    Policy Number: POL-123456
    Claimant: John Doe
    Incident Date: 01/15/2024
    Claim Amount: $5,000.00
    
    Description: Vehicle collision on highway...
    """
    
    extracted_attrs = processor.extract_claim_attributes(sample_document)
    missing_fields = processor.detect_missing_information(extracted_attrs)
    
    print("Extracted Attributes:", extracted_attrs)
    print("Missing Fields:", missing_fields)