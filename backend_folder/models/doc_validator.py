"""
Document Validation Module
Validates document authenticity using ML techniques
"""

import cv2
import numpy as np
from PIL import Image
import PyPDF2
import logging
from typing import Dict, Any, Tuple
import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class DocumentValidator:
    """AI-powered document validation system"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.getenv('MODEL_PATH', './models/')
        self.fraud_threshold = float(os.getenv('FRAUD_THRESHOLD', 0.7))
        
        # Initialize text analysis components
        self.text_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        
        # Known fraudulent document patterns (in production, this would be ML-trained)
        self.suspicious_patterns = [
            'lorem ipsum',
            'sample text',
            'placeholder',
            'test document',
            'fake document'
        ]
        
    def validate_document(self, file_path: str, document_type: str) -> Dict[str, Any]:
        """
        Main document validation function
        
        Args:
            file_path: Path to the document file
            document_type: Type of document (pdf, image, etc.)
            
        Returns:
            Dictionary with validation results
        """
        try:
            results = {
                'is_valid': True,
                'confidence_score': 1.0,
                'issues_found': [],
                'metadata': {},
                'risk_level': 'low'
            }
            
            # Validate file exists
            if not os.path.exists(file_path):
                results['is_valid'] = False
                results['issues_found'].append('File not found')
                return results
            
            # Get file metadata
            results['metadata'] = self._get_file_metadata(file_path)
            
            # Perform validation based on file type
            if document_type.lower() == 'pdf':
                pdf_results = self._validate_pdf(file_path)
                results.update(pdf_results)
            elif document_type.lower() in ['png', 'jpg', 'jpeg']:
                image_results = self._validate_image(file_path)
                results.update(image_results)
            else:
                results['issues_found'].append(f'Unsupported document type: {document_type}')
                results['is_valid'] = False
            
            # Calculate overall risk level
            results['risk_level'] = self._calculate_risk_level(results['confidence_score'])
            
            logger.info(f"Document validation completed for {file_path}: {results['risk_level']} risk")
            return results
            
        except Exception as e:
            logger.error(f"Error validating document {file_path}: {str(e)}")
            return {
                'is_valid': False,
                'confidence_score': 0.0,
                'issues_found': [f'Validation error: {str(e)}'],
                'metadata': {},
                'risk_level': 'high'
            }
    
    def _validate_pdf(self, file_path: str) -> Dict[str, Any]:
        """Validate PDF document"""
        results = {
            'is_valid': True,
            'confidence_score': 1.0,
            'issues_found': []
        }
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    results['issues_found'].append('Document is password protected')
                    results['confidence_score'] *= 0.7
                
                # Extract and analyze text
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text()
                
                if not text_content.strip():
                    results['issues_found'].append('No text content found')
                    results['confidence_score'] *= 0.5
                else:
                    # Analyze text for suspicious patterns
                    text_analysis = self._analyze_text_content(text_content)
                    results['confidence_score'] *= text_analysis['confidence_score']
                    results['issues_found'].extend(text_analysis['issues'])
                
                # Check PDF metadata for anomalies
                metadata_analysis = self._analyze_pdf_metadata(pdf_reader)
                results['confidence_score'] *= metadata_analysis['confidence_score']
                results['issues_found'].extend(metadata_analysis['issues'])
                
        except Exception as e:
            logger.error(f"Error validating PDF: {str(e)}")
            results['is_valid'] = False
            results['confidence_score'] = 0.0
            results['issues_found'].append(f'PDF validation error: {str(e)}')
        
        return results
    
    def _validate_image(self, file_path: str) -> Dict[str, Any]:
        """Validate image document"""
        results = {
            'is_valid': True,
            'confidence_score': 1.0,
            'issues_found': []
        }
        
        try:
            # Load image using OpenCV
            image = cv2.imread(file_path)
            if image is None:
                results['is_valid'] = False
                results['issues_found'].append('Could not load image')
                return results
            
            # Check image quality
            quality_analysis = self._analyze_image_quality(image)
            results['confidence_score'] *= quality_analysis['confidence_score']
            results['issues_found'].extend(quality_analysis['issues'])
            
            # Check for image manipulation
            manipulation_analysis = self._detect_image_manipulation(image)
            results['confidence_score'] *= manipulation_analysis['confidence_score']
            results['issues_found'].extend(manipulation_analysis['issues'])
            
            # Perform OCR if needed (placeholder for actual OCR implementation)
            # ocr_results = self._perform_ocr(image)
            # results['ocr_text'] = ocr_results.get('text', '')
            
        except Exception as e:
            logger.error(f"Error validating image: {str(e)}")
            results['is_valid'] = False
            results['confidence_score'] = 0.0
            results['issues_found'].append(f'Image validation error: {str(e)}')
        
        return results
    
    def _analyze_text_content(self, text: str) -> Dict[str, Any]:
        """Analyze text content for suspicious patterns"""
        results = {
            'confidence_score': 1.0,
            'issues': []
        }
        
        text_lower = text.lower()
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if pattern in text_lower:
                results['issues'].append(f'Suspicious pattern found: {pattern}')
                results['confidence_score'] *= 0.5
        
        # Check text length
        if len(text.strip()) < 50:
            results['issues'].append('Document contains very little text')
            results['confidence_score'] *= 0.6
        
        # Check for repeated content
        lines = text.split('\n')
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(lines) > 10 and len(unique_lines) / len(lines) < 0.5:
            results['issues'].append('High amount of repeated content')
            results['confidence_score'] *= 0.7
        
        return results
    
    def _analyze_pdf_metadata(self, pdf_reader) -> Dict[str, Any]:
        """Analyze PDF metadata for anomalies"""
        results = {
            'confidence_score': 1.0,
            'issues': []
        }
        
        try:
            metadata = pdf_reader.metadata
            if metadata:
                # Check creation date
                if '/CreationDate' in metadata:
                    creation_date = metadata['/CreationDate']
                    # Add logic to validate creation date
                
                # Check producer/creator
                if '/Producer' in metadata:
                    producer = metadata['/Producer']
                    # Check for known fraudulent software signatures
                    suspicious_producers = ['fake', 'test', 'sample']
                    if any(sus in producer.lower() for sus in suspicious_producers):
                        results['issues'].append('Suspicious document producer')
                        results['confidence_score'] *= 0.6
        
        except Exception as e:
            logger.warning(f"Could not analyze PDF metadata: {str(e)}")
        
        return results
    
    def _analyze_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image quality metrics"""
        results = {
            'confidence_score': 1.0,
            'issues': []
        }
        
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:  # Threshold for blurry images
                results['issues'].append('Image appears to be blurry or low quality')
                results['confidence_score'] *= 0.7
            
            # Check image dimensions
            height, width = gray.shape
            if height < 300 or width < 300:
                results['issues'].append('Image resolution is very low')
                results['confidence_score'] *= 0.8
            
        except Exception as e:
            logger.warning(f"Could not analyze image quality: {str(e)}")
        
        return results
    
    def _detect_image_manipulation(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect potential image manipulation"""
        results = {
            'confidence_score': 1.0,
            'issues': []
        }
        
        try:
            # Simple edge detection to find potential manipulation artifacts
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Calculate edge density
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            edge_density = edge_pixels / total_pixels
            
            # Very high edge density might indicate manipulation
            if edge_density > 0.3:
                results['issues'].append('Unusual edge patterns detected')
                results['confidence_score'] *= 0.8
            
        except Exception as e:
            logger.warning(f"Could not analyze image manipulation: {str(e)}")
        
        return results
    
    def _get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract file metadata"""
        try:
            stat = os.stat(file_path)
            return {
                'size': stat.st_size,
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'extension': os.path.splitext(file_path)[1].lower()
            }
        except Exception as e:
            logger.warning(f"Could not extract file metadata: {str(e)}")
            return {}
    
    def _calculate_risk_level(self, confidence_score: float) -> str:
        """Calculate risk level based on confidence score"""
        if confidence_score >= 0.8:
            return 'low'
        elif confidence_score >= 0.6:
            return 'medium'
        else:
            return 'high'