"""
AI Summarization Module
Handles document summarization and structured data extraction using NLP
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import re
import json
from datetime import datetime
from pathlib import Path

# NLP Libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# For advanced summarization (would use transformers in production)
# from transformers import pipeline

logger = logging.getLogger(__name__)

class DocumentSummarizer:
    """
    AI-powered document summarization for insurance claims
    """
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('taggers/averaged_perceptron_tagger')
            nltk.data.find('chunkers/maxent_ne_chunker')
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Insurance domain-specific patterns
        self.claim_patterns = {
            'policy_number': r'(?:Policy|Pol\.?)\s*(?:Number|No\.?|#):\s*([A-Z0-9\-]+)',
            'claim_number': r'(?:Claim|Clm\.?)\s*(?:Number|No\.?|#):\s*([A-Z0-9\-]+)',
            'date_patterns': r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            'amount_patterns': r'\$?([\d,]+\.?\d*)',
            'phone_patterns': r'(\(?[\d]{3}\)?[\s\-]?[\d]{3}[\s\-]?[\d]{4})',
            'email_patterns': r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            'address_patterns': r'(\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd))',
        }
        
        # Initialize transformer-based summarizer (if available)
        self.transformer_summarizer = None
        try:
            # Uncomment this in production with transformers installed
            # self.transformer_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            pass
        except Exception as e:
            logger.warning(f"Transformer summarizer not available: {str(e)}")
    
    def summarize_document(self, document_text: str, summary_ratio: float = 0.3) -> Dict[str, Any]:
        """
        Generate comprehensive summary of insurance claim document
        
        Args:
            document_text: Raw text from claim document
            summary_ratio: Proportion of original text to keep in summary
            
        Returns:
            Dictionary containing summary and extracted information
        """
        try:
            logger.info("Starting document summarization")
            
            if not document_text or len(document_text.strip()) == 0:
                return {'error': 'Empty document text provided'}
            
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(document_text)
            
            # Extract structured information
            structured_data = self.extract_structured_information(document_text)
            
            # Generate summary using multiple methods
            summaries = {}
            
            # Extractive summary (sentence-based)
            extractive_summary = self._extractive_summarization(cleaned_text, summary_ratio)
            summaries['extractive'] = extractive_summary
            
            # Keyword-based summary
            keyword_summary = self._keyword_based_summary(cleaned_text)
            summaries['keyword_based'] = keyword_summary
            
            # Statistical summary
            statistical_summary = self._generate_statistical_summary(document_text)
            
            # Try transformer-based summary if available
            if self.transformer_summarizer and len(cleaned_text) > 50:
                try:
                    # Note: This would work with transformers installed
                    # transformer_summary = self.transformer_summarizer(
                    #     cleaned_text[:1024],  # Limit input length
                    #     max_length=150,
                    #     min_length=50,
                    #     do_sample=False
                    # )[0]['summary_text']
                    # summaries['transformer'] = transformer_summary
                    pass
                except Exception as e:
                    logger.warning(f"Transformer summarization failed: {str(e)}")
            
            # Generate key insights
            insights = self._generate_insights(document_text, structured_data)
            
            # Compile final result
            result = {
                'document_length': len(document_text),
                'processed_length': len(cleaned_text),
                'summary_timestamp': datetime.now().isoformat(),
                'summaries': summaries,
                'structured_data': structured_data,
                'statistical_summary': statistical_summary,
                'key_insights': insights,
                'quality_score': self._calculate_summary_quality(summaries, structured_data)
            }
            
            logger.info("Document summarization completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in document summarization: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for summarization
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        try:
            # Remove extra whitespace and normalize
            text = ' '.join(text.split())
            
            # Remove special characters but keep punctuation for sentences
            text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', text)
            
            # Fix common OCR errors
            text = re.sub(r'\b(\w)\s+(\w)\b', r'\1\2', text)  # Fix scattered letters
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error in text preprocessing: {str(e)}")
            return text
    
    def _extractive_summarization(self, text: str, ratio: float) -> Dict[str, Any]:
        """
        Generate extractive summary by selecting most important sentences
        
        Args:
            text: Preprocessed text
            ratio: Ratio of sentences to keep
            
        Returns:
            Extractive summary dictionary
        """
        try:
            sentences = sent_tokenize(text)
            
            if len(sentences) <= 3:
                return {
                    'summary': text,
                    'selected_sentences': sentences,
                    'compression_ratio': 1.0
                }
            
            # Score sentences based on various factors
            sentence_scores = {}
            
            # Word frequency scoring
            word_freq = self._calculate_word_frequency(text)
            
            for sentence in sentences:
                sentence_scores[sentence] = 0
                words = word_tokenize(sentence.lower())
                
                # Score based on word frequency
                for word in words:
                    if word in word_freq:
                        sentence_scores[sentence] += word_freq[word]
                
                # Bonus for sentences with numbers (likely important in insurance)
                if re.search(r'\d+', sentence):
                    sentence_scores[sentence] += 5
                
                # Bonus for sentences with insurance keywords
                insurance_keywords = [
                    'claim', 'policy', 'damage', 'accident', 'injury', 'coverage',
                    'premium', 'deductible', 'liability', 'incident'
                ]
                for keyword in insurance_keywords:
                    if keyword.lower() in sentence.lower():
                        sentence_scores[sentence] += 3
                
                # Length normalization
                sentence_scores[sentence] = sentence_scores[sentence] / len(words) if words else 0
            
            # Select top sentences
            num_sentences = max(1, int(len(sentences) * ratio))
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
            
            # Order selected sentences by original appearance
            selected_sentences = []
            for sentence in sentences:
                if any(sentence == s[0] for s in top_sentences):
                    selected_sentences.append(sentence)
            
            summary = ' '.join(selected_sentences)
            
            return {
                'summary': summary,
                'selected_sentences': selected_sentences,
                'compression_ratio': len(summary) / len(text) if text else 0,
                'sentence_count': len(selected_sentences),
                'original_sentence_count': len(sentences)
            }
            
        except Exception as e:
            logger.error(f"Error in extractive summarization: {str(e)}")
            return {'summary': text[:500] + '...', 'error': str(e)}
    
    def _keyword_based_summary(self, text: str) -> Dict[str, Any]:
        """
        Generate summary based on key topics and entities
        
        Args:
            text: Preprocessed text
            
        Returns:
            Keyword-based summary
        """
        try:
            # Extract key phrases and entities
            sentences = sent_tokenize(text)
            keywords = []
            entities = []
            
            for sentence in sentences:
                # Extract named entities
                tokens = word_tokenize(sentence)
                pos_tags = pos_tag(tokens)
                chunks = ne_chunk(pos_tags)
                
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        entity = ' '.join([token for token, pos in chunk.leaves()])
                        entities.append((entity, chunk.label()))
            
            # Extract important keywords
            words = word_tokenize(text.lower())
            word_freq = {}
            for word in words:
                if word.isalpha() and word not in self.stop_words and len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top keywords
            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            keywords = [word for word, freq in top_keywords]
            
            # Create topic-based summary
            topic_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                keyword_count = sum(1 for keyword in keywords if keyword in sentence_lower)
                if keyword_count >= 2:  # Sentence contains at least 2 keywords
                    topic_sentences.append(sentence)
            
            if not topic_sentences:
                topic_sentences = sentences[:3]  # Fallback to first 3 sentences
            
            return {
                'summary': ' '.join(topic_sentences[:5]),  # Limit to 5 sentences
                'keywords': keywords,
                'named_entities': list(set(entities)),
                'topic_sentences_count': len(topic_sentences)
            }
            
        except Exception as e:
            logger.error(f"Error in keyword-based summary: {str(e)}")
            return {'summary': text[:500] + '...', 'error': str(e)}
    
    def _calculate_word_frequency(self, text: str) -> Dict[str, float]:
        """Calculate normalized word frequencies"""
        words = word_tokenize(text.lower())
        word_freq = {}
        
        for word in words:
            if word.isalpha() and word not in self.stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Normalize frequencies
        if word_freq:
            max_freq = max(word_freq.values())
            for word in word_freq:
                word_freq[word] = word_freq[word] / max_freq
        
        return word_freq
    
    def extract_structured_information(self, document_text: str) -> Dict[str, Any]:
        """
        Extract structured information from claim document
        
        Args:
            document_text: Raw document text
            
        Returns:
            Dictionary of extracted structured data
        """
        try:
            logger.info("Extracting structured information from document")
            extracted_data = {}
            
            # Extract using regex patterns
            for field, pattern in self.claim_patterns.items():
                matches = re.findall(pattern, document_text, re.IGNORECASE)
                if matches:
                    extracted_data[field] = matches
            
            # Extract specific claim information
            claim_info = self._extract_claim_specific_info(document_text)
            extracted_data.update(claim_info)
            
            # Extract contact information
            contact_info = self._extract_contact_information(document_text)
            extracted_data['contact_information'] = contact_info
            
            # Extract dates and format them
            dates = self._extract_and_format_dates(document_text)
            extracted_data['important_dates'] = dates
            
            # Extract monetary amounts
            amounts = self._extract_monetary_amounts(document_text)
            extracted_data['monetary_amounts'] = amounts
            
            logger.info(f"Extracted {len(extracted_data)} types of structured information")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting structured information: {str(e)}")
            return {}
    
    def _extract_claim_specific_info(self, text: str) -> Dict[str, Any]:
        """Extract claim-specific information"""
        info = {}
        
        # Incident type
        incident_keywords = {
            'auto': ['car', 'vehicle', 'automobile', 'collision', 'accident', 'traffic'],
            'property': ['property', 'home', 'house', 'building', 'fire', 'theft', 'burglary'],
            'health': ['medical', 'health', 'injury', 'hospital', 'doctor', 'treatment'],
            'liability': ['liability', 'damage', 'injury', 'lawsuit', 'legal']
        }
        
        text_lower = text.lower()
        for incident_type, keywords in incident_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                info['likely_incident_type'] = incident_type
                break
        
        # Severity indicators
        severity_keywords = {
            'high': ['severe', 'major', 'critical', 'extensive', 'total', 'catastrophic'],
            'medium': ['moderate', 'significant', 'considerable'],
            'low': ['minor', 'slight', 'small', 'minimal']
        }
        
        for severity, keywords in severity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                info['severity_indicators'] = severity
                break
        
        return info
    
    def _extract_contact_information(self, text: str) -> Dict[str, List[str]]:
        """Extract contact information"""
        contact_info = {}
        
        # Phone numbers
        phone_pattern = r'(\(?[\d]{3}\)?[\s\-\.]?[\d]{3}[\s\-\.]?[\d]{4})'
        phones = re.findall(phone_pattern, text)
        if phones:
            contact_info['phone_numbers'] = phones
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact_info['email_addresses'] = emails
        
        # Addresses (simplified)
        address_pattern = r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)'
        addresses = re.findall(address_pattern, text, re.IGNORECASE)
        if addresses:
            contact_info['addresses'] = addresses
        
        return contact_info
    
    def _extract_and_format_dates(self, text: str) -> List[Dict[str, str]]:
        """Extract and format dates"""
        date_patterns = [
            r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b',
            r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b',
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b'
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Try to parse and standardize the date
                    parsed_date = pd.to_datetime(match, errors='coerce')
                    if not pd.isna(parsed_date):
                        dates.append({
                            'original': match,
                            'standardized': parsed_date.strftime('%Y-%m-%d')
                        })
                except:
                    dates.append({'original': match, 'standardized': None})
        
        return dates
    
    def _extract_monetary_amounts(self, text: str) -> List[Dict[str, Any]]:
        """Extract monetary amounts"""
        amount_pattern = r'\$?([\d,]+\.?\d*)'
        matches = re.finditer(amount_pattern, text)
        
        amounts = []
        for match in matches:
            amount_str = match.group(1).replace(',', '')
            try:
                amount_value = float(amount_str)
                amounts.append({
                    'original': match.group(0),
                    'value': amount_value,
                    'position': match.start()
                })
            except ValueError:
                continue
        
        # Sort by value (descending)
        amounts.sort(key=lambda x: x['value'], reverse=True)
        return amounts[:10]  # Return top 10 amounts
    
    def _generate_statistical_summary(self, text: str) -> Dict[str, Any]:
        """Generate statistical summary of the document"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        return {
            'total_characters': len(text),
            'total_words': len(words),
            'total_sentences': len(sentences),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'avg_characters_per_word': len(text) / len(words) if words else 0,
            'readability_score': self._calculate_readability(text)
        }
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate simplified readability score"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        if not sentences or not words:
            return 0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Simplified readability (higher = more complex)
        complexity_score = avg_sentence_length / 20  # Normalize to 0-1 range roughly
        return min(complexity_score, 1.0)
    
    def _generate_insights(self, text: str, structured_data: Dict[str, Any]) -> List[str]:
        """Generate key insights from the document"""
        insights = []
        
        # Missing information insights
        expected_fields = ['policy_number', 'claim_number', 'date_patterns', 'amount_patterns']
        missing_fields = [field for field in expected_fields if field not in structured_data or not structured_data[field]]
        
        if missing_fields:
            insights.append(f"Missing critical information: {', '.join(missing_fields)}")
        
        # Amount insights
        if 'monetary_amounts' in structured_data and structured_data['monetary_amounts']:
            max_amount = max(structured_data['monetary_amounts'], key=lambda x: x['value'])
            if max_amount['value'] > 50000:
                insights.append(f"High-value claim detected: ${max_amount['value']:,.2f}")
        
        # Date insights
        if 'important_dates' in structured_data and structured_data['important_dates']:
            insights.append(f"Found {len(structured_data['important_dates'])} important dates")
        
        # Contact information
        contact_info = structured_data.get('contact_information', {})
        if contact_info:
            contact_types = list(contact_info.keys())
            insights.append(f"Contact information available: {', '.join(contact_types)}")
        
        # Document quality
        if len(text) < 500:
            insights.append("Document appears to be incomplete or very brief")
        
        return insights
    
    def _calculate_summary_quality(self, summaries: Dict[str, Any], structured_data: Dict[str, Any]) -> float:
        """Calculate overall quality score for the summarization"""
        score = 0.0
        
        # Summary completeness
        if summaries.get('extractive', {}).get('summary'):
            score += 0.3
        if summaries.get('keyword_based', {}).get('summary'):
            score += 0.2
        
        # Structured data extraction
        if structured_data:
            score += 0.3 * (len(structured_data) / 10)  # Up to 0.3 points for structured data
        
        # Information richness
        if structured_data.get('monetary_amounts'):
            score += 0.1
        if structured_data.get('important_dates'):
            score += 0.1
        
        return min(score, 1.0)


# Utility functions
def create_sample_claim_document() -> str:
    """Create a sample claim document for testing"""
    return """
    INSURANCE CLAIM REPORT
    
    Claim Number: CLM-2024-0156
    Policy Number: POL-789123
    Date of Loss: January 15, 2024
    
    Claimant Information:
    Name: John Smith
    Phone: (555) 123-4567
    Email: john.smith@email.com
    Address: 123 Main Street, Anytown, ST 12345
    
    Incident Description:
    On January 15, 2024, at approximately 3:30 PM, the insured vehicle was involved 
    in a collision at the intersection of Main Street and Oak Avenue. The claimant 
    was traveling eastbound on Main Street when another vehicle failed to stop at 
    the red light and collided with the driver's side of the insured vehicle.
    
    Damage Assessment:
    - Driver's side door: $2,500
    - Front bumper: $1,800
    - Headlight assembly: $650
    - Labor costs: $1,200
    
    Total Claim Amount: $6,150.00
    
    Police Report Filed: Yes
    Report Number: PR-2024-0789
    
    Witnesses:
    1. Mary Johnson - (555) 987-6543
    2. Robert Davis - (555) 456-7890
    
    Additional Notes:
    No injuries reported. Vehicle is drivable but requires immediate repair.
    Claimant has been with the company for 5 years with no prior claims.
    """


if __name__ == "__main__":
    # Example usage
    summarizer = DocumentSummarizer()
    
    # Test with sample document
    sample_doc = create_sample_claim_document()
    
    # Generate summary
    result = summarizer.summarize_document(sample_doc)
    
    # Display results
    print("=== DOCUMENT SUMMARIZATION RESULTS ===")
    print(f"Original length: {result['document_length']} characters")
    print(f"Quality score: {result['quality_score']:.2f}")
    
    print("\n=== EXTRACTIVE SUMMARY ===")
    extractive = result['summaries'].get('extractive', {})
    print(extractive.get('summary', 'Not available'))
    
    print("\n=== STRUCTURED DATA ===")
    for key, value in result['structured_data'].items():
        print(f"{key}: {value}")
    
    print("\n=== KEY INSIGHTS ===")
    for insight in result['key_insights']:
        print(f"• {insight}")
    
    print("\n=== STATISTICAL SUMMARY ===")
    stats = result['statistical_summary']
    print(f"Words: {stats['total_words']}")
    print(f"Sentences: {stats['total_sentences']}")
    print(f"Readability: {stats['readability_score']:.2f}")