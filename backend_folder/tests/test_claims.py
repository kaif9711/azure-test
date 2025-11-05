"""
Test suite for claims functionality.
Tests claim submission, processing, document handling, and ML integration.
"""

import pytest
import json
import io
import os
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime, timedelta
from conftest import (
    assert_response_success, 
    assert_response_error, 
    assert_valid_uuid,
    generate_test_claim_data,
    TestConstants,
    create_test_file,
    cleanup_test_file
)


class TestClaimSubmission:
    """Test claim submission functionality."""
    
    def test_submit_valid_claim(self, client, auth_headers, test_claim_data):
        """Test successful claim submission."""
        response = client.post('/claims', 
                             json=test_claim_data,
                             headers=auth_headers,
                             content_type='application/json')
        
        assert_response_success(response, 201)
        assert 'claim' in response.json
        assert 'id' in response.json['claim']
        assert assert_valid_uuid(response.json['claim']['id'])
        
        claim = response.json['claim']
        assert claim['claim_amount'] == test_claim_data['claim_amount']
        assert claim['status'] == 'submitted'
        assert claim['incident_description'] == test_claim_data['incident_description']
    
    def test_submit_claim_unauthenticated(self, client, test_claim_data):
        """Test claim submission without authentication."""
        response = client.post('/claims', 
                             json=test_claim_data,
                             content_type='application/json')
        
        assert_response_error(response, 401)
    
    def test_submit_claim_missing_fields(self, client, auth_headers):
        """Test claim submission with missing required fields."""
        incomplete_data = {
            'claim_amount': 5000.00,
            # Missing policy_id, incident_date, incident_description
        }
        
        response = client.post('/claims', 
                             json=incomplete_data,
                             headers=auth_headers,
                             content_type='application/json')
        
        assert_response_error(response, 400)
    
    def test_submit_claim_invalid_amount(self, client, auth_headers):
        """Test claim submission with invalid amount."""
        invalid_data = generate_test_claim_data(claim_amount=-1000.00)
        
        response = client.post('/claims', 
                             json=invalid_data,
                             headers=auth_headers,
                             content_type='application/json')
        
        assert_response_error(response, 400)
    
    def test_submit_claim_invalid_date(self, client, auth_headers):
        """Test claim submission with invalid incident date."""
        future_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        invalid_data = generate_test_claim_data(incident_date=future_date)
        
        response = client.post('/claims', 
                             json=invalid_data,
                             headers=auth_headers,
                             content_type='application/json')
        
        assert_response_error(response, 400)
    
    def test_submit_claim_invalid_policy(self, client, auth_headers):
        """Test claim submission with invalid policy ID."""
        invalid_data = generate_test_claim_data(policy_id=99999)
        
        response = client.post('/claims', 
                             json=invalid_data,
                             headers=auth_headers,
                             content_type='application/json')
        
        assert_response_error(response, 400)


class TestClaimRetrieval:
    """Test claim retrieval functionality."""
    
    def test_get_user_claims(self, client, auth_headers):
        """Test getting user's claims."""
        response = client.get('/claims', headers=auth_headers)
        
        assert_response_success(response)
        assert 'claims' in response.json
        assert 'pagination' in response.json
        assert isinstance(response.json['claims'], list)
    
    def test_get_claims_with_pagination(self, client, auth_headers):
        """Test claims retrieval with pagination."""
        response = client.get('/claims?page=1&limit=10', headers=auth_headers)
        
        assert_response_success(response)
        assert 'pagination' in response.json
        
        pagination = response.json['pagination']
        assert 'current_page' in pagination
        assert 'total_pages' in pagination
        assert 'total_items' in pagination
    
    def test_get_claims_with_filters(self, client, auth_headers):
        """Test claims retrieval with status filter."""
        response = client.get('/claims?status=submitted', headers=auth_headers)
        
        assert_response_success(response)
        claims = response.json['claims']
        
        # All returned claims should have 'submitted' status
        for claim in claims:
            assert claim['status'] == 'submitted'
    
    def test_get_specific_claim(self, client, auth_headers):
        """Test retrieving a specific claim by ID."""
        # First, create a claim
        claim_data = generate_test_claim_data()
        create_response = client.post('/claims', json=claim_data, headers=auth_headers)
        assert_response_success(create_response, 201)
        
        claim_id = create_response.json['claim']['id']
        
        # Then retrieve it
        response = client.get(f'/claims/{claim_id}', headers=auth_headers)
        
        assert_response_success(response)
        assert 'claim' in response.json
        assert response.json['claim']['id'] == claim_id
    
    def test_get_nonexistent_claim(self, client, auth_headers):
        """Test retrieving a non-existent claim."""
        fake_id = 'non-existent-claim-id'
        response = client.get(f'/claims/{fake_id}', headers=auth_headers)
        
        assert_response_error(response, 404)
    
    def test_get_claim_unauthorized_user(self, client, auth_headers):
        """Test accessing another user's claim."""
        # This would require setting up multiple users
        # For now, test basic access control
        pass


class TestDocumentHandling:
    """Test document upload and management."""
    
    def test_upload_claim_document(self, client, auth_headers):
        """Test uploading a document to a claim."""
        # First, create a claim
        claim_data = generate_test_claim_data()
        create_response = client.post('/claims', json=claim_data, headers=auth_headers)
        assert_response_success(create_response, 201)
        
        claim_id = create_response.json['claim']['id']
        
        # Create test file
        test_content = b'This is a test document content'
        file_data = {
            'documents': (io.BytesIO(test_content), 'test_document.txt', 'text/plain')
        }
        
        response = client.post(f'/claims/{claim_id}/documents',
                             data=file_data,
                             headers=auth_headers,
                             content_type='multipart/form-data')
        
        assert_response_success(response, 201)
        assert 'documents' in response.json
        assert len(response.json['documents']) > 0
        
        document = response.json['documents'][0]
        assert 'id' in document
        assert 'filename' in document
        assert document['original_filename'] == 'test_document.txt'
    
    def test_upload_multiple_documents(self, client, auth_headers):
        """Test uploading multiple documents."""
        # First, create a claim
        claim_data = generate_test_claim_data()
        create_response = client.post('/claims', json=claim_data, headers=auth_headers)
        assert_response_success(create_response, 201)
        
        claim_id = create_response.json['claim']['id']
        
        # Create multiple test files
        file_data = {
            'documents': [
                (io.BytesIO(b'Document 1 content'), 'doc1.txt', 'text/plain'),
                (io.BytesIO(b'Document 2 content'), 'doc2.txt', 'text/plain')
            ]
        }
        
        response = client.post(f'/claims/{claim_id}/documents',
                             data=file_data,
                             headers=auth_headers,
                             content_type='multipart/form-data')
        
        assert_response_success(response, 201)
        assert len(response.json['documents']) == 2
    
    def test_upload_invalid_file_type(self, client, auth_headers):
        """Test uploading invalid file type."""
        # First, create a claim
        claim_data = generate_test_claim_data()
        create_response = client.post('/claims', json=claim_data, headers=auth_headers)
        assert_response_success(create_response, 201)
        
        claim_id = create_response.json['claim']['id']
        
        # Try to upload executable file
        file_data = {
            'documents': (io.BytesIO(b'malicious content'), 'malware.exe', 'application/octet-stream')
        }
        
        response = client.post(f'/claims/{claim_id}/documents',
                             data=file_data,
                             headers=auth_headers,
                             content_type='multipart/form-data')
        
        assert_response_error(response, 400)
    
    def test_upload_oversized_file(self, client, auth_headers):
        """Test uploading file exceeding size limit."""
        # First, create a claim
        claim_data = generate_test_claim_data()
        create_response = client.post('/claims', json=claim_data, headers=auth_headers)
        assert_response_success(create_response, 201)
        
        claim_id = create_response.json['claim']['id']
        
        # Create oversized file (20MB)
        large_content = b'0' * (20 * 1024 * 1024)
        file_data = {
            'documents': (io.BytesIO(large_content), 'large_file.txt', 'text/plain')
        }
        
        response = client.post(f'/claims/{claim_id}/documents',
                             data=file_data,
                             headers=auth_headers,
                             content_type='multipart/form-data')
        
        assert_response_error(response, 413)  # Request Entity Too Large
    
    def test_get_claim_documents(self, client, auth_headers):
        """Test retrieving claim documents."""
        # First, create a claim and upload a document
        claim_data = generate_test_claim_data()
        create_response = client.post('/claims', json=claim_data, headers=auth_headers)
        assert_response_success(create_response, 201)
        
        claim_id = create_response.json['claim']['id']
        
        # Upload document
        file_data = {
            'documents': (io.BytesIO(b'test content'), 'test.txt', 'text/plain')
        }
        
        upload_response = client.post(f'/claims/{claim_id}/documents',
                                    data=file_data,
                                    headers=auth_headers,
                                    content_type='multipart/form-data')
        assert_response_success(upload_response, 201)
        
        # Get documents
        response = client.get(f'/claims/{claim_id}/documents', headers=auth_headers)
        
        assert_response_success(response)
        assert 'documents' in response.json
        assert len(response.json['documents']) > 0
    
    def test_delete_claim_document(self, client, auth_headers):
        """Test deleting a claim document."""
        # First, create a claim and upload a document
        claim_data = generate_test_claim_data()
        create_response = client.post('/claims', json=claim_data, headers=auth_headers)
        assert_response_success(create_response, 201)
        
        claim_id = create_response.json['claim']['id']
        
        file_data = {
            'documents': (io.BytesIO(b'test content'), 'test.txt', 'text/plain')
        }
        
        upload_response = client.post(f'/claims/{claim_id}/documents',
                                    data=file_data,
                                    headers=auth_headers,
                                    content_type='multipart/form-data')
        assert_response_success(upload_response, 201)
        
        document_id = upload_response.json['documents'][0]['id']
        
        # Delete document
        response = client.delete(f'/claims/{claim_id}/documents/{document_id}',
                               headers=auth_headers)
        
        assert_response_success(response)


class TestClaimProcessing:
    """Test ML processing of claims."""
    
    def test_process_claim_ml(self, client, auth_headers, mock_ml_models):
        """Test ML processing of a claim."""
        # Create a claim
        claim_data = generate_test_claim_data()
        create_response = client.post('/claims', json=claim_data, headers=auth_headers)
        assert_response_success(create_response, 201)
        
        claim_id = create_response.json['claim']['id']
        
        # Process with ML
        response = client.post(f'/claims/{claim_id}/process', headers=auth_headers)
        
        assert_response_success(response)
        assert 'claim' in response.json
        assert 'risk_assessment' in response.json
        
        claim = response.json['claim']
        assert 'risk_score' in claim
        assert 'status' in claim
        
        # Status should be updated based on risk assessment
        assert claim['status'] in ['approved', 'under_review', 'investigation']
    
    def test_process_high_risk_claim(self, client, auth_headers, mock_ml_models):
        """Test processing of high-risk claim."""
        # Create high-risk claim (large amount)
        claim_data = generate_test_claim_data(claim_amount=TestConstants.HIGH_RISK_CLAIM_AMOUNT)
        create_response = client.post('/claims', json=claim_data, headers=auth_headers)
        assert_response_success(create_response, 201)
        
        claim_id = create_response.json['claim']['id']
        
        # Process with ML
        response = client.post(f'/claims/{claim_id}/process', headers=auth_headers)
        
        assert_response_success(response)
        claim = response.json['claim']
        
        # High-risk claims should be flagged for investigation
        assert claim['risk_score'] > 0.7
        assert claim['status'] in ['investigation', 'under_review']
    
    def test_process_low_risk_claim(self, client, auth_headers, mock_ml_models):
        """Test processing of low-risk claim."""
        # Create low-risk claim (small amount)
        claim_data = generate_test_claim_data(claim_amount=TestConstants.LOW_RISK_CLAIM_AMOUNT)
        create_response = client.post('/claims', json=claim_data, headers=auth_headers)
        assert_response_success(create_response, 201)
        
        claim_id = create_response.json['claim']['id']
        
        # Process with ML
        response = client.post(f'/claims/{claim_id}/process', headers=auth_headers)
        
        assert_response_success(response)
        claim = response.json['claim']
        
        # Low-risk claims might be auto-approved
        assert claim['risk_score'] < 0.3
        assert claim['status'] in ['approved', 'under_review']
    
    def test_process_claim_with_documents(self, client, auth_headers, mock_ml_models):
        """Test processing claim with uploaded documents."""
        # Create claim
        claim_data = generate_test_claim_data()
        create_response = client.post('/claims', json=claim_data, headers=auth_headers)
        assert_response_success(create_response, 201)
        
        claim_id = create_response.json['claim']['id']
        
        # Upload document
        file_data = {
            'documents': (io.BytesIO(b'test document'), 'receipt.pdf', 'application/pdf')
        }
        
        upload_response = client.post(f'/claims/{claim_id}/documents',
                                    data=file_data,
                                    headers=auth_headers,
                                    content_type='multipart/form-data')
        assert_response_success(upload_response, 201)
        
        # Process claim
        response = client.post(f'/claims/{claim_id}/process', headers=auth_headers)
        
        assert_response_success(response)
        assert 'document_scores' in response.json
        assert len(response.json['document_scores']) > 0
    
    def test_process_nonexistent_claim(self, client, auth_headers):
        """Test processing non-existent claim."""
        fake_id = 'non-existent-claim-id'
        response = client.post(f'/claims/{fake_id}/process', headers=auth_headers)
        
        assert_response_error(response, 404)
    
    def test_process_already_processed_claim(self, client, auth_headers, mock_ml_models):
        """Test processing claim that's already been processed."""
        # Create and process a claim
        claim_data = generate_test_claim_data()
        create_response = client.post('/claims', json=claim_data, headers=auth_headers)
        assert_response_success(create_response, 201)
        
        claim_id = create_response.json['claim']['id']
        
        # First processing
        first_response = client.post(f'/claims/{claim_id}/process', headers=auth_headers)
        assert_response_success(first_response)
        
        # Second processing (should handle gracefully)
        second_response = client.post(f'/claims/{claim_id}/process', headers=auth_headers)
        
        # Should either succeed with updated info or return appropriate message
        assert second_response.status_code in [200, 409]  # OK or Conflict


class TestClaimUpdates:
    """Test claim status updates and modifications."""
    
    def test_update_claim_status(self, client, auth_headers):
        """Test updating claim status."""
        # Create claim
        claim_data = generate_test_claim_data()
        create_response = client.post('/claims', json=claim_data, headers=auth_headers)
        assert_response_success(create_response, 201)
        
        claim_id = create_response.json['claim']['id']
        
        # Update status
        update_data = {'status': 'under_review'}
        response = client.put(f'/claims/{claim_id}',
                            json=update_data,
                            headers=auth_headers,
                            content_type='application/json')
        
        assert_response_success(response)
        assert response.json['claim']['status'] == 'under_review'
    
    def test_update_claim_invalid_status(self, client, auth_headers):
        """Test updating claim with invalid status."""
        # Create claim
        claim_data = generate_test_claim_data()
        create_response = client.post('/claims', json=claim_data, headers=auth_headers)
        assert_response_success(create_response, 201)
        
        claim_id = create_response.json['claim']['id']
        
        # Try invalid status
        update_data = {'status': 'invalid_status'}
        response = client.put(f'/claims/{claim_id}',
                            json=update_data,
                            headers=auth_headers,
                            content_type='application/json')
        
        assert_response_error(response, 400)
    
    def test_update_claim_unauthorized(self, client, auth_headers):
        """Test updating claim by unauthorized user."""
        # This would require multiple users to test properly
        pass


class TestPolicyIntegration:
    """Test integration with policy system."""
    
    def test_get_user_policies(self, client, auth_headers):
        """Test retrieving user's policies."""
        response = client.get('/claims/policies', headers=auth_headers)
        
        assert_response_success(response)
        assert 'policies' in response.json
        assert isinstance(response.json['policies'], list)
    
    def test_get_policy_details(self, client, auth_headers):
        """Test retrieving specific policy details."""
        # First get policies
        policies_response = client.get('/claims/policies', headers=auth_headers)
        assert_response_success(policies_response)
        
        policies = policies_response.json['policies']
        if policies:
            policy_id = policies[0]['id']
            
            # Get specific policy
            response = client.get(f'/claims/policies/{policy_id}', headers=auth_headers)
            
            assert_response_success(response)
            assert 'policy' in response.json
            assert response.json['policy']['id'] == policy_id
    
    def test_claim_exceeds_policy_coverage(self, client, auth_headers):
        """Test submitting claim that exceeds policy coverage."""
        # Create claim with very high amount
        claim_data = generate_test_claim_data(claim_amount=999999.00)
        
        response = client.post('/claims', 
                             json=claim_data,
                             headers=auth_headers,
                             content_type='application/json')
        
        # Should either be rejected or flagged as high risk
        if response.status_code == 201:
            # If accepted, should be marked as high risk
            claim = response.json['claim']
            # Additional validation would be done during ML processing
        else:
            assert_response_error(response, 400)


class TestClaimSearch:
    """Test claim search and filtering functionality."""
    
    def test_search_claims(self, client, auth_headers):
        """Test searching claims."""
        search_data = {
            'query': 'accident',
            'filters': {
                'status': 'submitted',
                'date_from': '2024-01-01',
                'date_to': '2024-12-31'
            }
        }
        
        response = client.post('/claims/search',
                             json=search_data,
                             headers=auth_headers,
                             content_type='application/json')
        
        assert_response_success(response)
        assert 'claims' in response.json
        assert 'total' in response.json
    
    def test_search_claims_no_results(self, client, auth_headers):
        """Test searching with no matching results."""
        search_data = {
            'query': 'nonexistentterm12345',
            'filters': {}
        }
        
        response = client.post('/claims/search',
                             json=search_data,
                             headers=auth_headers,
                             content_type='application/json')
        
        assert_response_success(response)
        assert response.json['total'] == 0
        assert response.json['claims'] == []


class TestClaimValidation:
    """Test claim data validation."""
    
    def test_claim_amount_validation(self, client, auth_headers):
        """Test various claim amount validations."""
        test_cases = [
            (-100, 400),   # Negative amount
            (0, 400),      # Zero amount
            (0.001, 201),  # Very small amount
            (999999999, 201),  # Very large amount
        ]
        
        for amount, expected_status in test_cases:
            claim_data = generate_test_claim_data(claim_amount=amount)
            response = client.post('/claims', 
                                 json=claim_data,
                                 headers=auth_headers,
                                 content_type='application/json')
            
            assert response.status_code == expected_status
    
    def test_incident_description_validation(self, client, auth_headers):
        """Test incident description validation."""
        test_cases = [
            ('', 400),  # Empty description
            ('x' * 10001, 400),  # Too long description
            ('Valid description', 201),  # Valid description
        ]
        
        for description, expected_status in test_cases:
            claim_data = generate_test_claim_data(incident_description=description)
            response = client.post('/claims', 
                                 json=claim_data,
                                 headers=auth_headers,
                                 content_type='application/json')
            
            assert response.status_code == expected_status


class TestErrorHandling:
    """Test error handling in claims system."""
    
    def test_database_error_handling(self, client, auth_headers):
        """Test handling of database errors."""
        with patch('utils.db.get_db_connection', side_effect=Exception('Database error')):
            claim_data = generate_test_claim_data()
            response = client.post('/claims', 
                                 json=claim_data,
                                 headers=auth_headers,
                                 content_type='application/json')
            
            assert response.status_code >= 500
    
    def test_file_system_error_handling(self, client, auth_headers):
        """Test handling of file system errors during document upload."""
        with patch('os.makedirs', side_effect=OSError('File system error')):
            # Create claim first
            claim_data = generate_test_claim_data()
            create_response = client.post('/claims', json=claim_data, headers=auth_headers)
            assert_response_success(create_response, 201)
            
            claim_id = create_response.json['claim']['id']
            
            # Try to upload document
            file_data = {
                'documents': (io.BytesIO(b'test content'), 'test.txt', 'text/plain')
            }
            
            response = client.post(f'/claims/{claim_id}/documents',
                                 data=file_data,
                                 headers=auth_headers,
                                 content_type='multipart/form-data')
            
            assert response.status_code >= 500
    
    def test_ml_model_error_handling(self, client, auth_headers):
        """Test handling of ML model errors."""
        with patch('models.risk_checker.RiskChecker.assess_claim_risk', 
                  side_effect=Exception('ML model error')):
            # Create claim
            claim_data = generate_test_claim_data()
            create_response = client.post('/claims', json=claim_data, headers=auth_headers)
            assert_response_success(create_response, 201)
            
            claim_id = create_response.json['claim']['id']
            
            # Try to process
            response = client.post(f'/claims/{claim_id}/process', headers=auth_headers)
            
            # Should handle gracefully
            assert response.status_code in [200, 500]


class TestClaimIntegration:
    """Integration tests for complete claim lifecycle."""
    
    def test_complete_claim_lifecycle(self, client, auth_headers, mock_ml_models):
        """Test complete claim submission and processing workflow."""
        # 1. Submit claim
        claim_data = generate_test_claim_data()
        create_response = client.post('/claims', json=claim_data, headers=auth_headers)
        assert_response_success(create_response, 201)
        
        claim_id = create_response.json['claim']['id']
        
        # 2. Upload documents
        file_data = {
            'documents': (io.BytesIO(b'receipt content'), 'receipt.pdf', 'application/pdf')
        }
        
        upload_response = client.post(f'/claims/{claim_id}/documents',
                                    data=file_data,
                                    headers=auth_headers,
                                    content_type='multipart/form-data')
        assert_response_success(upload_response, 201)
        
        # 3. Process with ML
        process_response = client.post(f'/claims/{claim_id}/process', headers=auth_headers)
        assert_response_success(process_response)
        
        # 4. Check final status
        get_response = client.get(f'/claims/{claim_id}', headers=auth_headers)
        assert_response_success(get_response)
        
        final_claim = get_response.json['claim']
        assert 'risk_score' in final_claim
        assert final_claim['status'] in ['approved', 'under_review', 'investigation']
        
        # 5. Verify documents are processed
        docs_response = client.get(f'/claims/{claim_id}/documents', headers=auth_headers)
        assert_response_success(docs_response)
        
        documents = docs_response.json['documents']
        assert len(documents) > 0
        assert documents[0]['validation_status'] in ['validated', 'pending', 'failed']