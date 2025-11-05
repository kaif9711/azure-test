"""
Test suite for authentication functionality.
Tests login, registration, JWT token handling, and user management.
"""

import pytest
import json
from unittest.mock import patch, Mock
from datetime import datetime, timedelta
from conftest import (
    assert_response_success, 
    assert_response_error, 
    assert_valid_jwt_token,
    generate_test_user_data,
    TestConstants
)


class TestAuthRegistration:
    """Test user registration functionality."""
    
    def test_valid_registration(self, client):
        """Test successful user registration."""
        user_data = generate_test_user_data()
        
        response = client.post('/auth/register', 
                             json=user_data,
                             content_type='application/json')
        
        assert_response_success(response, 201)
        assert 'message' in response.json
        assert 'user' in response.json
        assert response.json['message'] == 'User registered successfully'
    
    def test_registration_duplicate_email(self, client):
        """Test registration with duplicate email."""
        user_data = generate_test_user_data(email=TestConstants.TEST_USER_EMAIL)
        
        response = client.post('/auth/register', 
                             json=user_data,
                             content_type='application/json')
        
        assert_response_error(response, 400)
        assert 'already exists' in response.json['error'].lower()
    
    def test_registration_missing_fields(self, client):
        """Test registration with missing required fields."""
        incomplete_data = {
            'email': 'incomplete@test.com',
            'first_name': 'Incomplete'
            # Missing last_name and password
        }
        
        response = client.post('/auth/register', 
                             json=incomplete_data,
                             content_type='application/json')
        
        assert_response_error(response, 400)
    
    def test_registration_invalid_email(self, client):
        """Test registration with invalid email format."""
        user_data = generate_test_user_data(email='invalid-email')
        
        response = client.post('/auth/register', 
                             json=user_data,
                             content_type='application/json')
        
        assert_response_error(response, 400)
    
    def test_registration_weak_password(self, client):
        """Test registration with weak password."""
        user_data = generate_test_user_data(password='123')
        
        response = client.post('/auth/register', 
                             json=user_data,
                             content_type='application/json')
        
        assert_response_error(response, 400)
    
    def test_registration_invalid_json(self, client):
        """Test registration with invalid JSON."""
        response = client.post('/auth/register', 
                             data='invalid json',
                             content_type='application/json')
        
        assert_response_error(response, 400)


class TestAuthLogin:
    """Test user login functionality."""
    
    def test_valid_login(self, client, test_user):
        """Test successful login."""
        login_data = {
            'email': TestConstants.TEST_USER_EMAIL,
            'password': TestConstants.TEST_USER_PASSWORD
        }
        
        response = client.post('/auth/login', 
                             json=login_data,
                             content_type='application/json')
        
        assert_response_success(response)
        assert 'access_token' in response.json
        assert 'user' in response.json
        assert_valid_jwt_token(response.json['access_token'])
        
        user_info = response.json['user']
        assert user_info['email'] == TestConstants.TEST_USER_EMAIL
        assert 'password_hash' not in user_info  # Password should not be returned
    
    def test_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        login_data = {
            'email': TestConstants.TEST_USER_EMAIL,
            'password': 'wrongpassword'
        }
        
        response = client.post('/auth/login', 
                             json=login_data,
                             content_type='application/json')
        
        assert_response_error(response, 401)
        assert 'invalid' in response.json['error'].lower()
    
    def test_login_nonexistent_user(self, client):
        """Test login with non-existent user."""
        login_data = {
            'email': 'nonexistent@test.com',
            'password': 'anypassword'
        }
        
        response = client.post('/auth/login', 
                             json=login_data,
                             content_type='application/json')
        
        assert_response_error(response, 401)
    
    def test_login_missing_fields(self, client):
        """Test login with missing fields."""
        # Missing password
        login_data = {'email': TestConstants.TEST_USER_EMAIL}
        
        response = client.post('/auth/login', 
                             json=login_data,
                             content_type='application/json')
        
        assert_response_error(response, 400)
    
    def test_login_inactive_user(self, client):
        """Test login with inactive user account."""
        # This would require setting up an inactive user in the test database
        # For now, we'll test the basic flow
        pass


class TestAuthProfile:
    """Test user profile management."""
    
    def test_get_profile_authenticated(self, client, auth_headers):
        """Test getting user profile when authenticated."""
        response = client.get('/auth/profile', headers=auth_headers)
        
        assert_response_success(response)
        assert 'user' in response.json
        
        user_info = response.json['user']
        assert 'email' in user_info
        assert 'first_name' in user_info
        assert 'last_name' in user_info
        assert 'role' in user_info
        assert 'password_hash' not in user_info
    
    def test_get_profile_unauthenticated(self, client):
        """Test getting profile without authentication."""
        response = client.get('/auth/profile')
        
        assert_response_error(response, 401)
    
    def test_update_profile(self, client, auth_headers):
        """Test updating user profile."""
        update_data = {
            'first_name': 'Updated',
            'last_name': 'Name'
        }
        
        response = client.put('/auth/profile', 
                            json=update_data,
                            headers=auth_headers,
                            content_type='application/json')
        
        assert_response_success(response)
        assert 'user' in response.json
        
        user_info = response.json['user']
        assert user_info['first_name'] == 'Updated'
        assert user_info['last_name'] == 'Name'
    
    def test_update_profile_invalid_data(self, client, auth_headers):
        """Test updating profile with invalid data."""
        update_data = {
            'email': 'invalid-email-format'
        }
        
        response = client.put('/auth/profile', 
                            json=update_data,
                            headers=auth_headers,
                            content_type='application/json')
        
        assert_response_error(response, 400)


class TestPasswordChange:
    """Test password change functionality."""
    
    def test_change_password_valid(self, client, auth_headers):
        """Test successful password change."""
        change_data = {
            'current_password': TestConstants.TEST_USER_PASSWORD,
            'new_password': 'newtestpassword123'
        }
        
        response = client.post('/auth/change-password', 
                             json=change_data,
                             headers=auth_headers,
                             content_type='application/json')
        
        assert_response_success(response)
        assert 'message' in response.json
    
    def test_change_password_wrong_current(self, client, auth_headers):
        """Test password change with wrong current password."""
        change_data = {
            'current_password': 'wrongcurrentpassword',
            'new_password': 'newtestpassword123'
        }
        
        response = client.post('/auth/change-password', 
                             json=change_data,
                             headers=auth_headers,
                             content_type='application/json')
        
        assert_response_error(response, 400)
    
    def test_change_password_weak_new(self, client, auth_headers):
        """Test password change with weak new password."""
        change_data = {
            'current_password': TestConstants.TEST_USER_PASSWORD,
            'new_password': '123'
        }
        
        response = client.post('/auth/change-password', 
                             json=change_data,
                             headers=auth_headers,
                             content_type='application/json')
        
        assert_response_error(response, 400)
    
    def test_change_password_unauthenticated(self, client):
        """Test password change without authentication."""
        change_data = {
            'current_password': TestConstants.TEST_USER_PASSWORD,
            'new_password': 'newtestpassword123'
        }
        
        response = client.post('/auth/change-password', 
                             json=change_data,
                             content_type='application/json')
        
        assert_response_error(response, 401)


class TestJWTTokenHandling:
    """Test JWT token functionality."""
    
    def test_jwt_token_structure(self, client):
        """Test JWT token has correct structure."""
        login_data = {
            'email': TestConstants.TEST_USER_EMAIL,
            'password': TestConstants.TEST_USER_PASSWORD
        }
        
        response = client.post('/auth/login', json=login_data)
        assert_response_success(response)
        
        token = response.json['access_token']
        assert_valid_jwt_token(token)
    
    def test_protected_endpoint_with_valid_token(self, client, auth_headers):
        """Test accessing protected endpoint with valid token."""
        response = client.get('/auth/profile', headers=auth_headers)
        assert_response_success(response)
    
    def test_protected_endpoint_without_token(self, client):
        """Test accessing protected endpoint without token."""
        response = client.get('/auth/profile')
        assert_response_error(response, 401)
    
    def test_protected_endpoint_with_invalid_token(self, client):
        """Test accessing protected endpoint with invalid token."""
        headers = {'Authorization': 'Bearer invalid_token'}
        response = client.get('/auth/profile', headers=headers)
        assert_response_error(response, 401)
    
    def test_protected_endpoint_with_malformed_header(self, client):
        """Test accessing protected endpoint with malformed auth header."""
        headers = {'Authorization': 'InvalidFormat token'}
        response = client.get('/auth/profile', headers=headers)
        assert_response_error(response, 401)


class TestAuthUtilFunctions:
    """Test authentication utility functions."""
    
    def test_password_hashing(self):
        """Test password hashing functionality."""
        from utils.auth_utils import hash_password, verify_password
        
        password = 'testpassword123'
        hashed = hash_password(password)
        
        assert hashed != password  # Hash should be different from original
        assert len(hashed) > 50  # Bcrypt hashes are typically 60 characters
        assert verify_password(password, hashed)  # Verification should work
        assert not verify_password('wrongpassword', hashed)  # Wrong password should fail
    
    def test_password_validation(self):
        """Test password strength validation."""
        from utils.auth_utils import validate_password_strength
        
        # Valid passwords
        assert validate_password_strength('StrongPass123!')['is_valid']
        assert validate_password_strength('GoodPassword1')['is_valid']
        
        # Invalid passwords
        assert not validate_password_strength('weak')['is_valid']
        assert not validate_password_strength('123456')['is_valid']
        assert not validate_password_strength('')['is_valid']
    
    def test_email_validation(self):
        """Test email validation functionality."""
        from utils.auth_utils import validate_email
        
        # Valid emails
        assert validate_email('user@example.com')['valid']
        assert validate_email('test.email+tag@domain.co.uk')['valid']

        # Invalid emails
        assert not validate_email('invalid-email')['valid']
        assert not validate_email('@domain.com')['valid']
        assert not validate_email('user@')['valid']
        assert not validate_email('')['valid']


class TestRoleBasedAccess:
    """Test role-based access control."""
    
    def test_user_role_access(self, client, auth_headers):
        """Test user role access to appropriate endpoints."""
        # Users should be able to access their profile
        response = client.get('/auth/profile', headers=auth_headers)
        assert_response_success(response)
        
        # Users should NOT be able to access admin endpoints
        response = client.get('/admin/users', headers=auth_headers)
        assert_response_error(response, 403)  # Forbidden
    
    def test_admin_role_access(self, client, admin_headers):
        """Test admin role access to all endpoints."""
        # Admins should be able to access their profile
        response = client.get('/auth/profile', headers=admin_headers)
        assert_response_success(response)
        
        # Admins should be able to access admin endpoints
        response = client.get('/admin/users', headers=admin_headers)
        assert_response_success(response)
    
    def test_role_inheritance(self, client, admin_headers):
        """Test that higher roles can access lower role functions."""
        # Admin should be able to do everything a user can do
        response = client.get('/auth/profile', headers=admin_headers)
        assert_response_success(response)


class TestAuthErrorHandling:
    """Test authentication error handling."""
    
    def test_database_error_handling(self, client):
        """Test handling of database errors during authentication."""
        with patch('utils.db.get_db_connection', side_effect=Exception('Database error')):
            login_data = {
                'email': TestConstants.TEST_USER_EMAIL,
                'password': TestConstants.TEST_USER_PASSWORD
            }
            
            response = client.post('/auth/login', json=login_data)
            assert response.status_code >= 500  # Server error
    
    def test_rate_limiting(self, client):
        """Test rate limiting for authentication attempts."""
        # This would require implementing rate limiting
        # For now, we'll test the basic concept
        login_data = {
            'email': TestConstants.TEST_USER_EMAIL,
            'password': 'wrongpassword'
        }
        
        # Multiple failed attempts
        for _ in range(5):
            response = client.post('/auth/login', json=login_data)
            assert_response_error(response, 401)
    
    def test_session_timeout(self, client):
        """Test JWT token expiration."""
        # This would require mocking time or using expired tokens
        # Implementation depends on JWT configuration
        pass


class TestSecurityFeatures:
    """Test security-related authentication features."""
    
    def test_password_not_logged(self, client, caplog):
        """Test that passwords are not logged in plain text."""
        login_data = {
            'email': TestConstants.TEST_USER_EMAIL,
            'password': TestConstants.TEST_USER_PASSWORD
        }
        
        response = client.post('/auth/login', json=login_data)
        
        # Check that password is not in logs
        assert TestConstants.TEST_USER_PASSWORD not in caplog.text
    
    def test_sql_injection_protection(self, client):
        """Test protection against SQL injection in login."""
        malicious_data = {
            'email': "admin@test.com'; DROP TABLE users; --",
            'password': 'anypassword'
        }
        
        response = client.post('/auth/login', json=malicious_data)
        
        # Should either fail authentication or handle gracefully
        assert response.status_code in [400, 401]
    
    def test_xss_protection(self, client):
        """Test protection against XSS in user input."""
        malicious_data = generate_test_user_data(
            first_name='<script>alert("xss")</script>',
            email='test_xss@test.com'
        )
        
        response = client.post('/auth/register', json=malicious_data)
        
        # Should either sanitize input or reject it
        if response.status_code == 201:
            # If registration succeeds, script should be sanitized
            assert '<script>' not in response.json.get('message', '')


class TestAuthIntegration:
    """Integration tests for authentication system."""
    
    def test_complete_user_lifecycle(self, client):
        """Test complete user registration, login, and profile management."""
        # 1. Register a new user
        user_data = generate_test_user_data(email='lifecycle@test.com')
        response = client.post('/auth/register', json=user_data)
        assert_response_success(response, 201)
        
        # 2. Login with new user
        login_data = {
            'email': user_data['email'],
            'password': user_data['password']
        }
        response = client.post('/auth/login', json=login_data)
        assert_response_success(response)
        
        token = response.json['access_token']
        headers = {'Authorization': f'Bearer {token}'}
        
        # 3. Get profile
        response = client.get('/auth/profile', headers=headers)
        assert_response_success(response)
        
        # 4. Update profile
        update_data = {'first_name': 'Updated'}
        response = client.put('/auth/profile', json=update_data, headers=headers)
        assert_response_success(response)
        
        # 5. Change password
        change_data = {
            'current_password': user_data['password'],
            'new_password': 'newpassword123'
        }
        response = client.post('/auth/change-password', json=change_data, headers=headers)
        assert_response_success(response)
        
        # 6. Login with new password
        login_data['password'] = 'newpassword123'
        response = client.post('/auth/login', json=login_data)
        assert_response_success(response)