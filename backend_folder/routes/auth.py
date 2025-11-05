# Deprecated legacy Flask auth routes.
# Replaced by FastAPI implementation in auth_fastapi.py.

"""
Authentication Routes
Handles user authentication, registration, and session management
"""

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, get_jwt
import bcrypt
from datetime import timedelta
import logging
from utils.db import get_db_connection
from utils.auth_utils import validate_password, hash_password, verify_password
import re

logger = logging.getLogger(__name__)
auth_bp = Blueprint('auth', __name__)

# Email validation regex
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['email', 'password', 'first_name', 'last_name']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        email = data['email'].lower().strip()
        password = data['password']
        first_name = data['first_name'].strip()
        last_name = data['last_name'].strip()
        role = data.get('role', 'user')  # Default role is 'user'
        
        # Validate email format
        if not EMAIL_REGEX.match(email):
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Validate password
        password_validation = validate_password(password)
        if not password_validation['valid']:
            return jsonify({'error': password_validation['message']}), 400
        
        # Validate role
        valid_roles = ['user', 'admin', 'investigator', 'supervisor']
        if role not in valid_roles:
            return jsonify({'error': 'Invalid role'}), 400
        
        # Check if user already exists
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            return jsonify({'error': 'User with this email already exists'}), 409
        
        # Hash password
        password_hash = hash_password(password)
        
        # Insert new user
        insert_query = """
            INSERT INTO users (email, password_hash, first_name, last_name, role, is_active, created_at)
            VALUES (%s, %s, %s, %s, %s, true, NOW())
            RETURNING id, email, first_name, last_name, role, created_at
        """
        
        cursor.execute(insert_query, (email, password_hash, first_name, last_name, role))
        user_data = cursor.fetchone()
        conn.commit()
        
        # Create access token
        access_token = create_access_token(
            identity=str(user_data[0]),
            additional_claims={
                'email': user_data[1],
                'role': user_data[4]
            }
        )
        
        logger.info(f"New user registered: {email}")
        
        return jsonify({
            'message': 'User registered successfully',
            'access_token': access_token,
            'user': {
                'id': user_data[0],
                'email': user_data[1],
                'first_name': user_data[2],
                'last_name': user_data[3],
                'role': user_data[4],
                'created_at': user_data[5].isoformat()
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500
    finally:
        if 'conn' in locals():
            conn.close()

@auth_bp.route('/login', methods=['POST'])
def login():
    """Authenticate user and return access token"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Email and password are required'}), 400
        
        email = data['email'].lower().strip()
        password = data['password']
        
        # Get user from database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT id, email, password_hash, first_name, last_name, role, is_active, last_login
            FROM users WHERE email = %s
        """
        cursor.execute(query, (email,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Check if user is active
        if not user[6]:  # is_active
            return jsonify({'error': 'Account is deactivated'}), 401
        
        # Verify password
        if not verify_password(password, user[2]):  # password_hash
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Update last login
        cursor.execute(
            "UPDATE users SET last_login = NOW() WHERE id = %s",
            (user[0],)
        )
        conn.commit()
        
        # Create access token
        access_token = create_access_token(
            identity=str(user[0]),
            additional_claims={
                'email': user[1],
                'role': user[5]
            }
        )
        
        logger.info(f"User logged in: {email}")
        
        return jsonify({
            'message': 'Login successful',
            'access_token': access_token,
            'user': {
                'id': user[0],
                'email': user[1],
                'first_name': user[3],
                'last_name': user[4],
                'role': user[5],
                'last_login': user[7].isoformat() if user[7] else None
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500
    finally:
        if 'conn' in locals():
            conn.close()

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """Logout user (in a real app, you'd blacklist the token)"""
    try:
        user_id = get_jwt_identity()
        logger.info(f"User logged out: {user_id}")
        
        return jsonify({'message': 'Logout successful'}), 200
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return jsonify({'error': 'Logout failed'}), 500

@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get current user profile"""
    try:
        user_id = get_jwt_identity()
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT id, email, first_name, last_name, role, created_at, last_login, is_active
            FROM users WHERE id = %s
        """
        cursor.execute(query, (user_id,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'user': {
                'id': user[0],
                'email': user[1],
                'first_name': user[2],
                'last_name': user[3],
                'role': user[4],
                'created_at': user[5].isoformat(),
                'last_login': user[6].isoformat() if user[6] else None,
                'is_active': user[7]
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Get profile error: {str(e)}")
        return jsonify({'error': 'Failed to get profile'}), 500
    finally:
        if 'conn' in locals():
            conn.close()

@auth_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update user profile"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        # Allowed fields to update
        allowed_fields = ['first_name', 'last_name', 'email']
        updates = {}
        
        for field in allowed_fields:
            if field in data:
                updates[field] = data[field].strip() if isinstance(data[field], str) else data[field]
        
        if not updates:
            return jsonify({'error': 'No valid fields to update'}), 400
        
        # Validate email if provided
        if 'email' in updates:
            email = updates['email'].lower()
            if not EMAIL_REGEX.match(email):
                return jsonify({'error': 'Invalid email format'}), 400
            updates['email'] = email
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if email is already taken (if updating email)
        if 'email' in updates:
            cursor.execute(
                "SELECT id FROM users WHERE email = %s AND id != %s",
                (updates['email'], user_id)
            )
            if cursor.fetchone():
                return jsonify({'error': 'Email is already taken'}), 409
        
        # Build update query
        set_clause = ', '.join([f"{field} = %s" for field in updates.keys()])
        values = list(updates.values()) + [user_id]
        
        update_query = f"""
            UPDATE users SET {set_clause}, updated_at = NOW()
            WHERE id = %s
            RETURNING id, email, first_name, last_name, role
        """
        
        cursor.execute(update_query, values)
        user_data = cursor.fetchone()
        conn.commit()
        
        logger.info(f"Profile updated for user: {user_id}")
        
        return jsonify({
            'message': 'Profile updated successfully',
            'user': {
                'id': user_data[0],
                'email': user_data[1],
                'first_name': user_data[2],
                'last_name': user_data[3],
                'role': user_data[4]
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Update profile error: {str(e)}")
        return jsonify({'error': 'Failed to update profile'}), 500
    finally:
        if 'conn' in locals():
            conn.close()

@auth_bp.route('/change-password', methods=['POST'])
@jwt_required()
def change_password():
    """Change user password"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        # Validate required fields
        if not data.get('current_password') or not data.get('new_password'):
            return jsonify({'error': 'Current password and new password are required'}), 400
        
        current_password = data['current_password']
        new_password = data['new_password']
        
        # Validate new password
        password_validation = validate_password(new_password)
        if not password_validation['valid']:
            return jsonify({'error': password_validation['message']}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get current password hash
        cursor.execute("SELECT password_hash FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Verify current password
        if not verify_password(current_password, user[0]):
            return jsonify({'error': 'Current password is incorrect'}), 401
        
        # Hash new password
        new_password_hash = hash_password(new_password)
        
        # Update password
        cursor.execute(
            "UPDATE users SET password_hash = %s, updated_at = NOW() WHERE id = %s",
            (new_password_hash, user_id)
        )
        conn.commit()
        
        logger.info(f"Password changed for user: {user_id}")
        
        return jsonify({'message': 'Password changed successfully'}), 200
        
    except Exception as e:
        logger.error(f"Change password error: {str(e)}")
        return jsonify({'error': 'Failed to change password'}), 500
    finally:
        if 'conn' in locals():
            conn.close()

@auth_bp.route('/verify-token', methods=['GET'])
@jwt_required()
def verify_token():
    """Verify if the current token is valid"""
    try:
        user_id = get_jwt_identity()
        claims = get_jwt()
        
        return jsonify({
            'valid': True,
            'user_id': user_id,
            'email': claims.get('email'),
            'role': claims.get('role')
        }), 200
        
    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        return jsonify({'valid': False, 'error': str(e)}), 401