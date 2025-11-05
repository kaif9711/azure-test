"""
Authentication utilities
Password hashing, validation, and JWT helpers
"""

import bcrypt
import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Password validation rules
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128
PASSWORD_REGEX = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)')

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    try:
        # Generate salt and hash password
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8')
    except Exception as e:
        logger.error(f"Error hashing password: {str(e)}")
        raise ValueError("Failed to hash password")

def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    except Exception as e:
        logger.error(f"Error verifying password: {str(e)}")
        return False

def validate_password(password: str) -> Dict[str, Any]:
    """
    Validate password strength
    
    Args:
        password: Password to validate
        
    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': True,
        'message': 'Password is valid',
        'errors': []
    }
    
    # Check length
    if len(password) < MIN_PASSWORD_LENGTH:
        result['errors'].append(f'Password must be at least {MIN_PASSWORD_LENGTH} characters long')
    
    if len(password) > MAX_PASSWORD_LENGTH:
        result['errors'].append(f'Password must be no more than {MAX_PASSWORD_LENGTH} characters long')
    
    # Check for at least one lowercase letter
    if not re.search(r'[a-z]', password):
        result['errors'].append('Password must contain at least one lowercase letter')
    
    # Check for at least one uppercase letter
    if not re.search(r'[A-Z]', password):
        result['errors'].append('Password must contain at least one uppercase letter')
    
    # Check for at least one digit
    if not re.search(r'\d', password):
        result['errors'].append('Password must contain at least one digit')
    
    # Check for at least one special character
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        result['errors'].append('Password must contain at least one special character')
    
    # Check for common weak passwords
    weak_passwords = [
        'password', 'password123', '12345678', 'qwerty123',
        'admin123', 'letmein', 'welcome123', 'password1'
    ]
    
    if password.lower() in weak_passwords:
        result['errors'].append('Password is too common and easily guessable')
    
    # Check for sequential characters
    if has_sequential_chars(password):
        result['errors'].append('Password should not contain sequential characters')
    
    # Check for repeated characters
    if has_repeated_chars(password):
        result['errors'].append('Password should not contain too many repeated characters')
    
    # Update result based on errors
    if result['errors']:
        result['valid'] = False
        result['message'] = '; '.join(result['errors'])
    
    return result

def has_sequential_chars(password: str, min_length: int = 3) -> bool:
    """Check if password contains sequential characters"""
    password_lower = password.lower()
    
    # Check for sequential letters
    for i in range(len(password_lower) - min_length + 1):
        chars = password_lower[i:i + min_length]
        if all(ord(chars[j]) == ord(chars[0]) + j for j in range(len(chars))):
            return True
    
    # Check for sequential numbers
    for i in range(len(password) - min_length + 1):
        chars = password[i:i + min_length]
        if chars.isdigit():
            if all(int(chars[j]) == int(chars[0]) + j for j in range(len(chars))):
                return True
    
    return False

def has_repeated_chars(password: str, max_repeats: int = 3) -> bool:
    """Check if password has too many repeated characters"""
    for i in range(len(password) - max_repeats + 1):
        char = password[i]
        if all(password[j] == char for j in range(i, i + max_repeats)):
            return True
    
    return False

def generate_password_strength_score(password: str) -> Dict[str, Any]:
    """
    Generate a password strength score (0-100)
    
    Args:
        password: Password to score
        
    Returns:
        Dictionary with score and breakdown
    """
    score = 0
    breakdown = {
        'length': 0,
        'lowercase': 0,
        'uppercase': 0,
        'numbers': 0,
        'special_chars': 0,
        'uniqueness': 0
    }
    
    # Length scoring (0-25 points)
    if len(password) >= 8:
        breakdown['length'] = min(25, len(password) * 2)
    
    # Character type scoring (15 points each)
    if re.search(r'[a-z]', password):
        breakdown['lowercase'] = 15
    
    if re.search(r'[A-Z]', password):
        breakdown['uppercase'] = 15
    
    if re.search(r'\d', password):
        breakdown['numbers'] = 15
    
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        breakdown['special_chars'] = 15
    
    # Uniqueness scoring (15 points)
    unique_chars = len(set(password.lower()))
    breakdown['uniqueness'] = min(15, unique_chars * 2)
    
    # Calculate total score
    score = sum(breakdown.values())
    
    # Determine strength level
    if score >= 85:
        strength = 'Very Strong'
    elif score >= 70:
        strength = 'Strong'
    elif score >= 55:
        strength = 'Good'
    elif score >= 40:
        strength = 'Fair'
    else:
        strength = 'Weak'
    
    return {
        'score': score,
        'strength': strength,
        'breakdown': breakdown,
        'max_score': 100
    }

def validate_email(email: str) -> Dict[str, Any]:
    """
    Validate email address format
    
    Args:
        email: Email to validate
        
    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': True,
        'message': 'Email is valid',
        'errors': []
    }
    
    # Basic format validation
    email_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    if not email_regex.match(email):
        result['errors'].append('Invalid email format')
    
    # Length validation
    if len(email) > 254:
        result['errors'].append('Email address is too long')
    
    if len(email) < 5:
        result['errors'].append('Email address is too short')
    
    # Local part validation (before @)
    local_part = email.split('@')[0] if '@' in email else email
    if len(local_part) > 64:
        result['errors'].append('Email local part is too long')
    
    # Domain validation
    if '@' in email:
        domain_part = email.split('@')[1]
        if len(domain_part) > 253:
            result['errors'].append('Email domain is too long')
        
        # Check for valid domain format
        domain_regex = re.compile(r'^[a-zA-Z0-9.-]+$')
        if not domain_regex.match(domain_part):
            result['errors'].append('Invalid characters in email domain')
        
        # Check for consecutive dots
        if '..' in domain_part:
            result['errors'].append('Email domain cannot contain consecutive dots')
        
        # Check domain starts or ends with dot or hyphen
        if domain_part.startswith('.') or domain_part.endswith('.') or \
           domain_part.startswith('-') or domain_part.endswith('-'):
            result['errors'].append('Email domain format is invalid')
    
    # Update result based on errors
    if result['errors']:
        result['valid'] = False
        result['message'] = '; '.join(result['errors'])
    
    return result

def is_password_compromised(password: str) -> bool:
    """
    Check if password appears in common breach databases
    (Simplified implementation - in production, use services like HaveIBeenPwned)
    
    Args:
        password: Password to check
        
    Returns:
        True if password is known to be compromised
    """
    # Common compromised passwords (simplified list)
    common_passwords = {
        'password', 'password123', '123456', 'password1', 'admin',
        'qwerty', 'letmein', 'welcome', 'monkey', '1234567890',
        '12345678', '123456789', 'qwerty123', 'password321',
        'admin123', 'root', 'toor', 'pass', 'test', 'guest',
        'user', 'login', 'changeme', 'secret', 'default'
    }
    
    return password.lower() in common_passwords

def generate_secure_token(length: int = 32) -> str:
    """
    Generate a secure random token
    
    Args:
        length: Length of the token
        
    Returns:
        Secure random token
    """
    import secrets
    import string
    
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def mask_email(email: str) -> str:
    """
    Mask email address for logging/display purposes
    
    Args:
        email: Email to mask
        
    Returns:
        Masked email address
    """
    if '@' not in email:
        return email
    
    local, domain = email.split('@', 1)
    
    if len(local) <= 2:
        masked_local = local
    else:
        masked_local = local[0] + '*' * (len(local) - 2) + local[-1]
    
    domain_parts = domain.split('.')
    if len(domain_parts) >= 2:
        domain_name = domain_parts[0]
        if len(domain_name) > 2:
            masked_domain_name = domain_name[0] + '*' * (len(domain_name) - 2) + domain_name[-1]
        else:
            masked_domain_name = domain_name
        
        masked_domain = masked_domain_name + '.' + '.'.join(domain_parts[1:])
    else:
        masked_domain = domain
    
    return f"{masked_local}@{masked_domain}"

def validate_user_role(role: str) -> bool:
    """
    Validate user role
    
    Args:
        role: Role to validate
        
    Returns:
        True if role is valid
    """
    valid_roles = ['user', 'admin', 'investigator', 'supervisor']
    return role in valid_roles

def check_rate_limit(user_id: str, action: str, max_attempts: int = 5, window_minutes: int = 15) -> Dict[str, Any]:
    """
    Check rate limiting for user actions
    (Simplified implementation - in production, use Redis or similar)
    
    Args:
        user_id: User identifier
        action: Action being performed
        max_attempts: Maximum attempts allowed
        window_minutes: Time window in minutes
        
    Returns:
        Dictionary with rate limit status
    """
    # This is a simplified implementation
    # In production, you would use Redis or a database to track attempts
    
    return {
        'allowed': True,
        'remaining_attempts': max_attempts,
        'reset_time': None,
        'blocked_until': None
    }