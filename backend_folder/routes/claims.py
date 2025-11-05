# Deprecated legacy Flask claims routes.
# Replaced by FastAPI implementation in claims_fastapi.py.

"""
Claims Routes
Handles claim submission, processing, and management
"""

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from werkzeug.utils import secure_filename
import os
import logging
from datetime import datetime
import uuid
from typing import Dict, Any

from utils.db import get_db_connection
from models.doc_validator import DocumentValidator
from models.risk_checker import RiskChecker
from models.supervisor import SupervisorAgent

logger = logging.getLogger(__name__)
claims_bp = Blueprint('claims', __name__)

# Initialize ML models
doc_validator = DocumentValidator()
risk_checker = RiskChecker()
supervisor = SupervisorAgent()

def allowed_file(filename):
    """Check if file has allowed extension"""
    allowed_extensions = current_app.config.get('ALLOWED_EXTENSIONS', {'pdf', 'png', 'jpg', 'jpeg', 'doc', 'docx'})
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@claims_bp.route('/', methods=['POST'])
@jwt_required()
def submit_claim():
    """Submit a new insurance claim"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['claim_amount', 'incident_date', 'incident_description', 'policy_number']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        # Validate claim amount
        try:
            claim_amount = float(data['claim_amount'])
            if claim_amount <= 0:
                return jsonify({'error': 'Claim amount must be positive'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid claim amount'}), 400
        
        # Validate incident date
        try:
            incident_date = datetime.fromisoformat(data['incident_date'])
            if incident_date > datetime.now():
                return jsonify({'error': 'Incident date cannot be in the future'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid incident date format'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verify policy exists and belongs to user
        cursor.execute("""
            SELECT id, policy_type, premium, coverage_amount, start_date 
            FROM policies 
            WHERE policy_number = %s AND user_id = %s AND is_active = true
        """, (data['policy_number'], user_id))
        
        policy = cursor.fetchone()
        if not policy:
            return jsonify({'error': 'Invalid policy number or policy not active'}), 400
        
        # Check if claim amount exceeds coverage
        if claim_amount > policy[3]:  # coverage_amount
            return jsonify({'error': 'Claim amount exceeds policy coverage'}), 400
        
        # Generate claim ID
        claim_id = str(uuid.uuid4())
        
        # Insert claim
        insert_query = """
            INSERT INTO claims (
                id, user_id, policy_id, claim_amount, incident_date, 
                incident_description, location, status, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'submitted', NOW())
            RETURNING id, created_at
        """
        
        cursor.execute(insert_query, (
            claim_id, user_id, policy[0], claim_amount, incident_date,
            data['incident_description'], data.get('location', ''), 
        ))
        
        claim_data = cursor.fetchone()
        conn.commit()
        
        logger.info(f"New claim submitted: {claim_id} by user {user_id}")
        
        return jsonify({
            'message': 'Claim submitted successfully',
            'claim_id': claim_data[0],
            'status': 'submitted',
            'created_at': claim_data[1].isoformat(),
            'next_steps': [
                'Upload supporting documents',
                'Await initial review',
                'Possible follow-up contact'
            ]
        }), 201
        
    except Exception as e:
        logger.error(f"Submit claim error: {str(e)}")
        return jsonify({'error': 'Failed to submit claim'}), 500
    finally:
        if 'conn' in locals():
            conn.close()

@claims_bp.route('/', methods=['GET'])
@jwt_required()
def get_claims():
    """Get claims list with pagination and filters"""
    try:
        user_id = get_jwt_identity()
        claims = get_jwt()
        user_role = claims.get('role', 'user')
        
        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)  # Max 100 per page
        status_filter = request.args.get('status')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build query based on user role
        if user_role in ['admin', 'supervisor', 'investigator']:
            # Admin/supervisor can see all claims
            base_query = """
                SELECT c.id, c.user_id, c.claim_amount, c.incident_date, c.incident_description,
                       c.location, c.status, c.created_at, c.updated_at, c.risk_score,
                       u.first_name, u.last_name, u.email, p.policy_number, p.policy_type
                FROM claims c
                JOIN users u ON c.user_id = u.id
                JOIN policies p ON c.policy_id = p.id
            """
        else:
            # Regular users can only see their own claims
            base_query = """
                SELECT c.id, c.user_id, c.claim_amount, c.incident_date, c.incident_description,
                       c.location, c.status, c.created_at, c.updated_at, c.risk_score,
                       u.first_name, u.last_name, u.email, p.policy_number, p.policy_type
                FROM claims c
                JOIN users u ON c.user_id = u.id
                JOIN policies p ON c.policy_id = p.id
                WHERE c.user_id = %s
            """
        
        # Add filters
        where_conditions = []
        params = [user_id] if user_role not in ['admin', 'supervisor', 'investigator'] else []
        
        if status_filter:
            where_conditions.append("c.status = %s")
            params.append(status_filter)
        
        if date_from:
            where_conditions.append("c.created_at >= %s")
            params.append(date_from)
        
        if date_to:
            where_conditions.append("c.created_at <= %s")
            params.append(date_to)
        
        if where_conditions:
            if user_role in ['admin', 'supervisor', 'investigator']:
                base_query += " WHERE " + " AND ".join(where_conditions)
            else:
                base_query += " AND " + " AND ".join(where_conditions)
        
        # Add pagination
        offset = (page - 1) * per_page
        query = f"{base_query} ORDER BY c.created_at DESC LIMIT %s OFFSET %s"
        params.extend([per_page, offset])
        
        cursor.execute(query, params)
        claims_data = cursor.fetchall()
        
        # Get total count for pagination
        count_query = base_query.replace(
            "SELECT c.id, c.user_id, c.claim_amount, c.incident_date, c.incident_description, c.location, c.status, c.created_at, c.updated_at, c.risk_score, u.first_name, u.last_name, u.email, p.policy_number, p.policy_type",
            "SELECT COUNT(*)"
        )
        cursor.execute(count_query, params[:-2])  # Remove LIMIT and OFFSET params
        total_claims = cursor.fetchone()[0]
        
        # Format response
        claims_list = []
        for claim in claims_data:
            claim_dict = {
                'id': claim[0],
                'user_id': claim[1],
                'claim_amount': float(claim[2]),
                'incident_date': claim[3].isoformat(),
                'incident_description': claim[4],
                'location': claim[5],
                'status': claim[6],
                'created_at': claim[7].isoformat(),
                'updated_at': claim[8].isoformat() if claim[8] else None,
                'risk_score': float(claim[9]) if claim[9] else None,
                'policy_number': claim[13],
                'policy_type': claim[14]
            }
            
            # Include user info for admin/supervisor views
            if user_role in ['admin', 'supervisor', 'investigator']:
                claim_dict['claimant'] = {
                    'first_name': claim[10],
                    'last_name': claim[11],
                    'email': claim[12]
                }
            
            claims_list.append(claim_dict)
        
        return jsonify({
            'claims': claims_list,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total_claims,
                'pages': (total_claims + per_page - 1) // per_page
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Get claims error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve claims'}), 500
    finally:
        if 'conn' in locals():
            conn.close()

@claims_bp.route('/<claim_id>', methods=['GET'])
@jwt_required()
def get_claim_details(claim_id):
    """Get detailed information about a specific claim"""
    try:
        user_id = get_jwt_identity()
        jwt_claims = get_jwt()
        user_role = jwt_claims.get('role', 'user')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build query based on user role
        if user_role in ['admin', 'supervisor', 'investigator']:
            # Admin can see all claims
            where_clause = "WHERE c.id = %s"
            params = [claim_id]
        else:
            # Regular users can only see their own claims
            where_clause = "WHERE c.id = %s AND c.user_id = %s"
            params = [claim_id, user_id]
        
        query = f"""
            SELECT c.id, c.user_id, c.policy_id, c.claim_amount, c.incident_date,
                   c.incident_description, c.location, c.status, c.created_at, c.updated_at,
                   c.risk_score, c.document_score, c.supervisor_decision, c.rejection_reason,
                   u.first_name, u.last_name, u.email, 
                   p.policy_number, p.policy_type, p.coverage_amount
            FROM claims c
            JOIN users u ON c.user_id = u.id
            JOIN policies p ON c.policy_id = p.id
            {where_clause}
        """
        
        cursor.execute(query, params)
        claim = cursor.fetchone()
        
        if not claim:
            return jsonify({'error': 'Claim not found'}), 404
        
        # Get claim documents
        cursor.execute("""
            SELECT id, filename, file_type, file_size, upload_date, validation_status, validation_score
            FROM claim_documents WHERE claim_id = %s ORDER BY upload_date DESC
        """, (claim_id,))
        documents = cursor.fetchall()
        
        # Get claim status history
        cursor.execute("""
            SELECT status, changed_at, changed_by_user_id, notes
            FROM claim_status_history 
            WHERE claim_id = %s 
            ORDER BY changed_at DESC
        """, (claim_id,))
        status_history = cursor.fetchall()
        
        # Format response
        claim_details = {
            'id': claim[0],
            'user_id': claim[1],
            'policy_id': claim[2],
            'claim_amount': float(claim[3]),
            'incident_date': claim[4].isoformat(),
            'incident_description': claim[5],
            'location': claim[6],
            'status': claim[7],
            'created_at': claim[8].isoformat(),
            'updated_at': claim[9].isoformat() if claim[9] else None,
            'risk_score': float(claim[10]) if claim[10] else None,
            'document_score': float(claim[11]) if claim[11] else None,
            'supervisor_decision': claim[12],
            'rejection_reason': claim[13],
            'claimant': {
                'first_name': claim[14],
                'last_name': claim[15],
                'email': claim[16]
            },
            'policy': {
                'policy_number': claim[17],
                'policy_type': claim[18],
                'coverage_amount': float(claim[19])
            },
            'documents': [
                {
                    'id': doc[0],
                    'filename': doc[1],
                    'file_type': doc[2],
                    'file_size': doc[3],
                    'upload_date': doc[4].isoformat(),
                    'validation_status': doc[5],
                    'validation_score': float(doc[6]) if doc[6] else None
                }
                for doc in documents
            ],
            'status_history': [
                {
                    'status': status[0],
                    'changed_at': status[1].isoformat(),
                    'changed_by_user_id': status[2],
                    'notes': status[3]
                }
                for status in status_history
            ]
        }
        
        return jsonify(claim_details), 200
        
    except Exception as e:
        logger.error(f"Get claim details error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve claim details'}), 500
    finally:
        if 'conn' in locals():
            conn.close()

@claims_bp.route('/<claim_id>/upload', methods=['POST'])
@jwt_required()
def upload_document(claim_id):
    """Upload supporting documents for a claim"""
    try:
        user_id = get_jwt_identity()
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verify claim exists and belongs to user (or user has admin access)
        jwt_claims = get_jwt()
        user_role = jwt_claims.get('role', 'user')
        
        if user_role in ['admin', 'supervisor']:
            cursor.execute("SELECT id, status FROM claims WHERE id = %s", (claim_id,))
        else:
            cursor.execute("SELECT id, status FROM claims WHERE id = %s AND user_id = %s", (claim_id, user_id))
        
        claim = cursor.fetchone()
        if not claim:
            return jsonify({'error': 'Claim not found'}), 404
        
        # Check if claim is in a state that allows document uploads
        if claim[1] in ['approved', 'rejected', 'closed']:
            return jsonify({'error': 'Cannot upload documents for claims in this status'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}_{filename}"
        
        upload_folder = current_app.config['UPLOAD_FOLDER']
        file_path = os.path.join(upload_folder, unique_filename)
        
        # Ensure upload directory exists
        os.makedirs(upload_folder, exist_ok=True)
        
        file.save(file_path)
        file_size = os.path.getsize(file_path)
        
        # Validate document
        validation_result = doc_validator.validate_document(file_path, file_extension)
        
        # Save document record
        doc_id = str(uuid.uuid4())
        insert_query = """
            INSERT INTO claim_documents (
                id, claim_id, filename, original_filename, file_type, file_size, 
                file_path, upload_date, uploaded_by_user_id, validation_status, 
                validation_score, validation_details
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s)
            RETURNING id, upload_date
        """
        
        cursor.execute(insert_query, (
            doc_id, claim_id, unique_filename, filename, file_extension,
            file_size, file_path, user_id, 
            'valid' if validation_result['is_valid'] else 'invalid',
            validation_result['confidence_score'],
            str(validation_result)
        ))
        
        doc_data = cursor.fetchone()
        conn.commit()
        
        logger.info(f"Document uploaded for claim {claim_id}: {filename}")
        
        return jsonify({
            'message': 'Document uploaded successfully',
            'document': {
                'id': doc_data[0],
                'filename': filename,
                'file_type': file_extension,
                'file_size': file_size,
                'upload_date': doc_data[1].isoformat(),
                'validation_status': 'valid' if validation_result['is_valid'] else 'invalid',
                'validation_score': validation_result['confidence_score'],
                'validation_details': {
                    'risk_level': validation_result.get('risk_level'),
                    'issues_found': validation_result.get('issues_found', [])
                }
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Upload document error: {str(e)}")
        return jsonify({'error': 'Failed to upload document'}), 500
    finally:
        if 'conn' in locals():
            conn.close()

@claims_bp.route('/<claim_id>/process', methods=['POST'])
@jwt_required()
def process_claim(claim_id):
    """Process claim through ML models for risk assessment"""
    try:
        jwt_claims = get_jwt()
        user_role = jwt_claims.get('role', 'user')
        
        # Only admin, supervisor, or investigator can process claims
        if user_role not in ['admin', 'supervisor', 'investigator']:
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get claim details
        cursor.execute("""
            SELECT c.id, c.user_id, c.claim_amount, c.incident_date, c.incident_description,
                   c.location, c.status, c.document_score, u.first_name, u.last_name, u.email,
                   p.policy_number, p.policy_type, p.premium,
                   (SELECT COUNT(*) FROM claims WHERE user_id = c.user_id AND id != c.id) as previous_claims,
                   EXTRACT(EPOCH FROM (NOW() - p.start_date))/2592000 as policy_duration_months
            FROM claims c
            JOIN users u ON c.user_id = u.id
            JOIN policies p ON c.policy_id = p.id
            WHERE c.id = %s
        """, (claim_id,))
        
        claim = cursor.fetchone()
        if not claim:
            return jsonify({'error': 'Claim not found'}), 404
        
        if claim[6] != 'submitted':  # status
            return jsonify({'error': 'Claim has already been processed'}), 400
        
        # Prepare claim data for ML models
        claim_data = {
            'claim_id': claim[0],
            'claim_amount': float(claim[2]),
            'claim_date': claim[3].isoformat(),
            'incident_description': claim[4],
            'location': claim[5],
            'document_confidence_score': float(claim[7]) if claim[7] else 1.0,
            'claimant_age': 35,  # Would be calculated from DOB in real system
            'policy_type': claim[12],
            'policy_premium': float(claim[13]),
            'previous_claims': int(claim[14]),
            'policy_duration_months': int(claim[15]) if claim[15] else 12
        }
        
        # Run risk assessment
        risk_assessment = risk_checker.assess_claim_risk(claim_data)
        
        # Run supervisor review
        supervisor_decision = supervisor.review_claim(claim_data, risk_assessment)
        
        # Update claim with results
        update_query = """
            UPDATE claims SET 
                risk_score = %s, 
                status = %s,
                supervisor_decision = %s,
                updated_at = NOW()
            WHERE id = %s
        """
        
        new_status = 'under_review'
        if supervisor_decision['decision'] == 'approve':
            new_status = 'approved'
        elif supervisor_decision['decision'] == 'reject':
            new_status = 'rejected'
        elif supervisor_decision['decision'] == 'investigate':
            new_status = 'investigation'
        
        cursor.execute(update_query, (
            risk_assessment['risk_score'],
            new_status,
            supervisor_decision['decision'],
            claim_id
        ))
        
        # Log status change
        cursor.execute("""
            INSERT INTO claim_status_history (claim_id, status, changed_at, changed_by_user_id, notes)
            VALUES (%s, %s, NOW(), %s, %s)
        """, (
            claim_id, 
            new_status, 
            get_jwt_identity(),
            f"Automated processing: {supervisor_decision['explanation']}"
        ))
        
        conn.commit()
        
        logger.info(f"Claim processed: {claim_id} -> {new_status}")
        
        return jsonify({
            'message': 'Claim processed successfully',
            'claim_id': claim_id,
            'new_status': new_status,
            'risk_assessment': risk_assessment,
            'supervisor_decision': supervisor_decision
        }), 200
        
    except Exception as e:
        logger.error(f"Process claim error: {str(e)}")
        return jsonify({'error': 'Failed to process claim'}), 500
    finally:
        if 'conn' in locals():
            conn.close()

@claims_bp.route('/<claim_id>/status', methods=['PUT'])
@jwt_required()
def update_claim_status(claim_id):
    """Update claim status (admin/supervisor only)"""
    try:
        jwt_claims = get_jwt()
        user_role = jwt_claims.get('role', 'user')
        user_id = get_jwt_identity()
        
        # Only admin or supervisor can update status
        if user_role not in ['admin', 'supervisor']:
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        data = request.get_json()
        new_status = data.get('status')
        notes = data.get('notes', '')
        
        valid_statuses = ['submitted', 'under_review', 'approved', 'rejected', 'investigation', 'closed']
        if new_status not in valid_statuses:
            return jsonify({'error': 'Invalid status'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Update claim status
        cursor.execute("""
            UPDATE claims SET status = %s, updated_at = NOW()
            WHERE id = %s
            RETURNING status
        """, (new_status, claim_id))
        
        result = cursor.fetchone()
        if not result:
            return jsonify({'error': 'Claim not found'}), 404
        
        # Log status change
        cursor.execute("""
            INSERT INTO claim_status_history (claim_id, status, changed_at, changed_by_user_id, notes)
            VALUES (%s, %s, NOW(), %s, %s)
        """, (claim_id, new_status, user_id, notes))
        
        conn.commit()
        
        logger.info(f"Claim status updated: {claim_id} -> {new_status} by {user_id}")
        
        return jsonify({
            'message': 'Claim status updated successfully',
            'claim_id': claim_id,
            'new_status': new_status
        }), 200
        
    except Exception as e:
        logger.error(f"Update claim status error: {str(e)}")
        return jsonify({'error': 'Failed to update claim status'}), 500
    finally:
        if 'conn' in locals():
            conn.close()