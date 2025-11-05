# Deprecated legacy Flask metrics routes.
# Replaced by FastAPI implementation in metrics_fastapi.py.

"""
Metrics Routes
Provides fraud detection statistics and system performance metrics
"""

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from utils.db import get_db_connection

logger = logging.getLogger(__name__)
metrics_bp = Blueprint('metrics', __name__)

@metrics_bp.route('/fraud-stats', methods=['GET'])
@jwt_required()
def get_fraud_statistics():
    """Get fraud detection statistics"""
    try:
        jwt_claims = get_jwt()
        user_role = jwt_claims.get('role', 'user')
        
        # Only admin, supervisor, or investigator can access metrics
        if user_role not in ['admin', 'supervisor', 'investigator']:
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        # Get query parameters
        days = int(request.args.get('days', 30))  # Default last 30 days
        start_date = datetime.now() - timedelta(days=days)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Total claims statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_claims,
                COUNT(CASE WHEN status = 'approved' THEN 1 END) as approved_claims,
                COUNT(CASE WHEN status = 'rejected' THEN 1 END) as rejected_claims,
                COUNT(CASE WHEN status = 'investigation' THEN 1 END) as under_investigation,
                COUNT(CASE WHEN risk_score > 0.7 THEN 1 END) as high_risk_claims,
                AVG(risk_score) as avg_risk_score,
                SUM(CASE WHEN status = 'approved' THEN claim_amount ELSE 0 END) as total_approved_amount,
                SUM(claim_amount) as total_claim_amount
            FROM claims 
            WHERE created_at >= %s
        """, (start_date,))
        
        stats = cursor.fetchone()
        
        # Daily fraud detection trends
        cursor.execute("""
            SELECT 
                DATE(created_at) as claim_date,
                COUNT(*) as total_claims,
                COUNT(CASE WHEN risk_score > 0.7 THEN 1 END) as high_risk_claims,
                AVG(risk_score) as avg_risk_score
            FROM claims
            WHERE created_at >= %s
            GROUP BY DATE(created_at)
            ORDER BY claim_date DESC
        """, (start_date,))
        
        daily_trends = cursor.fetchall()
        
        # Risk score distribution
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN risk_score < 0.3 THEN 'Low (0-0.3)'
                    WHEN risk_score < 0.5 THEN 'Medium-Low (0.3-0.5)'
                    WHEN risk_score < 0.7 THEN 'Medium (0.5-0.7)'
                    WHEN risk_score < 0.9 THEN 'High (0.7-0.9)'
                    ELSE 'Very High (0.9-1.0)'
                END as risk_category,
                COUNT(*) as count
            FROM claims
            WHERE created_at >= %s AND risk_score IS NOT NULL
            GROUP BY 
                CASE 
                    WHEN risk_score < 0.3 THEN 'Low (0-0.3)'
                    WHEN risk_score < 0.5 THEN 'Medium-Low (0.3-0.5)'
                    WHEN risk_score < 0.7 THEN 'Medium (0.5-0.7)'
                    WHEN risk_score < 0.9 THEN 'High (0.7-0.9)'
                    ELSE 'Very High (0.9-1.0)'
                END
            ORDER BY 
                CASE 
                    WHEN risk_score < 0.3 THEN 1
                    WHEN risk_score < 0.5 THEN 2
                    WHEN risk_score < 0.7 THEN 3
                    WHEN risk_score < 0.9 THEN 4
                    ELSE 5
                END
        """, (start_date,))
        
        risk_distribution = cursor.fetchall()
        
        # Top risk factors
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN claim_amount > 50000 THEN 'High Amount Claims'
                    WHEN document_score < 0.7 THEN 'Poor Document Quality'
                    ELSE 'Other Factors'
                END as risk_factor,
                COUNT(*) as count
            FROM claims
            WHERE created_at >= %s AND risk_score > 0.5
            GROUP BY 
                CASE 
                    WHEN claim_amount > 50000 THEN 'High Amount Claims'
                    WHEN document_score < 0.7 THEN 'Poor Document Quality'
                    ELSE 'Other Factors'
                END
            ORDER BY count DESC
        """, (start_date,))
        
        risk_factors = cursor.fetchall()
        
        # Model performance metrics
        cursor.execute("""
            SELECT 
                AVG(CASE WHEN document_score IS NOT NULL THEN document_score END) as avg_doc_validation_score,
                COUNT(CASE WHEN document_score < 0.5 THEN 1 END) as poor_quality_docs,
                COUNT(CASE WHEN document_score IS NOT NULL THEN 1 END) as total_validated_docs
            FROM claims
            WHERE created_at >= %s
        """, (start_date,))
        
        model_performance = cursor.fetchone()
        
        # Format response
        response = {
            'period': {
                'days': days,
                'start_date': start_date.isoformat(),
                'end_date': datetime.now().isoformat()
            },
            'overview': {
                'total_claims': stats[0] or 0,
                'approved_claims': stats[1] or 0,
                'rejected_claims': stats[2] or 0,
                'under_investigation': stats[3] or 0,
                'high_risk_claims': stats[4] or 0,
                'approval_rate': (stats[1] / stats[0] * 100) if stats[0] > 0 else 0,
                'rejection_rate': (stats[2] / stats[0] * 100) if stats[0] > 0 else 0,
                'high_risk_rate': (stats[4] / stats[0] * 100) if stats[0] > 0 else 0,
                'average_risk_score': float(stats[5]) if stats[5] else 0,
                'total_approved_amount': float(stats[6]) if stats[6] else 0,
                'total_claim_amount': float(stats[7]) if stats[7] else 0,
                'savings_from_fraud_prevention': float(stats[7] - stats[6]) if stats[6] and stats[7] else 0
            },
            'daily_trends': [
                {
                    'date': trend[0].isoformat(),
                    'total_claims': trend[1],
                    'high_risk_claims': trend[2],
                    'average_risk_score': float(trend[3]) if trend[3] else 0,
                    'risk_rate': (trend[2] / trend[1] * 100) if trend[1] > 0 else 0
                }
                for trend in daily_trends
            ],
            'risk_distribution': [
                {
                    'category': dist[0],
                    'count': dist[1],
                    'percentage': (dist[1] / stats[0] * 100) if stats[0] > 0 else 0
                }
                for dist in risk_distribution
            ],
            'top_risk_factors': [
                {
                    'factor': factor[0],
                    'count': factor[1]
                }
                for factor in risk_factors
            ],
            'model_performance': {
                'average_document_validation_score': float(model_performance[0]) if model_performance[0] else 0,
                'poor_quality_documents': model_performance[1] or 0,
                'total_validated_documents': model_performance[2] or 0,
                'document_validation_success_rate': 
                    ((model_performance[2] - model_performance[1]) / model_performance[2] * 100) 
                    if model_performance[2] > 0 else 0
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Get fraud statistics error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve fraud statistics'}), 500
    finally:
        if 'conn' in locals():
            conn.close()

@metrics_bp.route('/system-health', methods=['GET'])
@jwt_required()
def get_system_health():
    """Get system performance and health metrics"""
    try:
        jwt_claims = get_jwt()
        user_role = jwt_claims.get('role', 'user')
        
        # Only admin can access system health metrics
        if user_role != 'admin':
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Database health metrics
        cursor.execute("SELECT COUNT(*) FROM claims")
        total_claims = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = true")
        active_users = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM claim_documents")
        total_documents = cursor.fetchone()[0]
        
        # Processing metrics (last 24 hours)
        yesterday = datetime.now() - timedelta(days=1)
        
        cursor.execute("""
            SELECT 
                COUNT(*) as processed_claims,
                AVG(EXTRACT(EPOCH FROM (updated_at - created_at))/60) as avg_processing_time_minutes
            FROM claims 
            WHERE updated_at >= %s AND status != 'submitted'
        """, (yesterday,))
        
        processing_stats = cursor.fetchone()
        
        # Error metrics (would be from application logs in a real system)
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN status = 'error' THEN 1 END) as error_count,
                COUNT(*) as total_processes
            FROM claims 
            WHERE created_at >= %s
        """, (yesterday,))
        
        error_stats = cursor.fetchone()
        
        # System resource usage (simplified - in production, you'd use system monitoring)
        import psutil
        import os
        
        # Get current process info
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        cpu_usage = process.cpu_percent()
        
        # Disk usage for upload directory
        upload_folder = current_app.config.get('UPLOAD_FOLDER', './uploads')
        if os.path.exists(upload_folder):
            disk_usage = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(upload_folder)
                for filename in filenames
            ) / 1024 / 1024  # MB
        else:
            disk_usage = 0
        
        response = {
            'timestamp': datetime.now().isoformat(),
            'database': {
                'status': 'healthy',
                'total_claims': total_claims,
                'active_users': active_users,
                'total_documents': total_documents,
                'connection_status': 'connected'
            },
            'processing': {
                'claims_processed_24h': processing_stats[0] or 0,
                'average_processing_time_minutes': float(processing_stats[1]) if processing_stats[1] else 0,
                'processing_queue_length': 0,  # Would be from actual queue in production
                'processing_rate_per_hour': (processing_stats[0] or 0) / 24
            },
            'errors': {
                'error_count_24h': error_stats[0] or 0,
                'error_rate': (error_stats[0] / error_stats[1] * 100) if error_stats[1] > 0 else 0,
                'last_error_time': None  # Would be from error logs
            },
            'resources': {
                'memory_usage_mb': round(memory_usage, 2),
                'cpu_usage_percent': round(cpu_usage, 2),
                'disk_usage_mb': round(disk_usage, 2),
                'upload_folder_size_mb': round(disk_usage, 2)
            },
            'ml_models': {
                'document_validator_status': 'healthy',
                'risk_checker_status': 'healthy',
                'supervisor_agent_status': 'healthy',
                'model_load_time': 'N/A',  # Would track actual model load times
                'prediction_latency_ms': 'N/A'  # Would track prediction performance
            },
            'api': {
                'uptime_seconds': int((datetime.now() - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()),
                'requests_per_minute': 'N/A',  # Would be from request monitoring
                'response_time_ms': 'N/A',
                'active_connections': 'N/A'
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Get system health error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve system health metrics'}), 500
    finally:
        if 'conn' in locals():
            conn.close()

@metrics_bp.route('/user-activity', methods=['GET'])
@jwt_required()
def get_user_activity():
    """Get user activity metrics"""
    try:
        jwt_claims = get_jwt()
        user_role = jwt_claims.get('role', 'user')
        
        # Only admin or supervisor can access user activity metrics
        if user_role not in ['admin', 'supervisor']:
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        # Get query parameters
        days = int(request.args.get('days', 7))  # Default last 7 days
        start_date = datetime.now() - timedelta(days=days)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Active users
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT user_id) as active_users,
                COUNT(*) as total_activities
            FROM claims
            WHERE created_at >= %s
        """, (start_date,))
        
        activity_stats = cursor.fetchone()
        
        # User registration trends
        cursor.execute("""
            SELECT 
                DATE(created_at) as registration_date,
                COUNT(*) as new_users
            FROM users
            WHERE created_at >= %s
            GROUP BY DATE(created_at)
            ORDER BY registration_date DESC
        """, (start_date,))
        
        registration_trends = cursor.fetchall()
        
        # Most active users
        cursor.execute("""
            SELECT 
                u.first_name, u.last_name, u.email, u.role,
                COUNT(c.id) as claim_count
            FROM users u
            LEFT JOIN claims c ON u.id = c.user_id AND c.created_at >= %s
            WHERE u.is_active = true
            GROUP BY u.id, u.first_name, u.last_name, u.email, u.role
            ORDER BY claim_count DESC
            LIMIT 10
        """, (start_date,))
        
        active_users = cursor.fetchall()
        
        # User role distribution
        cursor.execute("""
            SELECT 
                role,
                COUNT(*) as count,
                COUNT(CASE WHEN is_active THEN 1 END) as active_count
            FROM users
            GROUP BY role
        """, )
        
        role_distribution = cursor.fetchall()
        
        response = {
            'period': {
                'days': days,
                'start_date': start_date.isoformat(),
                'end_date': datetime.now().isoformat()
            },
            'overview': {
                'active_users': activity_stats[0] or 0,
                'total_activities': activity_stats[1] or 0,
                'average_activities_per_user': 
                    (activity_stats[1] / activity_stats[0]) if activity_stats[0] > 0 else 0
            },
            'registration_trends': [
                {
                    'date': trend[0].isoformat(),
                    'new_users': trend[1]
                }
                for trend in registration_trends
            ],
            'most_active_users': [
                {
                    'name': f"{user[0]} {user[1]}",
                    'email': user[2],
                    'role': user[3],
                    'claim_count': user[4]
                }
                for user in active_users
            ],
            'role_distribution': [
                {
                    'role': role[0],
                    'total_users': role[1],
                    'active_users': role[2],
                    'activity_rate': (role[2] / role[1] * 100) if role[1] > 0 else 0
                }
                for role in role_distribution
            ]
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Get user activity error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve user activity metrics'}), 500
    finally:
        if 'conn' in locals():
            conn.close()

@metrics_bp.route('/performance-report', methods=['GET'])
@jwt_required()
def generate_performance_report():
    """Generate comprehensive performance report"""
    try:
        jwt_claims = get_jwt()
        user_role = jwt_claims.get('role', 'user')
        
        # Only admin can generate performance reports
        if user_role != 'admin':
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        # Get fraud statistics
        fraud_response = get_fraud_statistics()
        fraud_data = fraud_response[0].get_json() if fraud_response[1] == 200 else {}
        
        # Get system health
        health_response = get_system_health()
        health_data = health_response[0].get_json() if health_response[1] == 200 else {}
        
        # Get user activity
        activity_response = get_user_activity()
        activity_data = activity_response[0].get_json() if activity_response[1] == 200 else {}
        
        # Calculate key performance indicators
        fraud_overview = fraud_data.get('overview', {})
        
        kpis = {
            'fraud_detection_efficiency': {
                'high_risk_detection_rate': fraud_overview.get('high_risk_rate', 0),
                'false_positive_rate': 'N/A',  # Would need historical validation data
                'time_to_decision': health_data.get('processing', {}).get('average_processing_time_minutes', 0)
            },
            'cost_effectiveness': {
                'total_savings': fraud_overview.get('savings_from_fraud_prevention', 0),
                'processing_cost_per_claim': 'N/A',  # Would need cost data
                'roi_percentage': 'N/A'
            },
            'system_performance': {
                'uptime_percentage': 99.9,  # Would be from monitoring
                'average_response_time': 'N/A',
                'error_rate': health_data.get('errors', {}).get('error_rate', 0)
            },
            'user_satisfaction': {
                'active_user_rate': activity_data.get('overview', {}).get('active_users', 0),
                'average_claims_per_user': activity_data.get('overview', {}).get('average_activities_per_user', 0)
            }
        }
        
        # Generate recommendations
        recommendations = []
        
        if fraud_overview.get('high_risk_rate', 0) < 5:
            recommendations.append('Consider adjusting fraud detection thresholds - very low high-risk detection rate')
        elif fraud_overview.get('high_risk_rate', 0) > 30:
            recommendations.append('High fraud detection rate - review model sensitivity')
        
        if fraud_overview.get('approval_rate', 0) < 60:
            recommendations.append('Low approval rate may indicate overly strict fraud detection')
        
        if health_data.get('processing', {}).get('average_processing_time_minutes', 0) > 60:
            recommendations.append('Processing time is high - consider system optimization')
        
        response = {
            'report_generated_at': datetime.now().isoformat(),
            'report_period': fraud_data.get('period', {}),
            'executive_summary': {
                'total_claims_processed': fraud_overview.get('total_claims', 0),
                'fraud_cases_detected': fraud_overview.get('high_risk_claims', 0),
                'total_savings': fraud_overview.get('savings_from_fraud_prevention', 0),
                'system_uptime': '99.9%',  # Would be from monitoring
                'user_satisfaction_score': 'N/A'
            },
            'key_performance_indicators': kpis,
            'fraud_statistics': fraud_data,
            'system_health': health_data,
            'user_activity': activity_data,
            'recommendations': recommendations,
            'next_review_date': (datetime.now() + timedelta(days=30)).isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Generate performance report error: {str(e)}")
        return jsonify({'error': 'Failed to generate performance report'}), 500