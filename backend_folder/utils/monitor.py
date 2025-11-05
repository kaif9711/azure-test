"""
Monitoring and logging utilities
System health monitoring, performance tracking, and structured logging
"""

import logging
import time
import psutil
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.response_times = deque(maxlen=1000)  # Keep last 1000 response times
        self.active_requests = 0
        self.lock = threading.Lock()
        
        # Endpoint metrics
        self.endpoint_metrics = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'errors': 0,
            'avg_response_time': 0
        })
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network stats (if available)
            try:
                network = psutil.net_io_counters()
                network_stats = {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            except:
                network_stats = None
            
            # Application metrics
            uptime = time.time() - self.start_time
            
            with self.lock:
                avg_response_time = (
                    sum(self.response_times) / len(self.response_times)
                    if self.response_times else 0
                )
                error_rate = (
                    (self.error_count / self.request_count * 100)
                    if self.request_count > 0 else 0
                )
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory': {
                        'total': memory.total,
                        'available': memory.available,
                        'used': memory.used,
                        'percent': memory.percent
                    },
                    'disk': {
                        'total': disk.total,
                        'free': disk.free,
                        'used': disk.used,
                        'percent': (disk.used / disk.total * 100)
                    },
                    'network': network_stats
                },
                'application': {
                    'uptime_seconds': uptime,
                    'request_count': self.request_count,
                    'error_count': self.error_count,
                    'error_rate_percent': error_rate,
                    'average_response_time_ms': avg_response_time * 1000,
                    'active_requests': self.active_requests
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {'error': str(e)}
    
    def record_request(self, endpoint: str, method: str, response_time: float, status_code: int):
        """Record request metrics"""
        with self.lock:
            self.request_count += 1
            self.response_times.append(response_time)
            
            if status_code >= 400:
                self.error_count += 1
            
            # Update endpoint metrics
            key = f"{method} {endpoint}"
            metrics = self.endpoint_metrics[key]
            metrics['count'] += 1
            metrics['total_time'] += response_time
            metrics['avg_response_time'] = metrics['total_time'] / metrics['count']
            
            if status_code >= 400:
                metrics['errors'] += 1
    
    def get_endpoint_metrics(self) -> Dict[str, Any]:
        """Get endpoint-specific metrics"""
        with self.lock:
            return dict(self.endpoint_metrics)

# Global monitor instance
system_monitor = SystemMonitor()

def setup_monitoring(app):
    """Setup monitoring and logging for Flask app"""
    
    # Configure structured logging
    logging.basicConfig(
        level=getattr(logging, app.config.get('LOG_LEVEL', 'INFO')),
        format='%(asctime)s %(levelname)s %(name)s [%(pathname)s:%(lineno)d] %(message)s'
    )
    
    # Add file handler if log file is configured
    log_file = app.config.get('LOG_FILE')
    if log_file:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s %(levelname)s %(name)s [%(pathname)s:%(lineno)d] %(message)s'
            )
            file_handler.setFormatter(formatter)
            app.logger.addHandler(file_handler)
            
        except Exception as e:
            logger.warning(f"Could not setup file logging: {str(e)}")
    
    # Add request monitoring
    @app.before_request
    def before_request():
        """Record request start time"""
        from flask import g
        g.start_time = time.time()
        system_monitor.active_requests += 1
    
    @app.after_request
    def after_request(response):
        """Record request metrics"""
        from flask import g, request
        
        if hasattr(g, 'start_time'):
            response_time = time.time() - g.start_time
            system_monitor.record_request(
                request.endpoint or 'unknown',
                request.method,
                response_time,
                response.status_code
            )
        
        system_monitor.active_requests -= 1
        return response
    
    # Add health check endpoint
    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        from flask import jsonify
        
        metrics = system_monitor.get_system_metrics()
        
        # Determine health status
        health_status = 'healthy'
        issues = []
        
        # Check CPU usage
        if metrics.get('system', {}).get('cpu_percent', 0) > 80:
            health_status = 'degraded'
            issues.append('High CPU usage')
        
        # Check memory usage
        memory_percent = metrics.get('system', {}).get('memory', {}).get('percent', 0)
        if memory_percent > 85:
            health_status = 'unhealthy'
            issues.append('High memory usage')
        elif memory_percent > 70:
            health_status = 'degraded'
            issues.append('Elevated memory usage')
        
        # Check disk usage
        disk_percent = metrics.get('system', {}).get('disk', {}).get('percent', 0)
        if disk_percent > 90:
            health_status = 'unhealthy'
            issues.append('High disk usage')
        elif disk_percent > 80:
            health_status = 'degraded'
            issues.append('Elevated disk usage')
        
        # Check error rate
        error_rate = metrics.get('application', {}).get('error_rate_percent', 0)
        if error_rate > 10:
            health_status = 'unhealthy'
            issues.append('High error rate')
        elif error_rate > 5:
            health_status = 'degraded'
            issues.append('Elevated error rate')
        
        return jsonify({
            'status': health_status,
            'timestamp': datetime.now().isoformat(),
            'issues': issues,
            'metrics': metrics
        })
    
    logger.info("Monitoring setup completed")

def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.info(f"Function {func.__name__} executed in {execution_time:.4f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {execution_time:.4f}s: {str(e)}")
            raise
    
    return wrapper

class StructuredLogger:
    """Structured logging utility"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data"""
        self._log('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data"""
        self._log('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data"""
        self._log('ERROR', message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data"""
        self._log('DEBUG', message, **kwargs)
    
    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            **kwargs
        }
        
        log_method = getattr(self.logger, level.lower())
        log_method(json.dumps(log_data))

def audit_log(action: str, user_id: Optional[str] = None, resource_id: Optional[str] = None, 
              details: Optional[Dict[str, Any]] = None):
    """
    Log audit events
    
    Args:
        action: Action performed
        user_id: ID of user who performed action
        resource_id: ID of resource affected
        details: Additional details
    """
    audit_logger = StructuredLogger('audit')
    
    audit_data = {
        'action': action,
        'user_id': user_id,
        'resource_id': resource_id,
        'details': details or {},
        'timestamp': datetime.now().isoformat()
    }
    
    audit_logger.info('Audit event', **audit_data)

def security_log(event: str, user_id: Optional[str] = None, ip_address: Optional[str] = None,
                 user_agent: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
    """
    Log security events
    
    Args:
        event: Security event type
        user_id: ID of user involved
        ip_address: IP address of request
        user_agent: User agent string
        details: Additional details
    """
    security_logger = StructuredLogger('security')
    
    security_data = {
        'event': event,
        'user_id': user_id,
        'ip_address': ip_address,
        'user_agent': user_agent,
        'details': details or {},
        'timestamp': datetime.now().isoformat()
    }
    
    security_logger.warning('Security event', **security_data)

def performance_log(operation: str, duration: float, success: bool, 
                   details: Optional[Dict[str, Any]] = None):
    """
    Log performance metrics
    
    Args:
        operation: Operation performed
        duration: Duration in seconds
        success: Whether operation was successful
        details: Additional details
    """
    perf_logger = StructuredLogger('performance')
    
    perf_data = {
        'operation': operation,
        'duration_seconds': duration,
        'success': success,
        'details': details or {},
        'timestamp': datetime.now().isoformat()
    }
    
    perf_logger.info('Performance metric', **perf_data)

class MetricsCollector:
    """Collect and aggregate application metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.lock = threading.Lock()
    
    def record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        with self.lock:
            self.metrics[metric_name].append({
                'value': value,
                'timestamp': datetime.now().isoformat(),
                'tags': tags or {}
            })
    
    def get_metrics(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """Get recorded metrics"""
        with self.lock:
            if metric_name:
                return self.metrics.get(metric_name, [])
            return dict(self.metrics)
    
    def clear_metrics(self):
        """Clear all recorded metrics"""
        with self.lock:
            self.metrics.clear()

# Global metrics collector
metrics_collector = MetricsCollector()

def track_business_metric(metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """
    Track business metrics
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        tags: Optional tags for filtering
    """
    metrics_collector.record_metric(metric_name, value, tags)
    
    # Also log for external monitoring systems
    logger.info(f"Business metric: {metric_name}={value}", extra={
        'metric_name': metric_name,
        'metric_value': value,
        'metric_tags': tags or {}
    })

def get_application_health() -> Dict[str, Any]:
    """Get overall application health status"""
    metrics = system_monitor.get_system_metrics()
    
    # Determine overall health
    health_checks = {
        'database': check_database_health(),
        'system_resources': check_system_resources(metrics),
        'application': check_application_health(metrics)
    }
    
    overall_status = 'healthy'
    if any(check['status'] == 'unhealthy' for check in health_checks.values()):
        overall_status = 'unhealthy'
    elif any(check['status'] == 'degraded' for check in health_checks.values()):
        overall_status = 'degraded'
    
    return {
        'overall_status': overall_status,
        'timestamp': datetime.now().isoformat(),
        'checks': health_checks
    }

def check_database_health() -> Dict[str, Any]:
    """Check database connectivity"""
    try:
        from utils.db import test_connection
        
        start_time = time.time()
        connection_ok = test_connection()
        response_time = time.time() - start_time
        
        if connection_ok and response_time < 1.0:
            return {
                'status': 'healthy',
                'response_time_ms': response_time * 1000,
                'message': 'Database connection successful'
            }
        elif connection_ok:
            return {
                'status': 'degraded',
                'response_time_ms': response_time * 1000,
                'message': 'Database connection slow'
            }
        else:
            return {
                'status': 'unhealthy',
                'response_time_ms': response_time * 1000,
                'message': 'Database connection failed'
            }
            
    except Exception as e:
        return {
            'status': 'unhealthy',
            'message': f'Database health check error: {str(e)}'
        }

def check_system_resources(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Check system resource health"""
    system_metrics = metrics.get('system', {})
    
    issues = []
    status = 'healthy'
    
    # Check CPU
    cpu_percent = system_metrics.get('cpu_percent', 0)
    if cpu_percent > 90:
        status = 'unhealthy'
        issues.append(f'Critical CPU usage: {cpu_percent}%')
    elif cpu_percent > 80:
        status = 'degraded' if status == 'healthy' else status
        issues.append(f'High CPU usage: {cpu_percent}%')
    
    # Check memory
    memory_percent = system_metrics.get('memory', {}).get('percent', 0)
    if memory_percent > 95:
        status = 'unhealthy'
        issues.append(f'Critical memory usage: {memory_percent}%')
    elif memory_percent > 85:
        status = 'degraded' if status == 'healthy' else status
        issues.append(f'High memory usage: {memory_percent}%')
    
    # Check disk
    disk_percent = system_metrics.get('disk', {}).get('percent', 0)
    if disk_percent > 95:
        status = 'unhealthy'
        issues.append(f'Critical disk usage: {disk_percent}%')
    elif disk_percent > 85:
        status = 'degraded' if status == 'healthy' else status
        issues.append(f'High disk usage: {disk_percent}%')
    
    return {
        'status': status,
        'issues': issues,
        'metrics': system_metrics
    }

def check_application_health(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Check application-specific health"""
    app_metrics = metrics.get('application', {})
    
    issues = []
    status = 'healthy'
    
    # Check error rate
    error_rate = app_metrics.get('error_rate_percent', 0)
    if error_rate > 20:
        status = 'unhealthy'
        issues.append(f'Critical error rate: {error_rate}%')
    elif error_rate > 10:
        status = 'degraded' if status == 'healthy' else status
        issues.append(f'High error rate: {error_rate}%')
    
    # Check response time
    avg_response_time = app_metrics.get('average_response_time_ms', 0)
    if avg_response_time > 5000:  # 5 seconds
        status = 'unhealthy'
        issues.append(f'Critical response time: {avg_response_time}ms')
    elif avg_response_time > 2000:  # 2 seconds
        status = 'degraded' if status == 'healthy' else status
        issues.append(f'High response time: {avg_response_time}ms')
    
    return {
        'status': status,
        'issues': issues,
        'metrics': app_metrics
    }