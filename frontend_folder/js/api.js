/**
 * API Module for Fraudulent Claim Detection Agent
 * Handles all HTTP requests to the backend Flask API
 */

class APIClient {
    constructor() {
        // Allow a global override BEFORE loading this file via js/config.js
        // window.FRAUD_API_BASE_URL = 'http://192.168.1.10:8000'
        const override = window.FRAUD_API_BASE_URL && typeof window.FRAUD_API_BASE_URL === 'string'
            ? window.FRAUD_API_BASE_URL.trim() : '';
        if (override) {
            this.baseURL = override.replace(/\/$/, '');
        } else {
            const host = window.location.hostname;
            const isLocalHost = ['localhost', '127.0.0.1', '::1'].includes(host);
            // If user is accessing the frontend from another machine (host not localhost),
            // point to same host on port 8000 for API (Docker published port) to avoid broken localhost references.
            if (!isLocalHost) {
                this.baseURL = `${window.location.protocol}//${host}:8000`;
            } else {
                this.baseURL = 'http://localhost:8000';
            }
        }
        // Normalize (remove trailing slash)
        this.baseURL = this.baseURL.replace(/\/$/, '');

        this._fallbackTried = false; // track if we attempted fallback
        this.token = localStorage.getItem('token');

        // --- Defensive sanitization: fix any malformed baseURL like http://host:800/0 (caused by prior buggy regex) ---
        const originalBase = this.baseURL;
        // Detect pattern host:NNN/ digit -> recombine if matches 800/0 => 8000
        const splitPortMatch = this.baseURL.match(/^(https?:\/\/[^:]+:)(\d{2,3})\/(\d)(?=\b|\/)/);
        if (splitPortMatch) {
            const candidate = splitPortMatch[2] + splitPortMatch[3];
            // Only auto-fix if resulting 4-digit port seems plausible (e.g. 8000-8999)
            if (/^8\d{3}$/.test(candidate)) {
                this.baseURL = this.baseURL.replace(/:(\d{2,3})\/(\d)/, ':' + candidate);
            }
        }
        // Fallback: if still contains ':800/0' specifically, brute force replace
        if (this.baseURL.includes(':800/0')) {
            this.baseURL = this.baseURL.replace(':800/0', ':8000/');
        }
        if (this.baseURL !== originalBase) {
            console.warn('[API] Corrected malformed baseURL', { before: originalBase, after: this.baseURL });
        }

        // Request interceptor to add auth token
        this.setupInterceptors();
    }

    /**
     * Setup request interceptors for authentication
     */
    setupInterceptors() {
    const originalFetch = window.fetch;
    window.fetch = async (url, options = {}) => {
            // Add auth header if token exists
            if (this.token && !url.includes('/auth/login') && !url.includes('/auth/register')) {
                options.headers = {
                    ...options.headers,
                    'Authorization': `Bearer ${this.token}`,
                };
            }

            // Add content-type for JSON requests
            if (options.body && typeof options.body === 'string') {
                options.headers = {
                    'Content-Type': 'application/json',
                    ...options.headers,
                };
            }

            // Add CORS headers
            options.headers = {
                'Accept': 'application/json',
                ...options.headers,
            };

            // Build a fully qualified URL without breaking relative static assets
            let fullUrl;
            if (/^https?:\/\//i.test(url)) {
                // Absolute URL, forward as-is
                fullUrl = url;
            } else if (typeof url === 'string' && url.startsWith('/')) {
                // API-style absolute path → prefix with API base URL
                fullUrl = `${this.baseURL}${url}`;
            } else {
                // Relative static asset → resolve against current origin
                const rel = typeof url === 'string' ? url.replace(/^\//, '') : url;
                fullUrl = `${window.location.origin}/${rel}`;
            }

            // Previous implementation attempted to insert a slash after host:port using a regex.
            // That occasionally produced malformed URLs like http://localhost:800/0/auth (splitting port digits)
            // when a concatenation edge case occurred. We now rely on controlled joining.
            if (typeof fullUrl === 'string') {
                // Normalize any accidental duplicate slashes (except after protocol)
                fullUrl = fullUrl.replace(/([^:])\/+/g, (m, p1) => p1 + '/');
                // Basic sanity: ensure host:port is followed by a single '/'
                const m = fullUrl.match(/^(https?:\/\/[^\/]+)(.*)$/);
                if (m) {
                    const host = m[1];
                    let path = m[2];
                    if (!path.startsWith('/')) path = '/' + path;
                    fullUrl = host + path;
                }
                // Final defensive fix for any lingering :800/0 pattern in actual request URL
                if (fullUrl.includes(':800/0')) {
                    const fixed = fullUrl.replace(':800/0', ':8000/');
                    console.warn('[API] Corrected malformed request URL', { before: fullUrl, after: fixed });
                    fullUrl = fixed;
                }
            }

            try {
                const response = await originalFetch(fullUrl, options);
                // If backend not reachable (e.g., 404 root mismatch) and we haven't tried fallback,
                // attempt switching to legacy Flask on port 5000.
                // Only apply fallback for API calls (those going to current baseURL), not for static assets.
                if (
                    !response.ok &&
                    response.status === 404 &&
                    !this._fallbackTried &&
                    this.baseURL.includes(':8000') &&
                    typeof fullUrl === 'string' && fullUrl.startsWith(this.baseURL + '/')
                ) {
                    this._fallbackTried = true;
                    const previousBase = this.baseURL;
                    this.baseURL = 'http://localhost:5000';
                    console.warn(`[API] 404 from ${previousBase}. Falling back to ${this.baseURL}`);
                    const retryUrl = fullUrl.replace(previousBase + '/', this.baseURL + '/');
                    return await originalFetch(retryUrl, options);
                }
                // Handle token expiration
                if (response.status === 401 && this.token) {
                    this.logout();
                    window.location.reload();
                }
                return response;
            } catch (err) {
                // Network failure: attempt one-time fallback if on 8000, only for API calls
                if (
                    !this._fallbackTried &&
                    this.baseURL.includes(':8000') &&
                    typeof fullUrl === 'string' && fullUrl.startsWith(this.baseURL + '/')
                ) {
                    this._fallbackTried = true;
                    const previousBase = this.baseURL;
                    this.baseURL = 'http://localhost:5000';
                    console.warn(`[API] Network error from ${previousBase}. Retrying with fallback ${this.baseURL}`);
                    const retryUrl = fullUrl.replace(previousBase + '/', this.baseURL + '/');
                    return originalFetch(retryUrl, options);
                }
                throw err;
            }
        };
    }

    /**
     * Make HTTP request with error handling
     */
    async makeRequest(endpoint, options = {}) {
        try {
            const response = await fetch(endpoint, options);
            let data;
            const ct = response.headers.get('content-type') || '';
            if (ct.includes('application/json')) {
                try {
                    data = await response.json();
                } catch (jsonErr) {
                    console.warn('[API] Failed to parse JSON response', jsonErr);
                    data = { error: 'Invalid JSON response from server' };
                }
            } else {
                // Fallback to text for debugging (e.g., HTML error page, 502 proxy, etc.)
                const text = await response.text();
                data = { error: text.slice(0, 400) };
            }

            if (!response.ok) {
                const errMsg = data.error || data.message || `HTTP ${response.status}`;
                const debug = { status: response.status, url: endpoint, baseURL: this.baseURL };
                console.error('[API] Request failed', debug, 'payload:', options.body);
                throw new Error(errMsg);
            }

            return data;
        } catch (error) {
            console.error('API Request failed:', error);
            throw error;
        }
    }

    /**
     * Authentication Methods
     */
    async login(email, password) {
        const response = await this.makeRequest('/auth/login', {
            method: 'POST',
            body: JSON.stringify({ email, password })
        });
        
        if (response.access_token) {
            this.token = response.access_token;
            localStorage.setItem('token', this.token);
            localStorage.setItem('user', JSON.stringify(response.user));
        }
        
        return response;
    }

    async register(userData) {
        return await this.makeRequest('/auth/register', {
            method: 'POST',
            body: JSON.stringify(userData)
        });
    }

    async getProfile() {
        return await this.makeRequest('/auth/profile');
    }

    async updateProfile(profileData) {
        return await this.makeRequest('/auth/profile', {
            method: 'PUT',
            body: JSON.stringify(profileData)
        });
    }

    async changePassword(currentPassword, newPassword) {
        return await this.makeRequest('/auth/change-password', {
            method: 'POST',
            body: JSON.stringify({
                current_password: currentPassword,
                new_password: newPassword
            })
        });
    }

    logout() {
        this.token = null;
        localStorage.removeItem('token');
        localStorage.removeItem('user');
    }

    /**
     * Claims Methods
     */
    async getClaims(params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const endpoint = `/claims${queryString ? '?' + queryString : ''}`;
        return await this.makeRequest(endpoint);
    }

    async getClaimById(claimId) {
        return await this.makeRequest(`/claims/${claimId}`);
    }

    async submitClaim(claimData) {
        return await this.makeRequest('/claims', {
            method: 'POST',
            body: JSON.stringify(claimData)
        });
    }

    async updateClaim(claimId, updateData) {
        return await this.makeRequest(`/claims/${claimId}`, {
            method: 'PUT',
            body: JSON.stringify(updateData)
        });
    }

    async uploadClaimDocuments(claimId, files) {
        const formData = new FormData();
        
        for (let i = 0; i < files.length; i++) {
            formData.append('documents', files[i]);
        }

        // Remove content-type header to let browser set it with boundary
        const response = await fetch(`${this.baseURL}/claims/${claimId}/documents`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.token}`,
            },
            body: formData
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Failed to upload documents');
        }

        return data;
    }

    async getClaimDocuments(claimId) {
        return await this.makeRequest(`/claims/${claimId}/documents`);
    }

    async deleteClaimDocument(claimId, documentId) {
        return await this.makeRequest(`/claims/${claimId}/documents/${documentId}`, {
            method: 'DELETE'
        });
    }

    async processClaimML(claimId) {
        return await this.makeRequest(`/claims/${claimId}/process`, {
            method: 'POST'
        });
    }

    /**
     * Policies Methods
     */
    async getPolicies(params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const endpoint = `/claims/policies${queryString ? '?' + queryString : ''}`;
        return await this.makeRequest(endpoint);
    }

    async getPolicyById(policyId) {
        return await this.makeRequest(`/claims/policies/${policyId}`);
    }

    async createPolicy(data) {
        return await this.makeRequest('/claims/policies', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }
    /**
     * Metrics Methods
     */
    async getFraudStatistics() {
        return await this.makeRequest('/metrics/fraud-statistics');
    }

    async getSystemHealth() {
        return await this.makeRequest('/metrics/system-health');
    }

    async getPerformanceMetrics() {
        return await this.makeRequest('/metrics/performance');
    }

    async getClaimsAnalytics(timeframe = '30d') {
        return await this.makeRequest(`/metrics/claims-analytics?timeframe=${timeframe}`);
    }

    async getUserStatistics(userId = null) {
        const endpoint = userId ? `/metrics/user-statistics/${userId}` : '/metrics/user-statistics';
        return await this.makeRequest(endpoint);
    }

    /**
     * Admin Methods
     */
    async getAdminDashboard() {
        return await this.makeRequest('/admin/dashboard');
    }

    async getUsers(params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const endpoint = `/admin/users${queryString ? '?' + queryString : ''}`;
        return await this.makeRequest(endpoint);
    }

    async getUserById(userId) {
        return await this.makeRequest(`/admin/users/${userId}`);
    }

    async updateUser(userId, userData) {
        return await this.makeRequest(`/admin/users/${userId}`, {
            method: 'PUT',
            body: JSON.stringify(userData)
        });
    }

    async deactivateUser(userId) {
        return await this.makeRequest(`/admin/users/${userId}/deactivate`, {
            method: 'POST'
        });
    }

    async activateUser(userId) {
        return await this.makeRequest(`/admin/users/${userId}/activate`, {
            method: 'POST'
        });
    }

    async getAuditLogs(params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const endpoint = `/admin/audit-logs${queryString ? '?' + queryString : ''}`;
        return await this.makeRequest(endpoint);
    }

    async getSystemSettings() {
        return await this.makeRequest('/admin/system-settings');
    }

    async updateSystemSettings(settings) {
        return await this.makeRequest('/admin/system-settings', {
            method: 'PUT',
            body: JSON.stringify(settings)
        });
    }

    async exportData(dataType, params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const endpoint = `/admin/export/${dataType}${queryString ? '?' + queryString : ''}`;
        
        const response = await fetch(`${this.baseURL}${endpoint}`, {
            headers: {
                'Authorization': `Bearer ${this.token}`,
            }
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Export failed');
        }

        return response.blob();
    }

    /**
     * ML Model Methods
     */
    async getModelPerformance() {
        return await this.makeRequest('/metrics/ml-performance');
    }

    async retrainModel(modelName) {
        return await this.makeRequest(`/admin/retrain-model/${modelName}`, {
            method: 'POST'
        });
    }

    async getModelPrediction(claimData) {
        return await this.makeRequest('/claims/predict', {
            method: 'POST',
            body: JSON.stringify(claimData)
        });
    }

    /**
     * Health Check Methods
     */
    async healthCheck() {
        return await this.makeRequest('/health');
    }

    async checkEmailAvailability(email) {
        const url = `${this.baseURL}/auth/check-email?email=${encodeURIComponent(email)}`;
        const response = await fetch(url, { headers: { 'Accept': 'application/json' }});
        if (!response.ok) {
            // Treat non-200 as unknown (fail open so user can still try)
            return { email, available: true, error: true };
        }
        return await response.json();
    }

    async databaseHealth() {
        return await this.makeRequest('/health/database');
    }

    async cacheHealth() {
        return await this.makeRequest('/health/cache');
    }

    /**
     * Utility Methods
     */
    isAuthenticated() {
        return !!this.token;
    }

    getCurrentUser() {
        const userStr = localStorage.getItem('user');
        return userStr ? JSON.parse(userStr) : null;
    }

    getUserRole() {
        const user = this.getCurrentUser();
        return user ? user.role : null;
    }

    hasRole(role) {
        return this.getUserRole() === role;
    }

    hasAnyRole(roles) {
        const userRole = this.getUserRole();
        return roles.includes(userRole);
    }

    /**
     * File Download Helper
     */
    async downloadFile(url, filename) {
        try {
            const response = await fetch(`${this.baseURL}${url}`, {
                headers: {
                    'Authorization': `Bearer ${this.token}`,
                }
            });

            if (!response.ok) {
                throw new Error('Download failed');
            }

            const blob = await response.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(downloadUrl);
        } catch (error) {
            console.error('Download failed:', error);
            throw error;
        }
    }

    /**
     * Batch Operations
     */
    async batchUpdateClaims(claimIds, updateData) {
        return await this.makeRequest('/claims/batch-update', {
            method: 'PUT',
            body: JSON.stringify({
                claim_ids: claimIds,
                update_data: updateData
            })
        });
    }

    async batchDeleteDocuments(documentIds) {
        return await this.makeRequest('/claims/documents/batch-delete', {
            method: 'DELETE',
            body: JSON.stringify({
                document_ids: documentIds
            })
        });
    }

    /**
     * Search Methods
     */
    async searchClaims(query, filters = {}) {
        return await this.makeRequest('/claims/search', {
            method: 'POST',
            body: JSON.stringify({
                query: query,
                filters: filters
            })
        });
    }

    async searchUsers(query, filters = {}) {
        return await this.makeRequest('/admin/users/search', {
            method: 'POST',
            body: JSON.stringify({
                query: query,
                filters: filters
            })
        });
    }

    /**
     * Real-time Data Methods
     */
    async subscribeToNotifications(callback) {
        // This would be implemented with WebSockets or Server-Sent Events
        // For now, we'll use polling
        if (this.notificationInterval) {
            clearInterval(this.notificationInterval);
        }

        this.notificationInterval = setInterval(async () => {
            try {
                const notifications = await this.getNotifications();
                callback(notifications);
            } catch (error) {
                console.error('Failed to fetch notifications:', error);
            }
        }, 30000); // Poll every 30 seconds
    }

    async getNotifications() {
        return await this.makeRequest('/notifications');
    }

    async markNotificationAsRead(notificationId) {
        return await this.makeRequest(`/notifications/${notificationId}/read`, {
            method: 'PUT'
        });
    }

    unsubscribeFromNotifications() {
        if (this.notificationInterval) {
            clearInterval(this.notificationInterval);
            this.notificationInterval = null;
        }
    }

    /**
     * Error Handling Helper
     */
    handleError(error, context = '') {
        console.error(`API Error${context ? ` in ${context}` : ''}:`, error);
        
        // Show user-friendly error message
        const message = this.getUserFriendlyError(error.message);
        showAlert(message, 'danger');
    }

    getUserFriendlyError(errorMessage) {
        const errorMap = {
            'Network request failed': 'Connection error. Please check your internet connection.',
            'Failed to fetch': 'Unable to connect to server. Please try again later.',
            'Unauthorized': 'Session expired. Please log in again.',
            'Forbidden': 'You do not have permission to perform this action.',
            'Not Found': 'The requested resource was not found.',
            'Internal Server Error': 'Server error. Please try again later.',
            'Bad Request': 'Invalid request. Please check your input.',
        };

        for (const [key, value] of Object.entries(errorMap)) {
            if (errorMessage.includes(key)) {
                return value;
            }
        }

        return errorMessage || 'An unexpected error occurred. Please try again.';
    }
}

// Create global API instance
const API = new APIClient();

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = APIClient;
}