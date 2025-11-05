/**
 * Main JavaScript file for Fraudulent Claim Detection Agent
 * Handles UI interactions, navigation, and data visualization
 */

// Global variables
let currentUser = null;
let currentSection = 'home';
let claimsData = [];
let chartsInstances = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * Initialize the application
 */
async function initializeApp() {
    try {
        // Check if user is already logged in
        if (API.isAuthenticated()) {
            currentUser = API.getCurrentUser();
            updateUIForLoggedInUser();
            loadDashboardData();
        } else {
            updateUIForGuest();
        }

        // Load home page statistics
        await loadHomeStatistics();
        
        // Set up event listeners
        setupEventListeners();
        
        // Initialize Bootstrap tooltips and popovers
        initializeBootstrapComponents();
        
    } catch (error) {
        console.error('Failed to initialize app:', error);
        showAlert('Failed to initialize application. Please refresh the page.', 'danger');
    }
}

/**
 * Set up all event listeners
 */
function setupEventListeners() {
    // Authentication forms
    document.getElementById('login-form').addEventListener('submit', handleLogin);
    document.getElementById('register-form').addEventListener('submit', handleRegister);
    const regEmail = document.getElementById('register-email');
    if (regEmail) {
        regEmail.addEventListener('input', debounce(handleRegisterEmailInput, 500));
        regEmail.addEventListener('blur', handleRegisterEmailInput);
    }
    const createPolicyForm = document.getElementById('create-policy-form');
    if (createPolicyForm) {
        createPolicyForm.addEventListener('submit', handleCreatePolicySubmit);
    }
    document.getElementById('submit-claim-form').addEventListener('submit', handleSubmitClaim);
    
    // Search and filters
    const searchInput = document.getElementById('claims-search');
    if (searchInput) {
        searchInput.addEventListener('input', debounce(filterClaims, 300));
    }
    
    const statusFilter = document.getElementById('claims-status-filter');
    if (statusFilter) {
        statusFilter.addEventListener('change', filterClaims);
    }
    
    const sortSelect = document.getElementById('claims-sort');
    if (sortSelect) {
        sortSelect.addEventListener('change', filterClaims);
    }

    // Navigation handling
    window.addEventListener('popstate', handleBrowserNavigation);

    // Removed lessons refresh on language change
    
    // Auto-refresh for real-time data
    if (API.isAuthenticated()) {
        setInterval(refreshRealTimeData, 60000); // Refresh every minute
    }
}

/**
 * Initialize Bootstrap components
 */
function initializeBootstrapComponents() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

/**
 * Update UI for logged-in user
 */
function updateUIForLoggedInUser() {
    // Hide login/register buttons
    document.getElementById('nav-login').style.display = 'none';
    document.getElementById('nav-register').style.display = 'none';
    
    // Show user menu and navigation items
    document.getElementById('nav-user').style.display = 'block';
    document.getElementById('nav-dashboard').style.display = 'block';
    document.getElementById('nav-claims').style.display = 'block';
    
    // Show admin section if user is admin
    if (currentUser && (currentUser.role === 'admin' || currentUser.role === 'supervisor')) {
        document.getElementById('nav-admin').style.display = 'block';
    }
    
    // Update username display
    if (currentUser) {
        document.getElementById('username-display').textContent = 
            `${currentUser.first_name} ${currentUser.last_name}`;
    }
}

/**
 * Update UI for guest user
 */
function updateUIForGuest() {
    // Show login/register buttons
    document.getElementById('nav-login').style.display = 'block';
    document.getElementById('nav-register').style.display = 'block';
    
    // Hide user menu and navigation items
    document.getElementById('nav-user').style.display = 'none';
    document.getElementById('nav-dashboard').style.display = 'none';
    document.getElementById('nav-claims').style.display = 'none';
    document.getElementById('nav-admin').style.display = 'none';
    
    // Show home section
    showSection('home');
}

// --- Registration Email Availability Logic ---
let lastEmailCheck = { email: null, available: null };
let emailCheckInFlight = false;

async function handleRegisterEmailInput() {
    const emailInput = document.getElementById('register-email');
    const feedback = document.getElementById('register-email-feedback');
    const submitBtn = document.querySelector('#register-form button[type="submit"]');
    if (!emailInput || !feedback || !submitBtn) return;

    const email = emailInput.value.trim().toLowerCase();
    if (!email) {
        feedback.textContent = '';
        feedback.className = 'form-text';
        submitBtn.disabled = true;
        return;
    }
    // Basic format check
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
        feedback.textContent = 'Invalid email format';
        feedback.className = 'form-text text-danger';
        submitBtn.disabled = true;
        return;
    }
    // Avoid duplicate request
    if (email === lastEmailCheck.email && lastEmailCheck.available !== null) {
        applyEmailAvailabilityFeedback(lastEmailCheck.available, feedback, submitBtn);
        return;
    }
    if (emailCheckInFlight) return; // simple throttle
    emailCheckInFlight = true;
    feedback.textContent = 'Checking availability...';
    feedback.className = 'form-text text-muted';
    try {
        const res = await API.checkEmailAvailability(email);
        lastEmailCheck = { email, available: !!res.available };
        applyEmailAvailabilityFeedback(res.available, feedback, submitBtn);
    } catch (e) {
        feedback.textContent = 'Could not verify email (temporary issue)';
        feedback.className = 'form-text text-warning';
        // Allow attempt anyway
        submitBtn.disabled = false;
    } finally {
        emailCheckInFlight = false;
    }
}

function applyEmailAvailabilityFeedback(isAvailable, feedbackEl, submitBtn) {
    if (isAvailable) {
        feedbackEl.textContent = 'Email is available';
        feedbackEl.className = 'form-text text-success';
        submitBtn.disabled = false;
    } else {
        feedbackEl.textContent = 'Email already registered';
        feedbackEl.className = 'form-text text-danger';
        submitBtn.disabled = true;
    }
}


/**
 * Show specific section and hide others
 */
function showSection(sectionName) {
    // Hide all sections
    const sections = document.querySelectorAll('.content-section');
    sections.forEach(section => {
        section.style.display = 'none';
    });
    
    // Show selected section
    const targetSection = document.getElementById(`${sectionName}-section`);
    if (targetSection) {
        targetSection.style.display = 'block';
        currentSection = sectionName;
        
        // Update URL
        history.pushState({ section: sectionName }, '', `#${sectionName}`);
        
        // Update navigation
        updateNavigation(sectionName);
        
        // Load section-specific data
        loadSectionData(sectionName);
    }
}

/**
 * Update navigation active states
 */
function updateNavigation(activeSection) {
    // Remove active class from all nav links
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    navLinks.forEach(link => {
        link.classList.remove('active');
    });
    
    // Add active class to current section
    const activeLink = document.querySelector(`[onclick="showSection('${activeSection}')"]`);
    if (activeLink) {
        activeLink.classList.add('active');
    }
}

/**
 * Load section-specific data
 */
async function loadSectionData(sectionName) {
    try {
        showLoading(true);
        
        switch (sectionName) {
            case 'dashboard':
                await loadDashboardData();
                break;
            case 'claims':
                await loadClaimsData();
                break;
            case 'admin':
                if (API.hasAnyRole(['admin', 'supervisor'])) {
                    await loadAdminData();
                }
                break;
            // lessons section removed
            case 'home':
                await loadHomeStatistics();
                break;
        }
    } catch (error) {
        console.error(`Failed to load ${sectionName} data:`, error);
        showAlert(`Failed to load ${sectionName} data. Please try again.`, 'danger');
    } finally {
        showLoading(false);
    }
}

/**
 * Handle browser navigation (back/forward buttons)
 */
function handleBrowserNavigation(event) {
    const hash = window.location.hash.substring(1);
    if (hash && ['home', 'dashboard', 'claims', 'admin'].includes(hash)) {
        showSection(hash);
    }
}

/**
 * Authentication Functions
 */
async function handleLogin(event) {
    event.preventDefault();
    
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;
    
    try {
        showLoading(true);
        const response = await API.login(email, password);
        
        currentUser = response.user;
        updateUIForLoggedInUser();
        
        // Close modal
        const loginModal = bootstrap.Modal.getInstance(document.getElementById('loginModal'));
        loginModal.hide();
        
        // Show success message
        showAlert(`Welcome back, ${currentUser.first_name}!`, 'success');
        
        // Navigate to dashboard
        showSection('dashboard');
        
    } catch (error) {
        showAlert(error.message || 'Login failed. Please check your credentials.', 'danger');
    } finally {
        showLoading(false);
    }
}

async function handleRegister(event) {
    event.preventDefault();
    
    const formData = {
        first_name: document.getElementById('register-first-name').value,
        last_name: document.getElementById('register-last-name').value,
        email: document.getElementById('register-email').value,
        password: document.getElementById('register-password').value
    };
    
    const confirmPassword = document.getElementById('register-confirm-password').value;
    
    if (formData.password !== confirmPassword) {
        showAlert('Passwords do not match.', 'danger');
        return;
    }
    
    try {
        showLoading(true);
        const response = await API.register(formData);
        
        // Auto-login if token returned
        if (response && response.access_token) {
            API.token = response.access_token;
            localStorage.setItem('token', response.access_token);
            if (response.user) {
                localStorage.setItem('user', JSON.stringify(response.user));
                currentUser = response.user;
            }
            updateUIForLoggedInUser();
        }

        // Close modal
        const registerModal = bootstrap.Modal.getInstance(document.getElementById('registerModal'));
        registerModal.hide();
        
        showAlert('Registration successful! You are now logged in.', 'success');
        showSection('dashboard');
        
    } catch (error) {
        showAlert(error.message || 'Registration failed. Please try again.', 'danger');
    } finally {
        showLoading(false);
    }
}

function logout() {
    API.logout();
    currentUser = null;
    updateUIForGuest();
    showAlert('You have been logged out successfully.', 'info');
}

/**
 * Modal Functions
 */
function showLoginModal() {
    const modal = new bootstrap.Modal(document.getElementById('loginModal'));
    modal.show();
}

function showRegisterModal() {
    const modal = new bootstrap.Modal(document.getElementById('registerModal'));
    modal.show();
}

function showSubmitClaimModal() {
    if (!API.isAuthenticated()) {
        showAlert('Please log in to submit a claim.', 'warning');
        showLoginModal();
        return;
    }
    
    loadPoliciesForClaimForm();
    const modal = new bootstrap.Modal(document.getElementById('submitClaimModal'));
    modal.show();
}

function showProfileModal() {
    // This would load and show user profile editing modal
    // Implementation would be similar to login/register modals
    showAlert('Profile editing feature coming soon!', 'info');
}

/**
 * Claims Functions
 */
async function loadClaimsData() {
    try {
        const response = await API.getClaims();
        claimsData = response.claims || [];
        renderClaimsTable(claimsData);
        renderClaimsPagination(response.pagination || {});
    } catch (error) {
        API.handleError(error, 'loading claims');
    }
}

async function handleSubmitClaim(event) {
    event.preventDefault();
    const selectedPolicy = document.getElementById('claim-policy').value;
    if (!selectedPolicy) {
        showAlert('Please select a policy.', 'warning');
        return;
    }
    // The backend expects policy_number in JSON claim submission
    // We stored policy.id in option value, so we need to look up its policy_number.
    // For simplicity, reuse the text label before the first space or parse pattern "POL123 -".
    const policyOption = document.querySelector(`#claim-policy option[value='${selectedPolicy}']`);
    let policyNumber = policyOption ? policyOption.textContent.split(' - ')[0] : selectedPolicy;

    const formData = {
        policy_number: policyNumber,
        claim_amount: parseFloat(document.getElementById('claim-amount').value),
        incident_date: document.getElementById('incident-date').value,
        location: document.getElementById('incident-location').value,
        incident_description: document.getElementById('incident-description').value
    };
    
    try {
        showLoading(true);
    const response = await API.submitClaim(formData);
        
        // Handle file uploads if any
        const fileInput = document.getElementById('claim-documents');
        if (fileInput.files.length > 0) {
            // Response from FastAPI currently returns claim_id; adjust if structure differs
            const newClaimId = response.claim_id || (response.claim && response.claim.id);
            if (newClaimId) {
                await API.uploadClaimDocuments(newClaimId, fileInput.files);
            }
        }
        
        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('submitClaimModal'));
        modal.hide();
        
        showAlert('Claim submitted successfully!', 'success');
        
        // Refresh claims data
        if (currentSection === 'claims') {
            await loadClaimsData();
        }
        
        // Reset form
        document.getElementById('submit-claim-form').reset();
        
    } catch (error) {
        API.handleError(error, 'submitting claim');
    } finally {
        showLoading(false);
    }
}

async function loadPoliciesForClaimForm() {
    try {
        const user = API.getCurrentUser();
        const isAdmin = user && ['admin','supervisor'].includes(user.role);
        const response = await API.getPolicies(isAdmin ? { all: true } : {});
        const policySelect = document.getElementById('claim-policy');
        policySelect.innerHTML = '<option value="">Select a policy</option>';
        
        response.policies.forEach(policy => {
            const option = document.createElement('option');
            option.value = policy.id;
            option.textContent = `${policy.policy_number} - ${policy.policy_type} ($${policy.coverage_amount})`;
            policySelect.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load policies:', error);
    }
}

function renderClaimsTable(claims) {
    const tbody = document.querySelector('#claims-table tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    claims.forEach(claim => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>
                <small class="text-muted">${claim.id.substring(0, 8)}...</small>
            </td>
            <td>
                <span class="badge bg-secondary">${claim.policy_number}</span>
            </td>
            <td>
                <strong>$${formatNumber(claim.claim_amount)}</strong>
            </td>
            <td>${formatDate(claim.incident_date)}</td>
            <td>
                <span class="badge status-badge status-${claim.status}">
                    ${formatStatus(claim.status)}
                </span>
            </td>
            <td>
                <span class="risk-score ${getRiskClass(claim.risk_score)}">
                    ${claim.risk_score ? (claim.risk_score * 100).toFixed(1) + '%' : 'N/A'}
                </span>
            </td>
            <td>
                <span class="badge bg-info">${claim.document_count || 0}</span>
            </td>
            <td>
                <div class="btn-group btn-group-sm">
                    <button class="btn btn-outline-primary" onclick="viewClaim('${claim.id}')" 
                            title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="btn btn-outline-secondary" onclick="downloadClaimReport('${claim.id}')" 
                            title="Download Report">
                        <i class="fas fa-download"></i>
                    </button>
                </div>
            </td>
        `;
        tbody.appendChild(row);
    });
}

function renderClaimsPagination(pagination) {
    const paginationContainer = document.getElementById('claims-pagination');
    if (!paginationContainer || !pagination.pages) return;
    
    paginationContainer.innerHTML = '';
    
    // Previous button
    if (pagination.current_page > 1) {
        paginationContainer.innerHTML += `
            <li class="page-item">
                <a class="page-link" href="#" onclick="loadClaimsPage(${pagination.current_page - 1})">
                    <i class="fas fa-chevron-left"></i>
                </a>
            </li>
        `;
    }
    
    // Page numbers
    for (let i = 1; i <= pagination.pages; i++) {
        const isActive = i === pagination.current_page;
        paginationContainer.innerHTML += `
            <li class="page-item ${isActive ? 'active' : ''}">
                <a class="page-link" href="#" onclick="loadClaimsPage(${i})">${i}</a>
            </li>
        `;
    }
    
    // Next button
    if (pagination.current_page < pagination.pages) {
        paginationContainer.innerHTML += `
            <li class="page-item">
                <a class="page-link" href="#" onclick="loadClaimsPage(${pagination.current_page + 1})">
                    <i class="fas fa-chevron-right"></i>
                </a>
            </li>
        `;
    }
}

async function loadClaimsPage(page) {
    try {
        showLoading(true);
        const response = await API.getClaims({ page: page });
        claimsData = response.claims || [];
        renderClaimsTable(claimsData);
        renderClaimsPagination(response.pagination || {});
    } catch (error) {
        API.handleError(error, 'loading claims page');
    } finally {
        showLoading(false);
    }
}

function filterClaims() {
    const searchTerm = document.getElementById('claims-search')?.value.toLowerCase() || '';
    const statusFilter = document.getElementById('claims-status-filter')?.value || '';
    const sortBy = document.getElementById('claims-sort')?.value || 'created_at_desc';
    
    let filteredClaims = [...claimsData];
    
    // Apply search filter
    if (searchTerm) {
        filteredClaims = filteredClaims.filter(claim =>
            claim.id.toLowerCase().includes(searchTerm) ||
            claim.policy_number.toLowerCase().includes(searchTerm) ||
            claim.incident_description.toLowerCase().includes(searchTerm)
        );
    }
    
    // Apply status filter
    if (statusFilter) {
        filteredClaims = filteredClaims.filter(claim => claim.status === statusFilter);
    }
    
    // Apply sorting
    filteredClaims.sort((a, b) => {
        switch (sortBy) {
            case 'created_at_asc':
                return new Date(a.created_at) - new Date(b.created_at);
            case 'created_at_desc':
                return new Date(b.created_at) - new Date(a.created_at);
            case 'amount_asc':
                return a.claim_amount - b.claim_amount;
            case 'amount_desc':
                return b.claim_amount - a.claim_amount;
            case 'risk_score_asc':
                return (a.risk_score || 0) - (b.risk_score || 0);
            case 'risk_score_desc':
                return (b.risk_score || 0) - (a.risk_score || 0);
            default:
                return 0;
        }
    });
    
    renderClaimsTable(filteredClaims);
}

/**
 * Dashboard Functions
 */
async function loadDashboardData() {
    try {
        const claimsPromise = API.getClaims({ limit: 5 });
        let metricsResponse = null;
        const user = API.getCurrentUser();
        const metricsAllowed = user && ['admin','supervisor','investigator'].includes(user.role);
        if (metricsAllowed) {
            try {
                metricsResponse = await API.getFraudStatistics();
            } catch (err) {
                // Graceful degrade if metrics endpoint denies access or fails
                console.warn('[Dashboard] Metrics unavailable:', err.message);
            }
        }
        const claimsResponse = await claimsPromise;
        
        // If metrics unavailable or unauthorized, fabricate minimal stats from claims
        if (!metricsResponse) {
            const claims = claimsResponse.claims || [];
            metricsResponse = {
                total_claims: claims.length,
                approved_claims: claims.filter(c => c.status === 'approved').length,
                pending_claims: claims.filter(c => c.status === 'submitted' || c.status === 'under_review').length,
                rejected_claims: claims.filter(c => c.status === 'rejected').length,
                claims_trend: [],
                risk_distribution: {},
            };
        }

        updateDashboardStats(metricsResponse);
        
        // Render recent claims
        renderRecentClaims(claimsResponse.claims || []);
        
        // Create charts
        if (metricsResponse && metricsResponse.claims_trend) {
            createDashboardCharts(metricsResponse);
        }
        if (!metricsAllowed) {
            showAlert('Limited dashboard view. Request elevated role for full metrics.', 'info', 6000);
        }
        
    } catch (error) {
        API.handleError(error, 'loading dashboard');
    }
}

function updateDashboardStats(metrics) {
    document.getElementById('dashboard-total-claims').textContent = metrics.total_claims || 0;
    document.getElementById('dashboard-approved-claims').textContent = metrics.approved_claims || 0;
    document.getElementById('dashboard-pending-claims').textContent = metrics.pending_claims || 0;
    document.getElementById('dashboard-rejected-claims').textContent = metrics.rejected_claims || 0;
}

function renderRecentClaims(claims) {
    const tbody = document.querySelector('#recent-claims-table tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    claims.forEach(claim => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><small class="text-muted">${claim.id.substring(0, 8)}...</small></td>
            <td><strong>$${formatNumber(claim.claim_amount)}</strong></td>
            <td>${formatDate(claim.created_at)}</td>
            <td>
                <span class="badge status-badge status-${claim.status}">
                    ${formatStatus(claim.status)}
                </span>
            </td>
            <td>
                <span class="risk-score ${getRiskClass(claim.risk_score)}">
                    ${claim.risk_score ? (claim.risk_score * 100).toFixed(1) + '%' : 'N/A'}
                </span>
            </td>
            <td>
                <button class="btn btn-outline-primary btn-sm" onclick="viewClaim('${claim.id}')">
                    <i class="fas fa-eye"></i>
                </button>
            </td>
        `;
        tbody.appendChild(row);
    });
}

function createDashboardCharts(data) {
    // Claims trend chart
    createClaimsTrendChart(data.claims_trend || []);
    
    // Risk distribution chart
    createRiskDistributionChart(data.risk_distribution || {});
}

function createClaimsTrendChart(trendData) {
    const ctx = document.getElementById('claims-trend-chart');
    if (!ctx) return;
    
    // Destroy existing chart if it exists
    if (chartsInstances.claimsTrend) {
        chartsInstances.claimsTrend.destroy();
    }
    
    chartsInstances.claimsTrend = new Chart(ctx, {
        type: 'line',
        data: {
            labels: trendData.map(item => formatDate(item.date)),
            datasets: [{
                label: 'Claims Submitted',
                data: trendData.map(item => item.count),
                borderColor: 'rgb(0, 102, 204)',
                backgroundColor: 'rgba(0, 102, 204, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        precision: 0
                    }
                }
            }
        }
    });
}

function createRiskDistributionChart(distributionData) {
    const ctx = document.getElementById('risk-distribution-chart');
    if (!ctx) return;
    
    // Destroy existing chart if it exists
    if (chartsInstances.riskDistribution) {
        chartsInstances.riskDistribution.destroy();
    }
    
    chartsInstances.riskDistribution = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Low Risk', 'Medium Risk', 'High Risk'],
            datasets: [{
                data: [
                    distributionData.low || 0,
                    distributionData.medium || 0,
                    distributionData.high || 0
                ],
                backgroundColor: [
                    'rgb(40, 167, 69)',
                    'rgb(255, 193, 7)',
                    'rgb(220, 53, 69)'
                ],
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

/**
 * Admin Functions
 */
async function loadAdminData() {
    if (!API.hasAnyRole(['admin', 'supervisor'])) {
        showAlert('Access denied. Admin privileges required.', 'danger');
        return;
    }
    
    try {
        const [dashboardData, usersData, auditData, modelData] = await Promise.all([
            API.getAdminDashboard(),
            API.getUsers({ limit: 10 }),
            API.getAuditLogs({ limit: 20 }),
            API.getModelPerformance()
        ]);
        
        updateAdminStats(dashboardData);
        renderAdminUsers(usersData.users || []);
        renderAuditLog(auditData.logs || []);
        createMLPerformanceChart(modelData);
        
    } catch (error) {
        API.handleError(error, 'loading admin data');
    }
}

function updateAdminStats(data) {
    document.getElementById('admin-total-users').textContent = data.total_users || 0;
    document.getElementById('admin-fraud-rate').textContent = 
        (data.fraud_rate ? (data.fraud_rate * 100).toFixed(1) + '%' : '0%');
    document.getElementById('admin-accuracy').textContent = 
        (data.ml_accuracy ? (data.ml_accuracy * 100).toFixed(1) + '%' : '0%');
    document.getElementById('admin-investigations').textContent = data.active_investigations || 0;
}

function renderAdminUsers(users) {
    const tbody = document.querySelector('#admin-users-table tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    users.forEach(user => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${user.first_name} ${user.last_name}</td>
            <td>
                <span class="badge bg-primary">${user.role}</span>
            </td>
            <td>
                <span class="badge ${user.is_active ? 'bg-success' : 'bg-danger'}">
                    ${user.is_active ? 'Active' : 'Inactive'}
                </span>
            </td>
            <td>
                <div class="btn-group btn-group-sm">
                    <button class="btn btn-outline-primary" onclick="viewUser(${user.id})" 
                            title="View User">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="btn btn-outline-warning" onclick="editUser(${user.id})" 
                            title="Edit User">
                        <i class="fas fa-edit"></i>
                    </button>
                </div>
            </td>
        `;
        tbody.appendChild(row);
    });
}

function renderAuditLog(logs) {
    const tbody = document.querySelector('#audit-log-table tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    logs.forEach(log => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><small>${formatDateTime(log.timestamp)}</small></td>
            <td>${log.user_name || 'System'}</td>
            <td>
                <span class="badge bg-info">${log.action}</span>
            </td>
            <td>${log.resource_type} ${log.resource_id ? `(${log.resource_id})` : ''}</td>
            <td>
                <small class="text-muted">
                    ${log.details ? JSON.stringify(log.details).substring(0, 50) + '...' : 'No details'}
                </small>
            </td>
        `;
        tbody.appendChild(row);
    });
}

function createMLPerformanceChart(data) {
    const ctx = document.getElementById('ml-performance-chart');
    if (!ctx) return;
    
    // Destroy existing chart if it exists
    if (chartsInstances.mlPerformance) {
        chartsInstances.mlPerformance.destroy();
    }
    
    chartsInstances.mlPerformance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            datasets: [{
                label: 'Performance Metrics',
                data: [
                    (data.accuracy || 0) * 100,
                    (data.precision || 0) * 100,
                    (data.recall || 0) * 100,
                    (data.f1_score || 0) * 100
                ],
                backgroundColor: [
                    'rgba(0, 102, 204, 0.8)',
                    'rgba(40, 167, 69, 0.8)',
                    'rgba(255, 193, 7, 0.8)',
                    'rgba(220, 53, 69, 0.8)'
                ],
                borderColor: [
                    'rgb(0, 102, 204)',
                    'rgb(40, 167, 69)',
                    'rgb(255, 193, 7)',
                    'rgb(220, 53, 69)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

/**
 * Home Page Functions
 */
async function loadHomeStatistics() {
    try {
        // Load public statistics for home page
        const response = await API.healthCheck();
        
        // Update home page stats with some sample/demo data
        // In a real application, you'd have a public API endpoint for this
        document.getElementById('total-claims').textContent = '1,234';
        document.getElementById('accuracy-rate').textContent = '94.5%';
        document.getElementById('fraud-detected').textContent = '87';
        document.getElementById('money-saved').textContent = '$2.1M';
        
    } catch (error) {
        // Don't show error for home page stats as they might be public
        console.error('Failed to load home statistics:', error);
    }
}

/**
 * Utility Functions
 */
function showAlert(message, type = 'info', duration = 5000) {
    const alertContainer = document.getElementById('alert-container');
    const alertId = 'alert-' + Date.now();
    
    const alertHTML = `
        <div id="${alertId}" class="alert alert-${type} alert-dismissible fade show" role="alert">
            <i class="fas fa-${getAlertIcon(type)} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    alertContainer.insertAdjacentHTML('afterbegin', alertHTML);
    
    // Auto-dismiss after duration
    if (duration > 0) {
        setTimeout(() => {
            const alert = document.getElementById(alertId);
            if (alert) {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }
        }, duration);
    }
}

function getAlertIcon(type) {
    const icons = {
        success: 'check-circle',
        danger: 'exclamation-triangle',
        warning: 'exclamation-circle',
        info: 'info-circle'
    };
    return icons[type] || 'info-circle';
}

function showLoading(show) {
    const spinner = document.getElementById('loading-spinner');
    if (spinner) {
        spinner.classList.toggle('d-none', !show);
    }
}

function formatNumber(num) {
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(num);
}

function formatDate(dateString) {
    return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

function formatDateTime(dateString) {
    return new Date(dateString).toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatStatus(status) {
    const statusMap = { submitted: 'Submitted', under_review: 'Under Review', approved: 'Approved', rejected: 'Rejected', investigation: 'Investigation', closed: 'Closed' };
    return statusMap[status] || status;
}

function getRiskClass(riskScore) {
    if (!riskScore) return '';
    if (riskScore < 0.3) return 'risk-low';
    if (riskScore < 0.7) return 'risk-medium';
    return 'risk-high';
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Placeholder functions for future implementation
 */
async function viewClaim(claimId) {
    if (!API.isAuthenticated()) {
        showAlert('Please log in to submit a claim.', 'warning');
        return;
    }
    try {
        showLoading(true);
        const claim = await API.getClaimById(claimId);
        populateClaimDetailsModal(claim);
        const modalEl = document.getElementById('claimDetailsModal');
        const modal = new bootstrap.Modal(modalEl);
        modal.show();
    } catch (err) {
        console.error('Failed to load claim details', err);
        showAlert(err.message || 'Failed to load claim details', 'danger');
    } finally {
        showLoading(false);
    }
}

function populateClaimDetailsModal(claim) {
    const root = document.getElementById('claim-details-body');
    if (!root) return;
    const docs = (claim.documents || []).map(d => `
        <tr>
            <td>${d.filename}</td>
            <td>${d.file_type || ''}</td>
            <td>${(d.file_size/1024).toFixed(1)} KB</td>
            <td>${d.validation_status || ''}</td>
            <td>${d.validation_score ? (d.validation_score*100).toFixed(1)+'%' : ''}</td>
        </tr>
    `).join('') || '<tr><td colspan="5" class="text-muted">No documents</td></tr>';
    const history = (claim.status_history || []).map(h => `
        <tr>
            <td>${formatDateTime(h.changed_at)}</td>
            <td>${h.status}</td>
            <td>${h.changed_by_user_id || ''}</td>
            <td>${h.notes || ''}</td>
        </tr>
    `).join('') || '<tr><td colspan="4" class="text-muted">No history</td></tr>';
    const patternMatches = (claim.pattern_matches || []).map(pm => {
        let details = pm.match_details;
        try {
            if (details && details.trim().startsWith('{')) {
                const obj = JSON.parse(details);
                details = Object.entries(obj).map(([k,v]) => `${k}: ${v}`).join(', ');
            }
        } catch { /* ignore parse errors */ }
        const sevClass = pm.severity === 'high' ? 'danger' : pm.severity === 'medium' ? 'warning' : 'secondary';
        return `
            <tr>
                <td>${pm.pattern_name}</td>
                <td><span class="badge bg-${sevClass}">${pm.severity}</span></td>
                <td>${pm.match_score != null ? (pm.match_score*100).toFixed(1)+'%' : ''}</td>
                <td>${formatDateTime(pm.matched_at)}</td>
                <td class="text-truncate" style="max-width:220px" title="${details || ''}">${details || ''}</td>
            </tr>
        `;
    }).join('') || '<tr><td colspan="5" class="text-muted">No pattern matches</td></tr>';
    root.innerHTML = `
        <div class="mb-3">
            <h5 class="mb-1">Claim <small class="text-muted">${claim.id}</small></h5>
            <div class="d-flex flex-wrap gap-3 small">
                <div><strong>Status:</strong> <span class="badge status-badge status-${claim.status}">${formatStatus(claim.status)}</span></div>
                <div><strong>Amount:</strong> $${formatNumber(claim.claim_amount)}</div>
                <div><strong>Risk:</strong> ${claim.risk_score !== null && claim.risk_score !== undefined ? (claim.risk_score*100).toFixed(1)+'%' : 'N/A'}</div>
                <div><strong>Incident:</strong> ${formatDate(claim.incident_date)}</div>
                <div><strong>Location:</strong> ${claim.location || '—'}</div>
            </div>
        </div>
        <div class="mb-3">
            <h6>Description</h6>
            <p class="border rounded p-2 bg-light">${claim.incident_description}</p>
        </div>
        <div class="row g-3 mb-3">
            <div class="col-md-6">
                <div class="card h-100">
                    <div class="card-header py-2"><strong>Policy</strong></div>
                    <div class="card-body small">
                        <div><strong>Number:</strong> ${claim.policy?.policy_number || ''}</div>
                        <div><strong>Type:</strong> ${claim.policy?.policy_type || ''}</div>
                        <div><strong>Coverage:</strong> $${claim.policy?.coverage_amount ? formatNumber(claim.policy.coverage_amount) : '0.00'}</div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card h-100">
                    <div class="card-header py-2"><strong>Claimant</strong></div>
                    <div class="card-body small">
                        <div>${claim.claimant ? claim.claimant.first_name + ' ' + claim.claimant.last_name : ''}</div>
                        <div>${claim.claimant?.email || ''}</div>
                    </div>
                </div>
            </div>
        </div>
        <div class="mb-3">
            <h6 class="mb-2">Documents</h6>
            <div class="table-responsive">
                <table class="table table-sm align-middle">
                    <thead><tr><th>Name</th><th>Type</th><th>Size</th><th>Status</th><th>Score</th></tr></thead>
                    <tbody>${docs}</tbody>
                </table>
            </div>
        </div>
        <div class="mb-3">
            <h6 class="mb-2">Pattern Matches</h6>
            <div class="table-responsive">
                <table class="table table-sm align-middle">
                    <thead><tr><th>Pattern</th><th>Severity</th><th>Score</th><th>Matched At</th><th>Details</th></tr></thead>
                    <tbody>${patternMatches}</tbody>
                </table>
            </div>
        </div>
        <div class="mb-2">
            <h6 class="mb-2">Status History</h6>
            <div class="table-responsive">
                <table class="table table-sm align-middle">
                    <thead><tr><th>Changed At</th><th>Status</th><th>User</th><th>Notes</th></tr></thead>
                    <tbody>${history}</tbody>
                </table>
            </div>
        </div>
    `;
}

async function downloadClaimReport(claimId) {
    showAlert('Report download coming soon!', 'info');
}

async function viewUser(userId) {
    showAlert('User details view coming soon!', 'info');
}

async function editUser(userId) {
    showAlert('User editing coming soon!', 'info');
}

async function refreshRealTimeData() {
    if (currentSection === 'dashboard') {
        await loadDashboardData();
    } else if (currentSection === 'claims') {
        await loadClaimsData();
    } else if (currentSection === 'admin') {
        await loadAdminData();
    }
}

// Export functions for global access
window.showSection = showSection;
window.showLoginModal = showLoginModal;
window.showRegisterModal = showRegisterModal;
window.showSubmitClaimModal = showSubmitClaimModal;
window.showProfileModal = showProfileModal;
window.logout = logout;
window.viewClaim = viewClaim;
window.downloadClaimReport = downloadClaimReport;
window.viewUser = viewUser;
window.editUser = editUser;
window.loadClaimsPage = loadClaimsPage;
window.showCreatePolicyModal = showCreatePolicyModal;

function showCreatePolicyModal() {
    if (!API.isAuthenticated()) {
        showAlert('Please log in first.', 'warning');
        return;
    }
    const form = document.getElementById('create-policy-form');
    if (form) form.reset();
    const feedback = document.getElementById('create-policy-feedback');
    if (feedback) feedback.textContent = '';
    const modal = new bootstrap.Modal(document.getElementById('createPolicyModal'));
    modal.show();
}

async function handleCreatePolicySubmit(e) {
    e.preventDefault();
    const typeEl = document.getElementById('policy-type');
    const covEl = document.getElementById('policy-coverage');
    const premEl = document.getElementById('policy-premium');
    const numEl = document.getElementById('policy-number');
    const feedback = document.getElementById('create-policy-feedback');
    const submitBtn = document.getElementById('create-policy-submit');
    if (feedback) feedback.textContent = '';
    submitBtn.disabled = true;
    try {
        const payload = {
            policy_type: typeEl.value.trim().toLowerCase(),
            coverage_amount: parseFloat(covEl.value),
            premium: parseFloat(premEl.value)
        };
        if (numEl.value.trim()) payload.policy_number = numEl.value.trim();
        if (!payload.policy_type) throw new Error('Policy type required');
        if (!payload.coverage_amount || payload.coverage_amount <= 0) throw new Error('Coverage must be > 0');
        if (!payload.premium || payload.premium <= 0) throw new Error('Premium must be > 0');
        const res = await API.createPolicy(payload);
    showAlert('Policy created successfully', 'success');
        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('createPolicyModal'));
        modal.hide();
        // Reload policies and auto-select new one
        await loadPoliciesForClaimForm();
        const sel = document.getElementById('claim-policy');
        if (sel && res.policy && res.policy.id) {
            sel.value = String(res.policy.id);
        }
    } catch (err) {
        console.error('Create policy failed', err);
        if (feedback) feedback.textContent = err.message || 'Failed to create policy';
    } finally {
        submitBtn.disabled = false;
    }
}