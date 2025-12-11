/* ================================================================ */
/* JOBMATCH SEARCH PAGE - JAVASCRIPT INTELLIGENCE                   */
/* State Management • API Integration • Real-time Updates           */
/* ================================================================ */

// ================================================================
// STATE MANAGEMENT
// ================================================================

const AppState = {
    // Query state
    currentQuery: '',
    currentPage: 1,
    resultsPerPage: 20, // Increased from 10
    
    // Results state
    results: [],
    totalResults: 0,
    totalPages: 0,
    
    // Metadata
    searchTime: 0,
    status: 'idle', // idle, searching, success, warning, error
    confidence: 0,
    
    // UI state
    isLoading: false,
    
    // Update state and notify subscribers
    update(newState) {
        Object.assign(this, newState);
        this.notify();
    },
    
    // Subscribers for state changes
    subscribers: [],
    subscribe(callback) {
        this.subscribers.push(callback);
    },
    notify() {
        this.subscribers.forEach(callback => callback(this));
    }
};

// ================================================================
// DOM ELEMENTS
// ================================================================

const DOM = {
    // Sidebar
    searchInput: document.getElementById('searchInput'),
    searchBtn: document.getElementById('searchBtn'),
    clearBtn: document.getElementById('clearBtn'),
    
    // Status Card
    statusBadge: document.getElementById('statusBadge'),
    metricQuery: document.getElementById('metricQuery'),
    metricResults: document.getElementById('metricResults'),
    metricTime: document.getElementById('metricTime'),
    metricConfidence: document.getElementById('metricConfidence'),
    
    // Results
    resultsContainer: document.getElementById('resultsContainer'),
    statusBanner: document.getElementById('statusBanner'),
    bannerTitle: document.getElementById('bannerTitle'),
    bannerMessage: document.getElementById('bannerMessage'),
    bannerClose: document.getElementById('bannerClose'),
    
    resultsHeader: document.getElementById('resultsHeader'),
    resultsTitle: document.getElementById('resultsTitle'),
    resultsCount: document.getElementById('resultsCount'),
    
    emptyState: document.getElementById('emptyState'),
    jobsList: document.getElementById('jobsList'),
    
    // Pagination
    pagination: document.getElementById('pagination'),
    prevBtn: document.getElementById('prevBtn'),
    nextBtn: document.getElementById('nextBtn'),
    paginationPages: document.getElementById('paginationPages'),
    
    // Loading
    loadingOverlay: document.getElementById('loadingOverlay'),
    
    // Filters
    filtersToggle: document.getElementById('filtersToggle'),
    filtersBody: document.getElementById('filtersBody')
};

// ================================================================
// API SERVICE
// ================================================================

class APIService {
    static API_BASE = '/api';
    
    /**
     * Search jobs with intelligent query
     */
    static async search(query, page = 1, k = 10) {
        const response = await fetch(`${this.API_BASE}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query, page, k })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    /**
     * Get job details by ID
     */
    static async getJobDetails(jobId) {
        const response = await fetch(`${this.API_BASE}/job/${jobId}`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    /**
     * Health check
     */
    static async healthCheck() {
        const response = await fetch(`${this.API_BASE}/health`);
        return await response.json();
    }
}

// ================================================================
// SEARCH CONTROLLER
// ================================================================

class SearchController {
    constructor() {
        this.debounceTimer = null;
        this.debounceDelay = 300; // ms
        
        // Stop words to remove from queries
        this.stopWords = new Set([
            'i', 'need', 'want', 'looking', 'for', 'a', 'an', 'the', 
            'at', 'in', 'on', 'to', 'from', 'with', 'by', 'am', 'is',
            'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'my', 'me', 'we', 'us', 'our'
        ]);
    }
    
    /**
     * Clean query by removing stop words and noise
     */
    cleanQuery(rawQuery) {
        const original = rawQuery.trim();
        
        // Remove punctuation and extra spaces
        let cleaned = original
            .toLowerCase()
            .replace(/[^\w\s-]/g, ' ')  // Keep only alphanumeric, spaces, hyphens
            .replace(/\s+/g, ' ')        // Normalize spaces
            .trim();
        
        // Split into words
        const words = cleaned.split(' ');
        
        // Filter out stop words but keep important healthcare terms
        const filtered = words.filter(word => {
            // Keep if:
            // 1. Not a stop word
            // 2. OR it's a healthcare-specific term (even if short)
            return !this.stopWords.has(word) || word.length >= 3;
        });
        
        // Join back
        const result = filtered.join(' ').trim();
        
        // Log transformation for debugging
        if (original !== result) {
            console.log(`🧹 Query cleaned: "${original}" → "${result}"`);
        }
        
        return result || original; // Fallback to original if empty
    }
    
    /**
     * Perform search with loading states
     */
    async performSearch(query, page = 1) {
        // Validation
        if (!query || query.trim().length === 0) {
            UI.showBanner('error', 'Empty Query', 'Please enter a search term');
            return;
        }
        
        // Clean query (remove stop words)
        const cleanedQuery = this.cleanQuery(query);
        
        if (!cleanedQuery) {
            UI.showBanner('error', 'Invalid Query', 'Please enter a valid healthcare job title');
            return;
        }
        
        try {
            // Update state
            AppState.update({
                currentQuery: cleanedQuery,  // Use cleaned query
                currentPage: page,
                isLoading: true,
                status: 'searching'
            });
            
            // Show loading
            LoadingManager.show();
            
            // API Call with cleaned query
            const startTime = performance.now();
            const response = await APIService.search(cleanedQuery, page, AppState.resultsPerPage);
            const endTime = performance.now();
            const searchTime = Math.round(endTime - startTime);
            
            // Process response
            this.handleSearchResponse(response, searchTime);
            
        } catch (error) {
            console.error('Search error:', error);
            this.handleSearchError(error);
        } finally {
            // Hide loading
            LoadingManager.hide();
            
            // Update state
            AppState.update({ isLoading: false });
        }
    }
    
    /**
     * Handle successful search response
     */
    handleSearchResponse(response, searchTime) {
        const { status, message, results, pagination, metadata } = response;
        
        // Update state
        AppState.update({
            status: status,
            results: results || [],
            totalResults: pagination?.total_results || 0,
            totalPages: pagination?.total_pages || 0,
            searchTime: searchTime,
            confidence: metadata?.avg_ce_score || 0
        });
        
        // Show banner based on status
        if (status === 'success') {
            UI.showBanner('success', 'Results Found', message);
        } else if (status === 'low_confidence') {
            UI.showBanner('warning', 'Low Confidence', message);
        } else if (status === 'rejected') {
            UI.showBanner('error', 'Query Rejected', message);
        } else if (status === 'no_results') {
            UI.showBanner('info', 'No Results', message);
        }
        
        // Render results
        UI.renderResults(results);
        
        // Update pagination
        UI.renderPagination(pagination);
    }
    
    /**
     * Handle search error
     */
    handleSearchError(error) {
        AppState.update({
            status: 'error',
            results: [],
            totalResults: 0
        });
        
        UI.showBanner('error', 'Search Failed', error.message);
    }
    
    /**
     * Debounced search (for real-time search on typing)
     */
    debouncedSearch(query) {
        clearTimeout(this.debounceTimer);
        
        this.debounceTimer = setTimeout(() => {
            this.performSearch(query);
        }, this.debounceDelay);
    }
    
    /**
     * Clear search
     */
    clearSearch() {
        DOM.searchInput.value = '';
        DOM.clearBtn.classList.add('hidden');
        
        AppState.update({
            currentQuery: '',
            results: [],
            totalResults: 0,
            status: 'idle'
        });
        
        UI.showEmptyState();
    }
}

// ================================================================
// UI MANAGER
// ================================================================

class UI {
    /**
     * Show banner with type, title, message
     */
    static showBanner(type, title, message) {
        const banner = DOM.statusBanner;
        const bannerTitle = DOM.bannerTitle;
        const bannerMessage = DOM.bannerMessage;
        
        // Set type
        banner.setAttribute('data-type', type);
        
        // Set content
        bannerTitle.textContent = title;
        bannerMessage.textContent = message;
        
        // Show
        banner.classList.remove('hidden');
        
        // Re-initialize icons
        lucide.createIcons();
        
        // Auto-hide after 5s for success
        if (type === 'success') {
            setTimeout(() => {
                banner.classList.add('hidden');
            }, 5000);
        }
    }
    
    /**
     * Update status card metrics
     */
    static updateStatusCard(state) {
        const { currentQuery, totalResults, searchTime, status, confidence } = state;
        
        // Status badge
        const statusBadge = DOM.statusBadge;
        const statusText = statusBadge.querySelector('.status-text');
        
        statusBadge.setAttribute('data-status', status);
        
        const statusLabels = {
            idle: 'Ready',
            searching: 'Searching...',
            success: 'Success',
            warning: 'Low Confidence',
            error: 'Error',
            rejected: 'Rejected'
        };
        
        statusText.textContent = statusLabels[status] || 'Unknown';
        
        // Metrics
        DOM.metricQuery.textContent = currentQuery || '—';
        DOM.metricResults.textContent = totalResults > 0 ? totalResults : '—';
        DOM.metricTime.textContent = searchTime > 0 ? `${searchTime}ms` : '—';
        
        // Confidence
        if (confidence > 0) {
            const confidencePercent = Math.round((confidence / 10) * 100);
            const confidenceLabel = 
                confidencePercent >= 80 ? '🟢 Excellent' :
                confidencePercent >= 60 ? '🟡 Good' :
                '🟠 Low';
            DOM.metricConfidence.textContent = confidenceLabel;
        } else {
            DOM.metricConfidence.textContent = '—';
        }
    }
    
    /**
     * Render results
     */
    static renderResults(results) {
        if (!results || results.length === 0) {
            this.showEmptyState();
            return;
        }
        
        // Hide empty state
        DOM.emptyState.classList.add('hidden');
        
        // Show results
        DOM.resultsHeader.classList.remove('hidden');
        DOM.jobsList.classList.remove('hidden');
        
        // Update count
        DOM.resultsCount.textContent = `${AppState.totalResults} jobs found`;
        
        // Clear previous results
        DOM.jobsList.innerHTML = '';
        
        // Render each job card
        results.forEach((job, index) => {
            const card = this.createJobCard(job, index);
            DOM.jobsList.appendChild(card);
        });
        
        // Stagger animation
        this.animateJobCards();
    }
    
    /**
     * Create a job card element
     */
    static createJobCard(job, index) {
        const card = document.createElement('article');
        card.className = 'job-card';
        card.setAttribute('data-index', index);
        
        // Quality badge
        const qualityClass = 
            job.quality === 'excellent' ? 'quality-excellent' :
            job.quality === 'good' ? 'quality-good' :
            'quality-acceptable';
        
        // Score color
        const scoreColor = 
            job.ce_score >= 7 ? 'var(--success)' :
            job.ce_score >= 4 ? 'var(--info)' :
            'var(--warning)';
        
        card.innerHTML = `
            <div class="job-card-header">
                <div class="job-title-container">
                    <i data-lucide="briefcase" class="job-icon"></i>
                    <h3 class="job-title">${this.escapeHtml(job.title)}</h3>
                </div>
                <div class="job-score" style="color: ${scoreColor}">
                    <i data-lucide="star"></i>
                    <span class="mono">${job.ce_score.toFixed(2)}</span>
                </div>
            </div>
            
            <div class="job-card-body">
                <div class="job-meta">
                    <div class="job-meta-item">
                        <i data-lucide="map-pin"></i>
                        <span>${this.escapeHtml(job.location || 'Location N/A')}</span>
                    </div>
                    <div class="job-meta-item">
                        <i data-lucide="briefcase"></i>
                        <span>${this.escapeHtml(job.experience_level || 'N/A')}</span>
                    </div>
                </div>
                
                <p class="job-description">
                    ${this.escapeHtml(job.description || 'No description available')}
                </p>
                
                <div class="job-badges">
                    <span class="badge ${qualityClass}">
                        ${job.quality || 'N/A'}
                    </span>
                    <span class="badge badge-score">
                        Relevance: ${(job.final_score * 100).toFixed(0)}%
                    </span>
                </div>
            </div>
            
            <div class="job-card-footer">
                <button class="job-btn-details" data-job-id="${job.job_id}">
                    <span>View Details</span>
                    <i data-lucide="arrow-right"></i>
                </button>
            </div>
        `;
        
        // Add click listener
        const detailsBtn = card.querySelector('.job-btn-details');
        detailsBtn.addEventListener('click', () => this.showJobDetails(job.job_id));
        
        return card;
    }
    
    /**
     * Animate job cards (stagger)
     */
    static animateJobCards() {
        const cards = document.querySelectorAll('.job-card');
        
        cards.forEach((card, index) => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                card.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, index * 50); // Stagger by 50ms
        });
        
        // Re-initialize Lucide icons
        lucide.createIcons();
    }
    
    /**
     * Show job details (modal)
     */
    static async showJobDetails(jobId) {
        try {
            // Show modal with loading state
            Modal.show();
            
            const response = await APIService.getJobDetails(jobId);
            
            if (response.status === 'success') {
                Modal.showJobDetails(response.job);
            } else {
                Modal.showError('Failed to load job details');
            }
        } catch (error) {
            console.error('Error loading job details:', error);
            Modal.showError('Failed to load job details');
        }
    }
    
    /**
     * Show empty state
     */
    static showEmptyState() {
        DOM.emptyState.classList.remove('hidden');
        DOM.resultsHeader.classList.add('hidden');
        DOM.jobsList.classList.add('hidden');
        DOM.pagination.classList.add('hidden');
    }
    
    /**
     * Render pagination
     */
    static renderPagination(pagination) {
        if (!pagination || pagination.total_pages <= 1) {
            DOM.pagination.classList.add('hidden');
            return;
        }
        
        DOM.pagination.classList.remove('hidden');
        
        const { page, total_pages, has_prev, has_next } = pagination;
        
        // Update buttons
        DOM.prevBtn.disabled = !has_prev;
        DOM.nextBtn.disabled = !has_next;
        
        // Render page numbers
        DOM.paginationPages.innerHTML = '';
        
        const maxPages = 5;
        let startPage = Math.max(1, page - Math.floor(maxPages / 2));
        let endPage = Math.min(total_pages, startPage + maxPages - 1);
        
        if (endPage - startPage < maxPages - 1) {
            startPage = Math.max(1, endPage - maxPages + 1);
        }
        
        for (let i = startPage; i <= endPage; i++) {
            const btn = document.createElement('button');
            btn.className = 'page-btn';
            btn.textContent = i;
            
            if (i === page) {
                btn.classList.add('active');
            }
            
            btn.addEventListener('click', () => {
                searchController.performSearch(AppState.currentQuery, i);
                window.scrollTo({ top: 0, behavior: 'smooth' });
            });
            
            DOM.paginationPages.appendChild(btn);
        }
    }
    
    /**
     * Escape HTML to prevent XSS
     */
    static escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// ================================================================
// LOADING MANAGER
// ================================================================

class LoadingManager {
    static stages = ['searching', 'processing', 'rendering'];
    static currentStage = 0;
    static stageInterval = null;
    
    /**
     * Show loading overlay with 3-stage animation
     */
    static show() {
        DOM.loadingOverlay.classList.remove('hidden');
        this.currentStage = 0;
        this.showStage(0);
        
        // Auto-progress through stages
        this.stageInterval = setInterval(() => {
            this.currentStage = (this.currentStage + 1) % this.stages.length;
            this.showStage(this.currentStage);
        }, 800); // Change stage every 800ms
    }
    
    /**
     * Show specific loading stage
     */
    static showStage(stageIndex) {
        const stages = document.querySelectorAll('.loading-stage');
        
        stages.forEach((stage, index) => {
            if (index === stageIndex) {
                stage.classList.remove('hidden');
            } else {
                stage.classList.add('hidden');
            }
        });
    }
    
    /**
     * Hide loading overlay
     */
    static hide() {
        clearInterval(this.stageInterval);
        
        // Show final stage briefly before hiding
        this.showStage(2);
        
        setTimeout(() => {
            DOM.loadingOverlay.classList.add('hidden');
        }, 500);
    }
}

// ================================================================
// EVENT LISTENERS
// ================================================================

function initializeEventListeners() {
    // Search button
    DOM.searchBtn.addEventListener('click', () => {
        const query = DOM.searchInput.value.trim();
        if (query) {
            searchController.performSearch(query);
        }
    });
    
    // Search input - Enter key
    DOM.searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const query = DOM.searchInput.value.trim();
            if (query) {
                searchController.performSearch(query);
            }
        }
    });
    
    // Search input - Show/hide clear button
    DOM.searchInput.addEventListener('input', (e) => {
        const value = e.target.value;
        
        if (value.length > 0) {
            DOM.clearBtn.classList.remove('hidden');
        } else {
            DOM.clearBtn.classList.add('hidden');
        }
    });
    
    // Clear button
    DOM.clearBtn.addEventListener('click', () => {
        searchController.clearSearch();
    });
    
    // Banner close
    DOM.bannerClose.addEventListener('click', () => {
        DOM.statusBanner.classList.add('hidden');
    });
    
    // Filters toggle
    DOM.filtersToggle.addEventListener('click', () => {
        DOM.filtersToggle.classList.toggle('active');
        DOM.filtersBody.classList.toggle('hidden');
    });
    
    // Pagination
    DOM.prevBtn.addEventListener('click', () => {
        if (AppState.currentPage > 1) {
            searchController.performSearch(
                AppState.currentQuery, 
                AppState.currentPage - 1
            );
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    });
    
    DOM.nextBtn.addEventListener('click', () => {
        if (AppState.currentPage < AppState.totalPages) {
            searchController.performSearch(
                AppState.currentQuery, 
                AppState.currentPage + 1
            );
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    });
}

// ================================================================
// MODAL MANAGER
// ================================================================

class Modal {
    static DOM = {
        modal: document.getElementById('jobModal'),
        overlay: document.getElementById('modalOverlay'),
        closeBtn: document.getElementById('modalClose'),
        closeBtnFooter: document.getElementById('modalCloseBtn'),
        applyBtn: document.getElementById('modalApplyBtn'),
        title: document.getElementById('modalTitle'),
        body: document.getElementById('modalBody')
    };
    
    /**
     * Show modal with loading state
     */
    static show() {
        this.DOM.modal.classList.remove('hidden');
        this.DOM.body.innerHTML = `
            <div class="modal-loading">
                <div class="spinner"></div>
                <p>Loading job details...</p>
            </div>
        `;
        document.body.style.overflow = 'hidden';
    }
    
    /**
     * Hide modal
     */
    static hide() {
        this.DOM.modal.classList.add('hidden');
        document.body.style.overflow = '';
    }
    
    /**
     * Show job details in modal
     */
    static showJobDetails(job) {
        this.DOM.title.textContent = job.title || 'Job Details';
        
        // Helper to display field
        const displayField = (label, value) => {
            if (!value || value === 'N/A' || value === null || value === undefined) {
                return '';
            }
            return `
                <div class="job-detail-section">
                    <div class="job-detail-label">${label}</div>
                    <div class="job-detail-value">${this.escapeHtml(String(value))}</div>
                </div>
            `;
        };
        
        // Build modal content
        let content = '';
        
        // Location
        if (job.location) {
            content += `
                <div class="job-detail-section">
                    <div class="job-detail-label">Company / Location</div>
                    <div class="job-detail-value large">${this.escapeHtml(job.location)}</div>
                </div>
            `;
        }
        
        // Experience Level
        content += displayField('Experience Level', job.formatted_experience_level || job.experience_level);
        
        // Description
        if (job.description) {
            content += `
                <div class="job-detail-section">
                    <div class="job-detail-label">Description</div>
                    <div class="job-detail-value">${this.escapeHtml(job.description)}</div>
                </div>
            `;
        }
        
        // Additional fields (if available)
        content += displayField('Work Type', job.work_type);
        content += displayField('Shift', job.shift);
        content += displayField('Salary', job.salary);
        content += displayField('Benefits', job.benefits);
        
        // Job ID
        content += `
            <div class="job-detail-section">
                <div class="job-detail-label">Job ID</div>
                <div class="job-detail-value mono">${job.job_id}</div>
            </div>
        `;
        
        // Debug: Show all available fields
        console.log('Available job fields:', Object.keys(job));
        console.log('Job data:', job);
        
        this.DOM.body.innerHTML = content || '<p style="text-align: center; color: var(--text-muted);">No additional details available</p>';
        
        // Store job ID for apply button
        this.DOM.applyBtn.dataset.jobId = job.job_id;
        
        // Re-init icons
        lucide.createIcons();
    }
    
    /**
     * Show error message
     */
    static showError(message) {
        this.DOM.body.innerHTML = `
            <div class="modal-loading">
                <i data-lucide="alert-circle" style="width: 48px; height: 48px; color: var(--error);"></i>
                <p>${message}</p>
            </div>
        `;
        lucide.createIcons();
    }
    
    /**
     * Escape HTML
     */
    static escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    /**
     * Initialize event listeners
     */
    static init() {
        // Close buttons
        this.DOM.closeBtn.addEventListener('click', () => this.hide());
        this.DOM.closeBtnFooter.addEventListener('click', () => this.hide());
        
        // Overlay click
        this.DOM.overlay.addEventListener('click', () => this.hide());
        
        // ESC key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !this.DOM.modal.classList.contains('hidden')) {
                this.hide();
            }
        });
        
        // Apply button (placeholder)
        this.DOM.applyBtn.addEventListener('click', () => {
            const jobId = this.DOM.applyBtn.dataset.jobId;
            alert(`Apply feature coming soon for Job ID: ${jobId}`);
        });
    }
}

// ================================================================
// STATE SUBSCRIBER
// ================================================================

// Subscribe to state changes to update UI
AppState.subscribe((state) => {
    UI.updateStatusCard(state);
});

// ================================================================
// INITIALIZATION
// ================================================================

const searchController = new SearchController();

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    Modal.init();
    
    // Check URL params for initial search
    const urlParams = new URLSearchParams(window.location.search);
    const query = urlParams.get('q');
    
    if (query) {
        DOM.searchInput.value = query;
        searchController.performSearch(query);
    }
    
    console.log('🚀 JobMatch Search initialized');
});

// ================================================================
// EXPORTS (for testing)
// ================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        AppState,
        APIService,
        SearchController,
        UI,
        LoadingManager
    };
}