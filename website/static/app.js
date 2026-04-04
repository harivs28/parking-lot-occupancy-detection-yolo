/**
 * ParkVision AI — Client-Side Application Logic
 * Handles image upload, detection API calls, and results rendering
 */

// ── DOM Elements ────────────────────────────────────────────────────────────
const dropZone = document.getElementById('drop-zone');
const dropZoneContent = document.getElementById('drop-zone-content');
const fileInput = document.getElementById('file-input');
const previewImage = document.getElementById('preview-image');
const clearPreview = document.getElementById('clear-preview');
const detectBtn = document.getElementById('detect-btn');
const detectBtnText = document.getElementById('detect-btn-text');
const cameraSelect = document.getElementById('camera-select');
const confidenceSlider = document.getElementById('confidence-slider');
const overlapSlider = document.getElementById('overlap-slider');
const areaSlider = document.getElementById('area-slider');
const confidenceValue = document.getElementById('confidence-value');
const overlapValue = document.getElementById('overlap-value');
const areaValue = document.getElementById('area-value');
const resultsSection = document.getElementById('results-section');
const loadingOverlay = document.getElementById('loading-overlay');
const samplesGrid = document.getElementById('samples-grid');
const filterBar = document.getElementById('filter-bar');

// ── State ───────────────────────────────────────────────────────────────────
let selectedFile = null;
let samplesData = [];

// ── Initialize ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    setupUpload();
    setupSliders();
    setupFilters();
    loadSamples();
    setupSmoothScroll();
});

// ── Upload Handling ─────────────────────────────────────────────────────────
function setupUpload() {
    // Click to upload
    dropZone.addEventListener('click', (e) => {
        if (e.target === clearPreview || clearPreview.contains(e.target)) return;
        fileInput.click();
    });

    // File selected via input
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag & drop events
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Clear preview
    clearPreview.addEventListener('click', (e) => {
        e.stopPropagation();
        clearImage();
    });

    // Detect button
    detectBtn.addEventListener('click', runDetection);
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showToast('Please upload an image file (JPG, PNG, BMP)');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.classList.remove('hidden');
        clearPreview.classList.remove('hidden');
        dropZoneContent.classList.add('hidden');
        detectBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearImage() {
    selectedFile = null;
    previewImage.src = '';
    previewImage.classList.add('hidden');
    clearPreview.classList.add('hidden');
    dropZoneContent.classList.remove('hidden');
    detectBtn.disabled = true;
    fileInput.value = '';
}

// ── Slider Controls ─────────────────────────────────────────────────────────
function setupSliders() {
    confidenceSlider.addEventListener('input', () => {
        confidenceValue.textContent = `${confidenceSlider.value}%`;
    });

    overlapSlider.addEventListener('input', () => {
        overlapValue.textContent = `${overlapSlider.value}%`;
    });

    areaSlider.addEventListener('input', () => {
        areaValue.textContent = `${areaSlider.value}x`;
    });
}

// ── Detection ───────────────────────────────────────────────────────────────
async function runDetection() {
    if (!selectedFile) return;

    // Show loading
    loadingOverlay.classList.remove('hidden');
    detectBtn.disabled = true;
    detectBtnText.textContent = 'Processing...';

    try {
        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('camera', cameraSelect.value);
        formData.append('confidence', confidenceSlider.value);
        formData.append('overlap', overlapSlider.value);
        formData.append('area_threshold', areaSlider.value);

        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Detection failed');
        }

        // Display results
        displayResults(data);

    } catch (error) {
        showToast(`Error: ${error.message}`);
        console.error('Detection error:', error);
    } finally {
        loadingOverlay.classList.add('hidden');
        detectBtn.disabled = false;
        detectBtnText.textContent = 'Detect Parking Slots';
    }
}

function displayResults(data) {
    // Show results section
    resultsSection.classList.remove('hidden');

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    // Update stats with animation
    animateCounter('stat-total', data.stats.total);
    animateCounter('stat-empty', data.stats.empty);
    animateCounter('stat-occupied', data.stats.occupied);
    
    const rateEl = document.getElementById('stat-rate');
    rateEl.textContent = `${data.stats.occupancy_rate}%`;
    rateEl.style.animation = 'countUp 0.5s ease';

    // Update occupancy bar
    const fill = document.getElementById('occupancy-fill');
    fill.style.width = `${data.stats.occupancy_rate}%`;

    // Update images
    document.getElementById('result-original').src = data.input_image;
    document.getElementById('result-detected').src = data.output_image;
}

function animateCounter(elementId, target) {
    const el = document.getElementById(elementId);
    const duration = 800;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.round(eased * target);

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// ── Samples ─────────────────────────────────────────────────────────────────
async function loadSamples() {
    try {
        const response = await fetch('/api/samples');
        const data = await response.json();
        samplesData = data.samples;
        renderSamples(samplesData);
    } catch (error) {
        samplesGrid.innerHTML = `
            <div class="samples-loading">
                <p>Could not load sample images</p>
            </div>`;
    }
}

function renderSamples(samples) {
    if (samples.length === 0) {
        samplesGrid.innerHTML = `
            <div class="samples-loading">
                <p>No samples found</p>
            </div>`;
        return;
    }

    samplesGrid.innerHTML = samples.map((sample, idx) => `
        <div class="sample-card" data-weather="${sample.weather}" onclick="useSample('${sample.path}', ${sample.camera})" style="animation: fadeInUp 0.4s ease ${idx * 0.05}s both">
            <div class="sample-image-wrap">
                <img class="sample-image" src="/api/sample/${sample.path}" alt="${sample.filename}" loading="lazy">
            </div>
            <div class="sample-info">
                <div class="sample-meta">
                    <span class="sample-name">Camera ${sample.camera}</span>
                    <span class="sample-detail">${sample.date}</span>
                </div>
                <span class="sample-weather-badge weather-${sample.weather.toLowerCase()}">${sample.weather}</span>
            </div>
        </div>
    `).join('');
}

async function useSample(imagePath, cameraNum) {
    try {
        // Fetch the sample image
        const response = await fetch(`/api/sample/${imagePath}`);
        const blob = await response.blob();
        
        // Create a File object
        const filename = imagePath.split('/').pop();
        const file = new File([blob], filename, { type: 'image/jpeg' });
        
        // Set camera
        cameraSelect.value = cameraNum;

        // Handle as uploaded file
        handleFile(file);

        // Scroll to upload section
        document.getElementById('upload-section').scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        showToast('Failed to load sample image');
    }
}

// ── Filters ─────────────────────────────────────────────────────────────────
function setupFilters() {
    filterBar.addEventListener('click', (e) => {
        const btn = e.target.closest('.filter-btn');
        if (!btn) return;

        // Update active state
        filterBar.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const filter = btn.dataset.filter;
        if (filter === 'all') {
            renderSamples(samplesData);
        } else {
            renderSamples(samplesData.filter(s => s.weather === filter));
        }
    });
}

// ── Smooth Scroll ───────────────────────────────────────────────────────────
function setupSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.querySelector(link.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

// ── Toast Notification ──────────────────────────────────────────────────────
function showToast(message) {
    // Remove existing toast
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 2rem;
        left: 50%;
        transform: translateX(-50%);
        padding: 0.75rem 1.5rem;
        background: rgba(239, 68, 68, 0.9);
        color: white;
        border-radius: 12px;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
        z-index: 2000;
        animation: fadeInUp 0.3s ease;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    `;

    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
