/**
 * ParkVision AI — Client-Side Application Logic
 * Upload, paste, auto-detect, and render parking occupancy results.
 */

// ── DOM Elements ────────────────────────────────────────────────────────────
const dropZone = document.getElementById('drop-zone');
const dropZoneContent = document.getElementById('drop-zone-content');
const fileInput = document.getElementById('file-input');
const previewImage = document.getElementById('preview-image');
const clearPreview = document.getElementById('clear-preview');
const detectBtn = document.getElementById('detect-btn');
const detectBtnText = document.getElementById('detect-btn-text');
const pasteBtn = document.getElementById('paste-btn');
const heroPasteBtn = document.getElementById('hero-paste-btn');
const uploadStatus = document.getElementById('upload-status');
const resultsSection = document.getElementById('results-section');
const loadingOverlay = document.getElementById('loading-overlay');
const occupancyBarContainer = document.getElementById('occupancy-bar-container');
const resultNotice = document.getElementById('result-notice');

// ── State ───────────────────────────────────────────────────────────────────
const IMAGE_EXTENSION_PATTERN = /\.(bmp|gif|heic|heif|jpe?g|png|tiff?|webp)$/i;
let selectedFile = null;
let currentRequestController = null;
let currentRequestId = 0;

// ── Initialize ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    setupUpload();
    setupClipboard();
    setupSmoothScroll();
});

// ── Upload Handling ─────────────────────────────────────────────────────────
function setupUpload() {
    dropZone.addEventListener('click', (event) => {
        if (event.target === clearPreview || clearPreview.contains(event.target)) {
            return;
        }

        fileInput.click();
    });

    fileInput.addEventListener('change', async (event) => {
        if (event.target.files.length > 0) {
            await handleFile(event.target.files[0], 'browser');
        }
    });

    dropZone.addEventListener('dragover', (event) => {
        event.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', async (event) => {
        event.preventDefault();
        dropZone.classList.remove('drag-over');

        if (event.dataTransfer.files.length > 0) {
            await handleFile(event.dataTransfer.files[0], 'drop');
        }
    });

    clearPreview.addEventListener('click', (event) => {
        event.stopPropagation();
        clearImage();
    });

    detectBtn.addEventListener('click', () => {
        runDetection({ autoTriggered: false });
    });
}

function setupClipboard() {
    pasteBtn.addEventListener('click', readClipboardImage);
    heroPasteBtn.addEventListener('click', async () => {
        document.getElementById('upload-section').scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
        await readClipboardImage();
    });

    document.addEventListener('paste', async (event) => {
        const pastedImage = extractImageFromClipboard(event.clipboardData);
        if (!pastedImage) {
            return;
        }

        event.preventDefault();
        document.getElementById('upload-section').scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
        await handleFile(pastedImage, 'paste');
    });
}

async function handleFile(fileLike, source) {
    const normalizedFile = normalizeImageFile(fileLike, source);
    if (!isSupportedImageFile(normalizedFile)) {
        showToast('Please upload or paste a supported parking image file.');
        return;
    }

    selectedFile = normalizedFile;
    detectBtn.disabled = false;

    await showPreview(normalizedFile);
    uploadStatus.textContent = `${normalizedFile.name} selected from ${source}. Starting automatic analysis...`;
    await runDetection({ autoTriggered: true });
}

function normalizeImageFile(fileLike, source) {
    const mimeType = fileLike.type || 'image/png';
    const extension = getPreferredExtension(fileLike);
    const fallbackName = `parking-upload-${source}${extension}`;
    const filename = (fileLike.name && fileLike.name.trim()) || fallbackName;

    return new File([fileLike], filename, {
        type: mimeType,
        lastModified: fileLike.lastModified || Date.now()
    });
}

function getPreferredExtension(fileLike) {
    const mimeMap = {
        'image/bmp': '.bmp',
        'image/gif': '.gif',
        'image/heic': '.heic',
        'image/heif': '.heif',
        'image/jpeg': '.jpg',
        'image/jpg': '.jpg',
        'image/png': '.png',
        'image/tiff': '.tiff',
        'image/webp': '.webp'
    };

    if (fileLike.name && IMAGE_EXTENSION_PATTERN.test(fileLike.name)) {
        const match = fileLike.name.match(/\.[^.]+$/);
        if (match) {
            return match[0];
        }
    }

    return mimeMap[fileLike.type] || '.png';
}

function isSupportedImageFile(file) {
    if (file.type && file.type.startsWith('image/')) {
        return true;
    }

    return Boolean(file.name && IMAGE_EXTENSION_PATTERN.test(file.name));
}

function showPreview(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            previewImage.src = event.target.result;
            previewImage.classList.remove('hidden');
            clearPreview.classList.remove('hidden');
            dropZoneContent.classList.add('hidden');
            resolve();
        };
        reader.onerror = () => reject(new Error('Could not preview the selected image.'));
        reader.readAsDataURL(file);
    });
}

function clearImage() {
    if (currentRequestController) {
        currentRequestController.abort();
    }

    selectedFile = null;
    previewImage.src = '';
    previewImage.classList.add('hidden');
    clearPreview.classList.add('hidden');
    dropZoneContent.classList.remove('hidden');
    detectBtn.disabled = true;
    fileInput.value = '';
    uploadStatus.textContent = 'No image selected yet.';
}

function extractImageFromClipboard(clipboardData) {
    if (!clipboardData || !clipboardData.items) {
        return null;
    }

    for (const item of clipboardData.items) {
        if (item.type.startsWith('image/')) {
            return item.getAsFile();
        }
    }

    return null;
}

async function readClipboardImage() {
    if (!navigator.clipboard || !navigator.clipboard.read) {
        showToast('Clipboard read is unavailable here. Press Ctrl+V or Cmd+V to paste an image.');
        return;
    }

    try {
        const clipboardItems = await navigator.clipboard.read();
        for (const clipboardItem of clipboardItems) {
            const imageType = clipboardItem.types.find((type) => type.startsWith('image/'));
            if (!imageType) {
                continue;
            }

            const imageBlob = await clipboardItem.getType(imageType);
            await handleFile(imageBlob, 'clipboard');
            return;
        }

        showToast('Clipboard does not contain an image.');
    } catch (error) {
        showToast('Clipboard access was denied. You can still press Ctrl+V or Cmd+V on the page.');
    }
}

// ── Detection ───────────────────────────────────────────────────────────────
async function runDetection({ autoTriggered }) {
    if (!selectedFile) {
        return;
    }

    if (currentRequestController) {
        currentRequestController.abort();
    }

    const requestId = ++currentRequestId;
    currentRequestController = new AbortController();
    setLoadingState(true);
    uploadStatus.textContent = autoTriggered
        ? `Analyzing ${selectedFile.name} automatically...`
        : `Re-running analysis for ${selectedFile.name}...`;

    try {
        const formData = new FormData();
        formData.append('image', selectedFile, selectedFile.name);
        formData.append('camera', 'auto');

        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData,
            signal: currentRequestController.signal
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Detection failed.');
        }

        if (requestId !== currentRequestId) {
            return;
        }

        displayResults(data);
        uploadStatus.textContent = `${selectedFile.name} analyzed successfully.`;
    } catch (error) {
        if (error.name === 'AbortError') {
            return;
        }

        console.error('Detection error:', error);
        uploadStatus.textContent = 'Analysis failed. Please try another parking-lot image.';
        showToast(error.message || 'Detection failed.');
    } finally {
        if (requestId === currentRequestId) {
            currentRequestController = null;
            setLoadingState(false);
        }
    }
}

function setLoadingState(isLoading) {
    loadingOverlay.classList.toggle('hidden', !isLoading);
    detectBtn.disabled = isLoading || !selectedFile;
    detectBtnText.textContent = isLoading ? 'Analyzing...' : 'Analyze Again';
}

function displayResults(data) {
    const layoutSupported = Boolean(data.stats.layout_supported);
    resultsSection.classList.remove('hidden');

    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    if (layoutSupported) {
        setStatLabels('Total Slots', 'Empty Slots', 'Occupied Slots', 'Occupancy Rate');
        animateCounter('stat-total', data.stats.total);
        animateCounter('stat-empty', data.stats.empty);
        animateCounter('stat-occupied', data.stats.occupied);
        setStatValue('stat-rate', `${data.stats.occupancy_rate}%`);
        occupancyBarContainer.classList.remove('hidden');
        document.getElementById('occupancy-fill').style.width = `${data.stats.occupancy_rate}%`;
    } else {
        setStatLabels('Detected Vehicles', 'Empty Slots', 'Occupied Slots', 'Occupancy');
        setStatValue('stat-total', String(data.stats.detections));
        setStatValue('stat-empty', '--');
        setStatValue('stat-occupied', '--');
        setStatValue('stat-rate', 'N/A');
        occupancyBarContainer.classList.add('hidden');
        document.getElementById('occupancy-fill').style.width = '0%';
    }

    document.getElementById('result-original').src = data.input_image;
    document.getElementById('result-detected').src = data.output_image;
    document.getElementById('result-camera').textContent = `Profile: ${data.stats.camera_label}`;
    document.getElementById('result-mode').textContent = layoutSupported
        ? (data.stats.auto_profile ? 'Mode: Auto matched' : 'Mode: Manual')
        : 'Mode: Vehicle only';
    document.getElementById('result-detections').textContent = `Vehicles: ${data.stats.detections}`;
    document.getElementById('result-format').textContent = `Format: ${data.stats.input_format}`;

    if (data.stats.warning_message) {
        resultNotice.textContent = data.stats.warning_message;
        resultNotice.classList.remove('hidden');
    } else {
        resultNotice.textContent = '';
        resultNotice.classList.add('hidden');
    }
}

function animateCounter(elementId, target) {
    const element = document.getElementById(elementId);
    const duration = 800;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        element.textContent = Math.round(eased * target);

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

function setStatLabels(total, empty, occupied, rate) {
    document.getElementById('stat-total-label').textContent = total;
    document.getElementById('stat-empty-label').textContent = empty;
    document.getElementById('stat-occupied-label').textContent = occupied;
    document.getElementById('stat-rate-label').textContent = rate;
}

function setStatValue(elementId, value) {
    const element = document.getElementById(elementId);
    element.textContent = value;
    element.style.animation = 'countUp 0.5s ease';
}

// ── Smooth Scroll ───────────────────────────────────────────────────────────
function setupSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach((link) => {
        link.addEventListener('click', (event) => {
            event.preventDefault();
            const target = document.querySelector(link.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

// ── Toast Notification ──────────────────────────────────────────────────────
function showToast(message) {
    const existing = document.querySelector('.toast');
    if (existing) {
        existing.remove();
    }

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 2rem;
        left: 50%;
        transform: translateX(-50%);
        padding: 0.85rem 1.35rem;
        background: rgba(239, 68, 68, 0.92);
        color: white;
        border-radius: 12px;
        font-family: 'Inter', sans-serif;
        font-size: 0.92rem;
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
