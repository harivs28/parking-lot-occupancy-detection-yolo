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
const analysisModeSelect = document.getElementById('analysis-mode');
const cameraField = document.getElementById('camera-field');
const cameraSelect = document.getElementById('camera-select');
const detectionSizeSelect = document.getElementById('detection-size');
const autoRunToggle = document.getElementById('auto-run-toggle');
const confidenceThresholdInput = document.getElementById('confidence-threshold');
const confidenceValue = document.getElementById('confidence-value');
const emptySensitivityInput = document.getElementById('empty-sensitivity');
const emptySensitivityValue = document.getElementById('empty-sensitivity-value');
const inferEdgeSlotsInput = document.getElementById('infer-edge-slots');

// ── State ───────────────────────────────────────────────────────────────────
const IMAGE_EXTENSION_PATTERN = /\.(bmp|gif|heic|heif|jpe?g|png|tiff?|webp)$/i;
let selectedFile = null;
let currentRequestController = null;
let currentRequestId = 0;

// ── Initialize ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    setupControls();
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
    if (autoRunToggle.checked) {
        uploadStatus.textContent = `${normalizedFile.name} selected from ${source}. Starting automatic analysis...`;
        await runDetection({ autoTriggered: true });
        return;
    }

    uploadStatus.textContent = `${normalizedFile.name} selected from ${source}. Adjust settings if needed, then click Analyze Again.`;
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

function setupControls() {
    analysisModeSelect.addEventListener('change', syncControlState);
    confidenceThresholdInput.addEventListener('input', syncControlState);
    emptySensitivityInput.addEventListener('input', syncControlState);
    syncControlState();
}

function syncControlState() {
    const fixedModeSelected = analysisModeSelect.value === 'fixed';
    cameraField.classList.toggle('hidden', !fixedModeSelected);
    confidenceValue.textContent = Number(confidenceThresholdInput.value).toFixed(2);
    emptySensitivityValue.textContent = describeEmptySensitivity(emptySensitivityInput.value);
}

function describeEmptySensitivity(rawValue) {
    const value = Number(rawValue);
    if (value <= 35) {
        return 'Conservative';
    }
    if (value <= 65) {
        return 'Balanced';
    }
    return 'Aggressive';
}

function collectSettings() {
    return {
        analysisMode: analysisModeSelect.value,
        camera: analysisModeSelect.value === 'fixed' ? cameraSelect.value : 'auto',
        detectionSize: detectionSizeSelect.value,
        confidenceThreshold: Number(confidenceThresholdInput.value).toFixed(2),
        emptySensitivity: emptySensitivityInput.value,
        inferEdgeSlots: inferEdgeSlotsInput.checked
    };
}

function summarizeSettings(settings) {
    const modeLabelMap = {
        auto: 'Auto',
        fixed: `Fixed Camera ${settings.camera}`,
        generic: 'Generic estimate'
    };

    return `${modeLabelMap[settings.analysisMode]} • ${detectionSizeSelect.selectedOptions[0].text}`;
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
    const settings = collectSettings();
    uploadStatus.textContent = autoTriggered
        ? `Analyzing ${selectedFile.name} automatically with ${summarizeSettings(settings)}...`
        : `Re-running analysis for ${selectedFile.name} with ${summarizeSettings(settings)}...`;

    try {
        const formData = new FormData();
        formData.append('image', selectedFile, selectedFile.name);
        formData.append('camera', settings.camera);
        formData.append('analysis_mode', settings.analysisMode);
        formData.append('detection_size', settings.detectionSize);
        formData.append('confidence_threshold', settings.confidenceThreshold);
        formData.append('empty_sensitivity', settings.emptySensitivity);
        formData.append('infer_edge_slots', settings.inferEdgeSlots ? '1' : '0');

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
    const slotMode = data.stats.slot_mode || (data.stats.layout_supported ? 'fixed' : 'vehicle_only');
    const hasSlotEstimate = slotMode !== 'vehicle_only';
    resultsSection.classList.remove('hidden');

    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    if (slotMode === 'estimated') {
        setStatLabels('Estimated Slots', 'Estimated Empty', 'Estimated Occupied', 'Estimated Rate');
        animateCounter('stat-total', data.stats.total);
        animateCounter('stat-empty', data.stats.empty);
        animateCounter('stat-occupied', data.stats.occupied);
        setStatValue('stat-rate', `${data.stats.occupancy_rate}%`);
        occupancyBarContainer.classList.remove('hidden');
        document.getElementById('occupancy-fill').style.width = `${data.stats.occupancy_rate}%`;
    } else if (hasSlotEstimate) {
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
    document.getElementById('result-mode').textContent = `Mode: ${data.stats.result_mode_label || 'Waiting'}`;
    document.getElementById('result-detail').textContent = `Detail: ${data.stats.detection_profile_label || 'Auto'}`;
    document.getElementById('result-sensitivity').textContent = `Empty fill: ${data.stats.empty_sensitivity_label || 'Balanced'}`;
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
