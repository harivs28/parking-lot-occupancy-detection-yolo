/**
 * ParkVision AI client logic.
 * Handles image uploads and backend-managed live camera sessions.
 */

const IMAGE_EXTENSION_PATTERN = /\.(bmp|gif|heic|heif|jpe?g|png|tiff?|webp)$/i;
const LIVE_POLL_INTERVAL_MS = 2000;

const imageControls = {
    analysisModeSelect: document.getElementById('analysis-mode'),
    cameraField: document.getElementById('camera-field'),
    cameraSelect: document.getElementById('camera-select'),
    detectionSizeSelect: document.getElementById('detection-size'),
    confidenceThresholdInput: document.getElementById('confidence-threshold'),
    confidenceValue: document.getElementById('confidence-value'),
    emptySensitivityInput: document.getElementById('empty-sensitivity'),
    emptySensitivityValue: document.getElementById('empty-sensitivity-value'),
    inferEdgeSlotsInput: document.getElementById('infer-edge-slots')
};

const liveControls = {
    analysisModeSelect: document.getElementById('live-analysis-mode'),
    cameraField: document.getElementById('live-camera-field'),
    cameraSelect: document.getElementById('live-camera-select'),
    detectionSizeSelect: document.getElementById('live-detection-size'),
    confidenceThresholdInput: document.getElementById('live-confidence-threshold'),
    confidenceValue: document.getElementById('live-confidence-value'),
    emptySensitivityInput: document.getElementById('live-empty-sensitivity'),
    emptySensitivityValue: document.getElementById('live-empty-sensitivity-value'),
    inferEdgeSlotsInput: document.getElementById('live-infer-edge-slots')
};

const dom = {
    uploadSection: document.getElementById('upload-section'),
    resultsSection: document.getElementById('results-section'),
    liveSection: document.getElementById('live-section'),
    tabButtons: Array.from(document.querySelectorAll('.mode-tab')),
    dropZone: document.getElementById('drop-zone'),
    dropZoneContent: document.getElementById('drop-zone-content'),
    fileInput: document.getElementById('file-input'),
    previewImage: document.getElementById('preview-image'),
    clearPreview: document.getElementById('clear-preview'),
    detectBtn: document.getElementById('detect-btn'),
    detectBtnText: document.getElementById('detect-btn-text'),
    pasteBtn: document.getElementById('paste-btn'),
    heroPasteBtn: document.getElementById('hero-paste-btn'),
    uploadStatus: document.getElementById('upload-status'),
    loadingOverlay: document.getElementById('loading-overlay'),
    occupancyBarContainer: document.getElementById('occupancy-bar-container'),
    occupancyFill: document.getElementById('occupancy-fill'),
    resultNotice: document.getElementById('result-notice'),
    autoRunToggle: document.getElementById('auto-run-toggle'),
    resultOriginal: document.getElementById('result-original'),
    resultDetected: document.getElementById('result-detected'),
    resultCamera: document.getElementById('result-camera'),
    resultMode: document.getElementById('result-mode'),
    resultDetail: document.getElementById('result-detail'),
    resultSensitivity: document.getElementById('result-sensitivity'),
    resultDetections: document.getElementById('result-detections'),
    resultFormat: document.getElementById('result-format'),
    liveCameraUrl: document.getElementById('live-camera-url'),
    liveStartBtn: document.getElementById('live-start-btn'),
    liveStartBtnText: document.getElementById('live-start-btn-text'),
    liveStopBtn: document.getElementById('live-stop-btn'),
    liveConnectionStatus: document.getElementById('live-connection-status'),
    liveCameraLock: document.getElementById('live-camera-lock'),
    liveStartedAt: document.getElementById('live-started-at'),
    liveLastCapture: document.getElementById('live-last-capture'),
    liveNextCapture: document.getElementById('live-next-capture'),
    liveSessionStatus: document.getElementById('live-session-status'),
    liveResultNotice: document.getElementById('live-result-notice'),
    liveResultOriginal: document.getElementById('live-result-original'),
    liveOriginalPlaceholder: document.getElementById('live-original-placeholder'),
    liveResultDetected: document.getElementById('live-result-detected'),
    liveDetectedPlaceholder: document.getElementById('live-detected-placeholder'),
    liveStatTotal: document.getElementById('live-stat-total'),
    liveStatEmpty: document.getElementById('live-stat-empty'),
    liveStatOccupied: document.getElementById('live-stat-occupied'),
    liveStatRate: document.getElementById('live-stat-rate'),
    liveResultCamera: document.getElementById('live-result-camera'),
    liveResultMode: document.getElementById('live-result-mode'),
    liveResultDetail: document.getElementById('live-result-detail'),
    liveResultSensitivity: document.getElementById('live-result-sensitivity'),
    liveResultDetections: document.getElementById('live-result-detections'),
    liveResultFormat: document.getElementById('live-result-format'),
    liveCaptureInterval: document.getElementById('live-capture-interval')
};

const imageState = {
    selectedFile: null,
    currentRequestController: null,
    currentRequestId: 0
};

const liveState = {
    activeTab: 'image',
    sessionId: null,
    lastCaptureAt: null,
    pollTimer: null,
    pollInFlight: false,
    stopping: false,
    settings: null
};

document.addEventListener('DOMContentLoaded', () => {
    initializePanels();
    setupSettingsControls(imageControls);
    setupSettingsControls(liveControls);
    setupTabs();
    setupUpload();
    setupClipboard();
    setupLiveControls();
    setupSmoothScroll();
    resetLiveResults();
});

function initializePanels() {
    dom.liveSection.classList.remove('hidden');
    dom.liveSection.classList.add('tab-hidden');
}

function setupSettingsControls(controls) {
    controls.analysisModeSelect.addEventListener('change', () => syncSettingsControlState(controls));
    controls.confidenceThresholdInput.addEventListener('input', () => syncSettingsControlState(controls));
    controls.emptySensitivityInput.addEventListener('input', () => syncSettingsControlState(controls));
    syncSettingsControlState(controls);
}

function syncSettingsControlState(controls) {
    const fixedModeSelected = controls.analysisModeSelect.value === 'fixed';
    controls.cameraField.classList.toggle('hidden', !fixedModeSelected);
    controls.confidenceValue.textContent = Number(controls.confidenceThresholdInput.value).toFixed(2);
    controls.emptySensitivityValue.textContent = describeEmptySensitivity(controls.emptySensitivityInput.value);
}

function collectSettings(controls) {
    return {
        analysisMode: controls.analysisModeSelect.value,
        camera: controls.analysisModeSelect.value === 'fixed' ? controls.cameraSelect.value : 'auto',
        detectionSize: controls.detectionSizeSelect.value,
        confidenceThreshold: Number(controls.confidenceThresholdInput.value).toFixed(2),
        emptySensitivity: controls.emptySensitivityInput.value,
        inferEdgeSlots: controls.inferEdgeSlotsInput.checked
    };
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

function setupTabs() {
    dom.tabButtons.forEach((button) => {
        button.addEventListener('click', async () => {
            const targetTab = button.dataset.tabTarget;
            await switchTab(targetTab);
        });
    });
}

async function switchTab(targetTab) {
    if (targetTab === liveState.activeTab) {
        return;
    }

    if (liveState.activeTab === 'live' && liveState.sessionId) {
        await stopLiveSession({
            reason: 'Live session stopped because you left the Live 5G Camera tab.',
            showToastMessage: false
        });
    }

    liveState.activeTab = targetTab;
    dom.tabButtons.forEach((button) => {
        const isActive = button.dataset.tabTarget === targetTab;
        button.classList.toggle('active', isActive);
        button.setAttribute('aria-selected', String(isActive));
    });

    dom.uploadSection.classList.toggle('tab-hidden', targetTab !== 'image');
    dom.resultsSection.classList.toggle('tab-hidden', targetTab !== 'image');
    dom.liveSection.classList.toggle('tab-hidden', targetTab !== 'live');
}

function setupUpload() {
    dom.dropZone.addEventListener('click', (event) => {
        if (event.target === dom.clearPreview || dom.clearPreview.contains(event.target)) {
            return;
        }

        dom.fileInput.click();
    });

    dom.fileInput.addEventListener('change', async (event) => {
        if (event.target.files.length > 0) {
            await handleFile(event.target.files[0], 'browser');
        }
    });

    dom.dropZone.addEventListener('dragover', (event) => {
        event.preventDefault();
        dom.dropZone.classList.add('drag-over');
    });

    dom.dropZone.addEventListener('dragleave', () => {
        dom.dropZone.classList.remove('drag-over');
    });

    dom.dropZone.addEventListener('drop', async (event) => {
        event.preventDefault();
        dom.dropZone.classList.remove('drag-over');

        if (event.dataTransfer.files.length > 0) {
            await handleFile(event.dataTransfer.files[0], 'drop');
        }
    });

    dom.clearPreview.addEventListener('click', (event) => {
        event.stopPropagation();
        clearImage();
    });

    dom.detectBtn.addEventListener('click', () => {
        void runDetection({ autoTriggered: false });
    });
}

function setupClipboard() {
    dom.pasteBtn.addEventListener('click', () => {
        void readClipboardImage();
    });

    dom.heroPasteBtn.addEventListener('click', async () => {
        dom.uploadSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        await readClipboardImage();
    });

    document.addEventListener('paste', async (event) => {
        const pastedImage = extractImageFromClipboard(event.clipboardData);
        if (!pastedImage) {
            return;
        }

        event.preventDefault();
        dom.uploadSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        await handleFile(pastedImage, 'paste');
    });
}

async function handleFile(fileLike, source) {
    const normalizedFile = normalizeImageFile(fileLike, source);
    if (!isSupportedImageFile(normalizedFile)) {
        showToast('Please upload or paste a supported parking image file.');
        return;
    }

    imageState.selectedFile = normalizedFile;
    dom.detectBtn.disabled = false;

    await showPreview(normalizedFile);
    if (dom.autoRunToggle.checked) {
        dom.uploadStatus.textContent = `${normalizedFile.name} selected from ${source}. Starting automatic analysis...`;
        await runDetection({ autoTriggered: true });
        return;
    }

    dom.uploadStatus.textContent = `${normalizedFile.name} selected from ${source}. Adjust settings if needed, then click Analyze Again.`;
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
            dom.previewImage.src = event.target.result;
            dom.previewImage.classList.remove('hidden');
            dom.clearPreview.classList.remove('hidden');
            dom.dropZoneContent.classList.add('hidden');
            resolve();
        };
        reader.onerror = () => reject(new Error('Could not preview the selected image.'));
        reader.readAsDataURL(file);
    });
}

function clearImage() {
    if (imageState.currentRequestController) {
        imageState.currentRequestController.abort();
    }

    imageState.selectedFile = null;
    dom.previewImage.src = '';
    dom.previewImage.classList.add('hidden');
    dom.clearPreview.classList.add('hidden');
    dom.dropZoneContent.classList.remove('hidden');
    dom.detectBtn.disabled = true;
    dom.fileInput.value = '';
    dom.uploadStatus.textContent = 'No image selected yet.';
}

function summarizeImageSettings(settings) {
    const modeLabelMap = {
        auto: 'Auto',
        fixed: `Fixed Camera ${settings.camera}`,
        generic: 'Generic estimate'
    };

    return `${modeLabelMap[settings.analysisMode]} • ${imageControls.detectionSizeSelect.selectedOptions[0].text}`;
}

async function runDetection({ autoTriggered }) {
    if (!imageState.selectedFile) {
        return;
    }

    if (imageState.currentRequestController) {
        imageState.currentRequestController.abort();
    }

    const requestId = ++imageState.currentRequestId;
    imageState.currentRequestController = new AbortController();
    setLoadingState(true);
    const settings = collectSettings(imageControls);
    dom.uploadStatus.textContent = autoTriggered
        ? `Analyzing ${imageState.selectedFile.name} automatically with ${summarizeImageSettings(settings)}...`
        : `Re-running analysis for ${imageState.selectedFile.name} with ${summarizeImageSettings(settings)}...`;

    try {
        const formData = new FormData();
        formData.append('image', imageState.selectedFile, imageState.selectedFile.name);
        formData.append('camera', settings.camera);
        formData.append('analysis_mode', settings.analysisMode);
        formData.append('detection_size', settings.detectionSize);
        formData.append('confidence_threshold', settings.confidenceThreshold);
        formData.append('empty_sensitivity', settings.emptySensitivity);
        formData.append('infer_edge_slots', settings.inferEdgeSlots ? '1' : '0');

        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData,
            signal: imageState.currentRequestController.signal
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Detection failed.');
        }

        if (requestId !== imageState.currentRequestId) {
            return;
        }

        displayImageResults(data);
        dom.uploadStatus.textContent = `${imageState.selectedFile.name} analyzed successfully.`;
    } catch (error) {
        if (error.name === 'AbortError') {
            return;
        }

        console.error('Detection error:', error);
        dom.uploadStatus.textContent = 'Analysis failed. Please try another parking-lot image.';
        showToast(error.message || 'Detection failed.');
    } finally {
        if (requestId === imageState.currentRequestId) {
            imageState.currentRequestController = null;
            setLoadingState(false);
        }
    }
}

function setLoadingState(isLoading) {
    dom.loadingOverlay.classList.toggle('hidden', !isLoading);
    dom.detectBtn.disabled = isLoading || !imageState.selectedFile;
    dom.detectBtnText.textContent = isLoading ? 'Analyzing...' : 'Analyze Again';
}

function displayImageResults(data) {
    const slotMode = data.stats.slot_mode || (data.stats.layout_supported ? 'fixed' : 'vehicle_only');
    const hasSlotEstimate = slotMode !== 'vehicle_only';
    dom.resultsSection.classList.remove('hidden');

    setTimeout(() => {
        dom.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    if (slotMode === 'estimated') {
        setImageStatLabels('Estimated Slots', 'Estimated Empty', 'Estimated Occupied', 'Estimated Rate');
        animateCounter('stat-total', data.stats.total);
        animateCounter('stat-empty', data.stats.empty);
        animateCounter('stat-occupied', data.stats.occupied);
        setStatValue('stat-rate', `${data.stats.occupancy_rate}%`);
        dom.occupancyBarContainer.classList.remove('hidden');
        dom.occupancyFill.style.width = `${data.stats.occupancy_rate}%`;
    } else if (hasSlotEstimate) {
        setImageStatLabels('Total Slots', 'Empty Slots', 'Occupied Slots', 'Occupancy Rate');
        animateCounter('stat-total', data.stats.total);
        animateCounter('stat-empty', data.stats.empty);
        animateCounter('stat-occupied', data.stats.occupied);
        setStatValue('stat-rate', `${data.stats.occupancy_rate}%`);
        dom.occupancyBarContainer.classList.remove('hidden');
        dom.occupancyFill.style.width = `${data.stats.occupancy_rate}%`;
    } else {
        setImageStatLabels('Detected Vehicles', 'Empty Slots', 'Occupied Slots', 'Occupancy');
        setStatValue('stat-total', String(data.stats.detections));
        setStatValue('stat-empty', '--');
        setStatValue('stat-occupied', '--');
        setStatValue('stat-rate', 'N/A');
        dom.occupancyBarContainer.classList.add('hidden');
        dom.occupancyFill.style.width = '0%';
    }

    dom.resultOriginal.src = data.input_image;
    dom.resultDetected.src = data.output_image;
    dom.resultCamera.textContent = `Profile: ${data.stats.camera_label}`;
    dom.resultMode.textContent = `Mode: ${data.stats.result_mode_label || 'Waiting'}`;
    dom.resultDetail.textContent = `Detail: ${data.stats.detection_profile_label || 'Auto'}`;
    dom.resultSensitivity.textContent = `Empty fill: ${data.stats.empty_sensitivity_label || 'Balanced'}`;
    dom.resultDetections.textContent = `Vehicles: ${data.stats.detections}`;
    dom.resultFormat.textContent = `Format: ${data.stats.input_format}`;

    if (data.stats.warning_message) {
        dom.resultNotice.textContent = data.stats.warning_message;
        dom.resultNotice.classList.remove('hidden');
    } else {
        dom.resultNotice.textContent = '';
        dom.resultNotice.classList.add('hidden');
    }
}

function setImageStatLabels(total, empty, occupied, rate) {
    document.getElementById('stat-total-label').textContent = total;
    document.getElementById('stat-empty-label').textContent = empty;
    document.getElementById('stat-occupied-label').textContent = occupied;
    document.getElementById('stat-rate-label').textContent = rate;
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

function setStatValue(elementId, value) {
    const element = document.getElementById(elementId);
    element.textContent = value;
    element.style.animation = 'none';
    void element.offsetWidth;
    element.style.animation = 'countUp 0.5s ease';
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

function setupLiveControls() {
    dom.liveStartBtn.addEventListener('click', () => {
        void startLiveSession();
    });

    dom.liveStopBtn.addEventListener('click', () => {
        void stopLiveSession({
            reason: 'Live session stopped.',
            showToastMessage: false
        });
    });

    window.addEventListener('pagehide', () => {
        if (!liveState.sessionId) {
            return;
        }

        const sessionId = liveState.sessionId;
        stopLivePolling();
        liveState.sessionId = null;
        fetch(`/api/realtime/session/${sessionId}`, {
            method: 'DELETE',
            keepalive: true
        }).catch(() => {});
    });
}

async function startLiveSession() {
    const cameraUrl = dom.liveCameraUrl.value.trim();
    if (!cameraUrl) {
        showToast('Enter an RTSP or HTTP camera URL before starting the live session.');
        return;
    }

    if (liveState.sessionId) {
        await stopLiveSession({
            reason: '',
            showToastMessage: false
        });
    }

    const settings = collectSettings(liveControls);
    liveState.settings = settings;
    liveState.lastCaptureAt = null;
    resetLiveResults();
    setLiveRunningState(true);
    dom.liveSessionStatus.textContent = 'Starting live analysis session...';
    dom.liveConnectionStatus.textContent = 'Status: Starting';
    dom.liveConnectionStatus.className = 'result-pill live-status-pill status-starting';

    try {
        const response = await fetch('/api/realtime/session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                camera_url: cameraUrl,
                analysis_mode: settings.analysisMode,
                camera: settings.camera,
                detection_size: settings.detectionSize,
                confidence_threshold: settings.confidenceThreshold,
                empty_sensitivity: settings.emptySensitivity,
                infer_edge_slots: settings.inferEdgeSlots
            })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Could not start the live session.');
        }

        liveState.sessionId = data.session_id;
        dom.liveCaptureInterval.textContent = `${data.capture_interval_seconds} seconds`;
        dom.liveSessionStatus.textContent = 'Live session started. Waiting for the first decodable frame.';
        startLivePolling();
        await pollLiveSession();
    } catch (error) {
        console.error('Live session start error:', error);
        setLiveRunningState(false);
        dom.liveSessionStatus.textContent = 'Unable to start live analysis.';
        dom.liveConnectionStatus.textContent = 'Status: Idle';
        dom.liveConnectionStatus.className = 'result-pill live-status-pill status-idle';
        showToast(error.message || 'Could not start the live session.');
    }
}

function startLivePolling() {
    stopLivePolling();
    liveState.pollTimer = window.setInterval(() => {
        void pollLiveSession();
    }, LIVE_POLL_INTERVAL_MS);
}

function stopLivePolling() {
    if (liveState.pollTimer) {
        window.clearInterval(liveState.pollTimer);
        liveState.pollTimer = null;
    }
}

async function pollLiveSession() {
    if (!liveState.sessionId || liveState.pollInFlight || liveState.stopping) {
        return;
    }

    liveState.pollInFlight = true;
    try {
        const response = await fetch(`/api/realtime/session/${liveState.sessionId}`);
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Could not read the live session.');
        }

        updateLiveStatus(data);
        if (data.last_capture_at && data.last_capture_at !== liveState.lastCaptureAt) {
            liveState.lastCaptureAt = data.last_capture_at;
            displayLiveCapture(data);
        }
    } catch (error) {
        console.error('Live session poll error:', error);
        dom.liveSessionStatus.textContent = 'Live polling failed. The app will retry automatically.';
        updateLiveNotice(error.message || 'Live session polling failed.');
    } finally {
        liveState.pollInFlight = false;
    }
}

function updateLiveStatus(data) {
    dom.liveConnectionStatus.textContent = `Status: ${describeLiveStatus(data.status)}`;
    dom.liveConnectionStatus.className = `result-pill live-status-pill status-${normalizeLiveStatusClass(data.status)}`;
    dom.liveStartedAt.textContent = formatTimestamp(data.started_at, 'Not started');
    dom.liveLastCapture.textContent = formatTimestamp(data.last_capture_at, 'Waiting for first frame');
    dom.liveNextCapture.textContent = formatTimestamp(
        data.next_capture_at,
        'Starts after the first successful frame read'
    );

    if (data.camera_locked && data.locked_camera) {
        dom.liveCameraLock.textContent = `Camera lock: Camera ${data.locked_camera} locked`;
    } else if (liveState.settings && liveState.settings.analysisMode === 'fixed') {
        dom.liveCameraLock.textContent = `Camera lock: Manual Camera ${liveState.settings.camera}`;
    } else {
        dom.liveCameraLock.textContent = 'Camera lock: Auto-detect';
    }

    dom.liveSessionStatus.textContent = describeLiveSessionMessage(data);
    updateLiveNotice(data.error || (data.stats ? data.stats.warning_message : ''));
}

function displayLiveCapture(data) {
    const stats = data.stats || {};

    dom.liveStatTotal.textContent = String(stats.total ?? 0);
    dom.liveStatEmpty.textContent = String(stats.empty ?? 0);
    dom.liveStatOccupied.textContent = String(stats.occupied ?? 0);
    dom.liveStatRate.textContent = `${stats.occupancy_rate ?? 0}%`;

    dom.liveResultCamera.textContent = `Profile: ${stats.camera_label || 'Waiting'}`;
    dom.liveResultMode.textContent = `Mode: ${stats.result_mode_label || 'Waiting'}`;
    dom.liveResultDetail.textContent = `Detail: ${stats.detection_profile_label || 'Auto'}`;
    dom.liveResultSensitivity.textContent = `Empty fill: ${stats.empty_sensitivity_label || 'Conservative'}`;
    dom.liveResultDetections.textContent = `Vehicles: ${stats.detections ?? 0}`;
    dom.liveResultFormat.textContent = 'Source: Stream';

    setComparisonImage(dom.liveResultOriginal, dom.liveOriginalPlaceholder, data.input_image);
    setComparisonImage(dom.liveResultDetected, dom.liveDetectedPlaceholder, data.output_image);
    updateLiveNotice(data.error || stats.warning_message || '');
}

function resetLiveResults() {
    dom.liveStatTotal.textContent = '0';
    dom.liveStatEmpty.textContent = '0';
    dom.liveStatOccupied.textContent = '0';
    dom.liveStatRate.textContent = '0%';
    dom.liveResultCamera.textContent = 'Profile: Waiting';
    dom.liveResultMode.textContent = 'Mode: Waiting';
    dom.liveResultDetail.textContent = 'Detail: Auto';
    dom.liveResultSensitivity.textContent = 'Empty fill: Conservative';
    dom.liveResultDetections.textContent = 'Vehicles: 0';
    dom.liveResultFormat.textContent = 'Source: Stream';
    dom.liveStartedAt.textContent = 'Not started';
    dom.liveLastCapture.textContent = 'Waiting for first frame';
    dom.liveNextCapture.textContent = 'Starts after the first successful frame read';
    dom.liveCameraLock.textContent = 'Camera lock: Auto-detect';
    dom.liveSessionStatus.textContent = 'No live stream session running.';
    dom.liveConnectionStatus.textContent = 'Status: Idle';
    dom.liveConnectionStatus.className = 'result-pill live-status-pill status-idle';
    updateLiveNotice('');
    clearComparisonImage(dom.liveResultOriginal, dom.liveOriginalPlaceholder, 'Waiting for the first captured frame.');
    clearComparisonImage(dom.liveResultDetected, dom.liveDetectedPlaceholder, 'The annotated live result will appear here after analysis.');
}

function setComparisonImage(imageElement, placeholderElement, src) {
    if (!src) {
        clearComparisonImage(imageElement, placeholderElement, placeholderElement.textContent);
        return;
    }

    imageElement.src = src;
    imageElement.classList.remove('hidden');
    placeholderElement.classList.add('hidden');
}

function clearComparisonImage(imageElement, placeholderElement, message) {
    imageElement.src = '';
    imageElement.classList.add('hidden');
    placeholderElement.textContent = message;
    placeholderElement.classList.remove('hidden');
}

function updateLiveNotice(message) {
    if (message) {
        dom.liveResultNotice.textContent = message;
        dom.liveResultNotice.classList.remove('hidden');
    } else {
        dom.liveResultNotice.textContent = '';
        dom.liveResultNotice.classList.add('hidden');
    }
}

async function stopLiveSession({ reason, showToastMessage }) {
    if (!liveState.sessionId) {
        setLiveRunningState(false);
        if (reason) {
            dom.liveSessionStatus.textContent = reason;
        }
        return;
    }

    const sessionId = liveState.sessionId;
    liveState.stopping = true;
    liveState.sessionId = null;
    liveState.lastCaptureAt = null;
    stopLivePolling();
    setLiveRunningState(false);

    try {
        await fetch(`/api/realtime/session/${sessionId}`, {
            method: 'DELETE',
            keepalive: true
        });
        if (reason) {
            dom.liveSessionStatus.textContent = reason;
        }
        dom.liveConnectionStatus.textContent = 'Status: Idle';
        dom.liveConnectionStatus.className = 'result-pill live-status-pill status-idle';
        if (showToastMessage && reason) {
            showToast(reason);
        }
    } catch (error) {
        console.error('Live session stop error:', error);
        dom.liveSessionStatus.textContent = 'Live session stop request sent, but confirmation failed.';
    } finally {
        liveState.stopping = false;
    }
}

function setLiveRunningState(isRunning) {
    dom.liveStartBtn.disabled = isRunning;
    dom.liveStopBtn.disabled = !isRunning;
    dom.liveCameraUrl.disabled = isRunning;
    liveControls.analysisModeSelect.disabled = isRunning;
    liveControls.cameraSelect.disabled = isRunning;
    liveControls.detectionSizeSelect.disabled = isRunning;
    liveControls.confidenceThresholdInput.disabled = isRunning;
    liveControls.emptySensitivityInput.disabled = isRunning;
    liveControls.inferEdgeSlotsInput.disabled = isRunning;
    dom.liveStartBtnText.textContent = isRunning ? 'Starting...' : 'Start Live Analysis';
}

function describeLiveStatus(status) {
    const labelMap = {
        starting: 'Starting',
        connecting: 'Connecting',
        waiting_for_frame: 'Waiting For Frame',
        processing: 'Processing',
        running: 'Running',
        reconnecting: 'Reconnecting',
        stopped: 'Stopped',
        error: 'Error'
    };

    return labelMap[status] || 'Idle';
}

function normalizeLiveStatusClass(status) {
    if (!status) {
        return 'idle';
    }
    return status.replace(/_/g, '-');
}

function describeLiveSessionMessage(data) {
    const status = data.status || 'idle';
    if (status === 'waiting_for_frame') {
        return 'Connected to the stream. Waiting for the first decodable frame.';
    }
    if (status === 'processing') {
        return 'Analyzing the latest captured frame with the shared backend pipeline.';
    }
    if (status === 'running') {
        return data.last_capture_at
            ? 'Live session is active and polling for the next scheduled capture.'
            : 'Live session is running and waiting to produce the first capture.';
    }
    if (status === 'reconnecting') {
        return data.error || 'The stream is temporarily unavailable. ParkVision AI is retrying.';
    }
    if (status === 'error') {
        return data.error || 'The live session encountered an unrecoverable error.';
    }
    if (status === 'stopped') {
        return 'Live session stopped.';
    }
    if (status === 'connecting') {
        return 'Opening the live camera stream...';
    }
    return 'No live stream session running.';
}

function formatTimestamp(value, fallback) {
    if (!value) {
        return fallback;
    }

    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return value;
    }

    return date.toLocaleString();
}

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
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    `;

    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
