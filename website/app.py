"""
Parking Lot Occupancy Detection using YOLOv5 — Flask Web Application
Ported from parking_yolo.cpp by Daniele Ninni
"""

import base64
import csv
import os
import threading
import time
import traceback
import uuid
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import BytesIO

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request, send_from_directory

PIL_AVAILABLE = True
try:
    from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
except ImportError:  # pragma: no cover - Pillow is optional at runtime
    PIL_AVAILABLE = False
    Image = None
    ImageFile = None
    ImageOps = None
    UnidentifiedImageError = ValueError

try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - optional dependency for HEIC/HEIF
    register_heif_opener = None


if PIL_AVAILABLE and register_heif_opener is not None:
    register_heif_opener()

if PIL_AVAILABLE:
    ImageFile.LOAD_TRUNCATED_IMAGES = True


# ── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload


# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
YOLO_MODEL_PATH = os.path.join(PROJECT_DIR, "yolov5s.onnx")
CSV_DIR = os.path.join(PROJECT_DIR, "results")
REFERENCE_IMAGES_DIR = os.path.join(PROJECT_DIR, "results", "FULL_IMAGE_1000x750")


# ── Constants ────────────────────────────────────────────────────────────────
ORIGINAL_IMAGE_WIDTH = 2592.0
ORIGINAL_IMAGE_HEIGHT = 1944.0
DISPLAY_IMAGE_WIDTH = 1000
DISPLAY_IMAGE_HEIGHT = 750
YOLO_INPUT_WIDTH = 640
YOLO_INPUT_HEIGHT = 640

DEFAULT_CAMERA_NUMBER = 5
SUPPORTED_CAMERA_NUMBERS = tuple(range(1, 10))
DEFAULT_CONFIDENCE_THRESHOLD = 0.18
DEFAULT_NMS_IOU_THRESHOLD = 0.45
DEFAULT_SLOT_OVERLAP_THRESHOLD = 0.50
DEFAULT_DETECTION_AREA_THRESHOLD = 5
DEFAULT_ANALYSIS_MODE = "auto"
DEFAULT_DETECTION_SIZE = "auto"
DEFAULT_EMPTY_SENSITIVITY = 0.35
DEFAULT_INFER_EDGE_SLOTS = False
MIN_CONFIDENCE_THRESHOLD = 0.10
MAX_CONFIDENCE_THRESHOLD = 0.32
DETECTION_MAX_SIDE = 1920
DETECTION_UPSCALE_TARGET_SIDE = 1400
DETECTION_SECONDARY_UPSCALE_TARGET_SIDE = 1900
DETECTION_MAX_UPSCALE_FACTOR = 4.0
TILE_SIZE = 768
TILE_OVERLAP_RATIO = 0.35
TILE_TRIGGER_DETECTION_COUNT = 12
TILE_CONFIDENCE_THRESHOLD = 0.12
LAYOUT_MIN_FEATURE_SCORE = 120.0
LAYOUT_MIN_FEATURE_RATIO = 3.0
ROTATION_TRIGGER_DETECTION_COUNT = 8
UPSCALE_TRIGGER_IMAGE_SIDE = 1100
UPSCALE_TRIGGER_DETECTION_COUNT = 20
REALTIME_CAPTURE_INTERVAL_SECONDS = 20
LIVE_CAMERA_LOCK_MATCH_COUNT = 3
LIVE_RESULT_HISTORY_SIZE = 3
LIVE_STREAM_RETRY_DELAY_SECONDS = 2.0
LIVE_STREAM_READ_SLEEP_SECONDS = 0.2
LIVE_STREAM_REOPEN_AFTER_FAILURES = 6

DETECTION_SIZE_PRESETS = {
    "balanced": {
        "label": "Balanced",
        "max_side": DETECTION_MAX_SIDE,
        "primary_upscale_target": DETECTION_UPSCALE_TARGET_SIDE,
        "secondary_upscale_target": DETECTION_SECONDARY_UPSCALE_TARGET_SIDE,
        "upscale_trigger_side": UPSCALE_TRIGGER_IMAGE_SIDE,
        "upscale_trigger_detection_count": UPSCALE_TRIGGER_DETECTION_COUNT,
    },
    "high": {
        "label": "High detail",
        "max_side": 2300,
        "primary_upscale_target": 1800,
        "secondary_upscale_target": 2500,
        "upscale_trigger_side": 1400,
        "upscale_trigger_detection_count": 24,
    },
    "max": {
        "label": "Max detail",
        "max_side": 2800,
        "primary_upscale_target": 2100,
        "secondary_upscale_target": 3000,
        "upscale_trigger_side": 1650,
        "upscale_trigger_detection_count": 28,
    },
}

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

REFERENCE_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# Colors (BGR for OpenCV)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
PANEL_BG = (20, 28, 43)
PANEL_TEXT = (244, 247, 250)
DETECTION_BOX_COLOR = (255, 191, 0)


@dataclass(frozen=True)
class DetectionSettings:
    """User-adjustable detection and layout controls sent from the UI."""

    analysis_mode: str = DEFAULT_ANALYSIS_MODE
    detection_size: str = DEFAULT_DETECTION_SIZE
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    empty_sensitivity: float = DEFAULT_EMPTY_SENSITIVITY
    infer_edge_slots: bool = DEFAULT_INFER_EDGE_SLOTS


@dataclass
class FrameProcessingResult:
    """Shared processed-frame payload used by uploads and live sessions."""

    input_image: np.ndarray
    output_image: np.ndarray
    parking_lots: list
    detections: list[dict]
    stats: dict
    selected_camera: int | None


@dataclass
class LiveStreamSession:
    """Mutable state for one backend-managed realtime stream session."""

    session_id: str
    camera_url: str
    settings: DetectionSettings
    requested_camera: int | None = None
    capture_interval_seconds: int = REALTIME_CAPTURE_INTERVAL_SECONDS
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    state_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    history: deque = field(
        default_factory=lambda: deque(maxlen=LIVE_RESULT_HISTORY_SIZE),
        repr=False,
    )
    thread: threading.Thread | None = None
    status: str = "starting"
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_frame_at: datetime | None = None
    last_capture_at: datetime | None = None
    next_capture_at: datetime | None = None
    latest_result: dict | None = None
    error: str | None = None
    camera_locked: bool = False
    locked_camera: int | None = None
    last_candidate_camera: int | None = None
    consecutive_camera_matches: int = 0

    def snapshot(self) -> dict:
        """Return a thread-safe JSON-serializable view of this session."""
        with self.state_lock:
            latest_result = deepcopy(self.latest_result) if self.latest_result is not None else None
            return {
                "session_id": self.session_id,
                "status": self.status,
                "started_at": format_timestamp(self.started_at),
                "last_capture_at": format_timestamp(self.last_capture_at),
                "next_capture_at": format_timestamp(self.next_capture_at),
                "camera_locked": self.camera_locked,
                "locked_camera": self.locked_camera,
                "error": self.error,
                "input_image": None if latest_result is None else latest_result["input_image"],
                "output_image": None if latest_result is None else latest_result["output_image"],
                "stats": (
                    build_pending_live_stats(camera_locked=self.camera_locked)
                    if latest_result is None
                    else latest_result["stats"]
                ),
            }


class LiveStreamSessionManager:
    """Store, start, and stop live stream sessions by UUID."""

    def __init__(self):
        self._sessions: dict[str, LiveStreamSession] = {}
        self._lock = threading.Lock()

    def create(
        self,
        camera_url: str,
        settings: DetectionSettings,
        requested_camera: int | None = None,
    ) -> LiveStreamSession:
        session_id = str(uuid.uuid4())
        session = LiveStreamSession(
            session_id=session_id,
            camera_url=camera_url,
            settings=settings,
            requested_camera=requested_camera,
            next_capture_at=datetime.now(timezone.utc),
        )
        worker = threading.Thread(
            target=run_live_stream_session,
            args=(session,),
            name=f"live-stream-{session_id}",
            daemon=True,
        )
        session.thread = worker
        with self._lock:
            self._sessions[session_id] = session
        worker.start()
        return session

    def get(self, session_id: str) -> LiveStreamSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> LiveStreamSession | None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return None

        stop_live_stream_session(session)
        return session


# ── Load YOLO Model (once at startup) ───────────────────────────────────────
print(f"[INFO] Loading YOLOv5 model from: {YOLO_MODEL_PATH}")
ort_session = ort.InferenceSession(
    YOLO_MODEL_PATH,
    providers=["CPUExecutionProvider"],
)
input_name = ort_session.get_inputs()[0].name
INFERENCE_LOCK = threading.Lock()
print("[INFO] YOLOv5 model loaded successfully!")


def load_parking_lots(camera_number: int) -> list:
    """Load parking slot definitions from CSV for the given camera."""
    csv_path = os.path.join(CSV_DIR, f"camera{camera_number}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Camera CSV not found: {csv_path}")

    parking_lot_x_factor = DISPLAY_IMAGE_WIDTH / ORIGINAL_IMAGE_WIDTH
    parking_lot_y_factor = DISPLAY_IMAGE_HEIGHT / ORIGINAL_IMAGE_HEIGHT

    parking_lots = []
    with open(csv_path, "r", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            x = int(float(row["X"]) * parking_lot_x_factor)
            y = int(float(row["Y"]) * parking_lot_y_factor)
            w = int(float(row["W"]) * parking_lot_x_factor)
            h = int(float(row["H"]) * parking_lot_y_factor)
            parking_lots.append(
                {
                    "slot_id": row["SlotId"].strip(),
                    "bbox": (x, y, w, h),
                }
            )

    return parking_lots


def resize_to_display(image: np.ndarray) -> np.ndarray:
    """Resize an image to the layout size used by the parking-slot overlays."""
    interpolation = cv2.INTER_AREA
    if image.shape[1] < DISPLAY_IMAGE_WIDTH or image.shape[0] < DISPLAY_IMAGE_HEIGHT:
        interpolation = cv2.INTER_LINEAR

    return cv2.resize(
        image,
        (DISPLAY_IMAGE_WIDTH, DISPLAY_IMAGE_HEIGHT),
        interpolation=interpolation,
    )


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a numeric value into the given range."""
    return max(min_value, min(value, max_value))


def parse_bool(value, default: bool = False) -> bool:
    """Parse a checkbox-like form value."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_float_value(
    raw_value,
    *,
    default: float,
    min_value: float,
    max_value: float,
) -> float:
    """Parse and clamp a float form value."""
    if raw_value is None:
        return default
    if isinstance(raw_value, str):
        if raw_value.strip() == "":
            return default
        parsed_value = float(raw_value)
    else:
        parsed_value = float(raw_value)
    return clamp(parsed_value, min_value, max_value)


def parse_analysis_mode(raw_value: str | None) -> str:
    """Normalize the requested analysis mode."""
    normalized_value = str(raw_value or DEFAULT_ANALYSIS_MODE).strip().lower()
    if normalized_value not in {"auto", "fixed", "generic"}:
        raise ValueError("Analysis mode must be auto, fixed, or generic.")
    return normalized_value


def parse_detection_size(raw_value: str | None) -> str:
    """Normalize the requested detection detail preset."""
    normalized_value = str(raw_value or DEFAULT_DETECTION_SIZE).strip().lower()
    if normalized_value not in {"auto", *DETECTION_SIZE_PRESETS.keys()}:
        raise ValueError("Detection size must be auto, balanced, high, or max.")
    return normalized_value


def resolve_detection_profile(
    detection_size: str,
    image_shape: tuple[int, int, int],
) -> dict:
    """Choose the effective detection-resolution profile for this image."""
    requested_level = detection_size if detection_size != "auto" else DEFAULT_DETECTION_SIZE
    if detection_size == "auto":
        image_height, image_width = image_shape[:2]
        largest_side = max(image_height, image_width)
        megapixels = (image_height * image_width) / 1_000_000.0
        if largest_side <= 720 or megapixels <= 0.55:
            resolved_level = "max"
        elif largest_side <= 1100 or megapixels <= 1.20:
            resolved_level = "high"
        else:
            resolved_level = "balanced"
    else:
        resolved_level = detection_size

    profile = dict(DETECTION_SIZE_PRESETS[resolved_level])
    profile["requested_level"] = requested_level
    profile["resolved_level"] = resolved_level
    profile["display_label"] = (
        f"Auto -> {profile['label']}" if detection_size == "auto" else profile["label"]
    )
    return profile


def describe_empty_sensitivity(empty_sensitivity: float) -> str:
    """Return a short user-facing label for generic empty-slot aggressiveness."""
    if empty_sensitivity <= 0.35:
        return "Conservative"
    if empty_sensitivity <= 0.65:
        return "Balanced"
    return "Aggressive"


def utcnow() -> datetime:
    """Return the current UTC timestamp with timezone information."""
    return datetime.now(timezone.utc)


def format_timestamp(value: datetime | None) -> str | None:
    """Return an ISO-8601 string for API responses."""
    if value is None:
        return None
    return value.isoformat(timespec="seconds")


def parse_detection_settings(form_data) -> tuple[int | None, DetectionSettings]:
    """Parse analysis controls from the request form."""
    analysis_mode = parse_analysis_mode(form_data.get("analysis_mode"))
    detection_size = parse_detection_size(form_data.get("detection_size"))
    confidence_threshold = parse_float_value(
        form_data.get("confidence_threshold"),
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        min_value=MIN_CONFIDENCE_THRESHOLD,
        max_value=MAX_CONFIDENCE_THRESHOLD,
    )
    empty_sensitivity_percent = parse_float_value(
        form_data.get("empty_sensitivity"),
        default=DEFAULT_EMPTY_SENSITIVITY * 100.0,
        min_value=0.0,
        max_value=100.0,
    )
    infer_edge_slots = parse_bool(
        form_data.get("infer_edge_slots"),
        default=DEFAULT_INFER_EDGE_SLOTS,
    )

    requested_camera = None
    raw_camera_value = str(form_data.get("camera", "auto") or "auto").strip().lower()
    if analysis_mode == "fixed":
        if raw_camera_value and raw_camera_value != "auto":
            requested_camera = int(raw_camera_value)
        else:
            requested_camera = DEFAULT_CAMERA_NUMBER

        if requested_camera not in SUPPORTED_CAMERA_NUMBERS:
            raise ValueError("Camera number must be 1–9 when fixed profile mode is used.")

    settings = DetectionSettings(
        analysis_mode=analysis_mode,
        detection_size=detection_size,
        confidence_threshold=confidence_threshold,
        empty_sensitivity=empty_sensitivity_percent / 100.0,
        infer_edge_slots=infer_edge_slots,
    )
    return requested_camera, settings


def resize_for_detection(
    image: np.ndarray,
    max_side: int = DETECTION_MAX_SIDE,
) -> np.ndarray:
    """Keep more detail for detection while capping very large uploads."""
    image_height, image_width = image.shape[:2]
    largest_side = max(image_height, image_width)
    if largest_side <= max_side:
        return image.copy()

    scale = max_side / float(largest_side)
    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    return cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def compute_image_quality_metrics(image: np.ndarray) -> tuple[float, float]:
    """Return blur and contrast scores used to adapt image enhancement."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast_score = float(gray.std())
    return blur_score, contrast_score


def enhance_detection_input(image: np.ndarray) -> np.ndarray:
    """Improve low-quality images before running the detector."""
    blur_score, contrast_score = compute_image_quality_metrics(image)
    enhanced_image = image.copy()

    if contrast_score < 55.0:
        lab_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        clip_limit = 3.2 if contrast_score < 35.0 else 2.4
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        equalized_l = clahe.apply(l_channel)
        enhanced_image = cv2.cvtColor(
            cv2.merge((equalized_l, a_channel, b_channel)),
            cv2.COLOR_LAB2BGR,
        )

    if blur_score < 90.0:
        enhanced_image = cv2.detailEnhance(enhanced_image, sigma_s=8, sigma_r=0.15)

    if blur_score < 130.0 or contrast_score < 48.0:
        sharpen_strength = 0.30 if blur_score < 90.0 else 0.22
        enhanced_image = cv2.addWeighted(
            enhanced_image,
            1.0 + sharpen_strength,
            cv2.GaussianBlur(enhanced_image, (0, 0), 1.4),
            -sharpen_strength,
            0,
        )

    if contrast_score < 28.0:
        enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=1.10, beta=8)

    return enhanced_image


def build_upscaled_detection_views(
    image: np.ndarray,
    detection_profile: dict,
) -> list[tuple[np.ndarray, float]]:
    """Create extra high-resolution views for tiny vehicles in low-quality uploads."""
    image_height, image_width = image.shape[:2]
    largest_side = max(image_height, image_width)

    scale_targets = []
    if largest_side < detection_profile["upscale_trigger_side"]:
        scale_targets.append(detection_profile["primary_upscale_target"])
    if largest_side < detection_profile["primary_upscale_target"] * 0.75:
        scale_targets.append(detection_profile["secondary_upscale_target"])

    scaled_views = []
    seen_scales = {1.0}
    for target_side in scale_targets:
        scale = min(target_side / float(largest_side), DETECTION_MAX_UPSCALE_FACTOR)
        if scale <= 1.05:
            continue

        rounded_scale = round(scale, 2)
        if rounded_scale in seen_scales:
            continue
        seen_scales.add(rounded_scale)

        resized_width = max(1, int(round(image_width * scale)))
        resized_height = max(1, int(round(image_height * scale)))
        scaled_image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LANCZOS4,
        )
        scaled_views.append((enhance_detection_input(scaled_image), scale))

    return scaled_views


def decode_image(image_bytes: bytes) -> tuple[np.ndarray, str]:
    """Decode uploaded bytes into a BGR image while handling more formats."""
    if len(image_bytes) == 0:
        raise ValueError("Uploaded file is empty (0 bytes).")

    if PIL_AVAILABLE:
        try:
            with Image.open(BytesIO(image_bytes)) as pil_image:
                image_format = pil_image.format or "Unknown"
                pil_image = ImageOps.exif_transpose(pil_image)
                if getattr(pil_image, "is_animated", False):
                    pil_image.seek(0)
                rgb_image = np.array(pil_image.convert("RGB"))
                return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR), image_format.upper()
        except (UnidentifiedImageError, OSError, ValueError):
            pass

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if image is None:
        pillow_hint = ""
        if not PIL_AVAILABLE:
            pillow_hint = " Install Pillow to enable more formats such as GIF orientation fixes and HEIC/HEIF."
        raise ValueError(
            "Could not decode the uploaded image. Supported formats include JPG, "
            "PNG, WEBP, BMP, and TIFF through OpenCV."
            f"{pillow_hint}"
        )

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image, "UNKNOWN"


def create_feature_detector():
    return cv2.ORB_create(nfeatures=1400, fastThreshold=10)


def build_feature_descriptors(image: np.ndarray) -> np.ndarray | None:
    """Generate stable descriptors for camera-profile matching."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, descriptors = create_feature_detector().detectAndCompute(gray, None)
    return descriptors


def load_camera_references() -> dict[int, list[dict]]:
    """Preload reference descriptors for automatic camera-profile matching."""
    references = {camera: [] for camera in SUPPORTED_CAMERA_NUMBERS}
    if not os.path.exists(REFERENCE_IMAGES_DIR):
        return references

    for root, _, files in os.walk(REFERENCE_IMAGES_DIR):
        camera_name = os.path.basename(root).lower()
        if not camera_name.startswith("camera"):
            continue

        camera_number = int(camera_name.replace("camera", ""))
        if camera_number not in references:
            continue

        for filename in sorted(files):
            file_ext = os.path.splitext(filename)[1].lower()
            if (
                file_ext not in REFERENCE_IMAGE_EXTENSIONS
                or "_output" in filename
                or "_trackbars" in filename
            ):
                continue

            full_path = os.path.join(root, filename)
            image = cv2.imread(full_path, cv2.IMREAD_COLOR)
            if image is None:
                continue

            resized = resize_to_display(image)
            references[camera_number].append(
                {
                    "path": full_path,
                    "descriptors": build_feature_descriptors(resized),
                }
            )

    return references


def score_camera_profiles(image: np.ndarray) -> dict[int, float]:
    """Score the uploaded image against the supported camera layouts."""
    input_descriptors = build_feature_descriptors(image)
    scores = {camera: 0.0 for camera in SUPPORTED_CAMERA_NUMBERS}
    if input_descriptors is None:
        return scores

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    for camera_number, references in CAMERA_REFERENCES.items():
        best_score = 0.0
        for reference in references:
            reference_descriptors = reference["descriptors"]
            if reference_descriptors is None:
                continue

            matches = matcher.knnMatch(input_descriptors, reference_descriptors, k=2)
            good_matches = []
            for pair in matches:
                if len(pair) < 2:
                    continue
                first_match, second_match = pair
                if first_match.distance < 0.75 * second_match.distance:
                    good_matches.append(first_match)

            best_score = max(best_score, float(len(good_matches)))

        scores[camera_number] = best_score

    return scores


def letterbox_image(
    image: np.ndarray,
    target_shape: tuple[int, int] = (YOLO_INPUT_HEIGHT, YOLO_INPUT_WIDTH),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, int, int]:
    """Resize while preserving aspect ratio for YOLO input."""
    image_height, image_width = image.shape[:2]
    scale = min(target_shape[0] / image_height, target_shape[1] / image_width)

    resized_width = int(round(image_width * scale))
    resized_height = int(round(image_height * scale))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_width = target_shape[1] - resized_width
    pad_height = target_shape[0] - resized_height
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return padded, scale, pad_left, pad_top


def create_detection_variants(image: np.ndarray) -> list[np.ndarray]:
    """Create a small set of image variants to improve robustness."""
    variants = [image]

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_image = cv2.cvtColor(
        cv2.merge((enhanced_l, a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )
    variants.append(enhanced_image)

    sharpened_image = cv2.addWeighted(image, 1.20, cv2.GaussianBlur(image, (0, 0), 2.0), -0.20, 0)
    variants.append(sharpened_image)

    return variants


def parse_yolo_output(
    output: np.ndarray,
    scale: float,
    pad_left: int,
    pad_top: int,
    image_shape: tuple[int, int, int],
    confidence_threshold: float,
) -> list[dict]:
    """Convert raw YOLO predictions into filtered vehicle detections."""
    predictions = output.squeeze(axis=0)
    if predictions.ndim != 2:
        raise ValueError(f"Unexpected YOLO output shape: {output.shape}")

    if predictions.shape[1] < 6 and predictions.shape[0] >= 6:
        predictions = predictions.T
    elif predictions.shape[0] in (84, 85) and predictions.shape[1] > predictions.shape[0]:
        predictions = predictions.T

    image_height, image_width = image_shape[:2]
    detections = []
    for row in predictions:
        objectness = float(row[4])
        if objectness <= 0:
            continue

        class_scores = row[5:]
        class_id = int(np.argmax(class_scores))
        if class_id not in VEHICLE_CLASSES:
            continue

        class_confidence = float(class_scores[class_id])
        confidence = objectness * class_confidence
        if confidence < confidence_threshold:
            continue

        center_x, center_y, box_width, box_height = map(float, row[:4])

        left = (center_x - (box_width / 2) - pad_left) / scale
        top = (center_y - (box_height / 2) - pad_top) / scale
        width = box_width / scale
        height = box_height / scale

        left = int(max(0, min(round(left), image_width - 1)))
        top = int(max(0, min(round(top), image_height - 1)))
        width = int(max(1, min(round(width), image_width - left)))
        height = int(max(1, min(round(height), image_height - top)))

        if width < 4 or height < 4:
            continue

        detections.append(
            {
                "class_id": class_id,
                "label": VEHICLE_CLASSES[class_id],
                "confidence": confidence,
                "bbox": (left, top, width, height),
            }
        )

    return detections


def apply_non_max_suppression(
    detections: list[dict],
    confidence_threshold: float,
    iou_threshold: float,
) -> list[dict]:
    """Remove duplicate detections across image variants."""
    if not detections:
        return []

    boxes = [list(det["bbox"]) for det in detections]
    scores = [float(det["confidence"]) for det in detections]
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, iou_threshold)
    if indices is None or len(indices) == 0:
        return []

    selected_indices = np.array(indices).reshape(-1)
    return [detections[int(index)] for index in selected_indices]


def enumerate_tile_positions(length: int, tile_length: int, overlap_ratio: float) -> list[int]:
    """Generate start positions for overlapping detection tiles."""
    if length <= tile_length:
        return [0]

    step = max(64, int(round(tile_length * (1.0 - overlap_ratio))))
    positions = list(range(0, max(length - tile_length, 0) + 1, step))
    last_position = length - tile_length
    if positions[-1] != last_position:
        positions.append(last_position)
    return positions


def generate_detection_tiles(
    image: np.ndarray,
    tile_size: int = TILE_SIZE,
    overlap_ratio: float = TILE_OVERLAP_RATIO,
) -> list[tuple[np.ndarray, int, int]]:
    """Split large scenes into overlapping tiles so small vehicles stay visible."""
    image_height, image_width = image.shape[:2]
    tile_width = min(tile_size, image_width)
    tile_height = min(tile_size, image_height)

    tile_positions = []
    y_positions = enumerate_tile_positions(image_height, tile_height, overlap_ratio)
    x_positions = enumerate_tile_positions(image_width, tile_width, overlap_ratio)

    for top in y_positions:
        for left in x_positions:
            tile_positions.append(
                (
                    image[top: top + tile_height, left: left + tile_width].copy(),
                    left,
                    top,
                )
            )

    return tile_positions


def offset_detections(detections: list[dict], offset_x: int, offset_y: int) -> list[dict]:
    """Translate detections from tile coordinates into full-image coordinates."""
    translated = []
    for detection in detections:
        left, top, width, height = detection["bbox"]
        translated.append(
            {
                **detection,
                "bbox": (left + offset_x, top + offset_y, width, height),
            }
        )

    return translated


def run_yolo_on_region(
    image: np.ndarray,
    confidence_threshold: float,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[dict]:
    """Run YOLO on one region and translate boxes into full-image coordinates."""
    region_detections = []
    for variant_image in create_detection_variants(image):
        padded_image, scale, pad_left, pad_top = letterbox_image(variant_image)
        blob = cv2.dnn.blobFromImage(
            padded_image,
            scalefactor=1.0 / 255.0,
            size=(YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT),
            swapRB=True,
            crop=False,
        )

        outputs = ort_session.run(None, {input_name: blob})
        detections = parse_yolo_output(
            outputs[0],
            scale,
            pad_left,
            pad_top,
            variant_image.shape,
            confidence_threshold,
        )
        region_detections.extend(offset_detections(detections, offset_x, offset_y))

    return region_detections


def restore_bbox_from_rotation(
    bbox: tuple[int, int, int, int],
    original_shape: tuple[int, int, int],
    rotation_code: int | None,
) -> tuple[int, int, int, int]:
    """Map a detection bbox from a rotated image back to the original image."""
    if rotation_code is None:
        return bbox

    left, top, width, height = bbox
    image_height, image_width = original_shape[:2]
    corners = [
        (left, top),
        (left + width, top),
        (left + width, top + height),
        (left, top + height),
    ]

    restored_corners = []
    for x_pos, y_pos in corners:
        if rotation_code == cv2.ROTATE_90_CLOCKWISE:
            restored_x = y_pos
            restored_y = image_height - x_pos
        elif rotation_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
            restored_x = image_width - y_pos
            restored_y = x_pos
        elif rotation_code == cv2.ROTATE_180:
            restored_x = image_width - x_pos
            restored_y = image_height - y_pos
        else:  # pragma: no cover - defensive fallback
            restored_x = x_pos
            restored_y = y_pos

        restored_corners.append((restored_x, restored_y))

    xs = [point[0] for point in restored_corners]
    ys = [point[1] for point in restored_corners]

    restored_left = int(max(0, min(round(min(xs)), image_width - 1)))
    restored_top = int(max(0, min(round(min(ys)), image_height - 1)))
    restored_right = int(max(restored_left + 1, min(round(max(xs)), image_width)))
    restored_bottom = int(max(restored_top + 1, min(round(max(ys)), image_height)))
    return (
        restored_left,
        restored_top,
        restored_right - restored_left,
        restored_bottom - restored_top,
    )


def run_rotated_region_detections(
    image: np.ndarray,
    confidence_threshold: float,
) -> list[dict]:
    """Run detection on rotated views and map boxes back into the original image."""
    rotated_detections = []
    for rotation_code in [
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_180,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
    ]:
        rotated_image = cv2.rotate(image, rotation_code)
        detections = run_yolo_on_region(rotated_image, confidence_threshold)
        for detection in detections:
            rotated_detections.append(
                {
                    **detection,
                    "bbox": restore_bbox_from_rotation(
                        detection["bbox"],
                        image.shape,
                        rotation_code,
                    ),
                }
            )

    return rotated_detections


def rescale_detections(detections: list[dict], inverse_scale: float) -> list[dict]:
    """Project detections from an upscaled view back into the base image space."""
    scaled_detections = []
    for detection in detections:
        left, top, width, height = detection["bbox"]
        scaled_detections.append(
            {
                **detection,
                "bbox": (
                    int(round(left * inverse_scale)),
                    int(round(top * inverse_scale)),
                    max(1, int(round(width * inverse_scale))),
                    max(1, int(round(height * inverse_scale))),
                ),
            }
        )

    return scaled_detections


def run_vehicle_detection_on_single_view(
    image: np.ndarray,
    confidence_threshold: float,
    iou_threshold: float,
) -> list[dict]:
    """Run the base detector stack on one image resolution."""
    global_detections = run_yolo_on_region(image, confidence_threshold)
    deduplicated_global_detections = apply_non_max_suppression(
        global_detections,
        confidence_threshold,
        iou_threshold,
    )

    all_detections = list(global_detections)
    if len(deduplicated_global_detections) < ROTATION_TRIGGER_DETECTION_COUNT:
        all_detections.extend(
            run_rotated_region_detections(
                image,
                confidence_threshold,
            )
        )
        deduplicated_global_detections = apply_non_max_suppression(
            all_detections,
            confidence_threshold,
            iou_threshold,
        )

    if (
        max(image.shape[:2]) >= TILE_SIZE
        and len(deduplicated_global_detections) < TILE_TRIGGER_DETECTION_COUNT
    ):
        for tile_image, offset_x, offset_y in generate_detection_tiles(image):
            tile_detections = run_yolo_on_region(
                tile_image,
                TILE_CONFIDENCE_THRESHOLD,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            all_detections.extend(tile_detections)

    return apply_non_max_suppression(
        all_detections,
        min(confidence_threshold, TILE_CONFIDENCE_THRESHOLD),
        iou_threshold,
    )


def run_vehicle_detection(
    image: np.ndarray,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
    detection_profile: dict | None = None,
) -> list[dict]:
    """Run YOLOv5 inference and return deduplicated vehicle detections."""
    if detection_profile is None:
        detection_profile = resolve_detection_profile(DEFAULT_DETECTION_SIZE, image.shape)

    base_detections = run_vehicle_detection_on_single_view(
        image,
        confidence_threshold,
        iou_threshold,
    )

    all_detections = list(base_detections)
    if (
        max(image.shape[:2]) < detection_profile["upscale_trigger_side"]
        or len(base_detections) < detection_profile["upscale_trigger_detection_count"]
    ):
        auxiliary_confidence = max(0.10, confidence_threshold - 0.04)
        for upscaled_image, scale in build_upscaled_detection_views(image, detection_profile):
            auxiliary_detections = run_vehicle_detection_on_single_view(
                upscaled_image,
                auxiliary_confidence,
                iou_threshold,
            )
            all_detections.extend(rescale_detections(auxiliary_detections, 1.0 / scale))

    return apply_non_max_suppression(
        all_detections,
        max(0.10, confidence_threshold - 0.04),
        iou_threshold,
    )


def compute_intersection_area(rect1, rect2) -> int:
    """Compute intersection area between two rectangles (x, y, w, h)."""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    intersection_left = max(x1, x2)
    intersection_top = max(y1, y2)
    intersection_right = min(x1 + w1, x2 + w2)
    intersection_bottom = min(y1 + h1, y2 + h2)

    if intersection_right <= intersection_left or intersection_bottom <= intersection_top:
        return 0

    return (intersection_right - intersection_left) * (
        intersection_bottom - intersection_top
    )


def evaluate_parking_layout(
    parking_lots: list,
    detections: list,
    overlap_threshold: float,
    detection_area_threshold: int,
) -> dict:
    """Score a parking layout against the vehicle detections."""
    occupied_count = 0
    slot_details = []
    matched_detection_indices = set()
    overlap_sum = 0.0

    for parking_lot in parking_lots:
        slot_bbox = parking_lot["bbox"]
        slot_area = max(1, slot_bbox[2] * slot_bbox[3])
        best_match_index = None
        best_overlap = 0.0

        for detection_index, detection in enumerate(detections):
            det_bbox = detection["bbox"]
            det_area = det_bbox[2] * det_bbox[3]
            if det_area >= detection_area_threshold * slot_area:
                continue

            intersection = compute_intersection_area(slot_bbox, det_bbox)
            overlap_ratio = intersection / slot_area
            if overlap_ratio >= overlap_threshold and overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_match_index = detection_index

        is_occupied = best_match_index is not None
        if is_occupied:
            occupied_count += 1
            matched_detection_indices.add(best_match_index)
            overlap_sum += best_overlap

        slot_details.append(
            {
                "slot_id": parking_lot["slot_id"],
                "occupied": is_occupied,
                "overlap_ratio": round(best_overlap, 3),
            }
        )

    total_count = len(parking_lots)
    empty_count = total_count - occupied_count
    mean_overlap = overlap_sum / max(1, occupied_count)

    return {
        "total": total_count,
        "occupied": occupied_count,
        "empty": empty_count,
        "occupancy_rate": round(occupied_count / max(1, total_count) * 100, 1),
        "matched_detections": len(matched_detection_indices),
        "mean_overlap": round(mean_overlap, 3),
        "slots": slot_details,
    }


def scale_detections_to_display(
    detections: list[dict],
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> list[dict]:
    """Map detections from the detection image into the display image."""
    source_height, source_width = source_shape[:2]
    target_height, target_width = target_shape[:2]
    scale_x = target_width / max(1, source_width)
    scale_y = target_height / max(1, source_height)

    scaled = []
    for detection in detections:
        left, top, width, height = detection["bbox"]
        scaled_left = int(round(left * scale_x))
        scaled_top = int(round(top * scale_y))
        scaled_width = int(round(width * scale_x))
        scaled_height = int(round(height * scale_y))

        scaled_left = max(0, min(scaled_left, target_width - 1))
        scaled_top = max(0, min(scaled_top, target_height - 1))
        scaled_width = max(1, min(scaled_width, target_width - scaled_left))
        scaled_height = max(1, min(scaled_height, target_height - scaled_top))

        scaled.append(
            {
                **detection,
                "bbox": (scaled_left, scaled_top, scaled_width, scaled_height),
            }
        )

    return scaled


def get_detection_center(detection: dict) -> tuple[float, float]:
    left, top, width, height = detection["bbox"]
    return left + (width / 2.0), top + (height / 2.0)


def project_bbox_to_angle(width: float, height: float, angle_rad: float) -> tuple[float, float]:
    """Project an axis-aligned box onto a rotated along/cross coordinate system."""
    cos_angle = abs(np.cos(angle_rad))
    sin_angle = abs(np.sin(angle_rad))
    along_size = (width * cos_angle) + (height * sin_angle)
    cross_size = (width * sin_angle) + (height * cos_angle)
    return along_size, cross_size


def rotate_point(
    x: float,
    y: float,
    angle_rad: float,
    origin: tuple[float, float],
) -> tuple[float, float]:
    """Rotate a point around an origin."""
    origin_x, origin_y = origin
    translated_x = x - origin_x
    translated_y = y - origin_y
    rotated_x = (translated_x * np.cos(angle_rad)) - (translated_y * np.sin(angle_rad))
    rotated_y = (translated_x * np.sin(angle_rad)) + (translated_y * np.cos(angle_rad))
    return rotated_x + origin_x, rotated_y + origin_y


def estimate_candidate_row_angles(
    detections: list[dict],
    image_shape: tuple[int, int, int],
) -> list[float]:
    """Estimate likely parking-row directions from detection centers."""
    if len(detections) < 2:
        return [0.0, 90.0]

    centers = np.array([get_detection_center(detection) for detection in detections], dtype=np.float32)
    short_sides = np.array(
        [min(detection["bbox"][2], detection["bbox"][3]) for detection in detections],
        dtype=np.float32,
    )
    minimum_distance = max(18.0, float(np.median(short_sides) * 0.75))
    maximum_distance = max(image_shape[:2]) * 0.45

    angles = []
    for index, center in enumerate(centers):
        distances = np.linalg.norm(centers - center, axis=1)
        neighbor_indices = np.argsort(distances)[1:5]
        for neighbor_index in neighbor_indices:
            distance = float(distances[neighbor_index])
            if distance < minimum_distance or distance > maximum_distance:
                continue

            delta_x = centers[neighbor_index][0] - center[0]
            delta_y = centers[neighbor_index][1] - center[1]
            angle = float(np.degrees(np.arctan2(delta_y, delta_x)) % 180.0)
            angles.append(angle)

    if not angles:
        return [0.0, 90.0]

    histogram, edges = np.histogram(angles, bins=36, range=(0.0, 180.0))
    dominant_index = int(np.argmax(histogram))
    dominant_angle = float((edges[dominant_index] + edges[dominant_index + 1]) / 2.0)

    candidate_angles = []
    for angle in [dominant_angle, (dominant_angle + 90.0) % 180.0, 0.0, 90.0]:
        is_new_angle = True
        for existing_angle in candidate_angles:
            angular_distance = abs(((angle - existing_angle + 90.0) % 180.0) - 90.0)
            if angular_distance < 7.5:
                is_new_angle = False
                break

        if is_new_angle:
            candidate_angles.append(angle)

    return candidate_angles


def build_oriented_slot_bbox(
    center_u: float,
    center_v: float,
    along_span: float,
    cross_span: float,
    row_angle_rad: float,
    origin: tuple[float, float],
    image_shape: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    """Convert an oriented slot in rotated space back to an axis-aligned image bbox."""
    half_along = along_span / 2.0
    half_cross = cross_span / 2.0
    rotated_corners = [
        (center_u - half_along, center_v - half_cross),
        (center_u + half_along, center_v - half_cross),
        (center_u + half_along, center_v + half_cross),
        (center_u - half_along, center_v + half_cross),
    ]
    image_corners = [
        rotate_point(corner_u, corner_v, row_angle_rad, origin)
        for corner_u, corner_v in rotated_corners
    ]

    xs = [point[0] for point in image_corners]
    ys = [point[1] for point in image_corners]
    image_height, image_width = image_shape[:2]

    left = int(max(0, min(round(min(xs)), image_width - 1)))
    top = int(max(0, min(round(min(ys)), image_height - 1)))
    right = int(max(left + 1, min(round(max(xs)), image_width)))
    bottom = int(max(top + 1, min(round(max(ys)), image_height)))

    return left, top, right - left, bottom - top


def build_generic_layout_for_angle(
    detections: list[dict],
    image_shape: tuple[int, int, int],
    row_angle_deg: float,
    empty_sensitivity: float = DEFAULT_EMPTY_SENSITIVITY,
    infer_edge_slots: bool = DEFAULT_INFER_EDGE_SLOTS,
) -> tuple[list, dict, float]:
    """Estimate parking slots for a general parking image given a row angle."""
    if len(detections) < 2:
        return [], {}, 0.0

    row_angle_rad = float(np.radians(row_angle_deg))
    alignment_rotation = -row_angle_rad
    origin = (image_shape[1] / 2.0, image_shape[0] / 2.0)

    aligned_entries = []
    for detection_index, detection in enumerate(detections):
        center_x, center_y = get_detection_center(detection)
        aligned_u, aligned_v = rotate_point(center_x, center_y, alignment_rotation, origin)
        along_size, cross_size = project_bbox_to_angle(
            detection["bbox"][2],
            detection["bbox"][3],
            row_angle_rad,
        )
        aligned_entries.append(
            {
                "index": detection_index,
                "detection": detection,
                "u": aligned_u,
                "v": aligned_v,
                "along_size": along_size,
                "cross_size": cross_size,
            }
        )

    cross_sizes = [entry["cross_size"] for entry in aligned_entries]
    band_threshold = max(18.0, float(np.median(cross_sizes) * 0.8))
    sorted_entries = sorted(aligned_entries, key=lambda entry: entry["v"])

    row_clusters = []
    current_cluster = [sorted_entries[0]]
    for entry in sorted_entries[1:]:
        current_band = float(np.median([cluster_entry["v"] for cluster_entry in current_cluster]))
        if abs(entry["v"] - current_band) <= band_threshold:
            current_cluster.append(entry)
        else:
            row_clusters.append(current_cluster)
            current_cluster = [entry]
    row_clusters.append(current_cluster)

    image_corners = [
        rotate_point(0.0, 0.0, alignment_rotation, origin),
        rotate_point(float(image_shape[1]), 0.0, alignment_rotation, origin),
        rotate_point(0.0, float(image_shape[0]), alignment_rotation, origin),
        rotate_point(float(image_shape[1]), float(image_shape[0]), alignment_rotation, origin),
    ]
    rotated_min_u = min(point[0] for point in image_corners)
    rotated_max_u = max(point[0] for point in image_corners)

    parking_lots = []
    slot_details = []
    occupied_count = 0
    empty_count = 0
    inferred_empty_count = 0
    used_detection_indices = set()
    rows_with_multiple_cars = 0
    conservative_gap_factor = 1.95 - (empty_sensitivity * 0.55)
    min_row_support_for_empty_slots = 3 if empty_sensitivity < 0.75 else 2
    max_inferred_slots_per_gap = 1 if empty_sensitivity < 0.55 else 2

    for row_index, cluster in enumerate(row_clusters, start=1):
        if len(cluster) >= 2:
            rows_with_multiple_cars += 1

        ordered_cluster = sorted(cluster, key=lambda entry: entry["u"])
        row_v = float(np.median([entry["v"] for entry in ordered_cluster]))
        along_sizes = [entry["along_size"] for entry in ordered_cluster]
        cross_sizes = [entry["cross_size"] for entry in ordered_cluster]
        along_positions = [entry["u"] for entry in ordered_cluster]

        median_along = float(np.median(along_sizes))
        median_cross = float(np.median(cross_sizes))
        consecutive_gaps = np.diff(along_positions)
        valid_gaps = [
            float(gap)
            for gap in consecutive_gaps
            if gap >= max(12.0, median_along * 0.55)
        ]

        if valid_gaps:
            pitch = float(np.percentile(valid_gaps, 35))
            pitch = float(np.clip(pitch, median_along * 0.85, median_along * 3.0))
        else:
            pitch = median_along * 1.25

        slot_along_span = max(median_along * 1.08, pitch * 0.90)
        slot_cross_span = median_cross * 1.22
        row_span = ordered_cluster[-1]["u"] - ordered_cluster[0]["u"]
        row_inferred_empty_count = 0
        max_row_inferred_slots = max(
            1,
            min(
                3,
                int(round((empty_sensitivity * 2.0) + (len(ordered_cluster) / 3.0))),
            ),
        )
        can_infer_row_empties = (
            len(ordered_cluster) >= min_row_support_for_empty_slots
            and row_span >= pitch * 1.75
        )

        for slot_index, entry in enumerate(ordered_cluster, start=1):
            used_detection_indices.add(entry["index"])
            parking_lots.append(
                {
                    "slot_id": "",
                    "bbox": build_oriented_slot_bbox(
                        entry["u"],
                        row_v,
                        slot_along_span,
                        slot_cross_span,
                        row_angle_rad,
                        origin,
                        image_shape,
                    ),
                    "estimated": True,
                    "row": row_index,
                }
            )
            slot_details.append(
                {
                    "slot_id": "",
                    "occupied": True,
                    "overlap_ratio": 0.0,
                }
            )
            occupied_count += 1

            if slot_index < len(ordered_cluster):
                next_entry = ordered_cluster[slot_index]
                gap = next_entry["u"] - entry["u"]
                slot_steps = int(round(gap / max(pitch, 1.0)))
                if (
                    can_infer_row_empties
                    and slot_steps >= 2
                    and gap > pitch * conservative_gap_factor
                    and row_inferred_empty_count < max_row_inferred_slots
                ):
                    missing_slots = min(
                        slot_steps - 1,
                        max_inferred_slots_per_gap,
                        max_row_inferred_slots - row_inferred_empty_count,
                    )
                    if missing_slots <= 0:
                        continue

                    interpolated_step = gap / float(missing_slots + 1)
                    for missing_index in range(1, missing_slots + 1):
                        missing_u = entry["u"] + (interpolated_step * missing_index)
                        parking_lots.append(
                            {
                                "slot_id": "",
                                "bbox": build_oriented_slot_bbox(
                                    missing_u,
                                    row_v,
                                    slot_along_span,
                                    slot_cross_span,
                                    row_angle_rad,
                                    origin,
                                    image_shape,
                                ),
                                "estimated": True,
                                "row": row_index,
                            }
                        )
                        slot_details.append(
                            {
                                "slot_id": "",
                                "occupied": False,
                                "overlap_ratio": 0.0,
                            }
                        )
                        empty_count += 1
                        inferred_empty_count += 1
                        row_inferred_empty_count += 1

        if (
            infer_edge_slots
            and can_infer_row_empties
            and len(ordered_cluster) >= 3
            and row_inferred_empty_count < max_row_inferred_slots
        ):
            first_u = ordered_cluster[0]["u"]
            last_u = ordered_cluster[-1]["u"]
            available_edge_slots = min(
                2,
                max_row_inferred_slots - row_inferred_empty_count,
            )
            for edge_u in [first_u - pitch, last_u + pitch][:available_edge_slots]:
                if rotated_min_u + (pitch * 0.35) <= edge_u <= rotated_max_u - (pitch * 0.35):
                    parking_lots.append(
                        {
                            "slot_id": "",
                            "bbox": build_oriented_slot_bbox(
                                edge_u,
                                row_v,
                                slot_along_span,
                                slot_cross_span,
                                row_angle_rad,
                                origin,
                                image_shape,
                            ),
                            "estimated": True,
                            "row": row_index,
                        }
                    )
                    slot_details.append(
                        {
                            "slot_id": "",
                            "occupied": False,
                            "overlap_ratio": 0.0,
                        }
                    )
                    empty_count += 1
                    inferred_empty_count += 1
                    row_inferred_empty_count += 1

    for detection_index, detection in enumerate(detections):
        if detection_index in used_detection_indices:
            continue

        left, top, width, height = detection["bbox"]
        padding_x = max(4, int(round(width * 0.12)))
        padding_y = max(4, int(round(height * 0.12)))
        parking_lots.append(
            {
                "slot_id": "",
                "bbox": (
                    max(0, left - padding_x),
                    max(0, top - padding_y),
                    min(image_shape[1] - max(0, left - padding_x), width + (padding_x * 2)),
                    min(image_shape[0] - max(0, top - padding_y), height + (padding_y * 2)),
                ),
                "estimated": True,
                "row": None,
            }
        )
        slot_details.append(
            {
                "slot_id": "",
                "occupied": True,
                "overlap_ratio": 0.0,
            }
        )
        occupied_count += 1

    total_slots = occupied_count + empty_count
    effective_inferred_empty_count = min(
        inferred_empty_count,
        max(0, rows_with_multiple_cars * max(1, max_inferred_slots_per_gap)),
    )
    stats = {
        "total": total_slots,
        "occupied": occupied_count,
        "empty": empty_count,
        "occupancy_rate": round((occupied_count / max(1, total_slots)) * 100.0, 1),
        "matched_detections": occupied_count,
        "mean_overlap": 0.0,
        "slots": slot_details,
        "camera": None,
        "camera_label": "Estimated layout",
        "camera_mode": "generic_estimate",
        "camera_match_score": 0.0,
        "selection_score": round(
            rows_with_multiple_cars + (effective_inferred_empty_count * 0.08),
            3,
        ),
        "layout_supported": True,
        "slot_mode": "estimated",
        "warning_message": (
            "Estimated slot layout generated from detected vehicles for this parking-lot image."
        ),
        "estimated_row_angle": round(row_angle_deg, 1),
    }

    score = (
        rows_with_multiple_cars * 4.5
        + occupied_count * 1.6
        + effective_inferred_empty_count * (0.25 + (empty_sensitivity * 0.25))
    )
    return parking_lots, stats, score


def infer_generic_parking_layout(
    detections: list[dict],
    image_shape: tuple[int, int, int],
    settings: DetectionSettings | None = None,
) -> tuple[list, dict]:
    """Estimate parking slots for arbitrary parking-lot images."""
    if settings is None:
        settings = DetectionSettings()

    if not detections:
        return [], {
            "camera": None,
            "camera_label": "Vehicle detections only",
            "camera_mode": "vehicle_only",
            "camera_match_score": 0.0,
            "selection_score": 0.0,
            "layout_supported": False,
            "slot_mode": "vehicle_only",
            "warning_message": (
                "No supported parking layout was found and there were not enough vehicle "
                "detections to estimate parking slots automatically."
            ),
            "total": 0,
            "occupied": 0,
            "empty": 0,
            "occupancy_rate": 0.0,
            "matched_detections": 0,
            "mean_overlap": 0.0,
            "slots": [],
        }

    best_candidate = None
    for candidate_angle in estimate_candidate_row_angles(detections, image_shape):
        parking_lots, stats, score = build_generic_layout_for_angle(
            detections,
            image_shape,
            candidate_angle,
            empty_sensitivity=settings.empty_sensitivity,
            infer_edge_slots=settings.infer_edge_slots,
        )
        if best_candidate is None or score > best_candidate[2]:
            best_candidate = (parking_lots, stats, score)

    if best_candidate is None or best_candidate[2] < 6.0:
        return [], {
            "camera": None,
            "camera_label": "Vehicle detections only",
            "camera_mode": "vehicle_only",
            "camera_match_score": 0.0,
            "selection_score": 0.0,
            "layout_supported": False,
            "slot_mode": "vehicle_only",
            "warning_message": (
                "Detected vehicles were not arranged clearly enough to estimate parking slots "
                "for this image. Showing vehicle detections only."
            ),
            "total": 0,
            "occupied": 0,
            "empty": 0,
            "occupancy_rate": 0.0,
            "matched_detections": 0,
            "mean_overlap": 0.0,
            "slots": [],
        }

    return best_candidate[0], best_candidate[1]


def describe_result_mode(slot_mode: str, requested_analysis_mode: str) -> str:
    """Return a concise label describing how the final result was produced."""
    if slot_mode == "manual":
        return "Manual fixed profile"
    if slot_mode == "fixed":
        return "Auto matched" if requested_analysis_mode == "auto" else "Fixed profile"
    if slot_mode == "estimated":
        return (
            "Auto -> Generic estimate"
            if requested_analysis_mode == "auto"
            else "Generic estimate"
        )
    return "Vehicle only"


def select_best_camera(
    image: np.ndarray,
    detections: list,
    requested_camera: int | None = None,
    overlap_threshold: float = DEFAULT_SLOT_OVERLAP_THRESHOLD,
    detection_area_threshold: int = DEFAULT_DETECTION_AREA_THRESHOLD,
) -> tuple[int | None, list, dict]:
    """Auto-select the best matching camera profile for the uploaded image."""
    if requested_camera is not None:
        selected_layout = evaluate_parking_layout(
            PARKING_LOTS[requested_camera],
            detections,
            overlap_threshold,
            detection_area_threshold,
        )
        selected_layout.update(
            {
                "camera": requested_camera,
                "camera_label": f"Camera {requested_camera}",
                "camera_mode": "manual",
                "camera_match_score": 0.0,
                "selection_score": 1.0,
                "layout_supported": True,
                "slot_mode": "manual",
                "warning_message": (
                    ""
                    if selected_layout["matched_detections"] > 0
                    else (
                        f"Camera {requested_camera} was forced manually. "
                        "Switch to Auto or Generic estimate if this is a different parking layout."
                    )
                ),
            }
        )
        return requested_camera, PARKING_LOTS[requested_camera], selected_layout

    feature_scores = score_camera_profiles(image)
    ordered_scores = sorted(
        feature_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    max_feature_score = ordered_scores[0][1] if ordered_scores else 0.0
    second_best_feature_score = ordered_scores[1][1] if len(ordered_scores) > 1 else 0.0

    best_candidate = None
    for camera_number in SUPPORTED_CAMERA_NUMBERS:
        layout_stats = evaluate_parking_layout(
            PARKING_LOTS[camera_number],
            detections,
            overlap_threshold,
            detection_area_threshold,
        )

        normalized_feature = 0.0
        if max_feature_score > 0:
            normalized_feature = feature_scores[camera_number] / max_feature_score

        detection_support = layout_stats["matched_detections"] / max(1, len(detections))
        occupancy_support = layout_stats["occupied"] / max(1, layout_stats["total"])
        overlap_support = layout_stats["mean_overlap"]

        selection_score = (
            (normalized_feature * 0.70)
            + (detection_support * 0.20)
            + (occupancy_support * 0.05)
            + (overlap_support * 0.05)
        )

        candidate = (
            selection_score,
            camera_number,
            layout_stats,
            feature_scores[camera_number],
        )
        if best_candidate is None or candidate[0] > best_candidate[0]:
            best_candidate = candidate

    if best_candidate is None or best_candidate[0] <= 0:
        return None, [], {
            "camera": None,
            "camera_label": "Unsupported layout",
            "camera_mode": "vehicle_only",
            "camera_match_score": 0.0,
            "selection_score": 0.0,
            "layout_supported": False,
            "slot_mode": "vehicle_only",
            "warning_message": (
                "This image does not match any built-in parking layout closely enough "
                "for reliable slot occupancy. Showing vehicle detections only."
            ),
            "total": 0,
            "occupied": 0,
            "empty": 0,
            "occupancy_rate": 0.0,
            "matched_detections": 0,
            "mean_overlap": 0.0,
            "slots": [],
        }

    selection_score, camera_number, layout_stats, feature_score = best_candidate
    feature_ratio = feature_score / max(1.0, second_best_feature_score)
    layout_supported = (
        feature_score >= LAYOUT_MIN_FEATURE_SCORE
        and feature_ratio >= LAYOUT_MIN_FEATURE_RATIO
    )

    warning_message = ""
    if not layout_supported:
        warning_message = (
            f"The uploaded image is closest to Camera {camera_number}, but the match "
            "is not strong enough for trustworthy slot occupancy. Showing vehicle "
            "detections only."
        )

    layout_stats.update(
        {
            "camera": camera_number if layout_supported else None,
            "camera_label": f"Camera {camera_number}" if layout_supported else "Unsupported layout",
            "camera_mode": "auto" if layout_supported else "vehicle_only",
            "camera_match_score": round(feature_score, 2),
            "selection_score": round(selection_score, 3),
            "layout_supported": layout_supported,
            "slot_mode": "fixed" if layout_supported else "vehicle_only",
            "warning_message": warning_message,
        }
    )
    if not layout_supported:
        layout_stats.update(
            {
                "total": 0,
                "occupied": 0,
                "empty": 0,
                "occupancy_rate": 0.0,
                "slots": [],
            }
        )
        return None, [], layout_stats

    return camera_number, PARKING_LOTS[camera_number], layout_stats


def draw_parking_lots(image: np.ndarray, parking_lots: list, slot_details: list) -> np.ndarray:
    """Render parking-slot occupancy results on the image."""
    for parking_lot, slot_details_item in zip(parking_lots, slot_details):
        x, y, w, h = parking_lot["bbox"]
        is_occupied = slot_details_item["occupied"]
        color = RED if is_occupied else GREEN

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        if parking_lot["slot_id"]:
            cv2.putText(
                image,
                parking_lot["slot_id"],
                (x, max(12, y - 2)),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.55,
                YELLOW,
                1,
                cv2.LINE_AA,
            )

    return image


def draw_vehicle_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Render detected vehicles to help validate model behavior."""
    for detection in detections:
        left, top, width, height = detection["bbox"]
        cv2.rectangle(
            image,
            (left, top),
            (left + width, top + height),
            DETECTION_BOX_COLOR,
            2,
        )

    return image


def draw_result_header(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    """Add a small summary panel on top of the annotated image."""
    panel_width = 520
    panel_height = 56
    cv2.rectangle(image, (12, 12), (12 + panel_width, 12 + panel_height), PANEL_BG, -1)
    cv2.rectangle(image, (12, 12), (12 + panel_width, 12 + panel_height), (55, 70, 96), 1)

    cv2.putText(
        image,
        title,
        (28, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        PANEL_TEXT,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        subtitle,
        (28, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        PANEL_TEXT,
        1,
        cv2.LINE_AA,
    )
    return image


def encode_image_to_base64(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise ValueError("Could not encode the processed image.")
    return base64.b64encode(buffer).decode("utf-8")


def build_pending_live_stats(
    *,
    camera_locked: bool = False,
    warning_message: str = "",
) -> dict:
    """Return the placeholder stats payload for a live session."""
    return {
        "camera_label": "Waiting for live capture",
        "slot_mode": "pending",
        "total": 0,
        "empty": 0,
        "occupied": 0,
        "occupancy_rate": 0.0,
        "detections": 0,
        "warning_message": warning_message,
        "camera_locked": camera_locked,
    }


def build_result_subtitle(stats: dict, detections: list[dict]) -> str:
    """Return the small header subtitle for the annotated image."""
    if stats["slot_mode"] in {"fixed", "manual", "estimated"}:
        return f"Vehicles detected: {len(detections)}"
    return "Layout could not be estimated: showing vehicle detections only"


def render_annotated_result(
    display_image: np.ndarray,
    parking_lots: list,
    stats: dict,
    display_detections: list[dict],
) -> np.ndarray:
    """Render the final annotated frame for uploads and live captures."""
    annotated_image = display_image.copy()
    if stats["slot_mode"] in {"fixed", "manual", "estimated"}:
        annotated_image = draw_parking_lots(annotated_image, parking_lots, stats["slots"])
    annotated_image = draw_vehicle_detections(annotated_image, display_detections)
    return draw_result_header(
        annotated_image,
        stats["camera_label"],
        build_result_subtitle(stats, display_detections),
    )


def process_frame(
    image: np.ndarray,
    requested_camera: int | None = None,
    settings: DetectionSettings | None = None,
    *,
    input_format: str = "STREAM",
    allow_requested_camera_fallback: bool = False,
) -> FrameProcessingResult:
    """Shared detection pipeline for decoded uploads and live video frames."""
    if settings is None:
        settings = DetectionSettings()

    detection_profile = resolve_detection_profile(settings.detection_size, image.shape)
    detection_image = resize_for_detection(
        image,
        max_side=detection_profile["max_side"],
    )
    display_image = resize_to_display(image)

    with INFERENCE_LOCK:
        detections = run_vehicle_detection(
            detection_image,
            confidence_threshold=settings.confidence_threshold,
            detection_profile=detection_profile,
        )

    display_detections = scale_detections_to_display(
        detections,
        source_shape=detection_image.shape,
        target_shape=display_image.shape,
    )

    selected_camera = None
    parking_lots = []
    strong_camera_candidate = None

    if settings.analysis_mode == "generic":
        parking_lots, stats = infer_generic_parking_layout(
            display_detections,
            display_image.shape,
            settings=settings,
        )
    else:
        selected_camera, parking_lots, stats = select_best_camera(
            image=display_image,
            detections=display_detections,
            requested_camera=requested_camera,
            overlap_threshold=DEFAULT_SLOT_OVERLAP_THRESHOLD,
            detection_area_threshold=DEFAULT_DETECTION_AREA_THRESHOLD,
        )

        if settings.analysis_mode == "auto" and requested_camera is None:
            strong_camera_candidate = (
                selected_camera if stats["slot_mode"] in {"fixed", "manual"} else None
            )

        if (
            requested_camera is not None
            and allow_requested_camera_fallback
            and display_detections
            and stats["matched_detections"] == 0
        ):
            parking_lots, generic_stats = infer_generic_parking_layout(
                display_detections,
                display_image.shape,
                settings=settings,
            )
            locked_camera_warning = (
                f"Locked Camera {requested_camera} did not align cleanly with this frame. "
            )
            if generic_stats["slot_mode"] != "vehicle_only":
                generic_stats["warning_message"] = (
                    locked_camera_warning + generic_stats["warning_message"]
                )
                stats = generic_stats
                selected_camera = None
            else:
                stats["warning_message"] = locked_camera_warning + generic_stats["warning_message"]
        elif settings.analysis_mode == "auto" and requested_camera is None:
            if stats["slot_mode"] == "vehicle_only":
                parking_lots, generic_stats = infer_generic_parking_layout(
                    display_detections,
                    display_image.shape,
                    settings=settings,
                )
                if generic_stats["slot_mode"] != "vehicle_only":
                    stats = generic_stats
                    selected_camera = None
                else:
                    stats["warning_message"] = generic_stats["warning_message"]

    stats = deepcopy(stats)
    stats.update(
        {
            "detections": len(display_detections),
            "input_format": input_format,
            "image_size": f"{display_image.shape[1]}x{display_image.shape[0]}",
            "selected_camera": selected_camera,
            "auto_profile": settings.analysis_mode == "auto" and requested_camera is None,
            "analysis_mode_requested": settings.analysis_mode,
            "analysis_mode_label": {
                "auto": "Auto",
                "fixed": "Fixed profile",
                "generic": "Generic estimate",
            }[settings.analysis_mode],
            "result_mode_label": describe_result_mode(
                stats["slot_mode"],
                settings.analysis_mode,
            ),
            "detection_size_requested": settings.detection_size,
            "detection_profile_label": detection_profile["display_label"],
            "confidence_threshold": round(settings.confidence_threshold, 2),
            "empty_sensitivity": int(round(settings.empty_sensitivity * 100.0)),
            "empty_sensitivity_label": describe_empty_sensitivity(
                settings.empty_sensitivity
            ),
            "infer_edge_slots": settings.infer_edge_slots,
            "camera_lock_candidate": strong_camera_candidate,
        }
    )

    if not display_detections:
        no_detection_message = (
            "No vehicles were detected in this image. Try a sharper or higher-resolution "
            "parking-lot image for better results."
        )
        if stats["warning_message"]:
            stats["warning_message"] = f"{stats['warning_message']} {no_detection_message}"
        else:
            stats["warning_message"] = no_detection_message

    annotated_image = render_annotated_result(
        display_image,
        parking_lots,
        stats,
        display_detections,
    )
    return FrameProcessingResult(
        input_image=display_image,
        output_image=annotated_image,
        parking_lots=parking_lots,
        detections=display_detections,
        stats=stats,
        selected_camera=selected_camera,
    )


def normalize_slot_mode_for_smoothing(slot_mode: str) -> str:
    """Treat auto-fixed and manually fixed layouts as the same smoothing mode."""
    if slot_mode in {"fixed", "manual"}:
        return "fixed"
    return slot_mode


def can_majority_vote_slots(results: list[FrameProcessingResult]) -> bool:
    """Return True when the recent results share identical slot identities."""
    if len(results) < 2:
        return False

    latest_result = results[-1]
    latest_slot_mode = normalize_slot_mode_for_smoothing(
        latest_result.stats.get("slot_mode", "")
    )
    latest_slots = latest_result.stats.get("slots", [])
    latest_slot_ids = [slot["slot_id"] for slot in latest_slots]
    if latest_slot_mode not in {"fixed", "estimated"} or not latest_slot_ids:
        return False

    for result in results:
        if (
            normalize_slot_mode_for_smoothing(result.stats.get("slot_mode", ""))
            != latest_slot_mode
        ):
            return False
        if result.selected_camera != latest_result.selected_camera:
            return False
        if [slot["slot_id"] for slot in result.stats.get("slots", [])] != latest_slot_ids:
            return False

    return True


def smooth_live_result(
    session: LiveStreamSession,
    result: FrameProcessingResult,
) -> FrameProcessingResult:
    """Apply short-history slot voting so live counts do not oscillate as much."""
    history = list(session.history) + [result]
    if not can_majority_vote_slots(history):
        return result

    latest_result = history[-1]
    smoothed_stats = deepcopy(latest_result.stats)
    smoothed_slots = []

    for slot_index, latest_slot in enumerate(latest_result.stats["slots"]):
        occupied_votes = sum(
            1 for item in history if item.stats["slots"][slot_index]["occupied"]
        )
        overlap_samples = [
            float(item.stats["slots"][slot_index].get("overlap_ratio", 0.0))
            for item in history
        ]
        slot_detail = deepcopy(latest_slot)
        slot_detail["occupied"] = occupied_votes >= ((len(history) // 2) + 1)
        slot_detail["overlap_ratio"] = round(sum(overlap_samples) / len(overlap_samples), 3)
        smoothed_slots.append(slot_detail)

    occupied_count = sum(1 for slot in smoothed_slots if slot["occupied"])
    total_count = len(smoothed_slots)
    smoothed_stats["slots"] = smoothed_slots
    smoothed_stats["occupied"] = occupied_count
    smoothed_stats["empty"] = total_count - occupied_count
    smoothed_stats["total"] = total_count
    smoothed_stats["occupancy_rate"] = round(
        occupied_count / max(1, total_count) * 100.0,
        1,
    )
    smoothed_output_image = render_annotated_result(
        latest_result.input_image,
        latest_result.parking_lots,
        smoothed_stats,
        latest_result.detections,
    )
    return FrameProcessingResult(
        input_image=latest_result.input_image,
        output_image=smoothed_output_image,
        parking_lots=latest_result.parking_lots,
        detections=latest_result.detections,
        stats=smoothed_stats,
        selected_camera=latest_result.selected_camera,
    )


def build_live_response_payload(
    result: FrameProcessingResult,
    *,
    camera_locked: bool,
) -> dict:
    """Encode a processed live frame for JSON transport."""
    stats = deepcopy(result.stats)
    stats["camera_locked"] = camera_locked
    if camera_locked and stats["slot_mode"] == "manual":
        stats["result_mode_label"] = "Live locked profile"

    return {
        "input_image": f"data:image/jpeg;base64,{encode_image_to_base64(result.input_image)}",
        "output_image": f"data:image/jpeg;base64,{encode_image_to_base64(result.output_image)}",
        "stats": stats,
    }


def update_live_camera_lock(
    session: LiveStreamSession,
    result: FrameProcessingResult,
) -> None:
    """Lock a live session to one camera after three strong auto matches."""
    if session.settings.analysis_mode != "auto" or session.requested_camera is not None:
        return

    candidate_camera = result.stats.get("camera_lock_candidate")
    with session.state_lock:
        if session.camera_locked:
            return

        if candidate_camera is None:
            session.last_candidate_camera = None
            session.consecutive_camera_matches = 0
            return

        if session.last_candidate_camera == candidate_camera:
            session.consecutive_camera_matches += 1
        else:
            session.last_candidate_camera = candidate_camera
            session.consecutive_camera_matches = 1

        if session.consecutive_camera_matches >= LIVE_CAMERA_LOCK_MATCH_COUNT:
            session.camera_locked = True
            session.locked_camera = candidate_camera


def open_video_stream(camera_url: str) -> cv2.VideoCapture | None:
    """Open an RTSP or HTTP stream with a light preference for FFmpeg."""
    capture = None
    try:
        if hasattr(cv2, "CAP_FFMPEG"):
            capture = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            capture = cv2.VideoCapture(camera_url)
        if capture is not None and capture.isOpened():
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture
    except Exception:
        if capture is not None:
            capture.release()
        raise

    if capture is not None:
        capture.release()
    return None


def stop_live_stream_session(session: LiveStreamSession) -> None:
    """Signal the live worker to stop and wait briefly for cleanup."""
    session.stop_event.set()
    worker = session.thread
    if worker is not None and worker.is_alive() and worker is not threading.current_thread():
        worker.join(timeout=5.0)
    with session.state_lock:
        session.status = "stopped"
        session.next_capture_at = None


def run_live_stream_session(session: LiveStreamSession) -> None:
    """Read a live stream and analyze the most recent frame every 20 seconds."""
    capture = None
    consecutive_failures = 0

    with session.state_lock:
        session.status = "connecting"
        session.next_capture_at = session.started_at

    try:
        while not session.stop_event.is_set():
            if capture is None or not capture.isOpened():
                capture = open_video_stream(session.camera_url)
                if capture is None or not capture.isOpened():
                    with session.state_lock:
                        session.status = "reconnecting"
                        session.error = (
                            "Unable to open the live camera stream. Retrying shortly."
                        )
                    time.sleep(LIVE_STREAM_RETRY_DELAY_SECONDS)
                    continue

                with session.state_lock:
                    session.status = "waiting_for_frame"
                    session.error = None
                consecutive_failures = 0

            success, frame = capture.read()
            current_time = utcnow()

            if success and frame is not None and frame.size > 0:
                consecutive_failures = 0
                with session.state_lock:
                    session.last_frame_at = current_time
                    capture_due = (
                        session.next_capture_at is None
                        or current_time >= session.next_capture_at
                    )
                    if session.last_capture_at is None:
                        session.status = "waiting_for_frame"
                    elif session.status != "processing":
                        session.status = "running"

                if not capture_due:
                    time.sleep(LIVE_STREAM_READ_SLEEP_SECONDS)
                    continue

                with session.state_lock:
                    session.status = "processing"

                effective_requested_camera = (
                    session.locked_camera if session.camera_locked else session.requested_camera
                )
                processed_result = process_frame(
                    frame,
                    requested_camera=effective_requested_camera,
                    settings=session.settings,
                    input_format="STREAM",
                    allow_requested_camera_fallback=session.camera_locked,
                )
                update_live_camera_lock(session, processed_result)
                smoothed_result = smooth_live_result(session, processed_result)
                live_payload = build_live_response_payload(
                    smoothed_result,
                    camera_locked=session.camera_locked,
                )

                with session.state_lock:
                    session.history.append(smoothed_result)
                    session.latest_result = live_payload
                    session.last_capture_at = current_time
                    session.next_capture_at = current_time + timedelta(
                        seconds=session.capture_interval_seconds
                    )
                    session.error = None
                    session.status = "running"
                continue

            consecutive_failures += 1
            with session.state_lock:
                session.status = "reconnecting"
                session.error = (
                    "The live stream dropped or stopped producing frames. Reconnecting."
                )

            if consecutive_failures >= LIVE_STREAM_REOPEN_AFTER_FAILURES:
                if capture is not None:
                    capture.release()
                capture = None
                consecutive_failures = 0
                time.sleep(LIVE_STREAM_RETRY_DELAY_SECONDS)
            else:
                time.sleep(LIVE_STREAM_READ_SLEEP_SECONDS)
    except Exception as exc:  # pragma: no cover - worker failure path
        traceback.print_exc()
        with session.state_lock:
            session.status = "error"
            session.error = f"Live analysis failed: {str(exc)}"
            session.next_capture_at = None
    finally:
        if capture is not None:
            capture.release()
        with session.state_lock:
            if session.stop_event.is_set():
                session.status = "stopped"
                session.next_capture_at = None


def process_image(
    image_bytes: bytes,
    requested_camera: int | None = None,
    settings: DetectionSettings | None = None,
) -> tuple[str, str, dict]:
    """Backwards-compatible bytes wrapper around the shared frame pipeline."""
    if settings is None:
        settings = DetectionSettings()

    image, input_format = decode_image(image_bytes)
    processed_result = process_frame(
        image,
        requested_camera=requested_camera,
        settings=settings,
        input_format=input_format,
    )
    return (
        encode_image_to_base64(processed_result.input_image),
        encode_image_to_base64(processed_result.output_image),
        processed_result.stats,
    )


PARKING_LOTS = {
    camera_number: load_parking_lots(camera_number)
    for camera_number in SUPPORTED_CAMERA_NUMBERS
}
CAMERA_REFERENCES = load_camera_references()
LIVE_SESSION_MANAGER = LiveStreamSessionManager()
print(
    "[INFO] Loaded camera references:",
    sum(len(reference_group) for reference_group in CAMERA_REFERENCES.values()),
)


# ── Routes ───────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    """Serve the frontend."""
    return send_from_directory("static", "index.html")


@app.route("/api/detect", methods=["POST"])
def detect():
    """Process uploaded image and return detection results."""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file uploaded."}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        try:
            requested_camera, settings = parse_detection_settings(request.form)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        image_bytes = file.read()
        print(
            "[DEBUG] File:",
            file.filename,
            "Content-Type:",
            file.content_type,
            "Size:",
            len(image_bytes),
            "Analysis mode:",
            settings.analysis_mode,
            "Requested camera:",
            requested_camera or "auto",
            "Detection size:",
            settings.detection_size,
            "Confidence:",
            round(settings.confidence_threshold, 2),
            "Empty sensitivity:",
            round(settings.empty_sensitivity, 2),
            "Infer edge slots:",
            settings.infer_edge_slots,
        )

        input_b64, output_b64, stats = process_image(
            image_bytes,
            requested_camera,
            settings=settings,
        )
        return jsonify(
            {
                "success": True,
                "input_image": f"data:image/jpeg;base64,{input_b64}",
                "output_image": f"data:image/jpeg;base64,{output_b64}",
                "stats": stats,
            }
        )

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:  # pragma: no cover - Flask error path
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(exc)}"}), 500


@app.route("/api/realtime/session", methods=["POST"])
def create_realtime_session():
    """Start a backend-managed live stream session."""
    try:
        payload = request.get_json(silent=True) or {}
        camera_url = str(payload.get("camera_url", "") or "").strip()
        if not camera_url:
            return jsonify({"error": "camera_url is required."}), 400

        try:
            requested_camera, settings = parse_detection_settings(payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        session = LIVE_SESSION_MANAGER.create(
            camera_url=camera_url,
            settings=settings,
            requested_camera=requested_camera,
        )
        return jsonify(
            {
                "session_id": session.session_id,
                "status": session.status,
                "capture_interval_seconds": session.capture_interval_seconds,
            }
        ), 201
    except Exception as exc:  # pragma: no cover - Flask error path
        traceback.print_exc()
        return jsonify({"error": f"Could not start live session: {str(exc)}"}), 500


@app.route("/api/realtime/session/<session_id>", methods=["GET"])
def get_realtime_session(session_id: str):
    """Return the latest live stream session state."""
    session = LIVE_SESSION_MANAGER.get(session_id)
    if session is None:
        return jsonify({"error": "Live session not found."}), 404
    return jsonify(session.snapshot())


@app.route("/api/realtime/session/<session_id>", methods=["DELETE"])
def delete_realtime_session(session_id: str):
    """Stop and remove a live stream session."""
    session = LIVE_SESSION_MANAGER.delete(session_id)
    if session is None:
        return jsonify({"error": "Live session not found."}), 404
    return jsonify({"success": True, "status": "stopped"})


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Parking Lot Occupancy Detection — Web Server")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
