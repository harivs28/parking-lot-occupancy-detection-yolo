"""
Parking Lot Occupancy Detection using YOLOv5 — Flask Web Application
Ported from parking_yolo.cpp by Daniele Ninni
"""

import base64
import csv
import os
import traceback
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
DETECTION_MAX_SIDE = 1920
TILE_SIZE = 960
TILE_OVERLAP_RATIO = 0.35
TILE_TRIGGER_DETECTION_COUNT = 8
TILE_CONFIDENCE_THRESHOLD = 0.12
LAYOUT_MIN_FEATURE_SCORE = 120.0
LAYOUT_MIN_FEATURE_RATIO = 3.0

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


# ── Load YOLO Model (once at startup) ───────────────────────────────────────
print(f"[INFO] Loading YOLOv5 model from: {YOLO_MODEL_PATH}")
ort_session = ort.InferenceSession(
    YOLO_MODEL_PATH,
    providers=["CPUExecutionProvider"],
)
input_name = ort_session.get_inputs()[0].name
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


def run_vehicle_detection(
    image: np.ndarray,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
) -> list[dict]:
    """Run YOLOv5 inference and return deduplicated vehicle detections."""
    all_detections = run_yolo_on_region(image, confidence_threshold)

    if max(image.shape[:2]) >= TILE_SIZE and len(all_detections) < TILE_TRIGGER_DETECTION_COUNT:
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
                "warning_message": "",
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


def process_image(image_bytes: bytes, requested_camera: int | None = None) -> tuple[str, str, dict]:
    """Full detection pipeline: image bytes → annotated image + stats."""
    image, input_format = decode_image(image_bytes)
    detection_image = resize_for_detection(image)
    display_image = resize_to_display(image)

    detections = run_vehicle_detection(detection_image)
    display_detections = scale_detections_to_display(
        detections,
        source_shape=detection_image.shape,
        target_shape=display_image.shape,
    )

    selected_camera, parking_lots, stats = select_best_camera(
        image=display_image,
        detections=display_detections,
        requested_camera=requested_camera,
        overlap_threshold=DEFAULT_SLOT_OVERLAP_THRESHOLD,
        detection_area_threshold=DEFAULT_DETECTION_AREA_THRESHOLD,
    )

    annotated_image = display_image.copy()
    if stats["layout_supported"]:
        annotated_image = draw_parking_lots(annotated_image, parking_lots, stats["slots"])
    annotated_image = draw_vehicle_detections(annotated_image, display_detections)
    annotated_image = draw_result_header(
        annotated_image,
        stats["camera_label"],
        (
            f"Vehicles detected: {len(display_detections)}"
            if stats["layout_supported"]
            else "Layout unsupported: showing vehicle detections only"
        ),
    )

    stats.update(
        {
            "detections": len(display_detections),
            "input_format": input_format,
            "image_size": f"{display_image.shape[1]}x{display_image.shape[0]}",
            "selected_camera": selected_camera,
            "auto_profile": requested_camera is None,
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

    return (
        encode_image_to_base64(display_image),
        encode_image_to_base64(annotated_image),
        stats,
    )


PARKING_LOTS = {
    camera_number: load_parking_lots(camera_number)
    for camera_number in SUPPORTED_CAMERA_NUMBERS
}
CAMERA_REFERENCES = load_camera_references()
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

        requested_camera = None
        raw_camera_value = request.form.get("camera", "auto").strip().lower()
        if raw_camera_value and raw_camera_value != "auto":
            requested_camera = int(raw_camera_value)
            if requested_camera not in SUPPORTED_CAMERA_NUMBERS:
                return jsonify({"error": "Camera number must be 1–9 or auto."}), 400

        image_bytes = file.read()
        print(
            "[DEBUG] File:",
            file.filename,
            "Content-Type:",
            file.content_type,
            "Size:",
            len(image_bytes),
            "Requested camera:",
            requested_camera or "auto",
        )

        input_b64, output_b64, stats = process_image(image_bytes, requested_camera)
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


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Parking Lot Occupancy Detection — Web Server")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
