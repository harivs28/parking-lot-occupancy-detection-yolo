"""
Parking Lot Occupancy Detection using YOLOv5 — Flask Web Application
Ported from parking_yolo.cpp by Daniele Ninni
"""

import os
import csv
import base64
import traceback
from io import BytesIO

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, send_from_directory

# ── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max upload

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
YOLO_MODEL_PATH = os.path.join(PROJECT_DIR, "yolov5s.onnx")
CSV_DIR = os.path.join(PROJECT_DIR, "results")
SAMPLE_IMAGES_DIR = os.path.join(PROJECT_DIR, "results", "FULL_IMAGE_1000x750")

# ── Constants (matching parking_yolo.cpp) ────────────────────────────────────
ORIGINAL_IMAGE_WIDTH = 2592.0
ORIGINAL_IMAGE_HEIGHT = 1944.0
DOWNSAMPLED_IMAGE_WIDTH = 1000.0
DOWNSAMPLED_IMAGE_HEIGHT = 750.0
BLOB_WIDTH = 640.0
BLOB_HEIGHT = 640.0

# Colors (BGR for OpenCV)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)

# ── Load YOLO Model (once at startup) ───────────────────────────────────────
print(f"[INFO] Loading YOLOv5 model from: {YOLO_MODEL_PATH}")
ort_session = ort.InferenceSession(
    YOLO_MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
input_name = ort_session.get_inputs()[0].name
print("[INFO] YOLOv5 model loaded successfully!")


def load_parking_lots(camera_number: int) -> list:
    """Load parking slot definitions from CSV for the given camera."""
    csv_path = os.path.join(CSV_DIR, f"camera{camera_number}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Camera CSV not found: {csv_path}")

    parking_lot_x_factor = DOWNSAMPLED_IMAGE_WIDTH / ORIGINAL_IMAGE_WIDTH
    parking_lot_y_factor = DOWNSAMPLED_IMAGE_HEIGHT / ORIGINAL_IMAGE_HEIGHT

    parking_lots = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slot_id = row["SlotId"].strip()
            x = int(float(row["X"]) * parking_lot_x_factor)
            y = int(float(row["Y"]) * parking_lot_y_factor)
            w = int(float(row["W"]) * parking_lot_x_factor)
            h = int(float(row["H"]) * parking_lot_y_factor)
            parking_lots.append({
                "slot_id": slot_id,
                "bbox": (x, y, w, h)
            })

    return parking_lots


def yolo_detect(blob: np.ndarray, confidence_threshold: float,
                blob_x_factor: float, blob_y_factor: float) -> list:
    """Run YOLOv5 inference and return list of detections."""
    # Run inference
    outputs = ort_session.run(None, {input_name: blob})
    output = outputs[0]  # shape: (1, 25200, 85)
    data = output[0]     # shape: (25200, 85)

    detections = []
    for i in range(data.shape[0]):
        confidence = data[i, 4]

        if confidence >= confidence_threshold:
            # Get bounding box coordinates
            cx = data[i, 0]
            cy = data[i, 1]
            w = data[i, 2]
            h = data[i, 3]

            # Convert to actual image coordinates
            left = int((cx - 0.5 * w) * blob_x_factor)
            top = int((cy - 0.5 * h) * blob_y_factor)
            width = int(w * blob_x_factor)
            height = int(h * blob_y_factor)

            detections.append({
                "confidence": float(confidence),
                "bbox": (left, top, width, height)
            })

    return detections


def compute_intersection_area(rect1, rect2):
    """Compute intersection area between two rectangles (x, y, w, h)."""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Compute intersection bounds
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0

    return (ix2 - ix1) * (iy2 - iy1)


def draw_parking_lots(image: np.ndarray, parking_lots: list,
                      detections: list, overlap_threshold: float,
                      detection_area_threshold: int) -> dict:
    """Draw parking lot annotations and return occupancy statistics."""
    occupied_count = 0
    empty_count = 0
    slot_details = []

    for lot in parking_lots:
        slot_bbox = lot["bbox"]
        slot_area = slot_bbox[2] * slot_bbox[3]  # w * h
        is_occupied = False

        for det in detections:
            det_bbox = det["bbox"]
            det_area = det_bbox[2] * det_bbox[3]

            intersection = compute_intersection_area(slot_bbox, det_bbox)

            # Match conditions from parking_yolo.cpp:
            # 1) intersection area >= overlap_threshold * slot area
            # 2) detection area < detection_area_threshold * slot area
            if (intersection >= overlap_threshold * slot_area and
                    det_area < detection_area_threshold * slot_area):
                is_occupied = True
                break

        # Draw rectangle
        x, y, w, h = slot_bbox
        if is_occupied:
            cv2.rectangle(image, (x, y), (x + w, y + h), RED, 1)
            occupied_count += 1
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), GREEN, 1)
            empty_count += 1

        # Draw slot ID
        cv2.putText(image, lot["slot_id"], (x, y),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.35, YELLOW, 1,
                    cv2.LINE_AA)

        slot_details.append({
            "slot_id": lot["slot_id"],
            "occupied": is_occupied
        })

    return {
        "total": occupied_count + empty_count,
        "occupied": occupied_count,
        "empty": empty_count,
        "occupancy_rate": round(
            occupied_count / max(1, occupied_count + empty_count) * 100, 1
        ),
        "slots": slot_details
    }


def process_image(image_bytes: bytes, camera_number: int,
                  confidence_pct: int = 1, overlap_pct: int = 50,
                  area_threshold: int = 5) -> tuple:
    """Full detection pipeline: image bytes → annotated image + stats."""
    # Decode image
    print(f"[DEBUG] Received image bytes: {len(image_bytes)} bytes")
    if len(image_bytes) == 0:
        raise ValueError("Uploaded file is empty (0 bytes).")

    nparr = np.frombuffer(image_bytes, np.uint8)
    print(f"[DEBUG] numpy array shape: {nparr.shape}, dtype: {nparr.dtype}")
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        # Try reading with different flags as fallback
        image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if image is not None:
            # Convert to BGR if needed (e.g. RGBA or grayscale)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError(
                f"Could not decode the uploaded image. "
                f"Received {len(image_bytes)} bytes. "
                f"First 4 bytes: {image_bytes[:4].hex() if len(image_bytes) >= 4 else 'N/A'}. "
                f"Make sure the file is a valid image (JPG, PNG, BMP)."
            )
    print(f"[DEBUG] Decoded image shape: {image.shape}")

    # Resize to standard dimensions (1000 x 750)
    image = cv2.resize(image, (int(DOWNSAMPLED_IMAGE_WIDTH),
                                int(DOWNSAMPLED_IMAGE_HEIGHT)))

    # Create blob for YOLOv5
    blob = cv2.dnn.blobFromImage(
        image, 1.0 / 255.0,
        (int(BLOB_WIDTH), int(BLOB_HEIGHT)),
        swapRB=True, crop=False
    )

    # Scaling factors
    blob_x_factor = image.shape[1] / BLOB_WIDTH
    blob_y_factor = image.shape[0] / BLOB_HEIGHT

    # Normalize thresholds
    confidence_threshold = confidence_pct / 100.0
    overlap_threshold = overlap_pct / 100.0

    # Run YOLOv5 detection
    detections = yolo_detect(blob, confidence_threshold,
                             blob_x_factor, blob_y_factor)

    # Load parking lots
    parking_lots = load_parking_lots(camera_number)

    # Draw results
    output_image = image.copy()
    stats = draw_parking_lots(output_image, parking_lots, detections,
                              overlap_threshold, area_threshold)

    # Encode output image to base64
    _, buffer = cv2.imencode(".jpg", output_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    output_b64 = base64.b64encode(buffer).decode("utf-8")

    # Also encode original (resized) input for side-by-side
    _, orig_buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    input_b64 = base64.b64encode(orig_buffer).decode("utf-8")

    return input_b64, output_b64, stats


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend."""
    return send_from_directory("static", "index.html")


@app.route("/api/detect", methods=["POST"])
def detect():
    """Process uploaded image and return detection results."""
    try:
        # Validate file upload
        if "image" not in request.files:
            return jsonify({"error": "No image file uploaded."}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        # Read parameters
        camera_number = int(request.form.get("camera", 5))
        confidence = int(request.form.get("confidence", 1))
        overlap = int(request.form.get("overlap", 50))
        area_threshold = int(request.form.get("area_threshold", 5))

        # Validate camera number
        if camera_number < 1 or camera_number > 9:
            return jsonify({"error": "Camera number must be 1–9."}), 400

        # Process
        image_bytes = file.read()
        print(f"[DEBUG] File: {file.filename}, Content-Type: {file.content_type}, Size: {len(image_bytes)} bytes")
        input_b64, output_b64, stats = process_image(
            image_bytes, camera_number, confidence, overlap, area_threshold
        )

        return jsonify({
            "success": True,
            "input_image": f"data:image/jpeg;base64,{input_b64}",
            "output_image": f"data:image/jpeg;base64,{output_b64}",
            "stats": stats
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return list of available sample images."""
    samples = []
    for weather in ["OVERCAST", "RAINY", "SUNNY"]:
        weather_dir = os.path.join(SAMPLE_IMAGES_DIR, weather)
        if not os.path.exists(weather_dir):
            continue
        for date_folder in sorted(os.listdir(weather_dir)):
            date_path = os.path.join(weather_dir, date_folder)
            if not os.path.isdir(date_path):
                continue
            for cam_folder in sorted(os.listdir(date_path)):
                cam_path = os.path.join(date_path, cam_folder)
                if not os.path.isdir(cam_path):
                    continue
                camera_num = cam_folder.replace("camera", "")
                for img_file in sorted(os.listdir(cam_path)):
                    if img_file.endswith(".jpg") and "_output" not in img_file:
                        samples.append({
                            "weather": weather,
                            "date": date_folder,
                            "camera": int(camera_num),
                            "filename": img_file,
                            "path": f"{weather}/{date_folder}/{cam_folder}/{img_file}"
                        })
    return jsonify({"samples": samples})


@app.route("/api/sample/<path:image_path>", methods=["GET"])
def get_sample_image(image_path):
    """Serve a sample image file."""
    # Normalize path separators for the OS
    image_path = image_path.replace("/", os.sep)
    full_path = os.path.join(SAMPLE_IMAGES_DIR, image_path)
    if not os.path.exists(full_path):
        return jsonify({"error": "Sample image not found."}), 404
    directory = os.path.dirname(full_path)
    filename = os.path.basename(full_path)
    return send_from_directory(directory, filename)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Parking Lot Occupancy Detection — Web Server")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
