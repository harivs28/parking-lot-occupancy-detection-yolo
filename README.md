# ParkVision AI — Parking Lot Occupancy Detection using YOLOv5

A computer vision system that detects empty and occupied parking slots from aerial/camera images using **YOLOv5** object detection. Includes a modern web interface for uploading images and viewing results in real time.

<br>

## Abstract

This project presents a computer vision algorithm for parking lot occupancy detection. It uses **YOLOv5s**, a one-stage deep learning object detector (ONNX format), to detect vehicles in parking lot images. The detected vehicles are then matched against predefined parking slot coordinates to determine which slots are **occupied** (red) and which are **empty** (green).

The system is tested on the **CNRPark** dataset — images of the CNR (National Research Council) parking lot in Pisa, Italy. The images span 9 camera viewpoints across different dates, times, and weather conditions (Sunny, Overcast, Rainy), including challenging scenarios with shadows and occlusions.

<br>

## Features

- 🌐 **Web Interface** — Upload parking lot images via drag-and-drop and get instant results
- 🤖 **YOLOv5 Detection** — Real-time object detection using YOLOv5s ONNX model
- 📊 **Occupancy Statistics** — Total, empty, and occupied slot counts with occupancy rate
- 🎛️ **Adjustable Parameters** — Confidence threshold, overlap threshold, and detection area controls
- 🖼️ **Sample Gallery** — Pre-loaded images from 9 cameras across 3 weather conditions
- 📱 **Responsive Design** — Works on desktop and mobile browsers

<br>

## Project Structure

```
parking-lot-occupancy-detection-yolo/
├── README.md                   # This file
├── yolov5s.onnx                # YOLOv5s model (ONNX format)
├── parking_yolo.cpp            # Original C++ implementation (reference)
├── report.pdf                  # Project report
├── results/
│   ├── camera1.csv … camera9.csv   # Parking slot coordinates per camera
│   └── FULL_IMAGE_1000x750/        # Sample images & outputs
│       ├── OVERCAST/
│       ├── RAINY/
│       └── SUNNY/
└── website/                    # Web application
    ├── app.py                  # Flask backend (Python)
    ├── requirements.txt        # Python dependencies
    └── static/
        ├── index.html          # Frontend UI
        ├── style.css           # Dark glassmorphism design system
        └── app.js              # Client-side logic
```

<br>

## How to Run

### Prerequisites

- **Python 3.9+** installed on your system
- **pip** package manager

### 1. Install Dependencies

```bash
cd parking-lot-occupancy-detection-yolo
pip install -r website/requirements.txt
```

This installs:
- `flask` — Web server
- `opencv-python-headless` — Image processing
- `onnxruntime` — YOLOv5 ONNX model inference
- `numpy` — Numerical operations

### 2. Start the Server

```bash
python website/app.py
```

You should see:
```
[INFO] Loading YOLOv5 model from: .../yolov5s.onnx
[INFO] YOLOv5 model loaded successfully!

============================================================
  Parking Lot Occupancy Detection — Web Server
  Open http://localhost:5000 in your browser
============================================================
```

### 3. Open in Browser

Navigate to **http://localhost:5000** and:

1. **Upload** a parking lot image (drag-and-drop or click to browse)
2. **Select** the camera profile (1–9) matching your image's viewpoint
3. **Adjust** detection thresholds if needed (defaults work well)
4. **Click** "Detect Parking Slots"
5. **View** the annotated result with occupancy statistics

You can also click any **sample image** from the gallery to quickly test the system.

<br>

## Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Confidence Threshold | 1% | Minimum YOLOv5 detection confidence |
| Overlap Threshold | 50% | Minimum intersection-over-slot-area to mark as occupied |
| Detection Area | 5x | Maximum detection-to-slot area ratio |

<br>

## How It Works

1. **Image Input** — The uploaded image is resized to 1000×750 pixels
2. **YOLOv5 Inference** — A 640×640 blob is passed through YOLOv5s to detect all vehicles
3. **Slot Matching** — Each predefined parking slot (from CSV) is checked against detections:
   - If `intersection area ≥ overlap_threshold × slot area` AND `detection area < area_threshold × slot area` → **Occupied** (red box)
   - Otherwise → **Empty** (green box)
4. **Output** — Annotated image with colored bounding boxes and yellow slot IDs, plus statistics

<br>

## Results

|   |   |   |
|---|---|---|
![](results/FULL_IMAGE_1000x750/OVERCAST/2015-11-16/camera1/2015-11-16_0910_output.jpg) | ![](results/FULL_IMAGE_1000x750/OVERCAST/2015-11-29/camera2/2015-11-29_1614_output.jpg) | ![](results/FULL_IMAGE_1000x750/OVERCAST/2015-12-19/camera3/2015-12-19_1248_output.jpg)
![](results/FULL_IMAGE_1000x750/RAINY/2015-12-22/camera4/2015-12-22_0951_output.jpg) | ![](results/FULL_IMAGE_1000x750/RAINY/2016-01-09/camera5/2016-01-09_0927_output.jpg) | ![](results/FULL_IMAGE_1000x750/RAINY/2016-02-12/camera6/2016-02-12_1654_output.jpg)
![](results/FULL_IMAGE_1000x750/SUNNY/2015-11-12/camera7/2015-11-12_1647_output.jpg) | ![](results/FULL_IMAGE_1000x750/SUNNY/2015-12-17/camera8/2015-12-17_0941_output.jpg) | ![](results/FULL_IMAGE_1000x750/SUNNY/2016-01-16/camera9/2016-01-16_0940_output.jpg)

<br>

## Tech Stack

- **Backend**: Python, Flask, OpenCV, ONNX Runtime
- **Frontend**: HTML5, CSS3, JavaScript (vanilla)
- **Model**: YOLOv5s (ONNX format, ~29 MB)
- **Dataset**: CNRPark (CNR, Pisa)

<br>

## Credits

- Original C++ implementation by Daniele Ninni
- Based on the CNRPark dataset
- Computer Vision Project — University of Padua, A.Y. 2021/22

<p align="center">
  <img src="https://user-images.githubusercontent.com/62724611/166108149-7629a341-bbca-4a3e-8195-67f469a0cc08.png" alt="" height="70"/>
</p>