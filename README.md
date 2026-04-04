# ParkVision AI — Parking Lot Occupancy Detection using YOLOv5

A computer vision system that detects empty and occupied parking slots from aerial/camera images using **YOLOv5** object detection. The web interface now focuses on a simple upload-or-paste workflow with automatic layout matching and instant annotated results.

<br>

## Abstract

This project presents a computer vision algorithm for parking lot occupancy detection. It uses **YOLOv5s**, a one-stage deep learning object detector (ONNX format), to detect vehicles in parking lot images. The detected vehicles are filtered to vehicle classes, deduplicated with non-max suppression, and then matched against predefined parking slot coordinates to determine which slots are **occupied** (red) and which are **empty** (green).

The system is tested on the **CNRPark** dataset — images of the CNR (National Research Council) parking lot in Pisa, Italy. The images span 9 camera viewpoints across different dates, times, and weather conditions (Sunny, Overcast, Rainy), including challenging scenarios with shadows and occlusions.

<br>

## Features

- 🌐 **Automatic Web Interface** — Upload, drag-and-drop, or paste parking lot images and start detection immediately
- 🧠 **Auto Layout Matching** — Chooses the closest supported camera profile automatically
- 🖼️ **Wide Image Support** — Handles common image formats including JPG, PNG, WEBP, TIFF, BMP, GIF, plus HEIC/HEIF through Pillow support
- 🤖 **Improved YOLOv5 Detection** — Vehicle-class filtering and non-max suppression for cleaner detections
- 📊 **Occupancy Statistics** — Total, empty, and occupied slot counts with occupancy rate
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
- `Pillow` — Broader image-format decoding and EXIF orientation handling
- `pillow-heif` — HEIC and HEIF decoding support

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

1. **Upload** a parking lot image by browsing, dragging-and-dropping, or pasting from the clipboard
2. **Wait** for the app to automatically decode the image and start inference
3. **Review** the auto-selected parking profile, vehicle count, and occupancy statistics
4. **Re-run** detection on the same image if needed with the "Analyze Again" button

The application is designed so users do not need to change manual values before detection starts.

<br>

## How It Works

1. **Image Input** — The uploaded image is decoded through Pillow/OpenCV, EXIF orientation is normalized, and the image is resized to 1000×750 pixels
2. **YOLOv5 Inference** — A letterboxed 640×640 input is passed through YOLOv5s on one or more image variants
3. **Vehicle Filtering** — Only vehicle classes are retained and duplicate detections are removed with non-max suppression
4. **Auto Layout Match** — The image is compared against the 9 supported parking-camera layouts and the best matching profile is selected automatically
5. **Slot Matching** — Each predefined parking slot is checked against vehicle detections using overlap and detection-area rules
6. **Output** — Annotated image with colored parking-slot boxes, selected profile, and occupancy statistics

The current slot overlays are still based on the 9 CNRPark camera layouts, so best results come from parking-lot views that are visually close to those supported angles.

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
