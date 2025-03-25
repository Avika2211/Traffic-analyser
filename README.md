# Traffic Violation - Avika Joshi

A computer vision system to detect traffic violations like red light jumping and wrong lane driving using YOLOv8 and DeepSORT.

## Features
- Real-time vehicle detection using YOLOv8
- Vehicle tracking using DeepSORT
- Red light violation detection
- Wrong lane driving detection
- Web dashboard using Flask
- Core Technologies

## Tech Stack
Python - Primary programming language (v3.6+ recommended)

OpenCV (cv2) - Real-time video processing and computer vision operations

PyTorch - Deep learning framework that powers YOLOv8

Computer Vision & AI Components
YOLOv8 (Ultralytics) - For object detection (vehicles, pedestrians, etc.)

Using yolov8n.pt (nano version) for optimal performance

DeepSORT - For object tracking across video frames

Using OSNet (osnet_x0_25) as the ReID model

Backend & Web Interface
Flask - Lightweight web framework for creating the dashboard

Jinja2 - Templating engine for Flask (used in index.html)

Supporting Libraries
NumPy - Numerical operations and array handling

Threading - For running Flask server alongside video processing

Development & Deployment
Git/GitHub - Version control and code hosting

PIP - Package management (requirements.txt)

Optional/Recommended Extensions
Git LFS - For managing large model files (like yolov8n.pt)

Docker - For containerized deployment (can be added later)

SQLite/PostgreSQL - For violation logging (future enhancement)
