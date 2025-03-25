import cv2
import numpy as np
import time
import os
import threading
from flask import Flask, render_template, Response

# Ensure required modules are installed
try:
    import torch
    from ultralytics import YOLO
    from deep_sort.deep_sort import DeepSort
except ModuleNotFoundError as e:
    print("Required modules not found. Please install dependencies using: \n")
    print("pip install torch torchvision torchaudio ultralytics opencv-python numpy flask")
    raise e

# Load YOLOv8 model (Optimized for Speed)
yolo_model = YOLO("yolov8n.pt")  # Using YOLOv8 nano for efficiency
deep_sort = DeepSort("osnet_x0_25")  # DeepSORT tracker

# Flask app for live dashboard
app = Flask(__name__)

# Use CCTV Camera (RTSP) or Video File
VIDEO_SOURCE = "traffic_video.mp4"  # Change to RTSP link for live feed
cap = cv2.VideoCapture(VIDEO_SOURCE)
fps = cap.get(cv2.CAP_PROP_FPS)

# Violation Detection Parameters
violation_zones = {
    "red_light": [(100, 300), (500, 350)],  # Adjust as per intersection
    "wrong_lane": [(200, 400), (600, 450)]  # Define wrong lane areas
}

def detect_violations(detections):
    """Detect red light violations and wrong lane driving."""
    violations = {"red_light": [], "wrong_lane": []}
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        for zone, (start, end) in violation_zones.items():
            if start[0] < center_x < end[0] and start[1] < center_y < end[1]:
                violations[zone].append((x1, y1, x2, y2))

    return violations

def generate_frames():
    """Process video and return frames for web streaming."""
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model
        results = yolo_model(frame)
        detections = []

        for result in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2, conf, cls = result
            detections.append([x1, y1, x2, y2, conf, int(cls)])

        # DeepSORT tracking
        tracker_outputs = deep_sort.update(np.array(detections), frame)

        # Draw bounding boxes & IDs
        for track in tracker_outputs:
            x1, y1, x2, y2, track_id = track
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Detect traffic violations
        violations = detect_violations(detections)
        colors = {"red_light": (0, 0, 255), "wrong_lane": (255, 0, 0)}

        for category, color in colors.items():
            for (x1, y1, x2, y2) in violations[category]:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, category.replace("_", " ").upper(), (int(x1), int(y1)-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    threading.Thread(target=app.run, kwargs={'debug': True, 'use_reloader': False}).start()
