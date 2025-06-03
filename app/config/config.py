"""
Configuration module for the Flask YOLOv11m server.

This module contains all configurable parameters for the application,
making it easy to modify settings without changing the core code.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Application settings
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 5000))

# Video stream settings
DEFAULT_STREAM_SOURCE = os.getenv('STREAM_SOURCE', '0')  # 0 for webcam, URL for external stream
FRAME_WIDTH = int(os.getenv('FRAME_WIDTH', 640))
FRAME_HEIGHT = int(os.getenv('FRAME_HEIGHT', 480))
FPS = int(os.getenv('FPS', 30))

# YOLOv11m model settings
MODEL_PATH = os.getenv('MODEL_PATH', 'yolov11m.pt')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.5))
IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', 0.45))

# Firebase settings (for future implementation)
FIREBASE_CREDENTIALS = os.getenv('FIREBASE_CREDENTIALS', 'firebase-credentials.json')
FIREBASE_DATABASE_URL = os.getenv('FIREBASE_DATABASE_URL', 'https://your-project-id.firebaseio.com/')

# Detection storage settings
DETECTION_STORAGE_PATH = os.getenv('DETECTION_STORAGE_PATH', 'app/static/datasets/mission')
SAVE_DETECTIONS = os.getenv('SAVE_DETECTIONS', 'True').lower() == 'true'
