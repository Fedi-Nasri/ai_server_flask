"""
Data storage module for the Flask YOLOv11m server.

This module provides functionality for storing detection data,
including Firebase integration (commented out) and local storage.
"""

import os
import json
import time
from datetime import datetime
import cv2
import numpy as np
from app.config.config import DETECTION_STORAGE_PATH, SAVE_DETECTIONS
# Firebase imports
import firebase_admin
from firebase_admin import credentials, db

class DetectionStorage:
    """
    Detection storage class for managing detection data.
    
    This class handles storing detection data locally and provides
    commented-out Firebase integration code for future implementation.
    """
    
    def __init__(self, storage_path=DETECTION_STORAGE_PATH):
        """
        Initialize the detection storage handler.
        
        Args:
            storage_path (str): Path to store detection images and labels
        """
        self.storage_path = storage_path
        self.ensure_storage_directory()
        
        # Track already seen objects to identify new detections
        self.seen_objects = set()
        
        # Initialize Firebase
        self.firebase_app = None
        self.db_ref = None
        self.initialize_firebase()
        
    def ensure_storage_directory(self):
        """
        Ensure the storage directory exists.
        """
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)
            print(f"Created detection storage directory: {self.storage_path}")
            
    def initialize_firebase(self):
        """
        Initialize Firebase connection.
        """
        try:
            # Initialize Firebase app with credentials
            cred_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'firebase-credentials.json')
            cred = credentials.Certificate(cred_path)
            
            # Initialize the app
            self.firebase_app = firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://oceancleaner-741db-default-rtdb.firebaseio.com/'
            })
            
            # Get reference to the database
            self.db_ref = db.reference('detections')
            print("Firebase initialized successfully")
        except Exception as e:
            print(f"Error initializing Firebase: {e}")
        
    def store_detection(self, detection, frame):
        """
        Store detection data locally and in Firebase (commented out).
        
        Args:
            detection (dict): Detection data including class, confidence, bbox, track_id, timestamp
            frame (numpy.ndarray): Frame containing the detection
            
        Returns:
            bool: True if this is a new detection, False otherwise
        """
        # Create a unique identifier for this object
        object_id = f"{detection['class']}_{detection['track_id']}"
        
        # Check if this is a new detection
        is_new_detection = object_id not in self.seen_objects
        
        # Store in Firebase
        self.store_to_firebase(detection)
        
        # Print detection data
        print(f"Detection: {detection['class']} (ID: {detection['track_id']}) "
              f"with confidence {detection['confidence']:.2f} at {datetime.fromtimestamp(detection['timestamp'])}")
        
        # Save image and label if this is a new detection and saving is enabled
        if is_new_detection and SAVE_DETECTIONS:
            self.save_detection_image(detection, frame)
            self.save_detection_label(detection, frame)
            self.seen_objects.add(object_id)
            
        return is_new_detection
        
    def store_to_firebase(self, detection):
        """
        Store detection data to Firebase statistics/wasteTypes node.
        Increments the count for the detected waste type (metal, glass, plastic).
        Args:
            detection (dict): Detection data
        """
        if self.db_ref is None:
            print("Firebase not initialized, skipping data storage")
            return
        try:
            # Determine waste type
            waste_type = detection.get('class', '').lower()
            if waste_type not in ['metal', 'glass', 'plastic']:
                print(f"Unknown waste type '{waste_type}', skipping Firebase update.")
                return
            # Get count (default to 1 if not provided)
            count = detection.get('count', 1)
            # Reference to statistics/wasteTypes/<waste_type>
            waste_type_ref = db.reference('statistics/wasteTypes').child(waste_type)
            # Transaction to increment the count atomically
            def increment_count(current):
                if current is None:
                    return count
                return current + count
            waste_type_ref.transaction(increment_count)
            print(f"Updated Firebase: Added {count} to {waste_type} count.")
        except Exception as e:
            print(f"Error updating statistics/wasteTypes in Firebase: {e}")
        
    def save_detection_image(self, detection, frame):
        """
        Save detection image to disk.
        
        Args:
            detection (dict): Detection data
            frame (numpy.ndarray): Frame containing the detection
        """
        try:
            # Create filename with timestamp and detection info
            timestamp = int(detection['timestamp'])
            filename = f"{timestamp}_{detection['class']}_{detection['track_id']}.jpg"
            filepath = os.path.join(self.storage_path, filename)
            
            # Save the image
            cv2.imwrite(filepath, frame)
            print(f"Saved detection image: {filepath}")
        except Exception as e:
            print(f"Error saving detection image: {e}")
            
    def save_detection_label(self, detection, frame):
        """
        Save detection label in YOLO format.
        
        YOLO format: <class_id> <x_center> <y_center> <width> <height>
        where x, y, width, height are normalized to [0, 1]
        
        Args:
            detection (dict): Detection data
            frame (numpy.ndarray): Frame containing the detection
        """
        try:
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = detection['bbox']
            
            # Convert to YOLO format (normalized)
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            bbox_width = (x2 - x1) / width
            bbox_height = (y2 - y1) / height
            
            # Assume class_id is 0 for simplicity (can be mapped to actual class IDs later)
            class_id = 0  # Placeholder, should be mapped to actual class ID
            
            # Create YOLO format label
            yolo_label = f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}"
            
            # Create filename with timestamp and detection info
            timestamp = int(detection['timestamp'])
            filename = f"{timestamp}_{detection['class']}_{detection['track_id']}.txt"
            filepath = os.path.join(self.storage_path, filename)
            
            # Save the label
            with open(filepath, 'w') as f:
                f.write(yolo_label)
                
            print(f"Saved detection label: {filepath}")
        except Exception as e:
            print(f"Error saving detection label: {e}")
