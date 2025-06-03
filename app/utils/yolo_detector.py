"""
Utility module for YOLOv11m model loading and inference.

This module provides functions for loading the YOLOv11m model,
performing object detection, and tracking detected objects.
"""

import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
from app.config.config import MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD

class YOLODetector:
    """
    YOLOv11m detector class for object detection and tracking.
    
    This class handles loading the YOLOv11m model, performing inference
    on frames, and tracking objects across multiple frames.
    """
    
    def __init__(self, model_path=MODEL_PATH, conf_threshold=CONFIDENCE_THRESHOLD, 
                 iou_threshold=IOU_THRESHOLD):
        """
        Initialize the YOLOv11m detector.
        
        Args:
            model_path (str): Path to the YOLOv11m model weights
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IOU threshold for non-maximum suppression
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load the model
        try:
            self.model = YOLO(model_path)
            print(f"YOLOv11m model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading YOLOv11m model: {e}")
            # Fallback to a default model if available
            try:
                self.model = YOLO("yolov8n.pt")  # Fallback to YOLOv8n if YOLOv11m fails
                print("Fallback to YOLOv8n model")
            except Exception as e:
                print(f"Error loading fallback model: {e}")
                self.model = None
        
        # Initialize tracker
        self.tracker = None
        self.track_history = {}  # Dictionary to store tracking history
        
    def detect_and_track(self, frame):
        """
        Perform object detection and tracking on a frame.
        
        Args:
            frame (numpy.ndarray): Input frame for detection
            
        Returns:
            tuple: (processed_frame, detections)
                - processed_frame: Frame with bounding boxes and tracking info
                - detections: List of detection results with class, confidence, bbox, and tracking ID
        """
        if self.model is None:
            return frame, []
        
        # Perform detection with tracking
        results = self.model.track(frame, persist=True, conf=self.conf_threshold, 
                                  iou=self.iou_threshold, verbose=False)
        
        # Process results
        detections = []
        processed_frame = frame.copy()
        
        if results and len(results) > 0:
            # Get the first result (assuming single image input)
            result = results[0]
            
            # Check if tracking IDs are available
            if hasattr(result, 'boxes') and hasattr(result.boxes, 'id') and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().numpy()
                
                # Get class names
                class_names = result.names
                
                # Process each detection
                for box, cls_id, conf, track_id in zip(boxes, classes, confidences, track_ids):
                    x1, y1, x2, y2 = box.astype(int)
                    class_name = class_names[int(cls_id)]
                    
                    # Store detection info
                    detection = {
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': (x1, y1, x2, y2),
                        'track_id': int(track_id),
                        'timestamp': time.time()
                    }
                    detections.append(detection)
                    
                    # Draw bounding box and label
                    color = self._get_color(int(track_id))
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with class name, confidence, and tracking ID
                    label = f"{class_name} {conf:.2f} ID:{track_id}"
                    cv2.putText(processed_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Update tracking history
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    
                    # Add center point to track history
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    self.track_history[track_id].append(center)
                    
                    # Limit history length to avoid memory issues
                    if len(self.track_history[track_id]) > 30:
                        self.track_history[track_id].pop(0)
                    
                    # Draw tracking lines
                    if len(self.track_history[track_id]) > 1:
                        for i in range(1, len(self.track_history[track_id])):
                            cv2.line(processed_frame, 
                                    self.track_history[track_id][i-1],
                                    self.track_history[track_id][i], 
                                    color, 2)
            else:
                # Fallback to standard detection if tracking is not available
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_names = result.names
                
                for i, (box, cls_id, conf) in enumerate(zip(boxes, classes, confidences)):
                    x1, y1, x2, y2 = box.astype(int)
                    class_name = class_names[int(cls_id)]
                    
                    # Store detection info without tracking ID
                    detection = {
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': (x1, y1, x2, y2),
                        'track_id': i,  # Use index as placeholder
                        'timestamp': time.time()
                    }
                    detections.append(detection)
                    
                    # Draw bounding box and label
                    color = self._get_color(i)
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with class name and confidence
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(processed_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return processed_frame, detections
    
    def _get_color(self, idx):
        """
        Generate a consistent color based on object ID.
        
        Args:
            idx (int): Object ID or index
            
        Returns:
            tuple: BGR color tuple
        """
        idx = abs(int(idx)) * 3
        return ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
