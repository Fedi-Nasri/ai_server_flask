"""
Video stream handler module for the Flask YOLOv11m server.

This module provides functionality for handling video streams from
different sources (local camera or external stream) and processing frames.
"""

import cv2
import time
import threading
import numpy as np
from app.config.config import DEFAULT_STREAM_SOURCE, FRAME_WIDTH, FRAME_HEIGHT, FPS

class VideoStream:
    """
    Video stream handler class for managing video input sources.
    
    This class handles capturing video from different sources,
    processing frames, and providing a consistent interface
    regardless of the source type.
    """
    
    def __init__(self, source=DEFAULT_STREAM_SOURCE):
        """
        Initialize the video stream handler.
        
        Args:
            source (str or int): Video source - URL for external stream or camera index (int) for local camera
        """
        self.source = source
        self.stream = None
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.fps = FPS
        self.last_frame_time = 0
        self.frame_count = 0
        self.actual_fps = 0
        
    def start(self):
        """
        Start the video stream in a separate thread.
        
        Returns:
            VideoStream: self reference for method chaining
        """
        # Convert source to int if it's a digit string (camera index)
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)
            
        # Initialize video capture
        self.stream = cv2.VideoCapture(self.source)
        
        if not self.stream.isOpened():
            raise ValueError(f"Could not open video source: {self.source}")
            
        # Set resolution if possible
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        # Start the thread to read frames
        self.stopped = False
        threading.Thread(target=self._update, daemon=True).start()
        return self
        
    def _update(self):
        """
        Update method that runs in a thread to continuously fetch frames.
        """
        last_fps_calc_time = time.time()
        
        while not self.stopped:
            # Read the next frame
            ret, frame = self.stream.read()
            
            if not ret:
                print(f"Error reading frame from source: {self.source}")
                self.stopped = True
                break
                
            # Calculate FPS
            current_time = time.time()
            self.frame_count += 1
            
            # Update FPS calculation every second
            if current_time - last_fps_calc_time >= 1.0:
                self.actual_fps = self.frame_count / (current_time - last_fps_calc_time)
                self.frame_count = 0
                last_fps_calc_time = current_time
                
            # Update the frame with thread safety
            with self.lock:
                self.frame = frame
                self.last_frame_time = current_time
                
            # Control the update rate to avoid excessive CPU usage
            time_to_sleep = max(0, 1.0/self.fps - (time.time() - current_time))
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
                
    def read(self):
        """
        Read the current frame with thread safety.
        
        Returns:
            numpy.ndarray or None: Current frame if available, None otherwise
        """
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()
            
    def stop(self):
        """
        Stop the video stream and release resources.
        """
        self.stopped = True
        if self.stream is not None:
            self.stream.release()
            
    def is_stopped(self):
        """
        Check if the stream is stopped.
        
        Returns:
            bool: True if stopped, False otherwise
        """
        return self.stopped
        
    def change_source(self, new_source):
        """
        Change the video source.
        
        Args:
            new_source (str or int): New video source - URL for external stream or camera index for local camera
            
        Returns:
            bool: True if source changed successfully, False otherwise
        """
        # Stop current stream
        self.stop()
        
        # Update source
        self.source = new_source
        
        # Restart with new source
        try:
            self.start()
            return True
        except Exception as e:
            print(f"Error changing video source: {e}")
            return False
            
    def get_fps(self):
        """
        Get the actual FPS of the video stream.
        
        Returns:
            float: Actual frames per second
        """
        return self.actual_fps
