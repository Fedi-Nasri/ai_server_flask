"""
Main Flask application module for the YOLOv11m object detection server.

This module initializes the Flask application and provides routes for
video streaming, configuration, and detection visualization.
"""

import os
import cv2
import time
import threading
from flask import Flask, Response, render_template, request, jsonify
from flask_cors import CORS
import numpy as np

from app.utils.yolo_detector import YOLODetector
from app.utils.video_stream import VideoStream
from app.utils.detection_storage import DetectionStorage
from app.utils.dataset_downloader import DatasetDownloader
from app.config.config import HOST, PORT, DEFAULT_STREAM_SOURCE

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:8080"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Disposition"],
        "supports_credentials": True,
        "allow_credentials": True
    }
})


# Global variables
video_stream = None
yolo_detector = None
detection_storage = None
processing_thread = None
processing_frame = None
processing_lock = threading.Lock()
stream_active = False

def initialize_components():
    """
    Initialize all components required for the application.
    """
    global video_stream, yolo_detector, detection_storage
    
    # Initialize YOLOv11m detector
    yolo_detector = YOLODetector()
    
    # Initialize video stream
    video_stream = VideoStream(DEFAULT_STREAM_SOURCE).start()
    
    # Initialize detection storage
    detection_storage = DetectionStorage()
    
    print("All components initialized successfully")

def process_frames():
    """
    Process frames from the video stream in a separate thread.
    
    This function continuously reads frames from the video stream,
    applies object detection and tracking, and updates the global
    processing_frame variable with the processed frame.
    """
    global video_stream, yolo_detector, detection_storage, processing_frame, stream_active
    
    print("Frame processing thread started")
    
    while stream_active:
        # Read frame from video stream
        frame = video_stream.read()
        
        if frame is None:
            print("No frame available, waiting...")
            time.sleep(0.1)
            continue
        
        # Apply object detection and tracking
        processed_frame, detections = yolo_detector.detect_and_track(frame)
        
        # Process each detection
        for detection in detections:
            # Store detection data
            is_new = detection_storage.store_detection(detection, frame)
        
        # Add FPS information to the frame
        fps = video_stream.get_fps()
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Update the global processing frame with thread safety
        with processing_lock:
            processing_frame = processed_frame
            
        # Control the processing rate
        time.sleep(0.01)
    
    print("Frame processing thread stopped")

def generate_frames():
    """
    Generate frames for the video stream response.
    
    This function yields JPEG-encoded frames for the video stream
    response, ensuring that the frames are properly formatted for
    MJPEG streaming.
    
    Yields:
        bytes: JPEG-encoded frame data with multipart content type header
    """
    global processing_frame, processing_lock
    
    while True:
        # Get the current processing frame with thread safety
        with processing_lock:
            if processing_frame is None:
                # If no frame is available, yield a blank frame
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', blank_frame)
                frame_data = buffer.tobytes()
            else:
                # Encode the frame as JPEG
                _, buffer = cv2.imencode('.jpg', processing_frame)
                frame_data = buffer.tobytes()
        
        # Yield the frame with multipart content type header
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        
        # Control the frame rate
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    """
    Render the main page.
    
    Returns:
        str: Rendered HTML template
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Video streaming route.
    
    Returns:
        Response: Multipart response with video stream
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    """
    Start the video stream.
    
    Returns:
        dict: JSON response with status
    """
    global video_stream, processing_thread, stream_active
    
    # Check if stream is already active
    if stream_active:
        return jsonify({'status': 'error', 'message': 'Stream already active'})
    
    # Get stream source from request
    data = request.get_json()
    source = data.get('source', DEFAULT_STREAM_SOURCE)
    
    try:
        # Initialize or change video stream source
        if video_stream is None:
            video_stream = VideoStream(source).start()
        else:
            video_stream.change_source(source)
        
        # Start processing thread
        stream_active = True
        processing_thread = threading.Thread(target=process_frames, daemon=True)
        processing_thread.start()
        
        return jsonify({'status': 'success', 'message': f'Stream started with source: {source}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """
    Stop the video stream.
    
    Returns:
        dict: JSON response with status
    """
    global video_stream, stream_active
    
    # Check if stream is active
    if not stream_active:
        return jsonify({'status': 'error', 'message': 'No active stream to stop'})
    
    try:
        # Stop the stream
        stream_active = False
        
        # Wait for processing thread to stop
        if processing_thread is not None:
            processing_thread.join(timeout=1.0)
        
        # Stop video stream
        if video_stream is not None:
            video_stream.stop()
        
        return jsonify({'status': 'success', 'message': 'Stream stopped'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/status')
def status():
    """
    Get the current status of the system.
    
    Returns:
        dict: JSON response with status information
    """
    global video_stream, stream_active
    
    # Get stream status
    if video_stream is not None:
        fps = video_stream.get_fps()
        source = video_stream.source
        is_stopped = video_stream.is_stopped()
    else:
        fps = 0
        source = None
        is_stopped = True
    
    # Return status information
    return jsonify({
        'stream_active': stream_active,
        'fps': fps,
        'source': source,
        'is_stopped': is_stopped
    })

@app.route('/download-dataset/<dataset_type>')
def download_dataset(dataset_type):
    """
    Download a dataset as a zip file.
    
    Args:
        dataset_type (str): Type of dataset to download ('original', 'mission', or 'model')
        
    Returns:
        Response: Flask response with the zip file
    """
    try:
        downloader = DatasetDownloader()
        return downloader.get_dataset(dataset_type)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

def start_server():
    """
    Start the Flask server.
    """
    # Initialize components
    initialize_components()
    
    # Start the stream automatically
    global stream_active, processing_thread
    stream_active = True
    processing_thread = threading.Thread(target=process_frames, daemon=True)
    processing_thread.start()
    
    # Start the server
    app.run(host=HOST, port=PORT, debug=False, threaded=True)

if __name__ == '__main__':
    start_server()
