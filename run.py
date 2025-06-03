"""
Main entry point for the Flask YOLOv11m server.

This script initializes and starts the Flask server with YOLOv11m
object detection and tracking capabilities.
"""

from app import start_server

if __name__ == '__main__':
    # Start the Flask server
    start_server()
