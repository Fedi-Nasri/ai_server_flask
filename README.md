<<<<<<< HEAD
# Flask YOLOv11m Object Detection Server

A modular and well-documented Flask server for real-time object detection and tracking using YOLOv11m.

## Features

- **YOLOv11m Integration**: Real-time object detection and tracking
- **Video Streaming**: Support for both external video streams and local camera input
- **Object Detection & Tracking**: Detect objects, classify them, and track them across frames
- **Data Storage**: Store detection data with Firebase integration (commented out for manual implementation)
- **Dataset Expansion**: Automatically save detection images and YOLO-format labels for future training

## Project Structure

```
flask_yolo_server/
├── app/
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── static/
│   │   └── detections/
│   ├── templates/
│   │   └── index.html
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── detection_storage.py
│   │   ├── video_stream.py
│   │   └── yolo_detector.py
│   └── __init__.py
├── requirements.txt
├── run.py
└── todo.md
```

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the YOLOv11m model weights and place them in the project directory or configure the path in the `.env` file.

## Configuration

The application can be configured using environment variables or a `.env` file:

- `DEBUG`: Enable/disable debug mode (default: True)
- `HOST`: Host to bind the server to (default: 0.0.0.0)
- `PORT`: Port to run the server on (default: 5000)
- `STREAM_SOURCE`: Default video stream source (default: 0 for webcam)
- `FRAME_WIDTH`: Frame width for video processing (default: 640)
- `FRAME_HEIGHT`: Frame height for video processing (default: 480)
- `FPS`: Target frames per second (default: 30)
- `MODEL_PATH`: Path to the YOLOv11m model weights (default: yolov11m.pt)
- `CONFIDENCE_THRESHOLD`: Confidence threshold for detections (default: 0.5)
- `IOU_THRESHOLD`: IOU threshold for non-maximum suppression (default: 0.45)
- `FIREBASE_CREDENTIALS`: Path to Firebase credentials JSON file (default: firebase-credentials.json)
- `FIREBASE_DATABASE_URL`: Firebase database URL (default: https://your-project-id.firebaseio.com/)
- `DETECTION_STORAGE_PATH`: Path to store detection images and labels (default: app/static/detections)
- `SAVE_DETECTIONS`: Enable/disable saving detection images and labels (default: True)

## Usage

1. Start the server:

```bash
python run.py
```

2. Open a web browser and navigate to `http://localhost:5000`

3. Select a video source (local camera or external stream) and click "Start Stream"

## Firebase Integration

The code includes commented-out Firebase integration. To enable it:

1. Create a Firebase project and obtain credentials
2. Place the credentials JSON file in the `app/config` directory
3. Update the `FIREBASE_DATABASE_URL` in the configuration
4. Uncomment the Firebase integration code in `app/utils/detection_storage.py`

## Extending the Project

The modular design makes it easy to extend the project:

- Add new detection models by creating a new detector class in `app/utils`
- Implement additional storage backends by extending the storage module
- Add new visualization features by modifying the templates and static files

## License

This project is licensed under the MIT License - see the LICENSE file for details.
=======
# ai_server_flask
>>>>>>> d58f313b970b7b46d4b80d5e378489ab60b9e66f
