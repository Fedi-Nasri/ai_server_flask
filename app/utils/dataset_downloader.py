"""
Utility module for handling dataset downloads.
"""

import os
import zipfile
from flask import send_file, make_response
from datetime import datetime

class DatasetDownloader:
    def __init__(self):
        self.static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
        self.datasets_dir = os.path.join(self.static_dir, 'datasets')
        self.models_dir = os.path.join(self.static_dir, 'models')
        
        # Create directories if they don't exist
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def create_zip(self, source_path, zip_name):
        """
        Create a zip file from the given source path.
        
        Args:
            source_path (str): Path to the directory to be zipped
            zip_name (str): Name of the output zip file
            
        Returns:
            str: Path to the created zip file
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source path {source_path} does not exist")
            
        zip_path = os.path.join(self.static_dir, zip_name)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(source_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_path)
                    zipf.write(file_path, arcname)
                    
        return zip_path

    def get_dataset(self, dataset_type):
        """
        Get the specified dataset as a zip file.
        
        Args:
            dataset_type (str): Type of dataset to download ('original', 'mission', or 'object-detection')
            
        Returns:
            Response: Flask response with the zip file
        """
        try:
            if dataset_type == 'original':
                source_path = os.path.join(self.datasets_dir, 'original')
                zip_name = f'original_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
            elif dataset_type == 'mission':
                source_path = os.path.join(self.datasets_dir, 'mission')
                zip_name = f'mission_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
            elif dataset_type == 'object-detection':
                source_path = self.models_dir
                zip_name = f'yolo_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
            else:
                raise ValueError(f"Invalid dataset type: {dataset_type}")
            
            zip_path = self.create_zip(source_path, zip_name)
            
            response = make_response(send_file(
                zip_path,
                mimetype='application/zip',
                as_attachment=True,
                download_name=zip_name
            ))
            
            # Add CORS headers
            response.headers['Access-Control-Allow-Origin'] = 'http://localhost:8080'
            response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            
            return response
            
        except Exception as e:
            raise Exception(f"Error creating dataset zip: {str(e)}") 