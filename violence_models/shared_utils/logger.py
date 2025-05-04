import logging
import os
from datetime import datetime

class ViolenceDetectionLogger:
    def __init__(self):
        # Create logs directory if it doesn't exist
        self.logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger('ViolenceDetection')
        self.logger.setLevel(logging.DEBUG)
        
        # Create file handler
        log_file = os.path.join(self.logs_dir, f'violence_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_detection(self, frame_number, detections, confidence_scores):
        """Log detection results"""
        self.logger.info(f"Frame {frame_number}: Found {len(detections)} potential violent incidents")
        for i, (detection, score) in enumerate(zip(detections, confidence_scores)):
            self.logger.debug(f"Detection {i+1}: Box: {detection}, Confidence: {score:.2f}")
    
    def log_pose_analysis(self, frame_number, pose_features):
        """Log pose analysis results"""
        self.logger.debug(f"Frame {frame_number}: Pose Features - {pose_features}")
    
    def log_error(self, error_message):
        """Log error messages"""
        self.logger.error(f"Error: {error_message}")
    
    def log_model_performance(self, frame_count, processing_time):
        """Log model performance metrics"""
        fps = frame_count / processing_time if processing_time > 0 else 0
        self.logger.info(f"Performance: Processed {frame_count} frames in {processing_time:.2f} seconds ({fps:.2f} FPS)")

# Create a global logger instance
logger = ViolenceDetectionLogger()