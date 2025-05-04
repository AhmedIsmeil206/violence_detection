import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import os
from pathlib import Path
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoseProcessor:
    def __init__(self, model_path='yolov8x-pose.pt'):
        """Initialize pose processor with model path."""
        if isinstance(model_path, str) and not os.path.isabs(model_path):
            current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            model_path = current_dir.parent / model_path

        self.pose_model = YOLO(str(model_path))
        self.pose_model.to('cpu')  # Ensure model is on CPU
        self.motion_history = deque(maxlen=10)
        self.violence_threshold = 15.0

    def process_frame(self, frame):
        """Process a frame to detect poses."""
        try:
            # Run pose detection
            results = self.pose_model(frame, verbose=False)
            
            if not results or not results[0].keypoints:
                return []
                
            # Convert keypoints to CPU numpy arrays
            keypoints = results[0].keypoints.data
            if isinstance(keypoints, torch.Tensor):
                keypoints = keypoints.cpu().numpy()
            
            # Process each person's keypoints
            processed_results = []
            for person_kpts in keypoints:
                height = self._estimate_person_height(person_kpts)
                distance = self._categorize_distance(height, frame.shape[0])
                processed_results.append({
                    'keypoints': person_kpts,
                    'distance': distance,
                    'height': height,
                    'motion_score': self._calculate_motion_score(person_kpts)
                })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return []

    def _estimate_person_height(self, keypoints):
        """Estimate person height from keypoints."""
        if len(keypoints) < 17:  # COCO format has 17 keypoints
            return 0
            
        # Get nose and ankle points
        nose = keypoints[0]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        if nose is not None and (left_ankle is not None or right_ankle is not None):
            ankle = left_ankle if left_ankle is not None else right_ankle
            return float(abs(nose[1] - ankle[1]))
        return 0

    def _categorize_distance(self, height, frame_height):
        """Categorize distance based on relative height."""
        if frame_height == 0:
            return 'unknown'
            
        ratio = height / frame_height
        if ratio > 0.4:
            return 'near'
        elif ratio > 0.2:
            return 'medium'
        else:
            return 'far'

    def _calculate_motion_score(self, keypoints):
        """Calculate motion score for violence detection."""
        if len(self.motion_history) == 0:
            self.motion_history.append(keypoints)
            return 0
            
        prev_keypoints = self.motion_history[-1]
        motion_score = 0
        
        # Calculate motion between frames
        for curr, prev in zip(keypoints, prev_keypoints):
            if curr is not None and prev is not None:
                motion = np.sqrt(((curr - prev) ** 2).sum())
                motion_score = max(motion_score, motion)
        
        self.motion_history.append(keypoints)
        return float(motion_score)