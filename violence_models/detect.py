import cv2
import numpy as np
import torch
from ultralytics import YOLO
import logging
from deep_sort import nn_matching

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from violence_models.pose_estimation.pose_processor import PoseProcessor
from collections import deque
import os

class ViolenceDetector:
    def __init__(self, model_paths=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        if model_paths is None:
            # Use the specified trained model
            model_paths = [os.path.join(current_dir, 'datasets/violence/violence_detector_best.pt')]
        
        # Verify model files exist
        for path in model_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load YOLO model for violence detection
        self.model = YOLO(model_paths[0])
        self.class_names = ['normal', 'violence']
        self.violence_threshold = 0.25  # Lowered threshold for better long-distance detection
        self.pose_processor = PoseProcessor(os.path.join(current_dir, 'yolov8x-pose.pt'))
        self.temporal_window = deque(maxlen=5)
        self.min_pose_confidence = 0.5
        self.distance_confidence_adjustments = {
            'near': 1.0,
            'medium': 0.85,
            'far': 0.7
        }    
    def _ensemble_predictions(self, frame, results_list):
        """Combine predictions from multiple models using weighted averaging"""
        combined_boxes = []
        
        # Collect all predictions
        all_predictions = []
        for model_idx, results in enumerate(results_list):
            if hasattr(results, 'boxes') and len(results.boxes) > 0:
                boxes_data = results.boxes.data
                if boxes_data is not None and len(boxes_data) > 0:
                    for box_data in boxes_data:
                        conf = float(box_data[4].cpu().numpy() if box_data.is_cuda else box_data[4].numpy())
                        if conf > self.violence_threshold:
                            # Get coordinates using tensor data
                            coords = box_data[:4].cpu().numpy() if box_data.is_cuda else box_data[:4].numpy()
                            x1, y1, x2, y2 = map(int, coords)
                            # Give more weight to the primary model
                            weight = 0.7 if model_idx == 0 else 0.3
                            all_predictions.append((x1, y1, x2, y2, conf * weight))
                        # Give more weight to the primary (larger) model
                        weight = 0.7 if model_idx == 0 else 0.3
                        all_predictions.append((x1, y1, x2, y2, conf * weight))
            
            # Non-maximum suppression for overlapping boxes
            while all_predictions:
                best_pred = max(all_predictions, key=lambda x: x[4])
                combined_boxes.append(best_pred[:4])
                all_predictions = [
                    pred for pred in all_predictions
                    if self._calculate_iou(best_pred[:4], pred[:4]) < 0.5
                ]
            
            return combined_boxes
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _validate_with_pose(self, frame, box, pose_results):
        """Validate violence detection using pose information with distance awareness"""
        if not pose_results:
            return True
            
        x1, y1, x2, y2 = box
        roi_height = y2 - y1
        frame_height = frame.shape[0]
        
        # Determine distance category based on relative size
        height_ratio = roi_height / frame_height
        if height_ratio > 0.4:
            distance = 'near'
        elif height_ratio > 0.2:
            distance = 'medium'
        else:
            distance = 'far'
        
        violence_detected = False
        for pose_result in pose_results:
            keypoints = pose_result['keypoints']
            pose_distance = pose_result['distance']
            
            # Adjust confidence threshold based on distance
            confidence_adjustment = self.distance_confidence_adjustments[pose_distance]
            adjusted_threshold = self.min_pose_confidence * confidence_adjustment
            
            # Calculate motion dynamics with distance-aware thresholds
            wrist_velocity = self._calculate_keypoint_velocity(keypoints, [9, 10])  # Wrist points
            elbow_velocity = self._calculate_keypoint_velocity(keypoints, [7, 8])   # Elbow points
            
            # Adjust velocity thresholds based on distance
            velocity_multiplier = 1.0
            if distance == 'medium':
                velocity_multiplier = 1.3
            elif distance == 'far':
                velocity_multiplier = 1.5
            
            # Check for violent motion with distance-adjusted thresholds
            if (wrist_velocity > 15 * velocity_multiplier or 
                elbow_velocity > 12 * velocity_multiplier):
                violence_detected = True
                break
        
        return violence_detected
    
    def _calculate_keypoint_velocity(self, keypoints, indices):
        """Calculate velocity of keypoints with distance consideration"""
        total_velocity = 0
        valid_points = 0
        
        for idx in indices:
            if idx < len(keypoints) and keypoints[idx] is not None:
                kp = keypoints[idx]
                if kp.sum() > 0:  # Check if keypoint is valid
                    velocity = np.sqrt(kp[0]**2 + kp[1]**2)
                    total_velocity += velocity
                    valid_points += 1
        
        return total_velocity / valid_points if valid_points > 0 else 0

    def detect_violence(self, frame, conf_threshold=None):
        """
        Detect violence in a frame using YOLO model.
        Args:
            frame: Input image
            conf_threshold: Optional confidence threshold override (default: self.violence_threshold)
        Returns: 
            list of [x1, y1, x2, y2, confidence, class_id] for each detection
        """
        try:
            # Use provided threshold or default
            threshold = conf_threshold if conf_threshold is not None else self.violence_threshold
            
            # Run inference with YOLO model with increased size
            results = self.model(frame, imgsz=1280, verbose=False)[0]
            detections = []
            
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.data
                for box in boxes:
                    # Get box coordinates, confidence and class
                    x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                    class_id = int(cls)
                    
                    # Apply stricter filtering for false positives
                    # For knife and weapons, require higher confidence
                    if class_id in [0, 3]:  # knife or weapons
                        min_required_conf = 0.45  # Higher threshold for weapons
                        
                        # Calculate object area ratio (to filter out very small detections)
                        obj_width = x2 - x1
                        obj_height = y2 - y1
                        obj_area = obj_width * obj_height
                        frame_area = frame.shape[0] * frame.shape[1]
                        area_ratio = obj_area / frame_area
                        
                        # Filter out very small or very large detections
                        if area_ratio < 0.001 or area_ratio > 0.5:
                            continue
                            
                        # Filter by aspect ratio (weapons/knives shouldn't be too square)
                        aspect_ratio = obj_width / max(obj_height, 1)
                        if 0.7 < aspect_ratio < 1.3:  # Too square, likely a false positive
                            min_required_conf = 0.65  # Require even higher confidence for square objects
                            
                        # Only keep high-confidence detections for weapons/knives
                        if conf < min_required_conf:
                            continue
                    
                    # Only include detections above threshold
                    if conf > threshold:
                        detections.append([
                            int(x1), int(y1), int(x2), int(y2),
                            float(conf),
                            class_id
                        ])
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in violence detection: {str(e)}")
            return []

class ViolenceTypeDetector:
    def __init__(self, model_path=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        if model_path is None:
            # Use the specified trained model
            model_path = os.path.join(current_dir, 'datasets/type/type_detector_best.pt')
        
        # Verify model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load YOLO model for violence type detection
        self.model = YOLO(model_path)
        self.class_names = ['grenade', 'handgun', 'knife', 'theft mask']
        self.type_threshold = 0.35  # Default threshold for violence type detection
    
    def detect_violence_type(self, frame, conf_threshold=None):
        """
        Detect violence type in a frame using YOLO model.
        Args:
            frame: Input image
            conf_threshold: Optional confidence threshold override (default: self.type_threshold)
        Returns: 
            list of [x1, y1, x2, y2, confidence, class_id] for each detection
        """
        try:
            # Use provided threshold or default
            threshold = conf_threshold if conf_threshold is not None else self.type_threshold
            
            # Run inference with YOLO model
            results = self.model(frame, imgsz=1280, verbose=False)[0]
            detections = []
            
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.data
                for box in boxes:
                    # Get box coordinates, confidence and class
                    x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                    class_id = int(cls)
                    
                    # Apply filtering for false positives
                    obj_width = x2 - x1
                    obj_height = y2 - y1
                    obj_area = obj_width * obj_height
                    frame_area = frame.shape[0] * frame.shape[1]
                    area_ratio = obj_area / frame_area
                    
                    # Filter out very small or very large detections
                    if area_ratio < 0.0005 or area_ratio > 0.5:
                        continue
                    
                    # Only include detections above threshold
                    if conf > threshold:
                        detections.append([
                            int(x1), int(y1), int(x2), int(y2),
                            float(conf),
                            class_id
                        ])
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in violence type detection: {str(e)}")
            return []

def process_sequence(sequence_path, model_paths=None, output_dir='results/videos'):
    """Process a MOT17 sequence for violence detection.
    
    Args:
        sequence_path: Path to MOT17 sequence directory
        model_paths: List of paths to violence detection models
        output_dir: Directory to save output videos
    """
    detector = ViolenceDetector(model_paths)
    
    # Get sequence name from path
    sequence_name = os.path.basename(sequence_path)
    
    # Setup video input from sequence images
    img_dir = os.path.join(sequence_path, 'img1')
    if not os.path.exists(img_dir):
        logger.error(f"Image directory not found: {img_dir}")
        return
        
    # Get all frame images
    frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    if not frame_files:
        logger.error(f"No images found in {img_dir}")
        return
        
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(img_dir, frame_files[0]))
    height, width = first_frame.shape[:2]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Setup video writer
    output_path = os.path.join(output_dir, f"{sequence_name}_violence.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    
    # Process each frame
    violence_results = []
    for frame_idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(img_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            logger.error(f"Could not read frame: {frame_path}")
            continue
            
        # Detect violence in current frame
        violent_boxes = detector.detect_violence(frame)
        
        # Store results for this frame
        frame_results = {
            'frame': frame_idx + 1,  # MOT17 uses 1-based frame indices
            'violent_regions': violent_boxes,
            'violence_detected': len(violent_boxes) > 0
        }
        violence_results.append(frame_results)
        
        # Visualize results
        for box in violent_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Add violence label with distance indication
            height_ratio = (y2 - y1) / frame.shape[0]
            distance = "Near" if height_ratio > 0.4 else "Medium" if height_ratio > 0.2 else "Far"
            cv2.putText(frame, f"Violence ({distance})", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Write frame to video
        out.write(frame)
        
        # Log progress
        if (frame_idx + 1) % 100 == 0:
            logger.info(f"Processed {frame_idx + 1}/{len(frame_files)} frames in {sequence_name}")
    
    out.release()
    logger.info(f"Completed processing {sequence_name}. Output saved to {output_path}")
    return violence_results

def process_mot17_test(mot17_dir, output_dir='results/videos'):
    """Process all sequences in MOT17 test set.
    
    Args:
        mot17_dir: Path to MOT17 test directory
        output_dir: Directory to save output videos
    """
    test_sequences = [d for d in os.listdir(mot17_dir) if os.path.isdir(os.path.join(mot17_dir, d))]
    
    for seq in test_sequences:
        seq_path = os.path.join(mot17_dir, seq)
        logger.info(f"Processing sequence: {seq}")
        process_sequence(seq_path, output_dir=output_dir)
        
    logger.info("Completed processing all MOT17 test sequences")