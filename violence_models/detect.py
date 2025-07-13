import cv2
import numpy as np
import torch
from ultralytics import YOLO
import logging
from deep_sort import nn_matching
import time

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
        self.class_names = ['grenade', 'handgun', 'knife', 'theft mask']          # Enhanced class-specific thresholds for better confidence
        self.class_thresholds = {
            0: 0.65,  # grenade - significantly increased threshold to reduce false positives
            1: 0.30,  # handgun - medium threshold
            2: 0.15,  # knife - much lower threshold to detect knives better
            3: 0.80   # theft mask - very high threshold to prevent false detections
        }
        
        # Size validation constraints per class
        self.size_constraints = {
            'grenade': {'min_area_ratio': 0.0001, 'max_area_ratio': 0.02, 'aspect_ratio_range': (0.7, 1.4)},
            'handgun': {'min_area_ratio': 0.0001, 'max_area_ratio': 0.08, 'aspect_ratio_range': (1.2, 3.5)},
            'knife': {'min_area_ratio': 0.00001, 'max_area_ratio': 0.15, 'aspect_ratio_range': (1.0, 15.0)},  # More relaxed for knife
            'theft mask': {'min_area_ratio': 0.0005, 'max_area_ratio': 0.15, 'aspect_ratio_range': (0.8, 1.3)}
        }
        
        # Temporal consistency tracking
        from collections import deque
        self.detection_history = deque(maxlen=5)
        self.stable_detections = {}
        
        self.type_threshold = 0.30  # Default threshold for violence type detection

    def _validate_detection_size(self, box, class_id, frame_shape):
        """Validate detection based on expected size constraints"""
        x1, y1, x2, y2 = box[:4]
        width = x2 - x1
        height = y2 - y1
        area = width * height
        frame_area = frame_shape[0] * frame_shape[1]
        area_ratio = area / frame_area
        aspect_ratio = width / max(height, 1)
        
        class_name = self.class_names[class_id]
        constraints = self.size_constraints.get(class_name, {})
        
        # Check area constraints
        min_area = constraints.get('min_area_ratio', 0.00001)
        max_area = constraints.get('max_area_ratio', 0.5)
        
        if not (min_area <= area_ratio <= max_area):
            return False
        
        # Check aspect ratio constraints
        aspect_range = constraints.get('aspect_ratio_range', (0.1, 20.0))
        if not (aspect_range[0] <= aspect_ratio <= aspect_range[1]):
            return False
        
        return True
    
    def _adjust_confidence_by_context(self, detection, frame):
        """Adjust confidence based on contextual factors"""
        x1, y1, x2, y2, conf, class_id = detection[:6]
        class_name = self.class_names[class_id]
        
        # Size-based adjustment
        width = x2 - x1
        height = y2 - y1
        area = width * height
        frame_area = frame.shape[0] * frame.shape[1]
        area_ratio = area / frame_area
        
        size_multiplier = 1.0
        if area_ratio < 0.001:  # Very small objects
            size_multiplier = 0.85  # Slightly reduce confidence
        elif area_ratio > 0.05:  # Large objects
            size_multiplier = 1.1  # Increase confidence
        
        # Class-specific adjustments
        class_multiplier = 1.0
        aspect_ratio = width / max(height, 1)
        
        if class_name == 'knife' and 1.0 <= aspect_ratio <= 15.0:  # More relaxed aspect ratio for knife
            class_multiplier = 1.25  # Higher boost for knife detection
        elif class_name == 'handgun' and 1.2 <= aspect_ratio <= 3.0:
            class_multiplier = 1.2   # Good handgun aspect ratio
        elif class_name == 'grenade' and 0.8 <= aspect_ratio <= 1.3:
            class_multiplier = 1.1   # Round objects are likely grenades
        elif class_name == 'theft mask' and y1 < frame.shape[0] * 0.4:
            class_multiplier = 1.15  # Masks are typically on upper part of frame
        
        # Lighting-based adjustment
        roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
        if roi.size > 0:
            brightness = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
            if brightness < 60:  # Dark regions
                class_multiplier *= 0.9
            elif 80 <= brightness <= 180:  # Good lighting
                class_multiplier *= 1.05
        
        # Apply all adjustments
        adjusted_conf = conf * size_multiplier * class_multiplier
        return min(adjusted_conf, 0.98)  # Cap at 98%
    
    def _temporal_consistency_check(self, detections, frame_time):
        """Apply temporal consistency filtering"""
        current_frame_detections = {}
        
        for detection in detections:
            x1, y1, x2, y2, conf, class_id = detection[:6]
            
            # Create detection key based on location and class
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            det_key = f"{class_id}_{int(center_x/30)}_{int(center_y/30)}"
            
            current_frame_detections[det_key] = {
                'detection': detection,
                'timestamp': frame_time,
                'center': (center_x, center_y)
            }
        
        # Add to history
        self.detection_history.append(current_frame_detections)
        
        # Find stable detections (appeared in multiple recent frames)
        stable_detections = []
        min_appearances = 1  # Reduced to 1 for faster weapon detection
        
        for det_key, current_det in current_frame_detections.items():
            appearances = 1  # Current frame
            total_confidence = current_det['detection'][4]
            
            # Check recent history
            for hist_frame in list(self.detection_history)[:-1]:
                if det_key in hist_frame:
                    appearances += 1
                    total_confidence += hist_frame[det_key]['detection'][4]
            
            # If stable enough, add to results with averaged confidence
            if appearances >= min_appearances:
                avg_confidence = total_confidence / appearances
                detection = current_det['detection'].copy()
                detection[4] = avg_confidence * 1.1  # Boost for stability
                stable_detections.append(detection)
            elif current_det['detection'][4] > 0.5:  # Lower threshold for single detection
                stable_detections.append(current_det['detection'])
        
        return stable_detections

    def detect_violence_type(self, frame, conf_threshold=None):
        """
        Enhanced violence type detection with improved confidence filtering
        Args:
            frame: Input image
            conf_threshold: Optional confidence threshold override
        Returns: 
            list of [x1, y1, x2, y2, confidence, class_id] for each detection
        """
        try:
            import time
            frame_time = time.time()
            
            # Multi-scale detection for better small object detection
            detections = []
            
            # Standard scale detection
            results = self.model(frame, imgsz=1280, verbose=False)[0]
            standard_detections = self._process_results(results, frame, scale_factor=1.0)
            detections.extend(standard_detections)
            
            # Large scale for small objects (1.25x)
            height, width = frame.shape[:2]
            large_frame = cv2.resize(frame, (int(width*1.25), int(height*1.25)), interpolation=cv2.INTER_CUBIC)
            results_large = self.model(large_frame, imgsz=1280, verbose=False)[0]
            large_detections = self._process_results(results_large, large_frame, scale_factor=1/1.25)
            detections.extend(large_detections)
            
            # Apply Non-Maximum Suppression to remove duplicates
            detections = self._apply_nms(detections)
            
            # Filter and validate detections
            validated_detections = []
            for detection in detections:
                x1, y1, x2, y2, conf, class_id = detection[:6]
                
                # Size validation
                if not self._validate_detection_size(detection, class_id, frame.shape):
                    continue
                
                # Context-based confidence adjustment
                adjusted_conf = self._adjust_confidence_by_context(detection, frame)
                detection[4] = adjusted_conf
                
                # Apply class-specific threshold
                threshold = conf_threshold if conf_threshold else self.class_thresholds.get(class_id, 0.3)
                if adjusted_conf > threshold:
                    validated_detections.append(detection[:6])
            
            # Apply temporal consistency check
            stable_detections = self._temporal_consistency_check(validated_detections, frame_time)
            
            return stable_detections
            
        except Exception as e:
            logger.error(f"Error in enhanced violence type detection: {str(e)}")
            return []
    
    def _process_results(self, results, frame, scale_factor=1.0):
        """Process YOLO results with scale adjustment"""
        detections = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                class_id = int(cls)
                
                # Scale coordinates back if needed
                if scale_factor != 1.0:
                    x1, y1, x2, y2 = [coord * scale_factor for coord in [x1, y1, x2, y2]]
                
                # Apply basic threshold
                if conf > 0.15:  # Very low threshold for initial filtering
                    detections.append([
                        int(x1), int(y1), int(x2), int(y2),
                        float(conf), class_id
                    ])
        
        return detections
    
    def _apply_nms(self, detections, iou_threshold=0.5):
        """Apply Non-Maximum Suppression"""
        if not detections:
            return []
        
        # Group by class
        from collections import defaultdict
        class_detections = defaultdict(list)
        for det in detections:
            class_detections[det[5]].append(det)
        
        final_detections = []
        
        for class_id, class_dets in class_detections.items():
            if not class_dets:
                continue
                
            # Sort by confidence
            class_dets.sort(key=lambda x: x[4], reverse=True)
            
            # Apply NMS for this class
            keep = []
            while class_dets:
                best = class_dets.pop(0)
                keep.append(best)
                
                remaining = []
                for det in class_dets:
                    iou = self._calculate_iou(best[:4], det[:4])
                    if iou < iou_threshold:
                        remaining.append(det)
                
                class_dets = remaining
            
            final_detections.extend(keep)
        
        return final_detections
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

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

class UCFCrimeViolenceDetector:
    """Enhanced Violence Detector supporting UCF-Crime multi-class detection"""
    
    def __init__(self, model_paths=None, enable_pose_validation=True):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        if model_paths is None:
            # Try UCF-Crime model first, fallback to binary model
            ucf_model = os.path.join(current_dir, 'datasets/ucf_crime_yolo/ucf_crime_detector_best.pt')
            if os.path.exists(ucf_model):
                model_paths = [ucf_model]
            else:
                model_paths = [os.path.join(current_dir, 'datasets/violence/violence_detector_best.pt')]
        
        # Verify model files exist
        for path in model_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
          # Load YOLO model for violence detection
        self.model = YOLO(model_paths[0])        # Determine class names based on model - UCF-Crime multi-class classification
        try:
            # Try to get class names from model
            if hasattr(self.model, 'names'):
                model_class_names = list(self.model.names.values())
                print(f"Model has {len(model_class_names)} classes: {model_class_names}")
                
                # If model only has 2 classes, warn user about limited functionality
                if len(model_class_names) == 2:
                    print("⚠ WARNING: Model only has 2 classes (binary classification).")
                    print("   For full UCF-Crime multiclass detection, you need a model trained with data_multiclass.yaml")
                    print("   Currently detecting: non-violent vs violent")
                    print("   System will intelligently map binary outputs to specific violence types")
                    
                # Store both model and desired class names
                self.model_class_names = model_class_names
                self.class_names = model_class_names  # This can be overridden later
            else:
                # UCF-Crime multi-class - expand to include specific violence types
                self.class_names = ['non-violent', 'violent', 'abuse', 'arrest', 'arson', 'assault', 'burglary', 'explosion', 'fighting', 'robbery', 'shooting', 'shoplifting', 'stealing', 'vandalism']
                self.model_class_names = self.class_names
                print("Using default UCF-Crime multi-class labels")
        except:
            self.class_names = ['non-violent', 'violent']  # Fallback to binary
            self.model_class_names = self.class_names
            print("Fallback to binary classification")# UCF-Crime multi-class thresholds
        self.class_thresholds = {
            'non-violent': 0.6,
            'violent': 0.3,
            'abuse': 0.35,
            'arrest': 0.4,
            'arson': 0.3,
            'assault': 0.35,
            'burglary': 0.4,
            'explosion': 0.25,  # Lower threshold for critical events
            'fighting': 0.4,
            'robbery': 0.35,
            'shooting': 0.25,  # Lower threshold for critical violence
            'shoplifting': 0.4,
            'stealing': 0.4,
            'vandalism': 0.35,
            # Legacy fallbacks
            'normal': 0.6,
            'violence': 0.3
        }
        
        # Initialize pose processor if enabled
        self.enable_pose_validation = enable_pose_validation
        if enable_pose_validation:
            try:
                pose_model_path = os.path.join(current_dir, 'yolov8x-pose.pt')
                self.pose_processor = PoseProcessor(pose_model_path)
                logger.info("UCF-Crime pose processor initialized")
            except Exception as e:
                logger.warning(f"Could not initialize pose processor: {e}")
                self.pose_processor = None
        else:
            self.pose_processor = None
        
        # Temporal tracking
        self.temporal_window = deque(maxlen=8)
        self.min_pose_confidence = 0.4
        
        # Distance-based confidence adjustments
        self.distance_confidence_adjustments = {
            'near': 1.0,
            'medium': 0.85,
            'far': 0.7
        }
        
        logger.info(f"UCFCrimeViolenceDetector initialized with classes: {self.class_names}")

    def detect_violence(self, frame, conf_threshold=None):
        """
        Detect violence in frame with UCF-Crime multi-class support
        
        Args:
            frame: Input frame
            conf_threshold: Optional confidence threshold override
            
        Returns:
            List of detections [x1, y1, x2, y2, confidence, class_id]
        """
        try:
            if conf_threshold is None:
                conf_threshold = 0.25  # Base threshold
              # Run inference
            results = self.model(frame, imgsz=1280, verbose=False)[0]
            
            detections = []
            
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.data.cpu().numpy()
                
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box
                    class_id = int(cls)
                    
                    # Skip 'non-violent' class detections (class_id = 0)
                    if class_id == 0:
                        continue
                    
                    # Get class name
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else 'unknown'
                    
                    # Apply class-specific threshold
                    threshold = self.class_thresholds.get(class_name, conf_threshold)
                    
                    if conf > threshold:
                        # Validate with pose if enabled
                        if self.pose_processor and self.enable_pose_validation:
                            pose_validation = self._validate_with_pose(frame, [x1, y1, x2, y2], class_name)
                            if not pose_validation['valid']:
                                # Reduce confidence if pose validation fails
                                conf *= 0.7
                                if conf < threshold:
                                    continue
                        
                        # Apply temporal consistency
                        conf = self._apply_temporal_consistency(conf, class_id, [x1, y1, x2, y2])
                        
                        if conf > threshold:
                            detections.append([
                                int(x1), int(y1), int(x2), int(y2),
                                float(conf),
                                class_id
                            ])
            
            # Store for temporal consistency
            self.temporal_window.append(detections)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in UCF-Crime violence detection: {str(e)}")
            return []

    def _validate_with_pose(self, frame, bbox, violence_type):
        """Validate detection using pose analysis"""
        if not self.pose_processor:
            return {'valid': True, 'confidence': 0.5}
            
        try:
            # Use existing pose processing functionality
            pose_results = self.pose_processor.process_frame(frame)
            if pose_results:
                # Simple validation based on motion score
                max_motion = max([result.get('motion_score', 0) for result in pose_results])
                # Consider it valid if there's significant motion for violent activities
                is_valid = max_motion > 5.0 if violence_type in ['fighting', 'assault', 'shooting'] else True
                confidence = min(max_motion / 20.0, 1.0)  # Normalize motion to confidence
                return {'valid': is_valid, 'confidence': confidence}
            else:
                return {'valid': True, 'confidence': 0.5}
        except Exception as e:
            logger.error(f"Error in pose validation: {str(e)}")
            return {'valid': True, 'confidence': 0.5}

    def _apply_temporal_consistency(self, confidence, class_id, bbox):
        """Apply temporal consistency to reduce false positives"""
        try:
            if len(self.temporal_window) < 2:
                return confidence
            
            # Check recent detections for similar class and location
            recent_detections = []
            for recent_frame in list(self.temporal_window)[-3:]:  # Last 3 frames
                for det in recent_frame:
                    det_class_id = int(det[5])
                    if det_class_id == class_id:
                        # Check location similarity
                        det_bbox = det[:4]
                        overlap = self._calculate_bbox_overlap(bbox, det_bbox)
                        if overlap > 0.3:  # 30% overlap threshold
                            recent_detections.append(det[4])  # Store confidence
            
            # Boost confidence if consistently detected
            if len(recent_detections) >= 2:
                avg_recent_conf = np.mean(recent_detections)
                confidence = min(confidence * 1.2, max(confidence, avg_recent_conf * 1.1))
            elif len(recent_detections) == 0:
                # Reduce confidence for isolated detections
                confidence *= 0.8
                
            return confidence
            
        except Exception as e:
            logger.error(f"Error in temporal consistency: {str(e)}")
            return confidence

    def _calculate_bbox_overlap(self, bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes"""
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Calculate intersection
            ix1 = max(x1_1, x1_2)
            iy1 = max(y1_1, y1_2)
            ix2 = min(x2_1, x2_2)
            iy2 = min(y2_1, y2_2)
            
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            
            intersection = (ix2 - ix1) * (iy2 - iy1)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / max(union, 1)
            
        except Exception as e:
            return 0.0

    def get_class_name(self, class_id):
        """Get class name from class ID"""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return 'unknown'

    def is_violence_class(self, class_id):
        """Check if class ID represents violence (not normal)"""
        return class_id != 0  # All classes except 'normal' are violence
