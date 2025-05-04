import os
import cv2
import argparse
import numpy as np
from yolov8.yolov8_detector import YOLOv8Detector
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from market1501_reid import load_reid_model
from violence_models.detect import ViolenceDetector, ViolenceTypeDetector
import time

def realtime_tracking(camera_source=0, model_path=None, reid_model_path=None, violence_model_path=None, violence_type_model_path=None, output_path=None, show_display=True, weapon_persistence=0.5,
                     min_weapon_frames=1, max_weapon_frames_missing=5):
    """
    Perform real-time tracking and violence detection using a camera feed.
    
    Args:
        camera_source: Camera index (0 for default camera) or video file path
        model_path: Path to YOLOv8 model for human detection
        reid_model_path: Path to ReID model for person re-identification
        violence_model_path: Path to violence detection model
        violence_type_model_path: Path to violence type detection model
        output_path: Path to save the output video (optional)
        show_display: Whether to show the output in a window
        weapon_persistence: Time in seconds to keep showing weapon boxes after they disappear
        min_weapon_frames: Minimum frames a weapon needs to be detected before counting it
        max_weapon_frames_missing: Maximum frames a weapon can be missing before removing it
    """
    # Set default paths if not provided
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "human-detection", "Hcrowded_project.pt")
    
    if reid_model_path is None:
        reid_model_path = os.path.join(os.path.dirname(__file__), "market1501_reid.pth")
    
    if violence_model_path is None:
        violence_model_path = os.path.join(os.path.dirname(__file__), "violence_models", "datasets", "violence", "violence_detector_best.pt")
    
    if violence_type_model_path is None:
        violence_type_model_path = os.path.join(os.path.dirname(__file__), "violence_models", "datasets", "type", "type_detector_best.pt")
    
    # Ensure model paths are absolute
    model_path = os.path.abspath(model_path)
    reid_model_path = os.path.abspath(reid_model_path)
    violence_model_path = os.path.abspath(violence_model_path)
    violence_type_model_path = os.path.abspath(violence_type_model_path)
    
    # Model paths set
    
    # Initialize video capture
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"Error: Could not open camera source {camera_source}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30  # Default to 30 FPS if not available
    
    # Initialize video writer if output path is provided
    out = None
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"Saving output to: {output_path}")
    
    try:
        # Custom ReID model path
        custom_reid_model_path = reid_model_path
        reid_model = load_reid_model()
        print("ReID model loaded successfully")
    except Exception as e:
        print(f"Error loading ReID model: {e}")
        print("Falling back to tracking without ReID features")
        reid_model = None
    
    # Verify model files exist
    for path, name in [(model_path, "YOLOv8"), (violence_model_path, "Violence"), 
                       (violence_type_model_path, "Violence Type")]:
        if not os.path.exists(path):
            print(f"Error: {name} model not found at {path}")
            return

    # Initialize YOLOv8 detector for human detection
    try:
        detector = YOLOv8Detector(model_path, reid_model)
        print("YOLOv8 detector initialized successfully")
    except Exception as e:
        print(f"Error initializing YOLOv8 detector: {e}")
        return
    
    # Initialize DeepSORT tracker
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", 0.3, None)  # metric_type, matching_threshold, budget
    tracker = Tracker(metric, max_iou_distance=0.7, max_age=30, n_init=3)
    
    # Initialize Violence Detector with the specified model path
    try:
        violence_detector = ViolenceDetector([violence_model_path])
    except Exception as e:
        violence_detector = None
    
    # Initialize Violence Type Detector with the specified model path
    try:
        violence_type_detector = ViolenceTypeDetector(violence_type_model_path)
    except Exception as e:
        violence_type_detector = None
    
    # Class names from data.yaml files
    violence_class_names = ['normal', 'violence']
    violence_type_class_names = ['grenade', 'handgun', 'knife', 'theft mask']
    
    # Dictionary to store violence status per track
    violence_status = {}  # track_id -> {'status': class_name, 'confidence': score, 'class_id': id}
    
    # Dictionary to store violence type detections with timestamps
    violence_type_detections = {}  # detection_id -> {'bbox': [x1,y1,x2,y2], 'type': class_name, 'confidence': score, 'last_seen': timestamp}
    
    # Dictionary to store persistent weapon detections (for smoothing)
    persistent_weapons = {}  # weapon_id -> {'bbox': [x1,y1,x2,y2], 'type': class_name, 'confidence': score, 'last_seen': timestamp, 'track_count': int}
    
    # Store stable weapon counts - will only update when weapons truly appear/disappear
    stable_weapon_counts = {weapon_type: 0 for weapon_type in violence_type_class_names}
    total_stable_weapons = 0
    
    # For FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    
    print("Starting real-time tracking. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Update FPS calculation
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            fps_display = frame_count / elapsed_time
        
        # Current timestamp
        current_time = time.time()
        
        # Generate detections using YOLOv8
        detections = detector.detect(frame)
        
        # Format detections for DeepSORT
        deep_sort_detections = []
        for det in detections:
            if isinstance(det, Detection):
                deep_sort_detections.append(det)
            else:
                try:
                    x1, y1, x2, y2, confidence, class_id = det
                    if class_id != 0:  # Filter out non-human detections
                        continue
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    deep_sort_detections.append(Detection(bbox, confidence, np.array([])))
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
        
        # Update DeepSORT tracker
        tracker.predict()
        tracker.update(deep_sort_detections)
        
        # Run violence detection if available
        if violence_detector is not None:
            try:
                # Get violence detections
                violence_results = violence_detector.detect_violence(frame, conf_threshold=0.4)
                
                # Store violence detection results
                violence_detections = {}
                for i, det in enumerate(violence_results):
                    x1, y1, x2, y2, conf, class_id = det
                    
                    # Only process violence class (class_id = 1)
                    if class_id == 1:  # Violence class
                        violence_detections[i] = {
                            'bbox': [x1, y1, x2, y2],
                            'type': violence_class_names[class_id],
                            'confidence': conf,
                            'class_id': class_id
                        }
                
                # Run violence type detection if available
                if violence_type_detector is not None:
                    violence_type_results = violence_type_detector.detect_violence_type(frame, conf_threshold=0.35)
                    
                    # Clear the current frame's detections but keep track of persistent weapons
                    current_detections = {}
                    
                    # Store violence type detection results with timestamp
                    for i, det in enumerate(violence_type_results):
                        x1, y1, x2, y2, conf, class_id = det
                        
                        # First create a more stable detection ID that will persist across frames
                        # Use class_id and approximate location (divided by 20 to allow slight movement)
                        stable_x = int(x1) // 20
                        stable_y = int(y1) // 20
                        detection_id = f"{class_id}_{stable_x}_{stable_y}"
                        
                        # Create a new detection record
                        new_detection = {
                            'bbox': [x1, y1, x2, y2],
                            'type': violence_type_class_names[class_id],
                            'confidence': conf,
                            'class_id': class_id,
                            'last_seen': current_time
                        }
                        
                        # Update persistent weapons
                        if detection_id in persistent_weapons:
                            persistent_weapons[detection_id]['last_seen'] = current_time
                            persistent_weapons[detection_id]['track_count'] += 1
                            persistent_weapons[detection_id]['frames_missing'] = 0
                            
                            # DO NOT update the bbox at all after it's first detected
                            # Keep it completely static like human detection boxes
                            # Only update confidence if significantly higher
                            if conf > persistent_weapons[detection_id]['confidence'] + 0.2:
                                persistent_weapons[detection_id]['confidence'] = conf
                            
                            # Update stable counts immediately when detected
                            if not persistent_weapons[detection_id].get('counted', False):
                                weapon_type = persistent_weapons[detection_id]['type']
                                stable_weapon_counts[weapon_type] += 1
                                total_stable_weapons += 1
                                persistent_weapons[detection_id]['counted'] = True
                        else:
                            # Only create new weapon detection if not already tracking similar weapon nearby
                            # This prevents multiple boxes for the same weapon
                            create_new = True
                            
                            # Check if there's already a weapon of the same type nearby
                            for existing_id, existing_weapon in persistent_weapons.items():
                                if existing_weapon['class_id'] == class_id:  # Same type of weapon
                                    ex1, ey1, ex2, ey2 = existing_weapon['bbox']
                                    # Calculate centers
                                    ex_center = ((ex1 + ex2) / 2, (ey1 + ey2) / 2)
                                    new_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                                    # Calculate distance
                                    distance = np.sqrt((new_center[0] - ex_center[0])**2 + 
                                                      (new_center[1] - ex_center[1])**2)
                                    if distance < 100:  # If within 100 pixels
                                        # Update last_seen but don't create new detection
                                        existing_weapon['last_seen'] = current_time
                                        existing_weapon['frames_missing'] = 0
                                        create_new = False
                                        break
                            
                            # Only create new weapon detection if not near an existing one
                            if create_new:
                                # Create new weapon track
                                persistent_weapons[detection_id] = {
                                    'bbox': [x1, y1, x2, y2],
                                    'type': violence_type_class_names[class_id],
                                    'confidence': conf,
                                    'class_id': class_id,
                                    'last_seen': current_time,
                                    'track_count': 1,
                                    'frames_missing': 0,
                                    'counted': False,
                                    'original_bbox': [x1, y1, x2, y2]  # Store original bbox 
                                }
                        
                        # Store in current detections
                        current_detections[detection_id] = new_detection
                    
                    # Check for missing weapons and update their status
                    for weapon_id, weapon_data in persistent_weapons.items():
                        if weapon_id not in current_detections:
                            # Increment missing frames counter
                            persistent_weapons[weapon_id]['frames_missing'] += 1
                            
                            # If weapon has been missing for too long and was stable, update stable count
                            if (persistent_weapons[weapon_id]['frames_missing'] >= max_weapon_frames_missing and 
                                persistent_weapons[weapon_id]['track_count'] >= min_weapon_frames):
                                weapon_type = weapon_data['type']
                                stable_weapon_counts[weapon_type] = max(0, stable_weapon_counts[weapon_type] - 1)
                                total_stable_weapons = max(0, total_stable_weapons - 1)
                                print(f"{weapon_type} lost! Remaining: {stable_weapon_counts[weapon_type]}")
                                # Mark this weapon as unstable so we don't count it again
                                persistent_weapons[weapon_id]['track_count'] = 0
                    
                    # Clean up old persistent weapons
                    weapons_to_remove = []
                    for weapon_id, weapon_data in persistent_weapons.items():
                        time_since_last_seen = current_time - weapon_data['last_seen']
                        
                        # Remove if not seen for more than weapon_persistence seconds
                        if time_since_last_seen > weapon_persistence or weapon_data['frames_missing'] > max_weapon_frames_missing:
                            weapons_to_remove.append(weapon_id)
                    
                    # Remove old weapons
                    for weapon_id in weapons_to_remove:
                        del persistent_weapons[weapon_id]
                    
                    # Update current violence type detections
                    violence_type_detections = persistent_weapons
                
                # Reset status of all tracks to normal on each frame - they will be marked as violent again if needed
                for track in tracker.tracks:
                    if track.is_confirmed() and track.time_since_update <= 1:
                        track_id = track.track_id
                        # Default to normal for all tracks
                        violence_status[track_id] = {
                            'status': 'normal',
                            'confidence': 0.9,
                            'class_id': 0,
                            'has_weapon': False,
                            'frames_without_weapon': 0
                        }
                
                # Then check for weapons/violence and update only those tracks that currently have weapons/violence
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    
                    track_id = track.track_id
                    bbox = track.to_tlwh()
                    x1, y1, w, h = map(int, bbox)
                    track_box = [x1, y1, x1 + w, y1 + h]
                    
                    # Skip tracks without sufficient height (may be false positives)
                    if h < 50:
                        continue
                    
                    # Check overlap with violence detections with stricter criteria
                    is_violent = False
                    violence_evidence = 0  # Count evidences of violence
                    
                    for v_id, v_det in violence_detections.items():
                        v_box = v_det['bbox']
                        
                        # Calculate overlap
                        ix1 = max(track_box[0], v_box[0])
                        iy1 = max(track_box[1], v_box[1])
                        ix2 = min(track_box[2], v_box[2])
                        iy2 = min(track_box[3], v_box[3])
                        
                        # Require significant overlap with higher threshold
                        if ix2 > ix1 and iy2 > iy1:
                            intersection = (ix2 - ix1) * (iy2 - iy1)
                            track_area = w * h
                            overlap_ratio = intersection / track_area
                            
                            # Much stricter threshold (0.5 instead of 0.2)
                            if overlap_ratio > 0.5:
                                violence_evidence += 1
                                if violence_evidence >= 2:  # Require multiple evidences
                                    is_violent = True
                                    break
                    
                    # Check overlap with weapon detections - this part is more important
                    has_weapon = False
                    weapon_type = None
                    weapon_confidence = 0
                    weapon_bbox = None
                    max_weapon_overlap = 0
                    
                    # Define an expanded search area around the person, but smaller than before
                    expanded_x1 = max(0, x1 - int(w * 0.25))  # Slightly larger expansion
                    expanded_y1 = max(0, y1 - int(h * 0.05))
                    expanded_x2 = min(frame.shape[1], x1 + w + int(w * 0.25))
                    expanded_y2 = min(frame.shape[0], y1 + h + int(h * 0.05))
                    
                    # CRITICAL CHANGE: First check if any weapon is detected in the frame at all
                    if violence_type_detections:
                        for weapon_id, v_det in violence_type_detections.items():
                            v_box = v_det['bbox']
                            
                            # Less strict weapon size limits
                            weapon_width = v_box[2] - v_box[0]
                            weapon_height = v_box[3] - v_box[1]
                            if weapon_width > w * 0.8 or weapon_height > h * 0.8:  # Only filter very large weapons
                                continue
                            
                            # Skip weapons with very low confidence
                            if v_det['confidence'] < 0.25:  # Lower threshold to detect more weapons
                                continue
                            
                            # Check if weapon is within the expanded person area
                            wx1, wy1, wx2, wy2 = v_box
                            
                            # Simple overlap check - if any part of weapon is near the person
                            if ((expanded_x1 <= wx2 and expanded_x2 >= wx1) and
                                (expanded_y1 <= wy2 and expanded_y2 >= wy1)):
                                # Weapon is near person - mark as having weapon immediately
                                has_weapon = True
                                weapon_type = v_det['type']
                                weapon_confidence = v_det['confidence']
                                weapon_bbox = v_det['bbox']
                                break  # Stop once we find any weapon
                    
                    # Immediately mark the person as violent if a weapon is detected nearby
                    if has_weapon:
                        # Mark as violent only while weapon is detected
                        violence_status[track_id] = {
                            'status': 'violence',
                            'confidence': weapon_confidence,
                            'has_weapon': True,
                            'weapon_type': weapon_type,
                            'weapon_bbox': weapon_bbox,
                            'frames_without_weapon': 0
                        }
                        # Print violence detection information
                        print(f"VIOLENCE DETECTED: Person ID {track_id} with {weapon_type}")
                    elif is_violent:
                        # Only mark as violent if multiple frames of violence
                        if track_id in violence_status:
                            frames_violent = violence_status[track_id].get('frames_violent', 0) + 1
                            if frames_violent >= 3:  # Reduced from 5 to 3 frames
                                violence_status[track_id] = {
                                    'status': 'violence',
                                    'confidence': 1.0,
                                    'has_weapon': False,
                                    'weapon_type': None,
                                    'weapon_bbox': None,
                                    'frames_violent': frames_violent,
                                    'frames_without_weapon': 0
                                }
                                # Print violence detection without weapon
                                print(f"VIOLENCE DETECTED: Person ID {track_id} - physical violence")
                            else:
                                # Not enough frames yet, update counter
                                violence_status[track_id]['frames_violent'] = frames_violent
                        else:
                            # New detection, initialize counter
                            violence_status[track_id] = {
                                'status': 'normal',  # Start as normal
                                'confidence': 1.0,
                                'has_weapon': False,
                                'frames_violent': 1,
                                'frames_without_weapon': 0
                            }

            except Exception as e:
                pass  # Silent error handling
        
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Draw tracking boxes for all people
        incident_counts = {'normal': 0, 'violence': 0}
        weapon_type_counts = {weapon_type: 0 for weapon_type in violence_type_class_names}
        total_people = 0
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            total_people += 1
            bbox = track.to_tlwh()
            x1, y1, w, h = map(int, bbox)
            track_id = track.track_id
            
            # Get violence status
            if track_id in violence_status:
                status = violence_status[track_id]['status']
                has_weapon = violence_status[track_id].get('has_weapon', False)
                weapon_type = violence_status[track_id].get('weapon_type', None)
                incident_counts[status] += 1
                if weapon_type:
                    weapon_type_counts[weapon_type] += 1
            else:
                status = 'normal'
                has_weapon = False
                weapon_type = None
                incident_counts[status] += 1
            
            # Draw person tracking box - RED for ANY violent action (violence or weapon)
            if status == 'violence':
                color = (0, 0, 255)  # Red for violence
                border_thickness = 3  # Thicker border for violent people
            else:
                color = (0, 255, 0)  # Green for normal
                border_thickness = 2  # Normal border for normal people
                
            cv2.rectangle(display_frame, (x1, y1), (x1 + w, y1 + h), color, border_thickness)
            
            # Draw appropriate label including the violence type
            if status == 'violence' and has_weapon:
                label = f"ID: {track_id} - VIOLENT ({weapon_type.upper()})"
            elif status == 'violence':
                label = f"ID: {track_id} - VIOLENT"
            else:
                label = f"ID: {track_id} - NORMAL"
                
            # Make violent labels more visible with a background
            if status == 'violence':
                # Text size calculation
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                text_w, text_h = text_size
                
                # Draw background box for text
                cv2.rectangle(display_frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 0), -1)
                
                # Draw text with white color
                cv2.putText(display_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                # Normal label
                cv2.putText(display_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw weapon detections with their own labels - only if they're recent enough
        for weapon_id, v_det in violence_type_detections.items():
            # Only draw weapons that are currently visible or seen recently
            if current_time - v_det['last_seen'] <= weapon_persistence:
                # Use the original bounding box coordinates for complete stability
                if 'original_bbox' in v_det:
                    wx1, wy1, wx2, wy2 = map(int, v_det['original_bbox'])
                else:
                    wx1, wy1, wx2, wy2 = map(int, v_det['bbox'])
                
                weapon_type = v_det['type']
                confidence = v_det['confidence']
                track_count = v_det.get('track_count', 0)
                
                # Choose color based on weapon type
                if weapon_type == 'knife':
                    weapon_color = (255, 0, 255)  # Magenta for knife
                elif weapon_type == 'handgun':
                    weapon_color = (0, 165, 255)  # Orange for handgun
                elif weapon_type == 'grenade':
                    weapon_color = (0, 255, 255)  # Yellow for grenade
                else:  # theft mask
                    weapon_color = (255, 255, 0)  # Cyan for theft mask
                
                # Draw the weapon box - always use the original bbox for maximum stability
                cv2.rectangle(display_frame, 
                             (wx1, wy1), 
                             (wx2, wy2), 
                             weapon_color, 2)
                
                # Add weapon label with confidence
                weapon_label = f"{weapon_type.upper()} ({confidence:.2f})"
                cv2.putText(display_frame, weapon_label,
                           (wx1, wy1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, weapon_color, 2)
        
        # Display status information
        # Only count weapons that have been consistently detected (stable)
        weapon_count = total_stable_weapons
        violent_people = incident_counts.get('violence', 0)
        
        status_text = [
            f"People: {total_people}",
            f"Weapons: {weapon_count}",
            f"Violent: {violent_people}",
            f"FPS: {fps_display:.1f}"
        ]
        
        # Draw status text
        cv2.putText(display_frame, " | ".join(status_text), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add weapon type counts - using stable weapon counts
        if any(stable_weapon_counts.values()):
            weapon_counts_text = " | ".join([f"{weapon}: {count}" for weapon, count in stable_weapon_counts.items() if count > 0])
            cv2.putText(display_frame, weapon_counts_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add alert border if stable weapons are detected
        if weapon_count > 0:
            border_thickness = 10
            overlay = display_frame.copy()
            
            # Choose border color based on weapon type priority: handgun > knife > grenade > theft mask
            if stable_weapon_counts['handgun'] > 0:
                border_color = (0, 165, 255)  # Orange for handgun
                alert_text = "ALERT: HANDGUN DETECTED"
            elif stable_weapon_counts['knife'] > 0:
                border_color = (255, 0, 255)  # Magenta for knife
                alert_text = "ALERT: KNIFE DETECTED"
            elif stable_weapon_counts['grenade'] > 0:
                border_color = (0, 255, 255)  # Yellow for grenade
                alert_text = "ALERT: GRENADE DETECTED"
            elif stable_weapon_counts['theft mask'] > 0:
                border_color = (255, 255, 0)  # Cyan for theft mask
                alert_text = "ALERT: THEFT MASK DETECTED"
            else:
                border_color = (0, 0, 255)  # Red for general violence
                alert_text = "ALERT: VIOLENCE DETECTED"
            
            cv2.rectangle(overlay, (0, 0), (frame_width, frame_height),
                         border_color, border_thickness)
            cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)
            
            # Add flashing alert text
            if frame_count % 10 < 5:
                cv2.putText(display_frame, alert_text,
                           (frame_width // 2 - 200, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, border_color, 3)
        
        # Write frame to video if output is enabled
        if out:
            out.write(display_frame)
        
        # Display the frame if show_display is True
        if show_display:
            cv2.imshow('Real-time Tracking with Violence Detection', display_frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Clean up
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds")
    
    # Processing complete

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time person tracking and violence detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--video", type=str, help="Path to video file (overrides camera)")
    default_model = os.path.join(os.path.dirname(__file__), "human-detection", "Hcrowded_project.pt")
    default_reid = os.path.join(os.path.dirname(__file__), "market1501_reid.pth")
    default_violence = os.path.join(os.path.dirname(__file__), "violence_models", "datasets", "violence", "violence_detector_best.pt")
    default_violence_type = os.path.join(os.path.dirname(__file__), "violence_models", "datasets", "type", "type_detector_best.pt")
    
    parser.add_argument("--model", type=str, default=default_model, help="Path to YOLOv8 model")
    parser.add_argument("--reid", type=str, default=default_reid, help="Path to ReID model")
    parser.add_argument("--violence", type=str, default=default_violence, help="Path to violence detection model")
    parser.add_argument("--violence_type", type=str, default=default_violence_type, help="Path to violence type detection model")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video")
    parser.add_argument("--no-display", action="store_true", help="Disable display window")
    parser.add_argument("--weapon-persistence", type=float, default=0.5, help="Time in seconds to keep showing weapon boxes after they disappear (default: 0.5)")
    parser.add_argument("--min-weapon-frames", type=int, default=1, help="Minimum frames a weapon needs to be detected before counting it (default: 1)")
    parser.add_argument("--max-missing-frames", type=int, default=5, help="Maximum frames a weapon can be missing before removing it (default: 5)")
    
    args = parser.parse_args()
    
    # Determine the video source
    source = args.video if args.video else args.camera
    
    # Set weapon detection parameters from command line arguments
    weapon_persistence = args.weapon_persistence
    min_weapon_frames = args.min_weapon_frames
    max_weapon_frames_missing = args.max_missing_frames
    
    realtime_tracking(
        camera_source=source,
        model_path=args.model,
        reid_model_path=args.reid,
        violence_model_path=args.violence,
        violence_type_model_path=args.violence_type,
        output_path=args.output,
        show_display=not args.no_display,
        weapon_persistence=weapon_persistence,
        min_weapon_frames=min_weapon_frames,
        max_weapon_frames_missing=max_weapon_frames_missing
    ) 
    