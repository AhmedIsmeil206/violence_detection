import cv2
import numpy as np
from kinematic_tree import KinematicTree

class ViolenceVisualizer:
    @staticmethod
    def draw_poses(image, results, violent_tracks=[]):
        """Draw pose keypoints and connections"""
        if results.keypoints is None:
            return image
            
        keypoints = results.keypoints.xy.cpu().numpy()
        
        for i, kps in enumerate(keypoints):
            # Draw keypoints
            for j, kp in enumerate(kps):
                x, y = map(int, kp[:2])
                cv2.circle(image, (x, y), 3, (0, 255, 255), -1)
            
            # Draw connections
            for (start, end), color in zip(KinematicTree.CONNECTIONS, KinematicTree.COLORS):
                if start < len(kps) and end < len(kps):
                    x1, y1 = map(int, kps[start][:2])
                    x2, y2 = map(int, kps[end][:2])
                    cv2.line(image, (x1, y1), (x2, y2), color, 2)
        
        return image
    
    @staticmethod
    def draw_violence(image, boxes, tracks=[]):
        """Draw violence bounding boxes with track IDs"""
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Find matching track
            for track in tracks:
                track_box = track.to_tlbr()
                if (abs(track_box[0] - x1) < 10 and abs(track_box[1] - y1) < 10):
                    cv2.putText(image, f"VIOLENCE {track.track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    break
        
        return image