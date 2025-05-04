import numpy as np

class KinematicTree:
    # COCO Keypoint Connections
    CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (11, 12), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
    ]
    
    # Connection colors (BGR)
    COLORS = [
        (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),  # Head - Blue
        (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),  # Arms - Green
        (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255),  # Body - Red
        (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255)   # Legs - Red
    ]
    
    @classmethod
    def get_limb_angles(cls, keypoints):
        """Calculate angles between connected limbs"""
        angles = []
        for i, (start, end) in enumerate(cls.CONNECTIONS):
            if keypoints[start][2] > 0.1 and keypoints[end][2] > 0.1:  # Check confidence
                vec = keypoints[end][:2] - keypoints[start][:2]
                angle = np.degrees(np.arctan2(vec[1], vec[0]))
                angles.append((i, angle))
        return angles
    
    @classmethod
    def get_velocity(cls, prev_keypoints, curr_keypoints, fps):
        """Calculate movement velocity between frames"""
        velocities = []
        for i in range(len(curr_keypoints)):
            if curr_keypoints[i][2] > 0.1 and prev_keypoints[i][2] > 0.1:
                dist = np.linalg.norm(curr_keypoints[i][:2] - prev_keypoints[i][:2])
                velocities.append(dist * fps)
        return np.mean(velocities) if velocities else 0