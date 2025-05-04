# yolov8_detector.py
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from deep_sort.detection import Detection  # Add this import

class YOLOv8Detector:
    def __init__(self, model_path, reid_model=None, conf_thres=0.3, iou_thres=0.4, img_size=(1280, 720)):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_path: Path to YOLOv8 model
            reid_model: ReID model for feature extraction
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            img_size: Input image size (width, height)
        """
        self.model = YOLO(model_path)
        self.reid_model = reid_model
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.reid_model:
            self.reid_model = self.reid_model.to(self.device)
            self.reid_model.eval()
            self.transform = transforms.Compose([
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def get_reid_features(self, person_roi):
        """
        Extract ReID features from person ROI
        
        Args:
            person_roi: Person region of interest
            
        Returns:
            ReID features or empty array if reid_model is None
        """
        if self.reid_model is None or person_roi is None or person_roi.size == 0:
            return np.array([])
            
        try:
            # Convert to RGB (ReID models typically expect RGB)
            person_roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            img = Image.fromarray(person_roi_rgb)
            
            # Prepare transforms
            transform = transforms.Compose([
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Transform the image
            img_tensor = transform(img).unsqueeze(0)
            
            # Get features
            with torch.no_grad():
                features = self.reid_model(img_tensor, return_features=True)
                
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error extracting ReID features: {e}")
            return np.array([])

    def detect(self, frame):
        """
        Detect persons in frame
        
        Args:
            frame: Input frame
            
        Returns:
            List of detections in format [x1, y1, x2, y2, confidence, class_id]
            or Detection objects if reid_model is provided
        """
        # Resize frame to desired size
        frame_resized = cv2.resize(frame, self.img_size)
        
        # Run YOLOv8 inference
        results = self.model(frame_resized, verbose=False)[0]
        
        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.data
            
            # Convert boxes back to original frame size
            scale_x = frame.shape[1] / self.img_size[0]
            scale_y = frame.shape[0] / self.img_size[1]
            
            for box in boxes:
                # Get box coordinates and scores
                x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                
                # Only keep person detections (class 0) above confidence threshold
                if int(cls) == 0 and conf > self.conf_thres:
                    # Scale coordinates back to original frame size
                    x1, x2 = x1 * scale_x, x2 * scale_x
                    y1, y2 = y1 * scale_y, y2 * scale_y
                    
                    if self.reid_model is not None:
                        # Extract person ROI
                        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                        if 0 <= y1i < y2i and 0 <= x1i < x2i and y2i <= frame.shape[0] and x2i <= frame.shape[1]:
                            person_roi = frame[y1i:y2i, x1i:x2i]
                            
                            # Get ReID features
                            features = self.get_reid_features(person_roi)
                            
                            # Create Detection object with ReID features
                            bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to [x, y, w, h]
                            detection = Detection(bbox, conf, features)
                            detections.append(detection)
                        else:
                            # Skip invalid ROI
                            continue
                    else:
                        # Return basic detection format
                        detections.append([x1, y1, x2, y2, conf, 0])
        
        return detections
    
    