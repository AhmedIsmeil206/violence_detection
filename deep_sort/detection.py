# detection.py
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

class Detection(object):
    """
    This class represents a bounding box detection with ReID features.
    
    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this detection.
    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float64)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio, height)`."""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @classmethod
    def from_yolo(cls, yolo_detection, reid_model, img, device):
        """
        Create a Detection from YOLOv8 output with ReID features.
        
        Parameters
        ----------
        yolo_detection : tuple
            (x1, y1, x2, y2, confidence, class_id)
        reid_model : torch.nn.Module
            The ReID feature extractor model
        img : ndarray
            The full image frame
        device : torch.device
            Device to run ReID model on
            
        Returns
        -------
        Detection or None if extraction failed
        """
        x1, y1, x2, y2, confidence, class_id = yolo_detection
        tlwh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float64)
        
        # Skip if not a person detection
        if class_id != 0:
            return None
            
        # Extract image patch
        img_patch = img[int(y1):int(y2), int(x1):int(x2)]
        if img_patch.size == 0:
            return None
            
        # Convert BGR to RGB and resize for ReID model
        img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
        img_patch = Image.fromarray(img_patch)
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Extract features
        try:
            img_tensor = transform(img_patch).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = reid_model(img_tensor, return_features=True)
                feature = feature.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error extracting ReID features: {e}")
            feature = np.array([])
            
        return cls(tlwh, confidence, feature)