import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class ReIDWrapper:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model with correct architecture
        self.model = ReIDFeatureExtractor(
            feature_dim=checkpoint['feature_dim'],
            num_classes=checkpoint['num_classes']
        ).to(self.device)
        
        # Load state dict
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        self.feature_dim = checkpoint['feature_dim']
    
    def __call__(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(self.feature_dim)
        
        crop = Image.fromarray(crop).convert('RGB')
        crop = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(crop, return_features=True).cpu().numpy().flatten()
            features /= np.linalg.norm(features)  # L2 normalize
        return features

def _pdist(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2

def _cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)

def _nn_cosine_distance(x, y):
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            self.last_features[target] = feature
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        
        # Clean up samples for inactive targets
        active_set = set(active_targets)
        self.samples = {k: self.samples[k] for k in active_set if k in self.samples}
        self.last_features = {k: self.last_features[k] for k in active_set if k in self.last_features}

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            if target in self.samples:
                # Use both historical and last features
                all_features = np.array(self.samples[target])
                if target in self.last_features:
                    all_features = np.vstack([all_features, self.last_features[target]])
                
                # Calculate minimum distance to any stored feature
                cost_matrix[i, :] = self._metric(all_features, features)
            else:
                cost_matrix[i, :] = self.matching_threshold + 1e-5  # Prevent matching
        return cost_matrix