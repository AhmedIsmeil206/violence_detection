# market1501_reid.py (updated)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np

def load_reid_model():
    """
    Load pretrained ReID model
    """
    # Check if CUDA is available, if not use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = ReIDFeatureExtractor().to(device)
    
    # Load weights with proper mapping to CPU if CUDA is not available
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'market1501_reid.pth')
    try:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise
    
    # Set model to evaluation mode
    model.eval()
    return model

class ReIDFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=512, num_classes=751, model_path=None):
        super(ReIDFeatureExtractor, self).__init__()
        base_model = models.resnet50(pretrained=False)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.embedding = nn.Linear(2048, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim
        
        if model_path:
            self.load_state_dict(torch.load(model_path)['state_dict'])
        
    def forward(self, x, return_features=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.embedding(x)
        features = nn.functional.normalize(features, p=2, dim=1)
        
        if return_features:
            return features
        return self.classifier(features)



class Market1501Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or self.default_transform()
        self.image_paths = []
        self.pids = []
        
        # First pass: find all valid images and collect unique pids
        pid_set = set()
        for img_name in os.listdir(root_dir):
            if img_name.endswith('.jpg'):
                try:
                    pid = int(img_name.split('_')[0])
                    if pid >= 0:  # Only accept non-negative pids
                        pid_set.add(pid)
                        self.image_paths.append(os.path.join(root_dir, img_name))
                        self.pids.append(pid)
                except (ValueError, IndexError):
                    continue
        
        # Create mapping from original pids to continuous 0-based indices
        self.unique_pids = sorted(pid_set)
        self.pid_to_idx = {pid: idx for idx, pid in enumerate(self.unique_pids)}
        self.num_classes = len(self.unique_pids)
        
        # Convert all pids to 0-based indices
        self.pids = [self.pid_to_idx[pid] for pid in self.pids]
        
        print(f"Found {len(self.image_paths)} images from {self.num_classes} identities")
    
    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.pids[idx]

def train_reid_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize dataset first to get the actual number of classes
    train_dataset = Market1501Dataset('A:/AAST/grad proj/code/Versions/re_id_fune_tuning/deep_sort/Market_1501/bounding_box_train')
    
    # Initialize model with correct number of classes
    model = ReIDFeatureExtractor(num_classes=train_dataset.num_classes).to(device)
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, pids in train_loader:
            images, pids = images.to(device), pids.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, pids)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += pids.size(0)
            correct += (predicted == pids).sum().item()
        
        scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Acc: {100*correct/total:.2f}%')
    
    # Save the trained model
    torch.save({
        'state_dict': model.state_dict(),
        'feature_dim': model.feature_dim,
        'num_classes': train_dataset.num_classes
    }, 'deep_sort/market1501_reid.pth')
    print(f"Training complete. Model saved to market1501_reid.pth with {train_dataset.num_classes} classes")

if __name__ == '__main__':
    train_reid_model()