from ultralytics import YOLO
import cv2
import os

# Load model
model = YOLO('A:/AAST/grad proj/code/Versions/violence_pose/deep_sort/violence_models/pose_estimation/yolov8n-pose.pt')

# Specify your test image path
image_path = 'test2.jpg'  # Change this to your image filename

# Verify image exists
if not os.path.exists(image_path):
    print(f"Error: Image '{image_path}' not found!")
    print("Available files:", os.listdir('.'))
    exit()

# Run inference
results = model(image_path, save=True, conf=0.5, show=True)  # save=True saves output image

# Print keypoints information
for i, result in enumerate(results):
    print(f"\nDetection {i+1}:")
    print(f"Keypoints shape: {result.keypoints.xy.shape}")  # [num_people, 17, 2]
    print(f"First person keypoints:\n{result.keypoints.xy[0]}")  # First detected person

print(f"\nOutput saved to: {results[0].save_dir}")