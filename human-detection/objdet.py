import torch
from ultralytics import YOLO
import os

def main():
    # Check and set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.cuda.empty_cache()

    # Load the YOLO model
    model = YOLO("yolov8s.pt")  # Small model for better performance

    # Debug: Check trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable parameter: {name}")
        else:
            print(f"Frozen parameter: {name}")

    # Train the model
    model.train(
        data="./data.yaml",     # Path to dataset configuration
        epochs=5,               # Number of epochs
        batch=16,               # Batch size
        imgsz=640,              # Image size
        device='0',             # Use GPU (cuda:0)
        workers=4,              # Adjust based on your CPU
        name="crowd_project",   # Name of the training run
        lr0=0.01,               # Initial learning rate
        optimizer="SGD",        # Use SGD optimizer
        momentum=0.937,         # Momentum for optimizer
        patience=5,             # Early stopping after 25 epochs of no improvement
        freeze=10,              # Freeze the first 10 layers
    )

    # Validate the model
    metrics = model.val(device=device)  # Perform validation using the specified device

    # Print validation metrics
    print(metrics)

    # Save the trained model
    model.save("./Hcrowded_project.pt")

if __name__ == '__main__':
    main()