import os
import sys
import torch
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
import gc
from torch.cuda.amp import autocast, GradScaler

def train_violence_detector(device='0', batch=4):  # Increased batch size since YOLOv8s is lighter
    # Test CUDA availability and print detailed information
    print("\n" + "="*80)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch Version: {torch.__version__}")
    print("="*80 + "\n")
    
    # Optimize CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.cuda.empty_cache()
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit("Training requires a GPU-enabled environment.")

    # Set environment variables for better GPU memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Print GPU info
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()

    # Path to the data configuration
    data_yaml = "A:/AAST/grad proj/code/Versions/violence_try/deep_sort/violence_models/datasets/violence/data.yaml"
    
    # Get the violence dataset directory
    violence_dir = os.path.dirname(data_yaml)
    
    # Define output directory inside the violence dataset
    output_dir = os.path.join(violence_dir, "trained_models")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Training results will be saved to: {output_dir}")
    
    # Verify that data_yaml exists
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset file not found at {data_yaml}")
        sys.exit(f"Please check the path to your data.yaml file")
    else:
        print(f"Dataset file found at {data_yaml}")
    
    # Use fine-tuned YOLO model with absolute path
    model_path = "A:/AAST/grad proj/code/Versions/violence_pose/deep_sort/human-detection/Hcrowded_project.pt"
    
    # Verify that model_path exists
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print("Falling back to yolov8s.pt")
        model_path = "yolov8s.pt"
    else:
        print(f"Using pre-trained model from {model_path}")
    
    model = YOLO(model_path)

    # Configure training parameters
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=1280,  
        batch=batch,
        device=device,  # Using GPU
        name="violence_detector_gpu",
        project=output_dir,  # Set the project directory to the violence dataset
        exist_ok=True,
        patience=10,  # Early stopping after 3 epochs without improvement
        save=True,
        plots=True,
        cache=True,
        amp=True,    # Enable mixed precision training
        workers=4,
        optimizer="SGD",  # Use SGD optimizer
        cos_lr=True,
        lr0=0.02,
        lrf=0.0005,
        momentum=0.95,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        box=10,
        cls=1.0,
        dfl=1.5,
        mixup=0.5,
        mosaic=1.0,
        scale=0.5,
        fliplr=0.5,
        translate=0.1,
        copy_paste=0.5,
        degrees=0.373,
        perspective=0.0007,
        rect=False,
        mask_ratio=4,
        nbs=128,
        single_cls=False,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fraction=1.0,
        profile=True,
        val=True,
        save_period=10
    )

    print(f"\nTraining complete. Results saved in: {results.save_dir}")
    print("\nGPU Memory Summary:")
    print(torch.cuda.memory_summary())
    
    # Copy the best model to a more accessible location
    best_model = os.path.join(results.save_dir, "weights", "best.pt")
    if os.path.exists(best_model):
        final_model = os.path.join(violence_dir, "violence_detector_best.pt")
        import shutil
        shutil.copy(best_model, final_model)
        print(f"\nBest model copied to: {final_model}")
    
    return results.save_dir

if __name__ == "__main__":
    # Test CUDA availability at script start
    print("\nChecking CUDA availability before training...")
    if torch.cuda.is_available():
        print("CUDA is available! GPU will be used for training.")
    else:
        print("CUDA is NOT available! Please check your GPU drivers and PyTorch installation.")
        print("You can install PyTorch with CUDA support using:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        sys.exit("Training requires a GPU-enabled environment.")
    
    # Set CUDA device
    cuda_device = '0'
    print(f"\nInitializing training on CUDA device {cuda_device}")
    
    try:
        train_violence_detector(device=cuda_device, batch=4)
    except torch.cuda.OutOfMemoryError:
        print("\nGPU out of memory detected. Attempting to recover...")
        torch.cuda.empty_cache()
        gc.collect()
        print("\nRetrying with reduced batch size...")
        train_violence_detector(device=cuda_device, batch=2)  # Fallback to smaller batch size if needed
    except FileNotFoundError as e:
        print(f"\nFile not found error: {str(e)}")
        print("Please check the paths to your data.yaml and model files.")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise