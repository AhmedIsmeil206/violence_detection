import os
import sys
import torch
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
import gc
import glob
import shutil
from torch.cuda.amp import autocast, GradScaler

def clear_cache_files(dataset_path):
    """Clear any existing cache files to prevent EOF errors"""
    cache_files = glob.glob(os.path.join(dataset_path, "*.cache"))
    if cache_files:
        print(f"Removing {len(cache_files)} cache files to prevent errors...")
        for cache_file in cache_files:
            try:
                os.remove(cache_file)
                print(f"Removed cache file: {cache_file}")
            except Exception as e:
                print(f"Failed to remove {cache_file}: {str(e)}")

def train_violence_type_detector(device='0', batch=4):  # Batch size for YOLOv8s
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

    # Path to the data configuration for violence type detection
    data_yaml = "A:/AAST/grad proj/code/Versions/violence_try/deep_sort/violence_models/datasets/type/data.yaml"
    
    # Get the violence type dataset directory
    type_dir = os.path.dirname(data_yaml)
    
    # Clear any existing cache files that might be corrupted
    clear_cache_files(type_dir)
    
    # Get the train directory path to clear its cache too
    import yaml
    try:
        with open(data_yaml, 'r') as f:
            data_info = yaml.safe_load(f)
            train_dir = os.path.join(type_dir, data_info.get('train', '').replace('../', ''))
            if os.path.exists(os.path.dirname(train_dir)):
                clear_cache_files(os.path.dirname(train_dir))
    except Exception as e:
        print(f"Warning: Error reading train path from yaml: {str(e)}")
    
    # Use the type directory directly for output
    output_dir = type_dir
    print(f"Training results will be saved directly to: {output_dir}")
    
    # Verify that data_yaml exists
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset file not found at {data_yaml}")
        sys.exit(f"Please check the path to your data.yaml file")
    else:
        print(f"Dataset file found at {data_yaml}")
        # Print class information
        try:
            with open(data_yaml, 'r') as f:
                data_info = yaml.safe_load(f)
                print(f"Classes to detect: {data_info.get('names', [])}")
        except Exception as e:
            print(f"Warning: Could not read class information: {str(e)}")
    
    # Use custom model path for violence type detection
    model_path = "A:/AAST/grad proj/code/Versions/violence_pose/deep_sort/human-detection/Hcrowded_project.pt"
    
    # Verify that model_path exists
    if not os.path.exists(model_path):
        print(f"Warning: Custom model file not found at {model_path}")
        print("Falling back to yolov8s.pt")
        model_path = "yolov8s.pt"
    else:
        print(f"Using custom pre-trained model from {model_path}")
    
    model = YOLO(model_path)

    # Configure training parameters for violence type detection
    results = model.train(
        data=data_yaml,
        epochs=50,            # Train for more epochs for better accuracy
        imgsz=1280,           # Higher resolution for better detection of small objects
        batch=batch,
        device=device,        # Using GPU
        name="type_detector",  # Simpler name for folder
        project=output_dir,   # Set the project directory to the type dataset root
        exist_ok=True,
        patience=10,          # Early stopping after 10 epochs without improvement
        save=True,
        plots=True,
        cache=False,          # Disable cache to prevent EOF errors
        amp=True,             # Enable mixed precision training
        workers=4,
        optimizer="SGD",      # Use SGD optimizer
        cos_lr=True,
        lr0=0.01,             # Learning rate
        lrf=0.001,
        momentum=0.95,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        box=10,
        cls=1.0,
        dfl=1.5,
        mixup=0.5,            # Data augmentation
        mosaic=1.0,           # Data augmentation
        scale=0.5,
        fliplr=0.5,
        translate=0.1,
        copy_paste=0.3,       # Add copy-paste augmentation
        degrees=0.373,
        perspective=0.0007,
        rect=False,
        mask_ratio=4,
        nbs=128,
        single_cls=False,     # Multi-class detection
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fraction=1.0,
        profile=True,
        val=True,
        save_period=10         # Save checkpoints every 10 epochs
    )

    print(f"\nTraining complete. Results saved in: {results.save_dir}")
    print("\nGPU Memory Summary:")
    print(torch.cuda.memory_summary())
    
    # Copy the best model directly to the type directory with a descriptive name
    best_model = os.path.join(results.save_dir, "weights", "best.pt")
    if os.path.exists(best_model):
        final_model = os.path.join(type_dir, "type_detector_best.pt")
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
        # Start with a more conservative batch size to prevent memory issues
        train_violence_type_detector(device=cuda_device, batch=4)
    except torch.cuda.OutOfMemoryError:
        print("\nGPU out of memory detected. Attempting to recover...")
        torch.cuda.empty_cache()
        gc.collect()
        print("\nRetrying with reduced batch size...")
        train_violence_type_detector(device=cuda_device, batch=2)  # Further reduce batch size
    except FileNotFoundError as e:
        print(f"\nFile not found error: {str(e)}")
        print("Please check the paths to your data.yaml and model files.")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        # Print more detailed traceback for debugging
        import traceback
        traceback.print_exc()
        raise 