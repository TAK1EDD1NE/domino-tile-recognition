#!/usr/bin/env python3
"""
YOLOv8 Domino Tile Detection Training Script
Fine-tunes a pretrained YOLOv8 model on a custom domino tile dataset.
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

def setup_training_environment():
    """Set up the training environment and check requirements."""
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set up directories
    results_dir = Path("runs/detect")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    return device

def validate_dataset_structure(data_yaml_path):
    """Validate the dataset structure and data.yaml file."""
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")
    
    # Load and validate data.yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    required_keys = ['train', 'val', 'nc', 'names']
    for key in required_keys:
        if key not in data_config:
            raise KeyError(f"Missing required key '{key}' in data.yaml")
    
    # Check if train and validation directories exist
    train_path = Path(data_config['train'])
    val_path = Path(data_config['val'])
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_path}")
    
    print(f"Dataset validation passed:")
    print(f"  - Classes: {data_config['nc']} ({data_config['names']})")
    print(f"  - Train path: {train_path}")
    print(f"  - Validation path: {val_path}")
    
    return data_config

def train_yolov8_model(
    data_yaml_path="data.yaml",
    model_size="n",  # n, s, m, l, x
    epochs=200,  # Increased epochs
    batch_size=16,
    img_size=640,
    learning_rate=0.001,  # Lower learning rate for better convergence
    patience=0,  # Disable early stopping
    save_period=10,
    project_name="domino_detection"
):
    """
    Train YOLOv8 model on domino tile dataset.
    
    Args:
        data_yaml_path: Path to data.yaml configuration file
        model_size: YOLOv8 model size (n, s, m, l, x)
        epochs: Maximum number of training epochs
        batch_size: Training batch size
        img_size: Input image size
        learning_rate: Initial learning rate
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
        project_name: Project name for saving results
    """
    
    # Setup environment
    device = setup_training_environment()
    
    # Validate dataset
    data_config = validate_dataset_structure(data_yaml_path)
    
    # Load pretrained YOLOv8 model
    model_name = f"yolov8{model_size}.pt"
    print(f"Loading pretrained model: {model_name}")
    model = YOLO(model_name)
    
    # Configure training parameters optimized for small datasets
    train_args = {
        'data': data_yaml_path,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'lr0': learning_rate,
        'lrf': 0.1,  # Final learning rate (lr0 * lrf)
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Classification loss gain
        'dfl': 1.5,  # Distribution focal loss gain
        'pose': 12.0,  # Pose loss gain
        'kobj': 1.0,  # Keypoint objective loss gain
        'label_smoothing': 0.0,
        'nbs': 64,  # Nominal batch size
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        'save': True,
        'save_period': save_period,
        'cache': False,  # Set to True if you have enough RAM
        'device': device,
        'workers': 8,
        'project': project_name,
        'name': f'yolov8{model_size}_domino_tiles',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'SGD',  # SGD, Adam, AdamW, NAdam, RAdam, RMSProp
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,  # Disable mosaic augmentation for last N epochs
        'resume': False,
        'amp': True,  # Automatic Mixed Precision
        'fraction': 1.0,
        'profile': False,
        'freeze': None,  # Freeze first N layers, or list of layer indices
        'multi_scale': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'split': 'val',
        'save_json': False,
        'save_hybrid': False,
        'conf': None,
        'iou': 0.7,
        'max_det': 300,
        'half': False,
        'dnn': False,
        'plots': True,
        'source': None,
        'vid_stride': 1,
        'stream_buffer': False,
        'visualize': False,
        'augment': False,
        'agnostic_nms': False,
        'classes': None,
        'retina_masks': False,
        'embed': None,
        'show': False,
        'save_frames': False,
        'save_txt': False,
        'save_conf': False,
        'save_crop': False,
        'show_labels': True,
        'show_conf': True,
        'show_boxes': True,
        'line_width': None,
        'format': 'torchscript',
        'keras': False,
        'optimize': False,
        'int8': False,
        'dynamic': False,
        'simplify': False,
        'opset': None,
        'workspace': 4,
        'nms': False,
        'batch': 1,
        'patience': patience
    }
    
    # Remove keys that are not valid for training
    valid_train_args = {
        'data': data_yaml_path,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'lr0': learning_rate,
        'lrf': 0.1,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'label_smoothing': 0.0,
        'nbs': 64,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        'save': True,
        'save_period': save_period,
        'cache': False,
        'device': device,
        'workers': 8,
        'project': project_name,
        'name': f'yolov8{model_size}_domino_tiles',
        'exist_ok': True,
        'optimizer': 'SGD',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'patience': patience
    }
    
    print("\nStarting training with the following configuration:")
    print(f"  - Model: YOLOv8{model_size}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Image size: {img_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Early stopping: {'DISABLED' if patience == 0 else f'{patience} epochs'}")
    print(f"  - Optimizer: AdamW")
    print(f"  - Cosine LR scheduler: Enabled")
    print(f"  - Device: {device}")
    
    # Start training
    try:
        results = model.train(**valid_train_args)
        print("\nTraining completed successfully!")
        return results
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

def evaluate_model(model_path, data_yaml_path):
    """Evaluate the trained model on validation set."""
    print(f"\nEvaluating model: {model_path}")
    model = YOLO(model_path)
    
    # Run validation
    val_results = model.val(data=data_yaml_path)
    
    print("Validation results:")
    print(f"  - mAP50: {val_results.box.map50:.4f}")
    print(f"  - mAP50-95: {val_results.box.map:.4f}")
    
    return val_results

def main():
    """Main training pipeline."""
    # Configuration
    config = {
        'data_yaml_path': 'data.yaml',
        'model_size': 'n',  # Start with nano model for small dataset
        'epochs': 200,  # Increased epochs since early stopping is disabled
        'batch_size': 16,  # Adjust based on your GPU memory
        'img_size': 640,
        'learning_rate': 0.001,  # Lower learning rate for better convergence
        'patience': 0,  # 0 = disabled early stopping
        'save_period': 20,  # Save less frequently due to more epochs
        'project_name': 'domino_detection'
    }
    
    print("=== YOLOv8 Domino Tile Detection Training ===")
    print(f"Configuration: {config}")
    
    # Train the model
    results = train_yolov8_model(**config)
    
    # Find the best model path
    best_model_path = Path(config['project_name']) / f"yolov8{config['model_size']}_domino_tiles_v2" / 'weights' / 'best.pt'
    
    if best_model_path.exists():
        print(f"\nBest model saved at: {best_model_path}")
        
        # Evaluate the best model
        evaluate_model(str(best_model_path), config['data_yaml_path'])
        
        # Optional: Test inference on a sample image
        print(f"\nModel ready for inference!")
        print(f"To use the trained model:")
        print(f"  from ultralytics import YOLO")
        print(f"  model = YOLO('{best_model_path}')")
        print(f"  results = model('path/to/your/image.jpg')")
        
    else:
        print("Warning: Best model not found. Check training logs.")

if __name__ == "__main__":
    main()