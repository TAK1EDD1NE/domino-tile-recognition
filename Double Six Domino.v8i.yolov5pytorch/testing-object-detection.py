#!/usr/bin/env python3
"""
YOLOv8 Model Testing and Visualization Script
Tests a trained YOLOv8 model on training data and visualizes detection results.
"""

import os
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from ultralytics import YOLO
import torch
from collections import defaultdict
import random
from PIL import Image

def load_dataset_config(data_yaml_path):
    """Load dataset configuration from data.yaml."""
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_image_files(image_dir):
    """Get all image files from directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
        image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    return sorted(image_files)

def load_yolo_labels(label_path, img_width, img_height):
    """Load YOLO format labels and convert to pixel coordinates."""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    # Convert to x1, y1, x2, y2
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    boxes.append({
                        'class_id': class_id,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': 1.0  # Ground truth has confidence 1.0
                    })
    return boxes

def run_inference_on_dataset(model_path, data_yaml_path, confidence_threshold=0.25):
    """Run inference on all images in the training dataset."""
    # Load model
    model = YOLO(model_path)
    
    # Load dataset config
    config = load_dataset_config(data_yaml_path)
    train_dir = Path(config['train'])
    
    # Make train_dir absolute if it's relative
    if not train_dir.is_absolute():
        train_dir = Path(data_yaml_path).parent / train_dir
    train_dir = train_dir.resolve()
    
    print(f"Training directory: {train_dir}")
    
    # Get all image files
    image_files = get_image_files(train_dir)
    
    print(f"Found {len(image_files)} images in training directory")
    
    # Debug: Print first few image paths
    if image_files:
        print("First few image paths:")
        for i, img_path in enumerate(image_files[:3]):
            print(f"  {i+1}: {img_path}")
            print(f"      Exists: {img_path.exists()}")
    
    results_data = []
    
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        
        # Check if file exists before processing
        if not img_path.exists():
            print(f"  ERROR: File does not exist: {img_path}")
            continue
        
        try:
            # Run inference - use absolute path
            results = model(str(img_path.resolve()), conf=confidence_threshold)
            
            # Load image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ERROR: Could not load image: {img_path}")
                continue
                
            img_height, img_width = img.shape[:2]
            
            # Find corresponding label file
            # Try multiple possible label directory structures
            possible_label_paths = [
                train_dir / 'labels' / f"{img_path.stem}.txt",  # Same parent directory
                train_dir.parent / 'labels' / f"{img_path.stem}.txt",  # Parent directory
                train_dir.parent / 'train' / 'labels' / f"{img_path.stem}.txt",  # Common structure
            ]
            
            label_path = None
            for path in possible_label_paths:
                if path.exists():
                    label_path = path
                    break
            
            ground_truth = []
            if label_path:
                ground_truth = load_yolo_labels(str(label_path), img_width, img_height)
            else:
                print(f"  WARNING: No label file found for {img_path.name}")
            
            # Extract predictions
            predictions = []
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    predictions.append({
                        'class_id': int(cls),
                        'bbox': box.tolist(),
                        'confidence': float(conf)
                    })
            
            results_data.append({
                'image_path': str(img_path),
                'image_name': img_path.name,
                'predictions': predictions,
                'ground_truth': ground_truth,
                'image_size': (img_width, img_height)
            })
            
        except Exception as e:
            print(f"  ERROR processing {img_path.name}: {e}")
            continue
    
    return results_data

def calculate_metrics(results_data, iou_threshold=0.5):
    """Calculate detection metrics."""
    total_gt = 0
    total_pred = 0
    true_positives = 0
    
    for result in results_data:
        gt_boxes = result['ground_truth']
        pred_boxes = result['predictions']
        
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        
        # Simple TP calculation (could be improved with proper IoU matching)
        for pred in pred_boxes:
            for gt in gt_boxes:
                if calculate_iou(pred['bbox'], gt['bbox']) > iou_threshold:
                    true_positives += 1
                    break
    
    precision = true_positives / total_pred if total_pred > 0 else 0
    recall = true_positives / total_gt if total_gt > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'total_ground_truth': total_gt,
        'total_predictions': total_pred,
        'true_positives': true_positives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def visualize_results(results_data, num_samples=100, save_dir="visualization_results"):
    """Visualize detection results with ground truth comparison and save each image individually."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Randomly sample images for visualization
    sample_results = random.sample(results_data, min(num_samples, len(results_data)))
    
    for result in sample_results:
        # Create figure for this single image
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Load and display image
        img = cv2.imread(result['image_path'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        
        # Set title with image name and counts
        ax.set_title(f"{result['image_name']}\nGT: {len(result['ground_truth'])}, Pred: {len(result['predictions'])}", 
                    fontsize=10)
        ax.axis('off')
        
        # Draw ground truth boxes (green)
        for gt in result['ground_truth']:
            x1, y1, x2, y2 = gt['bbox']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='green', facecolor='none', 
                                   label='Ground Truth')
            ax.add_patch(rect)
        
        # Draw prediction boxes (red)
        for pred in result['predictions']:
            x1, y1, x2, y2 = pred['bbox']
            conf = pred['confidence']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='red', facecolor='none',
                                   linestyle='--',
                                   label='Prediction')
            ax.add_patch(rect)
            
            # Add confidence score
            ax.text(x1, y1-5, f'{conf:.2f}', color='red', fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
        # Add legend once per image
        ax.legend(loc='upper right')
        
        # Save individual image file
        filename = os.path.join(save_dir, f"{os.path.splitext(result['image_name'])[0]}_result.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close figure to free memory
    
    print(f"Saved {len(sample_results)} visualizations to {save_dir}/")

def create_detailed_report(results_data, metrics, save_dir="test_results"):
    """Create a detailed testing report."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate confidence distribution
    all_confidences = []
    for result in results_data:
        for pred in result['predictions']:
            all_confidences.append(pred['confidence'])
    
    # Calculate detection count distribution
    detection_counts = [len(result['predictions']) for result in results_data]
    gt_counts = [len(result['ground_truth']) for result in results_data]
    
    # Create summary report
    report = f"""
YOLOv8 Domino Tile Detection - Test Results Report
================================================

Dataset Statistics:
- Total Images: {len(results_data)}
- Total Ground Truth Boxes: {metrics['total_ground_truth']}
- Total Predicted Boxes: {metrics['total_predictions']}
- Average GT boxes per image: {np.mean(gt_counts):.2f}
- Average predicted boxes per image: {np.mean(detection_counts):.2f}

Performance Metrics:
- True Positives: {metrics['true_positives']}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1-Score: {metrics['f1_score']:.4f}

Confidence Statistics:
- Mean Confidence: {np.mean(all_confidences):.4f}
- Median Confidence: {np.median(all_confidences):.4f}
- Min Confidence: {np.min(all_confidences):.4f}
- Max Confidence: {np.max(all_confidences):.4f}

Detection Distribution:
- Images with 0 detections: {sum(1 for x in detection_counts if x == 0)}
- Images with 1-5 detections: {sum(1 for x in detection_counts if 1 <= x <= 5)}
- Images with 6-10 detections: {sum(1 for x in detection_counts if 6 <= x <= 10)}
- Images with >10 detections: {sum(1 for x in detection_counts if x > 10)}
"""
    
    # Save report
    with open(f"{save_dir}/test_report.txt", 'w') as f:
        f.write(report)
    
    print(report)
    
    # Create confidence histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_confidences, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Confidence Scores')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/confidence_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detection count comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(gt_counts, bins=15, alpha=0.7, label='Ground Truth', color='green')
    plt.hist(detection_counts, bins=15, alpha=0.7, label='Predictions', color='red')
    plt.xlabel('Number of Boxes per Image')
    plt.ylabel('Frequency')
    plt.title('Detection Count Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(gt_counts, detection_counts, alpha=0.6)
    plt.xlabel('Ground Truth Count')
    plt.ylabel('Predicted Count')
    plt.title('GT vs Predicted Count Correlation')
    plt.plot([0, max(max(gt_counts), max(detection_counts))], 
             [0, max(max(gt_counts), max(detection_counts))], 'r--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/detection_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main testing pipeline."""
    # Configuration
    model_path = "domino_detection/yolov8n_domino_tiles/weights/best.pt"  # Update this path
    data_yaml_path = "data.yaml"
    confidence_threshold = 0.25
    
    print("=== YOLOv8 Domino Tile Detection Testing ===")
    print(f"Working directory: {os.getcwd()}")
    print(f"Model path: {model_path}")
    print(f"Data YAML path: {data_yaml_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please update the model_path variable with the correct path to your trained model.")
        
        # Try to find the model automatically
        possible_paths = [
            "runs/detect/yolov8n_domino_tiles/weights/best.pt",
            "runs/detect/train/weights/best.pt",
            "runs/detect/exp/weights/best.pt",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found model at: {path}")
                model_path = path
                break
        else:
            print("Could not find model. Please check the path.")
            return
    
    # Check if data.yaml exists
    if not os.path.exists(data_yaml_path):
        print(f"Error: data.yaml not found at {data_yaml_path}")
        return
    
    # Load and display dataset config
    try:
        config = load_dataset_config(data_yaml_path)
        print(f"Dataset configuration:")
        print(f"  Train path: {config.get('train', 'Not specified')}")
        print(f"  Val path: {config.get('val', 'Not specified')}")
        print(f"  Classes: {config.get('nc', 'Not specified')}")
        print(f"  Names: {config.get('names', 'Not specified')}")
    except Exception as e:
        print(f"Error loading data.yaml: {e}")
        return
    
    # Run inference on all training images
    print(f"\nRunning inference with confidence threshold: {confidence_threshold}")
    results_data = run_inference_on_dataset(model_path, data_yaml_path, confidence_threshold)
    
    if not results_data:
        print("No results obtained. Please check your dataset paths and image files.")
        return
    
    # Calculate metrics
    print(f"\nCalculating metrics on {len(results_data)} images...")
    metrics = calculate_metrics(results_data, iou_threshold=0.5)
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_results(results_data, num_samples=6)
    
    # Generate detailed report
    print("\nGenerating detailed report...")
    create_detailed_report(results_data, metrics)
    
    print("\nTesting completed!")
    print(f"Results saved in 'test_results' and 'visualization_results' directories")

if __name__ == "__main__":
    main()