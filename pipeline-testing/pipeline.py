import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import argparse
import os
from typing import List, Tuple, Dict

class DominoPipeline:
    """
    Complete pipeline for detecting and classifying domino tiles in images.
    
    Combines YOLO detection with CNN classification to identify individual
    domino tiles and calculate their scores.
    """
    
    def __init__(self, yolo_model_path: str, cnn_model_path: str):
        """
        Initialize the pipeline with pre-trained models.
        
        Args:
            yolo_model_path: Path to the YOLO model (.pt file)
            cnn_model_path: Path to the CNN model (.h5 or .pt file)
        """
        self.yolo_model = self._load_yolo_model(yolo_model_path)
        self.cnn_model = self._load_cnn_model(cnn_model_path)
        self.class_id_to_tile = self._create_class_mapping()
        
    def _load_yolo_model(self, model_path: str) -> YOLO:
        """Load the YOLO model from disk."""
        try:
            model = YOLO(model_path)
            print(f"YOLO model loaded from {model_path}")
            return model
        except Exception as e:
            raise Exception(f"Failed to load YOLO model: {e}")
    
    def _load_cnn_model(self, model_path: str):
        """Load the CNN model from disk (supports both TensorFlow and PyTorch)."""
        try:
            if model_path.endswith('.keras'):
                # TensorFlow/Keras model
                model = tf.keras.models.load_model(model_path)
                print(f"CNN model (TensorFlow) loaded from {model_path}")
            elif model_path.endswith('.pt'):
                # PyTorch model - you'll need to adapt this based on your model architecture
                import torch
                model = torch.load(model_path, map_location='cpu')
                model.eval()
                print(f"CNN model (PyTorch) loaded from {model_path}")
            else:
                raise ValueError("CNN model must be .keras (TensorFlow) or .pt (PyTorch)")
            return model
        except Exception as e:
            raise Exception(f"Failed to load CNN model: {e}")
    
    def _create_class_mapping(self) -> Dict[int, Tuple[int, int]]:
        """Create mapping from class ID to domino tile values."""
        mapping = {}
        class_id = 0
        for i in range(7):  # 0 to 6 dots
            for j in range(i, 7):  # Only upper triangular to avoid duplicates
                mapping[class_id] = (i, j)
                class_id += 1
        return mapping
    
    def detect_tiles(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect domino tiles in the image using YOLO.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of bounding boxes as (x1, y1, x2, y2) tuples
        """
        results = self.yolo_model(image, verbose=False)
        bboxes = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Extract coordinates and convert to integers
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    bboxes.append((x1, y1, x2, y2))
        
        print(f"Detected {len(bboxes)} domino tiles")
        return bboxes
    
    def crop_tile(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop a single tile from the image based on bounding box.
        
        Args:
            image: Original image
            bbox: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            Cropped tile image
        """
        x1, y1, x2, y2 = bbox
        # Add small padding to ensure we get the full tile
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        return image[y1:y2, x1:x2]
    
    def preprocess_for_cnn(self, tile_image: np.ndarray, target_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """
        Preprocess cropped tile for CNN classification.
        Matches the preprocessing used during training: resize to 100x100 and rescale by 1/255.
        
        Args:
            tile_image: Cropped tile image
            target_size: Target size for CNN input (100x100 to match training)
            
        Returns:
            Preprocessed image ready for CNN
        """
        # Resize to target size (100x100 as used in training)
        resized = cv2.resize(tile_image, target_size)
        
        # Convert BGR to RGB (OpenCV uses BGR, but your model likely expects RGB)
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Rescale by 1.0/255 (same as your training preprocessing)
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension to match model input shape: (1, 100, 100, 3)
        return np.expand_dims(normalized, axis=0)
    
    def classify_tile(self, tile_image: np.ndarray) -> Tuple[Tuple[int, int], float]:
        """
        Classify a single domino tile using the CNN.
        
        Args:
            tile_image: Cropped tile image
            
        Returns:
            Tuple of (tile_values, confidence_score)
        """
        # Preprocess the image
        processed_image = self.preprocess_for_cnn(tile_image)
        
        # Make prediction
        if hasattr(self.cnn_model, 'predict'):  # TensorFlow model
            predictions = self.cnn_model.predict(processed_image, verbose=0)
            print(predictions)
            class_id = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
        else:  # PyTorch model
            import torch
            with torch.no_grad():
                tensor_image = torch.from_numpy(processed_image).permute(0, 3, 1, 2)
                predictions = self.cnn_model(tensor_image)
                class_id = torch.argmax(predictions[0]).item()
                confidence = torch.softmax(predictions[0], dim=0).max().item()
        
        # Map class ID to tile values
        tile_values = self.class_id_to_tile.get(class_id, (0, 0))
        
        return tile_values, confidence
    def calculate_score(self, tile_values: Tuple[int, int]) -> int:
        """Calculate the score for a domino tile (sum of both numbers)."""
        return sum(tile_values)
    
    def visualize_results(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]], 
                         classifications: List[Tuple[Tuple[int, int], float]], 
                         total_score: int, save_path: str = None) -> np.ndarray:
        """
        Visualize the results by drawing bounding boxes and labels on the image.
        
        Args:
            image: Original image
            bboxes: List of bounding boxes
            classifications: List of (tile_values, confidence) tuples
            total_score: Total score of all tiles
            save_path: Optional path to save the result image
            
        Returns:
            Image with visualizations
        """
        # Create a copy of the image for drawing
        result_image = image.copy()
        
        # Define colors and font settings
        bbox_color = (0, 255, 0)  # Green for bounding boxes
        text_color = (255, 255, 255)  # White for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Draw bounding boxes and labels for each tile
        for bbox, (tile_values, confidence) in zip(bboxes, classifications):
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), bbox_color, thickness)
            
            # Create label text
            label = f"{tile_values[0]}–{tile_values[1]}"
            score = self.calculate_score(tile_values)
            
            # Calculate text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(result_image, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width + 10, y1), 
                         bbox_color, -1)
            
            # Draw label text
            cv2.putText(result_image, label, (x1 + 5, y1 - 5), 
                       font, font_scale, text_color, thickness)
        
        # Draw total score at the top of the image
        total_text = f"Total Score: {total_score}"
        (total_width, total_height), _ = cv2.getTextSize(total_text, font, 1.0, 3)
        
        # Draw background for total score
        cv2.rectangle(result_image, (10, 10), (total_width + 20, total_height + 20), 
                     (0, 0, 0), -1)
        
        # Draw total score text
        cv2.putText(result_image, total_text, (15, 35), 
                   font, 1.0, (0, 255, 255), 3)  # Yellow text
        
        # Save image if path provided
        if save_path:
            cv2.imwrite(save_path, result_image)
            print(f"Result saved to {save_path}")
        
        return result_image
    
    def process_image(self, image_path: str, save_path: str = None, display: bool = True) -> Dict:
        """
        Complete pipeline to process a single image.
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save result
            display: Whether to display the result using matplotlib
            
        Returns:
            Dictionary with processing results
        """
        print(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Detect tiles
        bboxes = self.detect_tiles(image)
        
        if not bboxes:
            print("No domino tiles detected in the image")
            return {"total_score": 0, "tiles": [], "image": image}
        
        # Classify each detected tile
        classifications = []
        total_score = 0
        
        print("Classifying detected tiles...")
        for i, bbox in enumerate(bboxes):
            # Crop tile
            tile_crop = self.crop_tile(image, bbox)
            
            # Classify tile
            tile_values, confidence = self.classify_tile(tile_crop)
            classifications.append((tile_values, confidence))
            
            # Calculate score
            score = self.calculate_score(tile_values)
            total_score += score
            
            print(f"  Tile {i+1}: {tile_values[0]}–{tile_values[1]} (score: {score}, confidence: {confidence:.3f})")
        
        print(f"Total score: {total_score}")
        
        # Visualize results
        result_image = self.visualize_results(image, bboxes, classifications, total_score, save_path)
        
        # Display result if requested
        if display:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Domino Tile Detection and Classification (Total Score: {total_score})")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return {
            "total_score": total_score,
            "tiles": [{"bbox": bbox, "values": values, "confidence": conf, "score": self.calculate_score(values)} 
                     for bbox, (values, conf) in zip(bboxes, classifications)],
            "image": result_image
        }


def main():
    """Main function to run the pipeline from command line."""
    parser = argparse.ArgumentParser(description="Domino Tile Detection and Classification Pipeline")
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument("--yolo", "-y", default="best.pt", help="Path to YOLO model (default: best.pt)")
    parser.add_argument("--cnn", "-c", required=True, help="Path to CNN model (.h5 or .pt)")
    parser.add_argument("--output", "-o", help="Path to save output image")
    parser.add_argument("--no-display", action="store_true", help="Don't display the result")
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = DominoPipeline(args.yolo, args.cnn)
        
        # Process image
        results = pipeline.process_image(
            image_path=args.image,
            save_path=args.output,
            display=not args.no_display
        )
        
        print(f"\n{'='*50}")
        print(f"FINAL RESULTS:")
        print(f"Total Score: {results['total_score']}")
        print(f"Tiles Detected: {len(results['tiles'])}")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Example usage when running as script
    # python domino_pipeline.py -i image.jpg -c cnn_model.h5 -o result.jpg
    
    # Or use directly as a class:
    # pipeline = DominoPipeline("best.pt", "cnn_model.h5")
    # results = pipeline.process_image("image.jpg")
    
    exit(main())