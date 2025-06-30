import numpy as np
import os
from keras import models
from tensorflow.keras.preprocessing.image import load_img, img_to_array #type:ignore
import tensorflow as tf

# Import functions from the main testing script
from testing_cnn_model import all_classes  # Import the class definitions

# Configuration
IMAGE_SIZE = (100, 100)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def test_single_image(model_path, image_path):
    """Test a single image and return prediction."""
    
    # Load the trained model
    print(f"Loading model from: {model_path}")
    model = models.load_model(model_path)
    
    # Load and preprocess the image
    print(f"Loading image: {image_path}")
    
    # Load image with same preprocessing as training
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Same rescaling as training
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    # Get class name
    predicted_class = all_classes[predicted_class_index]
    
    # Print results
    print(f"\nPrediction Results:")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    # Show top 3 predictions
    print(f"\nTop 3 Predictions:")
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    for i, idx in enumerate(top_3_indices, 1):
        class_name = all_classes[idx]
        conf = predictions[0][idx]
        print(f"{i}. {class_name}: {conf:.4f} ({conf*100:.2f}%)")
    
    return predicted_class, confidence

def main():
    """Main function to test a single image."""
    
    # Paths - update these as needed
    model_path = "best_model_1000_epochs.keras"
    image_path = input("Enter the path to the image you want to test: ").strip()
    
    # Remove quotes if user copied path with quotes
    image_path = image_path.strip('"').strip("'")
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    # Test the image
    try:
        predicted_class, confidence = test_single_image(model_path, image_path)
        print(f"\n{'='*50}")
        print(f"FINAL RESULT: {predicted_class} (Confidence: {confidence*100:.2f}%)")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()