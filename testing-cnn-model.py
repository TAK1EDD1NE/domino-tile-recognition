import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration - match your training setup
BATCH_SIZE = 20
IMAGE_SIZE = (100, 100)

def flow_from_datagenerator(datagen, data, batch_size=BATCH_SIZE, shuffle=False, all_classes=None):
    """Returns a generator from an ImageDataGenerator and a dataframe."""
    
    # Print debug information
    print(f"DataFrame shape: {data.shape}")
    print(f"Categories found: {data['category'].unique()}")
    print(f"Expected classes: {all_classes}")
    
    # Check if paths exist
    existing_paths = data['path'].apply(lambda x: Path(x).exists())
    print(f"Existing paths: {existing_paths.sum()}/{len(data)}")
    
    if not existing_paths.all():
        print("Some paths don't exist. First few missing:")
        missing = data[~existing_paths]['path'].head()
        for path in missing:
            print(f"  Missing: {path}")
    
    return datagen.flow_from_dataframe(
        dataframe=data,
        x_col="path",
        y_col="category",
        class_mode='categorical',
        batch_size=batch_size,
        target_size=IMAGE_SIZE,
        shuffle=shuffle,
        classes=all_classes,
        validate_filenames=False  # This might help with path issues
    )

def load_test_data(test_folder_path):
    """
    Load test data from folder structure or CSV file.
    Modify this function based on your data structure.
    """
    # Option 1: If you have a CSV file with paths and categories
    # test_data = pd.read_csv('test_data.csv')
    
    # Option 2: If you have folder structure (category/image.jpg)
    test_data = []
    test_folder = Path(test_folder_path)
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    print(f"Scanning folder: {test_folder}")
    
    for category_folder in test_folder.iterdir():
        if category_folder.is_dir():
            category = category_folder.name
            print(f"Found category folder: {category}")
            
            image_count = 0
            for image_file in category_folder.iterdir():
                if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                    # Use absolute path to ensure it's found
                    abs_path = str(image_file.resolve())
                    test_data.append({
                        'path': abs_path,
                        'category': category
                    })
                    image_count += 1
            
            print(f"  Found {image_count} images in {category}")
    
    df = pd.DataFrame(test_data)
    
    # Verify paths exist
    if len(df) > 0:
        missing_files = []
        for idx, row in df.iterrows():
            if not Path(row['path']).exists():
                missing_files.append(row['path'])
        
        if missing_files:
            print(f"Warning: {len(missing_files)} files don't exist!")
            print("First few missing files:", missing_files[:5])
    
    return df

def evaluate_model(model_path, test_data_path, all_classes):
    """
    Comprehensive model evaluation on test dataset.
    """
    # Load the trained model
    print("Loading trained model...")
    model = load_model(model_path)
    
    # Load test data
    print("Loading test data...")
    test_data = load_test_data(test_data_path)
    print(f"Found {len(test_data)} test images")
    
    # Create test data generator (no augmentation, no shuffle)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = flow_from_datagenerator(test_datagen, test_data, shuffle=False)
    
    # Calculate steps
    test_steps = test_generator.n // test_generator.batch_size
    if test_generator.n % test_generator.batch_size != 0:
        test_steps += 1  # Include the last partial batch
    
    print(f"Test samples: {test_generator.n}")
    print(f"Test steps: {test_steps}")
    
    # Reset generator
    test_generator.reset()
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps, verbose=1)
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Get predictions
    print("\nGenerating predictions...")
    test_generator.reset()
    predictions = model.predict(test_generator, steps=test_steps, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_labels = test_generator.classes[:len(predicted_classes)]
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_classes, 
                              target_names=all_classes, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=all_classes, yticklabels=all_classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(all_classes):
        print(f"{class_name}: {per_class_accuracy[i]:.4f}")
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_labels': true_labels,
        'confusion_matrix': cm
    }

def predict_single_image(model_path, image_path, all_classes):
    """
    Predict a single image.
    """
    from tensorflow.keras.preprocessing import image
    
    model = load_model(model_path)
    
    # Load and preprocess image
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Predict
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction)
    predicted_class = all_classes[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx]
    
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    
    # Show top 3 predictions
    top_3_idx = np.argsort(prediction[0])[-3:][::-1]
    print("\nTop 3 predictions:")
    for i, idx in enumerate(top_3_idx):
        print(f"{i+1}. {all_classes[idx]}: {prediction[0][idx]:.4f}")
    
    return predicted_class, confidence

# Example usage
if __name__ == "__main__":
    # Define your classes (should match training)
    all_classes = ['0x0','1x0','1x1','2x0','2x1','2x2','3x0','3x1','3x2','3x3','4x0','4x1','4x2','4x3','4x4','5x0','5x1','5x2','5x3','5x4','5x5','6x0','6x1','6x2','6x3','6x4','6x5','6x6']  # Replace with your actual classes
    
    # Paths
    model_path = 'C:\\Users\\pc\\Desktop\\domino recognition\\best_model_1000_epochs.keras'  # Your trained model
    test_data_path = 'C:\\Users\\pc\\Desktop\\domino recognition\\archive2\\images'       # Your test data folder
    
    # Run comprehensive evaluation
    results = evaluate_model(model_path, test_data_path, all_classes)
    
    # Optional: Test single image
    # single_image_path = 'path/to/single/test/image.jpg'
    # predict_single_image(model_path, single_image_path, all_classes)
    
    print("\nTesting completed!")