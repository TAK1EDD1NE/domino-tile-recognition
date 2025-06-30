import numpy as np
import pandas as pd
import os
from keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from plot_saving_funcs import save_all_results

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Configuration - same as training
DATA_DIRECTORY = os.path.abspath("C:\\Users\\pc\\Desktop\\domino recognition\\archive\\data")
BATCH_SIZE = 20
IMAGE_SIZE = (100, 100)

# Define all classes in the same order as training
all_classes = [f"{i}x{j}" for i in range(7) for j in range(0, i + 1)]
print("All classes:", all_classes)
print(f"Total number of classes: {len(all_classes)}")

def categorized_from_directory(path):
    """Returns a Pandas dataframe with the `category` and `path` of each image."""
    rows = []
    supported_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG'}
    
    for category in os.listdir(path):
        category_path = os.path.join(path, category)
        if os.path.isdir(category_path):
            for image in os.listdir(category_path):
                # Check if file has supported extension
                _, ext = os.path.splitext(image)
                if ext in supported_extensions:
                    image_path = os.path.join(category_path, image)
                    rows.append({'category': category, 'path': image_path})
    return pd.DataFrame(rows)

def flow_from_datagenerator(datagen, data, batch_size=BATCH_SIZE, shuffle=False):
    """Returns a generator from an ImageDataGenerator and a dataframe."""
    return datagen.flow_from_dataframe(
        dataframe=data,
        x_col="path",
        y_col="category", 
        class_mode='categorical',
        batch_size=batch_size,
        target_size=IMAGE_SIZE,
        shuffle=shuffle,  # Set to False for testing to maintain order
        classes=all_classes)

def load_and_prepare_test_data():
    """Load and prepare all test data from the data directory."""
    print("Loading test data from:", DATA_DIRECTORY)
    
    # Load all data from the directory
    test_data = categorized_from_directory(DATA_DIRECTORY)
    
    print(f"Total images found: {len(test_data)}")
    print("\nImages per class:")
    class_counts = test_data.groupby('category').size()
    print(class_counts)
    
    # Create test data generator (same preprocessing as training, but no augmentation)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = flow_from_datagenerator(test_datagen, test_data, shuffle=False)
    
    return test_data, test_generator

def evaluate_model(model_path, test_generator, test_data):
    """Load model and evaluate on test data."""
    print(f"\nLoading model from: {model_path}")
    
    try:
        model = models.load_model(model_path)
        print("Model loaded successfully!")
        print("\nModel summary:")
        model.summary()
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Calculate steps for complete evaluation
    test_steps = len(test_data) // BATCH_SIZE
    if len(test_data) % BATCH_SIZE != 0:
        test_steps += 1  # Add one more step for remaining samples
    
    print(f"\nEvaluating model on {len(test_data)} images...")
    print(f"Test steps: {test_steps}")
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps, verbose=1)
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    return model, test_loss, test_accuracy

def detailed_predictions(model, test_generator, test_data):
    """Generate detailed predictions and classification report."""
    print("\nGenerating detailed predictions...")
    
    # Calculate steps for complete prediction
    test_steps = len(test_data) // BATCH_SIZE
    if len(test_data) % BATCH_SIZE != 0:
        test_steps += 1
    
    # Get predictions
    predictions = model.predict(test_generator, steps=test_steps, verbose=1)
    
    # Get predicted classes
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true classes
    true_classes = test_generator.classes[:len(predicted_classes)]
    
    # Get class labels
    class_labels = list(test_generator.class_indices.keys())
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Number of predictions: {len(predicted_classes)}")
    print(f"Number of true labels: {len(true_classes)}")
    
    # Calculate accuracy manually
    correct_predictions = np.sum(predicted_classes == true_classes)
    accuracy = correct_predictions / len(true_classes)
    print(f"Manual accuracy calculation: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\nClassification Report:")
    report = classification_report(true_classes, predicted_classes, 
                                 target_names=class_labels, 
                                 output_dict=True)
    print(classification_report(true_classes, predicted_classes, 
                              target_names=class_labels))
    
    return predicted_classes, true_classes, class_labels, report

def plot_confusion_matrix(true_classes, predicted_classes, class_labels, save_path="confusion_matrix.png"):
    """Plot confusion matrix."""
    print("\nGenerating confusion matrix...")
    
    # Create confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix - Domino Tile Classification')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm

def analyze_class_performance(report, class_labels):
    """Analyze per-class performance."""
    print("\nPer-class Performance Analysis:")
    print("-" * 60)
    
    class_metrics = []
    for class_name in class_labels:
        if class_name in report:
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1_score = report[class_name]['f1-score']
            support = report[class_name]['support']
            
            class_metrics.append({
                'Class': class_name,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1_score,
                'Support': support
            })
            
            print(f"{class_name:>6}: Precision={precision:.3f}, Recall={recall:.3f}, "
                  f"F1={f1_score:.3f}, Support={support}")
    
    # Convert to DataFrame for easy analysis
    metrics_df = pd.DataFrame(class_metrics)
    
    print(f"\nOverall Statistics:")
    print(f"Mean Precision: {metrics_df['Precision'].mean():.3f}")
    print(f"Mean Recall: {metrics_df['Recall'].mean():.3f}")
    print(f"Mean F1-Score: {metrics_df['F1-Score'].mean():.3f}")
    print(f"Total Support: {metrics_df['Support'].sum()}")
    
    # Find best and worst performing classes
    best_f1 = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
    worst_f1 = metrics_df.loc[metrics_df['F1-Score'].idxmin()]
    
    print(f"\nBest performing class: {best_f1['Class']} (F1={best_f1['F1-Score']:.3f})")
    print(f"Worst performing class: {worst_f1['Class']} (F1={worst_f1['F1-Score']:.3f})")

    return metrics_df

def main():
    """Main testing function."""
    print("=" * 60)
    print("DOMINO TILE CNN MODEL TESTING")
    print("=" * 60)
    
    # Load and prepare test data
    test_data, test_generator = load_and_prepare_test_data()
    
    # Path to your trained model
    model_path = "best_model_1000_epochs.keras"  # Update this path if needed
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please make sure the model file is in the correct location.")
        return
    
    # Evaluate model
    model, test_loss, test_accuracy = evaluate_model(model_path, test_generator, test_data)
    
    if model is None:
        return
    
    # Get detailed predictions
    predicted_classes, true_classes, class_labels, report = detailed_predictions(model, test_generator, test_data)
    
    # Plot confusion matrix
    confusion_matrix_result = plot_confusion_matrix(true_classes, predicted_classes, class_labels, "domino_confusion_matrix.png")
    
    # Analyze class performance
    metrics_df = analyze_class_performance(report, class_labels)

    #analytics
    save_all_results(metrics_df, test_accuracy, test_loss, len(test_data))

    
    print("\n" + "=" * 60)
    print("TESTING COMPLETED!")
    print("=" * 60)
    print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Total images tested: {len(test_data)}")

if __name__ == "__main__":
    main()