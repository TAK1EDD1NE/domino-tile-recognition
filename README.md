🁢 Domino Tile Recognition
This project implements a computer vision pipeline that detects and classifies domino tiles from an image. It combines object detection to locate the tiles and a Convolutional Neural Network (CNN) to predict the identity (number) of each tile.

🧠 How It Works
📷 Image Input: A photo with multiple domino tiles is fed into the pipeline.

🔍 Detection Phase: An object detection model (YOLO) locates individual domino tiles in the image.

🧩 Classification Phase: Each detected tile is passed to a CNN model trained to classify all possible domino configurations.

🖼️ Output: The system returns each tile’s bounding box and predicted label and the total points.

📊 Model Visualization
## cnn model:
the cnn model have an accuracy of 97%


![Prediction Example](/domino_confusion_matrix.png)
![Prediction Example](/domino_accuracy_summary.png)
![Prediction Example](/domino_class_performance.png)

to run the cnn model:
`
python testing_cnn_model.py
`
## object detection model(yolo):
Dataset Statistics:
- Total Images: 174
- Total Ground Truth Boxes: 756
- Total Predicted Boxes: 780
- Average GT boxes per image: 4.34
- Average predicted boxes per image: 4.48

Performance Metrics:
- True Positives: 762
- Precision: 0.9769
- Recall: 1.0079
- F1-Score: 0.9922

Confidence Statistics:
- Mean Confidence: 0.9273
- Median Confidence: 0.9801
- Min Confidence: 0.2549
- Max Confidence: 0.9999

Detection Distribution:
- Images with 0 detections: 0
- Images with 1-5 detections: 138
- Images with 6-10 detections: 0
- Images with >10 detections: 36

![object detection Example](/Double%20Six%20Domino.v8i.yolov5pytorch/testing-object-detection.py)
![object detection Example](/Double%20Six%20Domino.v8i.yolov5pytorch/train-object-detection-model.py)

to run the cnn model:
`
python testing-object-detection.py
`



# Step 1: Clone the repo
git clone https://github.com/TAK1EDD1NE/domino-tile-recognition.git
cd your-repo

# Step 2: Install dependencies

# Step 3: Run the pipeline
`
cd pipeline
python pipeline.py -i image.jpg -c best_model_1000_epochs.keras -o result.jpg
`
⚠️ Known Issues
The CNN model currently shows low accuracy on real-world images, especially under variable lighting or rotation.

It performs better on clean, synthetic or preprocessed tile images.