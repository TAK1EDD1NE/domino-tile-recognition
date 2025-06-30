🁢 Domino Tile Recognition
This project implements a computer vision pipeline that detects and classifies domino tiles from an image. It combines object detection to locate the tiles and a Convolutional Neural Network (CNN) to predict the identity (number) of each tile.

🧠 How It Works
📷 Image Input: A photo with multiple domino tiles is fed into the pipeline.

🔍 Detection Phase: An object detection model (YOLO) locates individual domino tiles in the image.

🧩 Classification Phase: Each detected tile is passed to a CNN model trained to classify all possible domino configurations.

🖼️ Output: The system returns each tile’s bounding box and predicted label and the total points.

📊 Model Visualization
the cnn model have an accuracy of 97%

domino_confusion_matrix.png

📌 Recommended: add screenshots showing the detection + classification results here.

🖥️ Running the Project
bash
Copy
Edit
# Step 1: Clone the repo
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the pipeline
python main.py --input path/to/image.jpg
Replace main.py and CLI arguments with your actual entry point.

⚠️ Known Issues
The CNN model currently shows low accuracy on real-world images, especially under variable lighting or rotation.

It performs better on clean, synthetic or preprocessed tile images.