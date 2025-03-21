# Underwater Object Detection using YOLOv8

## Project Overview
The goal of this project is to build an object detection system capable of identifying various marine creatures in underwater images using the YOLOv8 object detection model. This system can help with marine conservation, monitoring, and research.

### Features:
- **YOLOv8 Model**: Utilizes the latest version of the YOLO architecture.
- **Classes**: Detects seven underwater classes: fish, jellyfish, penguin, puffin, shark, starfish, and stingray.
- **Custom Dataset**: Trained on a custom dataset with annotated underwater images.

## Dataset
The dataset is organized as follows:


### Classes:
The dataset contains the following classes:
- Fish
- Jellyfish
- Penguin
- Puffin
- Shark
- Starfish
- Stingray

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/ShahidHasib586/Underwater-object-detection-using-yolov8.git


Install the required dependencies:

The required dependencies include:

PyTorch
Ultralytics (YOLOv8)
OpenCV
Matplotlib
Seaborn
Install YOLOv8 directly from the Ultralytics package:

pip install ultralytics


Training
Prepare the Dataset: Ensure your dataset follows the structure shown in the Dataset section.

Configure the data.yaml file: Edit the data.yaml file to match your dataset paths and classes.

Train the model: Run the following command to start training:

from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # or 'yolov8s.pt', 'yolov8m.pt', etc.

# Train the model
model.train(data='data.yaml', epochs=100, imgsz=640, batch=8, device='cpu')


You can adjust parameters such as epochs, imgsz, and batch size based on your requirements and hardware capabilities.

Evaluation
To evaluate the trained model on the validation or test set, run:

results = model.val(data='data.yaml', split='val')


This will calculate metrics like precision, recall, and mAP (mean Average Precision).

Visualization
You can visualize random samples from the dataset along with the bounding boxes by using the provided function visualize_image_with_annotation_bboxes().

# Visualize 12 sample images with bounding boxes
visualize_image_with_annotation_bboxes(train_images, train_labels)
Results
After training, the model will output the performance metrics, including mAP scores, and save the best model weights in the runs/detect/ folder.

Acknowledgments
This project leverages the Ultralytics YOLOv8 framework for object detection. Special thanks to the open-source community for developing and maintaining this powerful tool.
