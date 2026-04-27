# Construction Safety Vision: Real-Time Helmet Detection
This is a Computer Vision and Image Processing project for Computer Science. It is a comprehensive construction site safety system that monitors employees and ensures they are wearing safety helmets.

## Features
- **Real-Time Detection**: Detects people, heads, helmets, and unsafe conditions.
- **Two Detection Modes**:
  - **YOLOv8**: Ultra-fast and accurate real-time detection.
  - **Faster R-CNN**: Robust, research-grade detection with heatmap visualization.
- **Side-by-Side Comparison**: Evaluate and compare the performance of both models.
- **Visualization**: Heatmaps, bounding boxes, and confidence scores.
- **CPU Optimized**: Efficient implementations for running on standard hardware.

## Setup

### 1. Dependencies
Install required libraries for both YOLOv8 and Faster R-CNN.
```bash
pip install torch torchvision opencv-python matplotlib

# For YOLOv8 specific features
pip install ultralytics seaborn
```

### 2. Dataset Preparation
Ensure your dataset is organized in the following structure:
```
dataset/
├── train/
│   ├── images/
│   ├── labels/
│   └── _annotations.coco.json
├── valid/
│   ├── images/
│   ├── labels/
│   └── _annotations.coco.json
└── test/
    ├── images/
    ├── labels/
    └── _annotations.coco.json
```

Create a `data.yaml` file in the `dataset/` directory:
```yaml
names:
  0: head
  1: helmet
  2: no_helmet
nc: 3
train: ../dataset/train/images
val: ../dataset/valid/images
test: ../dataset/test/images
```

## Usage

### 1. Train YOLOv8
Train the YOLOv8 model on your dataset.
```bash
python yolov8/train_yolo.py
```
- Output: `runs/detect/results/yolov8/helmet_detection/weights/best.pt`

### 2. Train Faster R-CNN
Train the Faster R-CNN model on your dataset.
```bash
python faster_rcnn/train_rcnn.py
```
- Output: `faster_rcnn/results/best_model.pth`

### 3. Run Inference
Use the trained models to detect objects in images.

**YOLOv8 Inference:**
```bash
# Detect on a single image
python yolov8/detect_yolo.py --source path/to/image.jpg

# Detect on a video file
python yolov8/detect_yolo.py --source path/to/video.mp4

# Real-time detection using webcam
python yolov8/detect_yolo.py --source webcam
```

**Faster R-CNN Inference:**
```bash
# Detect on a single image
python faster_rcnn/detect_rcnn.py --image path/to/image.jpg

# Detect with custom threshold (e.g., 0.5)
python faster_rcnn/detect_rcnn.py --image path/to/image.jpg --threshold 0.5

# Detect on a whole folder of images
python faster_rcnn/detect_rcnn.py --folder dataset/test/images/
```

## Evaluation & Comparison
Compare the performance of both models on the test set.
```bash
python evaluate_compare.py
```
This script will generate:
- Precision, Recall, F1-score for both models.
- Speed comparison (FPS).
- mAP (mean Average Precision) estimates.
- Comparative plots for easy visualization.

## Project Structure
```
construction-safety-vision/
├── yolov8/                     # YOLOv8 implementation
│   ├── train_yolo.py
│   ├── detect_yolo.py
│   └── ...
├── faster_rcnn/                # Faster R-CNN implementation
│   ├── train_rcnn.py
│   ├── detect_rcnn.py
│   └── ...
├── dataset/                    # Training dataset
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── evaluate_compare.py         # Model comparison script
└── README.md
```
