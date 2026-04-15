"""
========================================
 Safety Helmet Detection — YOLOv8
 Algorithm 1: Single-Stage Detector
========================================

SETUP (run once in terminal / Colab):
    pip install ultralytics opencv-python matplotlib

DATASET:
    Download from: https://public.roboflow.com/object-detection/hard-hat-workers
    Choose export format: YOLOv8
    Extract to:  dataset/
    Your folder should look like:
        dataset/
            data.yaml
            train/images/  train/labels/
            valid/images/  valid/labels/
            test/images/   test/labels/
"""

import os
from ultralytics import YOLO
import yaml

# ─────────────────────────────────────────
# CONFIG — edit these paths if needed
# ─────────────────────────────────────────
DATASET_YAML  = "dataset/data.yaml"   # path to dataset config
MODEL_SIZE    = "yolov8n.pt"          # n=nano(fast), s=small, m=medium
EPOCHS        = 50
IMG_SIZE      = 640
BATCH_SIZE    = 16
PROJECT_NAME  = "results/yolov8"
RUN_NAME      = "helmet_detection"

def train():
    print("=" * 50)
    print("  YOLOv8 — Training Safety Helmet Detector")
    print("=" * 50)

    # Load pre-trained YOLOv8 model (downloads automatically)
    model = YOLO(MODEL_SIZE)

    # Train
    results = model.train(
        data      = DATASET_YAML,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH_SIZE,
        project   = PROJECT_NAME,
        name      = RUN_NAME,
        patience  = 10,          # early stopping
        save      = True,
        plots     = True,        # saves training curves
        verbose   = True,
    )

    print("\n✅ Training complete!")
    print(f"   Best model saved at: {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    return results

def validate():
    """Run validation on the test set and print metrics."""
    best_model_path = f"{PROJECT_NAME}/{RUN_NAME}/weights/best.pt"

    if not os.path.exists(best_model_path):
        print("❌ No trained model found. Run train() first.")
        return

    print("\n── Validating on test set ──")
    model = YOLO(best_model_path)
    metrics = model.val(data=DATASET_YAML, split="test")

    print(f"\n📊 YOLOv8 Test Metrics:")
    print(f"   mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"   mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"   Precision    : {metrics.box.mp:.4f}")
    print(f"   Recall       : {metrics.box.mr:.4f}")

    return metrics

if __name__ == "__main__":
    train()
    validate()