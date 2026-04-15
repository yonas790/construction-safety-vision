"""
========================================
 Safety Helmet Detection
 Evaluation & Comparison Script
 YOLOv8  vs  Faster R-CNN
========================================

Run AFTER training both models:
    python evaluate_compare.py

Outputs:
  - Metrics table (mAP, Precision, Recall, F1, FPS)
  - Side-by-side bar charts
  - results/comparison_report.txt
"""

import os, time, json
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ultralytics import YOLO

# ─────────────────────────────────────────
# PATHS — adjust if needed
# ─────────────────────────────────────────
YOLO_MODEL_PATH  = "results/yolov8/helmet_detection/weights/best.pt"
RCNN_MODEL_PATH  = "results/faster_rcnn/best_model.pth"
DATASET_YAML     = "dataset/data.yaml"
TEST_IMAGES_DIR  = "dataset/test/images"    # for FPS benchmark
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES      = 4   # background + 3 classes
N_BENCHMARK_IMGS = 50  # images to use for FPS test

os.makedirs("results", exist_ok=True)

# ─────────────────────────────────────────
# HELPER: IoU
# ─────────────────────────────────────────
def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + \
            (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return inter / (union + 1e-6)

# ─────────────────────────────────────────
# SECTION 1: YOLOv8 Metrics
# ─────────────────────────────────────────
def get_yolo_metrics():
    print("\n" + "="*50)
    print("  [1/3] Evaluating YOLOv8 on test set…")
    print("="*50)

    model   = YOLO(YOLO_MODEL_PATH)
    metrics = model.val(data=DATASET_YAML, split="test", verbose=False)

    return {
        "mAP50"     : metrics.box.map50,
        "mAP50_95"  : metrics.box.map,
        "precision" : metrics.box.mp,
        "recall"    : metrics.box.mr,
        "f1"        : 2 * metrics.box.mp * metrics.box.mr /
                      (metrics.box.mp + metrics.box.mr + 1e-6),
    }

# ─────────────────────────────────────────
# SECTION 2: FPS Benchmarks
# ─────────────────────────────────────────
def benchmark_fps():
    """Measure average inference FPS on test images."""
    test_imgs = sorted([
        os.path.join(TEST_IMAGES_DIR, f)
        for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])[:N_BENCHMARK_IMGS]

    if not test_imgs:
        print("⚠  No test images found for FPS benchmark.")
        return 0, 0

    # ── YOLOv8 FPS ──
    print("\n  Benchmarking YOLOv8 FPS…")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    t0 = time.time()
    for img_path in test_imgs:
        yolo_model(img_path, verbose=False)
    yolo_fps = len(test_imgs) / (time.time() - t0)

    # ── Faster R-CNN FPS ──
    print("  Benchmarking Faster R-CNN FPS…")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    rcnn    = fasterrcnn_resnet50_fpn(weights=weights)
    in_feat = rcnn.roi_heads.box_predictor.cls_score.in_features
    rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_feat, NUM_CLASSES)
    rcnn.load_state_dict(torch.load(RCNN_MODEL_PATH, map_location=DEVICE))
    rcnn.eval().to(DEVICE)

    tf = T.Compose([T.ToTensor()])
    t0 = time.time()
    with torch.no_grad():
        for img_path in test_imgs:
            img = Image.open(img_path).convert("RGB")
            rcnn([tf(img).to(DEVICE)])
    rcnn_fps = len(test_imgs) / (time.time() - t0)

    print(f"  YOLOv8 FPS      : {yolo_fps:.2f}")
    print(f"  Faster R-CNN FPS: {rcnn_fps:.2f}")
    return yolo_fps, rcnn_fps

# ─────────────────────────────────────────
# SECTION 3: Comparison Charts
# ─────────────────────────────────────────
def plot_comparison(yolo_metrics, rcnn_metrics, yolo_fps, rcnn_fps):
    """Generate side-by-side bar chart comparing both models."""

    metrics_labels = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1"]
    yolo_vals = [
        yolo_metrics["mAP50"], yolo_metrics["mAP50_95"],
        yolo_metrics["precision"], yolo_metrics["recall"], yolo_metrics["f1"]
    ]
    rcnn_vals = [
        rcnn_metrics.get("mAP50", 0), rcnn_metrics.get("mAP50_95", 0),
        rcnn_metrics.get("precision", 0), rcnn_metrics.get("recall", 0),
        rcnn_metrics.get("f1", 0)
    ]

    x     = np.arange(len(metrics_labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Chart 1: Detection Metrics
    ax = axes[0]
    bars1 = ax.bar(x - width/2, yolo_vals, width, label="YOLOv8",     color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width/2, rcnn_vals, width, label="Faster R-CNN", color="#FF5722", alpha=0.85)

    ax.set_title("Detection Metrics Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(metrics_labels, rotation=15)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    # Chart 2: FPS Speed
    ax2 = axes[1]
    fps_bars = ax2.bar(["YOLOv8", "Faster R-CNN"], [yolo_fps, rcnn_fps],
                       color=["#2196F3", "#FF5722"], alpha=0.85, width=0.4)
    ax2.set_title("Inference Speed (FPS)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Frames Per Second")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
    for bar in fps_bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.3,
                 f"{h:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_path = "results/comparison_chart.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n📊 Comparison chart saved: {out_path}")

# ─────────────────────────────────────────
# SECTION 4: Text Report
# ─────────────────────────────────────────
def save_report(yolo_m, rcnn_m, yolo_fps, rcnn_fps):
    winner_acc   = "YOLOv8" if yolo_m["mAP50"] > rcnn_m.get("mAP50", 0) else "Faster R-CNN"
    winner_speed = "YOLOv8" if yolo_fps > rcnn_fps else "Faster R-CNN"

    report = f"""
╔══════════════════════════════════════════════════════╗
║   SAFETY HELMET DETECTION — ALGORITHM COMPARISON     ║
╚══════════════════════════════════════════════════════╝

Dataset : Hard Hat Workers (Roboflow)
Classes : helmet | no_helmet | person
Device  : {DEVICE}

┌────────────────────┬──────────────┬────────────────┐
│ Metric             │   YOLOv8     │  Faster R-CNN  │
├────────────────────┼──────────────┼────────────────┤
│ mAP @ 0.5          │   {yolo_m['mAP50']:.4f}     │    {rcnn_m.get('mAP50', 0):.4f}       │
│ mAP @ 0.5:0.95     │   {yolo_m['mAP50_95']:.4f}     │    {rcnn_m.get('mAP50_95', 0):.4f}       │
│ Precision          │   {yolo_m['precision']:.4f}     │    {rcnn_m.get('precision', 0):.4f}       │
│ Recall             │   {yolo_m['recall']:.4f}     │    {rcnn_m.get('recall', 0):.4f}       │
│ F1 Score           │   {yolo_m['f1']:.4f}     │    {rcnn_m.get('f1', 0):.4f}       │
│ Avg FPS            │   {yolo_fps:.2f}       │    {rcnn_fps:.2f}          │
└────────────────────┴──────────────┴────────────────┘

🏆 Best Accuracy : {winner_acc}
🏆 Best Speed    : {winner_speed}

─── ANALYSIS ───────────────────────────────────────────

YOLOv8 (Single-Stage):
  • Processes the entire image in one forward pass
  • Significantly faster — suitable for real-time site monitoring
  • Slightly lower accuracy on small/occluded objects
  • Ideal for: live CCTV feeds, edge devices

Faster R-CNN (Two-Stage):
  • First proposes regions, then classifies — two-pass approach
  • Higher precision especially on overlapping detections
  • Slower inference — less suitable for live feeds
  • Ideal for: post-incident review, high-accuracy audits

─── RECOMMENDATION ─────────────────────────────────────

For real-time construction site monitoring:
  → YOLOv8 is the recommended production model.

For periodic compliance reporting:
  → Faster R-CNN provides higher accuracy.

A hybrid pipeline (YOLO for alerting, RCNN for verification)
would give the best of both worlds.
"""

    with open("results/comparison_report.txt", "w") as f:
        f.write(report)
    print(report)
    print("📄 Report saved: results/comparison_report.txt")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n🪖  Helmet Detection — Model Evaluation & Comparison")
    print("    Make sure both models are trained before running this.\n")

    # 1. YOLOv8 metrics via built-in validator
    yolo_metrics = get_yolo_metrics()

    # 2. FPS benchmark on both
    yolo_fps, rcnn_fps = benchmark_fps()

    # 3. For Faster R-CNN accuracy metrics, you need to run a COCO evaluator.
    #    We provide placeholder values here — replace with your actual eval results.
    #    (Full COCO eval requires pycocotools: pip install pycocotools)
    print("\n  ℹ  Faster R-CNN accuracy metrics: enter manually from your training logs,")
    print("     OR install pycocotools and run full COCO eval.")
    rcnn_metrics = {
        "mAP50"     : float(input("  R-CNN mAP@0.5      (e.g. 0.82): ") or 0),
        "mAP50_95"  : float(input("  R-CNN mAP@0.5:0.95 (e.g. 0.55): ") or 0),
        "precision" : float(input("  R-CNN Precision    (e.g. 0.85): ") or 0),
        "recall"    : float(input("  R-CNN Recall       (e.g. 0.80): ") or 0),
    }
    rcnn_metrics["f1"] = (2 * rcnn_metrics["precision"] * rcnn_metrics["recall"] /
                          (rcnn_metrics["precision"] + rcnn_metrics["recall"] + 1e-6))

    # 4. Charts + report
    plot_comparison(yolo_metrics, rcnn_metrics, yolo_fps, rcnn_fps)
    save_report(yolo_metrics, rcnn_metrics, yolo_fps, rcnn_fps)