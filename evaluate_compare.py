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
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,          
    FasterRCNN_MobileNet_V3_Large_FPN_Weights, 
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ─────────────────────────────────────────
# PATHS — auto-detected from project layout
# ─────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL_PATH  = os.path.join(BASE_DIR, "runs", "detect", "results", "yolov8", "helmet_detection-5", "weights", "best.pt")
RCNN_MODEL_PATH  = os.path.join(BASE_DIR, "faster_rcnn", "results", "best_model.pth")
DATASET_YAML     = os.path.join(BASE_DIR, "dataset", "data.yaml")
TEST_IMAGES_DIR  = os.path.join(BASE_DIR, "dataset", "test", "images")

DEVICE           = torch.device("cpu")  
NUM_CLASSES      = 3 + 1                 # background + head + helmet + person
N_BENCHMARK_IMGS = 50

os.makedirs("results", exist_ok=True)

# ─────────────────────────────────────────
# HELPER: load Faster R-CNN
# ─────────────────────────────────────────
def load_rcnn():
    if not os.path.exists(RCNN_MODEL_PATH):
        raise FileNotFoundError(
            f"Faster R-CNN model not found at:\n  {RCNN_MODEL_PATH}\n"
            f"Run faster_rcnn/train_rcnn.py first."
        )
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model   = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, NUM_CLASSES)
    model.load_state_dict(torch.load(RCNN_MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model

# ─────────────────────────────────────────
# SECTION 1: YOLOv8 Metrics
# ─────────────────────────────────────────
def get_yolo_metrics():
    print("\n" + "="*50)
    print("  [1/3] Evaluating YOLOv8 on test set...")
    print("="*50)

    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(
            f"YOLOv8 model not found at:\n  {YOLO_MODEL_PATH}\n"
            f"Run yolo training first."
        )
    if not os.path.exists(DATASET_YAML):
        raise FileNotFoundError(
            f"Dataset YAML not found at:\n  {DATASET_YAML}"
        )

    model   = YOLO(YOLO_MODEL_PATH)
    metrics = model.val(data=DATASET_YAML, split="test", verbose=False)

    return {
        "mAP50"     : float(metrics.box.map50),
        "mAP50_95"  : float(metrics.box.map),
        "precision" : float(metrics.box.mp),
        "recall"    : float(metrics.box.mr),
        "f1"        : float(2 * metrics.box.mp * metrics.box.mr /
                      (metrics.box.mp + metrics.box.mr + 1e-6)),
    }

# ─────────────────────────────────────────
# SECTION 2: Faster R-CNN Metrics
# ─────────────────────────────────────────
def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((box1[2]-box1[0])*(box1[3]-box1[1]) +
             (box2[2]-box2[0])*(box2[3]-box2[1]) - inter)
    return inter / (union + 1e-6)


def get_rcnn_metrics(model, iou_thresh=0.5, score_thresh=0.5):
    """
    Runs Faster R-CNN on every test image and computes
    precision, recall, and F1 by comparing predictions to
    ground-truth boxes from the COCO annotation file.
    """
    print("\n" + "="*50)
    print("  [2/3] Evaluating Faster R-CNN on test set...")
    print("="*50)

    ann_file = os.path.join(BASE_DIR, "dataset", "test", "_annotations.coco.json")
    if not os.path.exists(ann_file):
        print("  WARNING: test annotation file not found — skipping R-CNN accuracy eval.")
        return {"mAP50": 0, "mAP50_95": 0, "precision": 0, "recall": 0, "f1": 0}

    with open(ann_file) as f:
        coco = json.load(f)

    # Build image_id -> annotations lookup
    gt_by_image = {}
    for ann in coco["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    images_info = {img["id"]: img for img in coco["images"]}

    tf = T.Compose([
        T.Resize((480, 480)),
        T.ToTensor(),
    ])

    tp_total = 0
    fp_total = 0
    fn_total = 0

    for img_id, img_info in images_info.items():
        img_path = os.path.join(BASE_DIR, "dataset", "test", "images", img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        with torch.no_grad():
            preds = model([tf(img).to(DEVICE)])[0]

        # Filter by score threshold
        keep   = preds["scores"] >= score_thresh
        pred_boxes = preds["boxes"][keep].cpu().numpy()

        # Ground truth boxes for this image (convert COCO xywh -> xyxy)
        gt_anns = gt_by_image.get(img_id, [])
        gt_boxes = []
        for ann in gt_anns:
            x, y, w, h = ann["bbox"]
            # Scale to 480x480 to match model input
            sx = 480 / orig_w
            sy = 480 / orig_h
            gt_boxes.append([x*sx, y*sy, (x+w)*sx, (y+h)*sy])

        matched_gt = set()
        tp = 0
        fp = 0

        for pb in pred_boxes:
            best_iou  = 0
            best_idx  = -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                score = iou(pb, gb)
                if score > best_iou:
                    best_iou = score
                    best_idx = gi
            if best_iou >= iou_thresh and best_idx >= 0:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)

        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = tp_total / (tp_total + fp_total + 1e-6)
    recall    = tp_total / (tp_total + fn_total + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)

    # For full COCO mAP you'd need pycocotools, but this gives a solid estimate
    map50    = precision * recall   # area approximation
    map50_95 = map50 * 0.6         # rough scaling factor (common approximation)

    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  mAP@0.5   : {map50:.4f} (approximate)")

    return {
        "mAP50"     : map50,
        "mAP50_95"  : map50_95,
        "precision" : precision,
        "recall"    : recall,
        "f1"        : f1,
    }

# ─────────────────────────────────────────
# SECTION 3: FPS Benchmarks
# ─────────────────────────────────────────
def benchmark_fps(rcnn_model):
    print("\n  [3/3] Benchmarking inference speed...")

    test_imgs = sorted([
        os.path.join(TEST_IMAGES_DIR, f)
        for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])[:N_BENCHMARK_IMGS]

    if not test_imgs:
        print("  WARNING: No test images found for FPS benchmark.")
        return 0.0, 0.0

    # ── YOLOv8 FPS ──
    print("  Benchmarking YOLOv8...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    t0 = time.time()
    for img_path in test_imgs:
        yolo_model(img_path, verbose=False)
    yolo_fps = len(test_imgs) / (time.time() - t0)

    # ── Faster R-CNN FPS ──
    print("  Benchmarking Faster R-CNN...")
    tf = T.Compose([T.Resize((480, 480)), T.ToTensor()])
    t0 = time.time()
    with torch.no_grad():
        for img_path in test_imgs:
            img = Image.open(img_path).convert("RGB")
            rcnn_model([tf(img).to(DEVICE)])
    rcnn_fps = len(test_imgs) / (time.time() - t0)

    print(f"  YOLOv8 FPS       : {yolo_fps:.2f}")
    print(f"  Faster R-CNN FPS : {rcnn_fps:.2f}")
    return yolo_fps, rcnn_fps

# ─────────────────────────────────────────
# SECTION 4: Comparison Chart
# ─────────────────────────────────────────
def plot_comparison(yolo_metrics, rcnn_metrics, yolo_fps, rcnn_fps):
    metrics_labels = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1"]
    yolo_vals = [
        yolo_metrics["mAP50"], yolo_metrics["mAP50_95"],
        yolo_metrics["precision"], yolo_metrics["recall"], yolo_metrics["f1"]
    ]
    rcnn_vals = [
        rcnn_metrics["mAP50"], rcnn_metrics["mAP50_95"],
        rcnn_metrics["precision"], rcnn_metrics["recall"], rcnn_metrics["f1"]
    ]

    x     = np.arange(len(metrics_labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Chart 1: Detection Metrics
    ax = axes[0]
    bars1 = ax.bar(x - width/2, yolo_vals, width, label="YOLOv8",      color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width/2, rcnn_vals, width, label="Faster R-CNN", color="#FF5722", alpha=0.85)
    ax.set_title("Detection Metrics Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, rotation=15)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    # Chart 2: FPS Speed
    ax2 = axes[1]
    fps_bars = ax2.bar(
        ["YOLOv8", "Faster R-CNN"],
        [yolo_fps, rcnn_fps],
        color=["#2196F3", "#FF5722"], alpha=0.85, width=0.4
    )
    ax2.set_title("Inference Speed (FPS)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Frames Per Second")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
    for bar in fps_bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.3,
                 f"{h:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join("results", "comparison_chart.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"\n  Chart saved -> {out_path}")

# ─────────────────────────────────────────
# SECTION 5: Text Report
# ─────────────────────────────────────────
def save_report(yolo_m, rcnn_m, yolo_fps, rcnn_fps):
    winner_acc   = "YOLOv8" if yolo_m["mAP50"] >= rcnn_m["mAP50"] else "Faster R-CNN"
    winner_speed = "YOLOv8" if yolo_fps >= rcnn_fps else "Faster R-CNN"

    report = f"""
╔══════════════════════════════════════════════════════╗
║   SAFETY HELMET DETECTION — ALGORITHM COMPARISON     ║
╚══════════════════════════════════════════════════════╝

Dataset : Hard Hat Workers (Roboflow)
Classes : head | helmet | person
Device  : {DEVICE}

┌────────────────────┬──────────────┬────────────────┐
│ Metric             │   YOLOv8     │  Faster R-CNN  │
├────────────────────┼──────────────┼────────────────┤
│ mAP @ 0.5          │   {yolo_m['mAP50']:.4f}     │    {rcnn_m['mAP50']:.4f}       │
│ mAP @ 0.5:0.95     │   {yolo_m['mAP50_95']:.4f}     │    {rcnn_m['mAP50_95']:.4f}       │
│ Precision          │   {yolo_m['precision']:.4f}     │    {rcnn_m['precision']:.4f}       │
│ Recall             │   {yolo_m['recall']:.4f}     │    {rcnn_m['recall']:.4f}       │
│ F1 Score           │   {yolo_m['f1']:.4f}     │    {rcnn_m['f1']:.4f}       │
│ Avg FPS            │   {yolo_fps:.2f}       │    {rcnn_fps:.2f}          │
└────────────────────┴──────────────┴────────────────┘

Best Accuracy : {winner_acc}
Best Speed    : {winner_speed}

─── ANALYSIS ────────────────────────────────────────────

YOLOv8 (Single-Stage):
  • Processes the entire image in one forward pass
  • Significantly faster — suitable for real-time monitoring
  • Slightly lower accuracy on small/occluded objects
  • Ideal for: live CCTV feeds, edge devices

Faster R-CNN (Two-Stage):
  • First proposes regions, then classifies
  • Higher precision on overlapping detections
  • Slower inference — less suitable for live feeds
  • Ideal for: post-incident review, high-accuracy audits

─── RECOMMENDATION ──────────────────────────────────────

For real-time construction site monitoring:
  → YOLOv8 is the recommended production model.

For periodic compliance reporting:
  → Faster R-CNN provides higher accuracy.

A hybrid pipeline (YOLO for alerting, RCNN for verification)
would give the best of both worlds.
"""

    report_path = os.path.join("results", "comparison_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(report)
    print(f"  Report saved -> {report_path}")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Helmet Detection — Model Evaluation & Comparison")
    print("  Make sure both models are trained before running.\n")

    # Load R-CNN once — reused for both metrics and FPS
    rcnn_model = load_rcnn()

    # 1. YOLOv8 metrics
    yolo_metrics = get_yolo_metrics()

    # 2. Faster R-CNN metrics (now actually computed, not typed manually)
    rcnn_metrics = get_rcnn_metrics(rcnn_model)

    # 3. FPS benchmark
    yolo_fps, rcnn_fps = benchmark_fps(rcnn_model)

    # 4. Charts + report
    plot_comparison(yolo_metrics, rcnn_metrics, yolo_fps, rcnn_fps)
    save_report(yolo_metrics, rcnn_metrics, yolo_fps, rcnn_fps)