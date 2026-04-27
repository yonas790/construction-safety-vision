"""
========================================
 Safety Helmet Detection — Inference
 Run this AFTER training is complete
========================================

Usage:
    python detect.py --image path/to/photo.jpg
    python detect.py --image path/to/photo.jpg --threshold 0.5
    python detect.py --folder path/to/folder/of/images/
"""

import os
import argparse
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------------------------
# CONFIG
# -----------------------------------------

NUM_CLASSES  = 3 + 1       # head, helmet, person + background
CLASS_NAMES  = ["__background__", "head", "helmet", "person"]
DEVICE       = torch.device("cpu")
IMG_SIZE     = 480
THRESHOLD    = 0.5          # only show detections above 50% confidence

# Path trained model
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, "faster_rcnn", "results", "best_model.pth")
OUTPUT_DIR   = os.path.join(BASE_DIR, "faster_rcnn", "results", "detections")

# Colors per class (BGR for drawing)
CLASS_COLORS = {
    "__background__" : "gray",
    "head"           : "#378ADD",   # blue   — uncertain
    "helmet"         : "#1D9E75",   # green  — safe
    "person"         : "#D85A30",   # orange — no helmet / at risk
}

# -----------------------------------------
# MODEL
# -----------------------------------------

def build_model(num_classes):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model   = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model(model_path):
    """Load trained weights into the model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}\n"
            f"Make sure you have run train_rcnn.py first."
        )
    model = build_model(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()   # switch to detection mode (not training mode)
    print(f"Model loaded from: {model_path}")
    return model


# -----------------------------------------
# DETECTION
# -----------------------------------------

def detect_image(model, image_path, threshold=THRESHOLD):
    """
    Run detection on a single image.
    Returns: original PIL image + detection results dict
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load and preprocess
    img_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img_pil.size

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    img_tensor = tf(img_pil).unsqueeze(0).to(DEVICE)  # add batch dimension

    # Run inference
    with torch.no_grad():
        predictions = model(img_tensor)

    pred = predictions[0]

    # Filter by confidence threshold
    keep = pred["scores"] >= threshold

    results = {
        "boxes"  : pred["boxes"][keep].cpu().numpy(),
        "labels" : pred["labels"][keep].cpu().numpy(),
        "scores" : pred["scores"][keep].cpu().numpy(),
        "orig_size" : (orig_w, orig_h),
    }

    return img_pil, results


def draw_detections(img_pil, results, save_path=None, show=True):
    """
    Draw bounding boxes on image and optionally save/show it.
    """
    orig_w, orig_h = results["orig_size"]
    boxes   = results["boxes"]
    labels  = results["labels"]
    scores  = results["scores"]

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_pil)

    # Scale boxes back to original image size
    scale_x = orig_w / IMG_SIZE
    scale_y = orig_h / IMG_SIZE

    helmet_count   = 0
    no_helmet_count = 0
    head_count     = 0

    for box, label, score in zip(boxes, labels, scores):
        class_name = CLASS_NAMES[label]
        color      = CLASS_COLORS.get(class_name, "white")

        # Scale box coordinates back to original image size
        x1 = box[0] * scale_x
        y1 = box[1] * scale_y
        x2 = box[2] * scale_x
        y2 = box[3] * scale_y
        w  = x2 - x1
        h  = y2 - y1

        # Draw box
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        # Draw label
        label_text = f"{class_name} {score:.0%}"
        ax.text(
            x1, y1 - 5,
            label_text,
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8)
        )

        # Count per class
        if class_name == "helmet":
            helmet_count += 1
        elif class_name == "person":
            no_helmet_count += 1
        elif class_name == "head":
            head_count += 1

    # Summary title
    title = (f"✅ Helmets: {helmet_count}   "
             f"⚠️  No Helmet: {no_helmet_count}   "
             f"❓ Uncertain: {head_count}")
    ax.set_title(title, fontsize=13, pad=10,
                 color="white",
                 bbox=dict(boxstyle="round", facecolor="#222", alpha=0.8))
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")

    if show:
        plt.show()

    plt.close()

    return {
        "helmet_count"   : helmet_count,
        "no_helmet_count": no_helmet_count,
        "head_count"     : head_count,
        "total"          : len(boxes),
    }


# -----------------------------------------
# MAIN
# -----------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Helmet detector inference")
    parser.add_argument("--image",     type=str, help="Path to a single image")
    parser.add_argument("--folder",    type=str, help="Path to a folder of images")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="Confidence threshold (default 0.5)")
    parser.add_argument("--no-show",   action="store_true",
                        help="Don't display image, just save")
    args = parser.parse_args()

    if not args.image and not args.folder:
        print("ERROR: Please provide --image or --folder")
        print("  Example: python detect.py --image test.jpg")
        return

    # Load model once
    model = load_model(MODEL_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Single image
    if args.image:
        print(f"\nRunning detection on: {args.image}")
        img_pil, results = detect_image(model, args.image, args.threshold)

        fname     = os.path.splitext(os.path.basename(args.image))[0]
        save_path = os.path.join(OUTPUT_DIR, f"{fname}_detected.jpg")

        summary = draw_detections(img_pil, results,
                                  save_path=save_path,
                                  show=not args.no_show)

        print(f"\n  Results:")
        print(f"    ✅ Helmets     : {summary['helmet_count']}")
        print(f"    ⚠️  No helmet   : {summary['no_helmet_count']}")
        print(f"    ❓ Uncertain   : {summary['head_count']}")
        print(f"    Total detected : {summary['total']}")

        if summary["no_helmet_count"] > 0:
            print(f"\n  🚨 ALERT: {summary['no_helmet_count']} person(s) detected WITHOUT a helmet!")

    # Folder of images
    if args.folder:
        if not os.path.isdir(args.folder):
            print(f"ERROR: Folder not found: {args.folder}")
            return

        exts   = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [f for f in os.listdir(args.folder)
                  if os.path.splitext(f)[1].lower() in exts]

        if not images:
            print(f"No images found in: {args.folder}")
            return

        print(f"\nRunning detection on {len(images)} images in: {args.folder}\n")

        total_helmets    = 0
        total_no_helmets = 0

        for fname in images:
            img_path  = os.path.join(args.folder, fname)
            save_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(fname)[0]}_detected.jpg")

            try:
                img_pil, results = detect_image(model, img_path, args.threshold)
                summary = draw_detections(img_pil, results,
                                          save_path=save_path,
                                          show=False)
                total_helmets    += summary["helmet_count"]
                total_no_helmets += summary["no_helmet_count"]
                print(f"  {fname}: {summary['helmet_count']} helmet(s), "
                      f"{summary['no_helmet_count']} no-helmet(s)")

            except Exception as e:
                print(f"  ERROR on {fname}: {e}")

        print(f"\n{'='*50}")
        print(f"  Folder summary:")
        print(f"  Total helmets detected    : {total_helmets}")
        print(f"  Total no-helmets detected : {total_no_helmets}")
        print(f"  Output saved to           : {OUTPUT_DIR}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()