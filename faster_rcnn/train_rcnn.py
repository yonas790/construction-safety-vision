"""
========================================
 Safety Helmet Detection — Faster R-CNN
 CPU-OPTIMIZED VERSION
========================================

SETUP (run once):
    pip install torch torchvision opencv-python matplotlib pillow

DATASET structure:
    dataset/
        train/
            images/
            _annotations.coco.json
        valid/
            images/
            _annotations.coco.json
        test/
            images/
            _annotations.coco.json
"""

import os
import json
import time
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------------------
# CPU OPTIMIZATIONS — applied immediately
# -----------------------------------------
torch.set_num_threads(os.cpu_count())          # use all available CPU cores
torch.backends.quantized.engine = "qnnpack"   # optimized quantization engine for CPU

# -----------------------------------------
# CONFIG
# -----------------------------------------

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset")

NUM_CLASSES  = 3 + 1   # head, helmet, no_helmet  +  background (index 0)
EPOCHS       = 10      # ↓ from 20  — enough for good results on CPU
BATCH_SIZE   = 4       # ↑ from 2   — larger batches are faster on CPU
LR           = 0.01    # ↑ from 0.005 — converges faster
DEVICE       = torch.device("cpu")

# Image size for training — smaller = much faster
IMG_SIZE     = 480     # ↓ from ~640 default — good speed/accuracy balance

RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SAVE_PATH    = os.path.join(RESULTS_DIR, "best_model.pth")

print(f"Dataset root   : {DATASET_ROOT}")
print(f"Results dir    : {RESULTS_DIR}")
print(f"Device         : {DEVICE}")
print(f"CPU threads    : {os.cpu_count()}")
print(f"Image size     : {IMG_SIZE}px")
print(f"Backbone       : MobileNetV3  (3–5x faster than ResNet50 on CPU)")
print()

# -----------------------------------------
# DATASET
# -----------------------------------------

class HelmetCOCODataset(Dataset):
    """
    Loads images + COCO-format annotations.

    Folder layout per split:
        <root>/<split>/images/   <- .jpg files
        <root>/<split>/_annotations.coco.json

    Class indices:
        0            = background (reserved by Faster R-CNN)
        1, 2, 3, ... = your actual classes
    """

    def __init__(self, root, split="train", transforms=None):
        self.img_dir    = os.path.join(root, split, "images")
        self.transforms = transforms

        ann_file = os.path.join(root, split, "_annotations.coco.json")

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.exists(ann_file):
            raise FileNotFoundError(
                f"Annotation file not found: {ann_file}\n"
                f"Make sure you exported your Roboflow dataset in COCO format."
            )

        with open(ann_file) as f:
            coco = json.load(f)

        self.images      = {img["id"]: img for img in coco["images"]}
        self.img_ids     = [img["id"] for img in coco["images"]]
        self.annotations = {}

        for ann in coco["annotations"]:
            self.annotations.setdefault(ann["image_id"], []).append(ann)

        # category_id -> 1-based class index (0 is background)
        self.cat_map = {
            cat["id"]: idx + 1
            for idx, cat in enumerate(coco["categories"])
        }

        self.class_names = ["__background__"] + [
            cat["name"] for cat in coco["categories"]
        ]
        print(f"  [{split}] {len(self.img_ids)} images | "
              f"classes: {self.class_names[1:]}")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        anns   = self.annotations.get(img_id, [])
        boxes  = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:          # skip degenerate boxes
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_map[ann["category_id"]])

        if boxes:
            boxes  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)

        target = {
            "boxes"    : boxes,
            "labels"   : labels,
            "image_id" : torch.tensor([img_id]),
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


# -----------------------------------------
# MODEL  —  MobileNetV3 backbone (fast!)
# -----------------------------------------

def build_model(num_classes):
    """
    MobileNetV3-Large backbone instead of ResNet50.
    3–5x faster on CPU with only a small accuracy trade-off.
    """
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model   = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    # Replace the classification head for our dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def freeze_backbone(model):
    """
    Freeze everything except the RPN and ROI heads.
    Only the detection layers train → 2–3x fewer gradient ops.
    """
    for name, param in model.named_parameters():
        if "roi_heads" not in name and "rpn" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params : {trainable:,} / {total:,}  "
          f"({100 * trainable / total:.1f}%)")


# -----------------------------------------
# TRAINING
# -----------------------------------------

def train():
    # Resize + ToTensor — smaller images = faster forward pass
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    print("\n" + "=" * 56)
    print("  Faster R-CNN — CPU-Optimized Helmet Detector")
    print("=" * 56)

    # -- datasets --
    try:
        train_ds = HelmetCOCODataset(DATASET_ROOT, "train", transforms=tf)
        valid_ds = HelmetCOCODataset(DATASET_ROOT, "valid", transforms=tf)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return

    # num_workers=4  — parallel data loading on CPU
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # -- model --
    model = build_model(NUM_CLASSES).to(DEVICE)

    # Freeze backbone — only train RPN + ROI heads
    print("\nFreezing backbone (training detection head only):")
    freeze_backbone(model)

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_loss    = float("inf")
    train_losses = []
    valid_losses = []

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\nStarting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):

        # ── train ─────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs    = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss      = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Show batch progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"    Batch [{batch_idx + 1}/{len(train_loader)}]  "
                      f"Loss: {loss.item():.4f}", end="\r")

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ── validation loss ────────────────────────────────────────────────
        # model.train() is intentional — Faster R-CNN only returns loss dicts
        # in training mode; torch.no_grad() skips gradient computation.
        model.train()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in valid_loader:
                imgs    = [img.to(DEVICE) for img in imgs]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                loss_dict = model(imgs, targets)
                val_loss += sum(loss_dict.values()).item()

        avg_val_loss = val_loss / len(valid_loader)
        valid_losses.append(avg_val_loss)

        scheduler.step()

        elapsed = time.time() - t0
        print(f"  Epoch [{epoch:02d}/{EPOCHS}]  "
              f"Train: {avg_train_loss:.4f}  "
              f"Val: {avg_val_loss:.4f}  "
              f"Time: {elapsed:.1f}s      ")

        # ── save best ──────────────────────────────────────────────────────
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"    -> Saved best model  (val loss: {best_loss:.4f})")

    # ── loss curve ─────────────────────────────────────────────────────────
    plt.figure(figsize=(9, 4))
    plt.plot(range(1, EPOCHS + 1), train_losses, marker="o", label="Train Loss")
    plt.plot(range(1, EPOCHS + 1), valid_losses, marker="s", label="Val Loss")
    plt.title("Faster R-CNN (CPU-Optimized) — Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    curve_path = os.path.join(RESULTS_DIR, "training_loss.png")
    plt.savefig(curve_path)

    print(f"\n{'=' * 56}")
    print(f"  Training complete!")
    print(f"  Loss curve  -> {curve_path}")
    print(f"  Best model  -> {SAVE_PATH}")
    print(f"  Best val loss: {best_loss:.4f}")
    print(f"{'=' * 56}")


if __name__ == "__main__":
    train()