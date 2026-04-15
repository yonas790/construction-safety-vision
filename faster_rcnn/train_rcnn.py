"""
========================================
 Safety Helmet Detection — Faster R-CNN
 Algorithm 2: Two-Stage Detector
========================================

SETUP (run once):
    pip install torch torchvision opencv-python matplotlib pillow

DATASET structure expected (same Roboflow download, COCO format):
    dataset/
        train/  _annotations.coco.json  + images/
        valid/  _annotations.coco.json  + images/
        test/   _annotations.coco.json  + images/

NOTE: On Roboflow, choose "COCO" as export format for this script.
"""

import os, json, time, copy
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATASET_ROOT  = "dataset"
NUM_CLASSES   = 3 + 1       # helmet, no_helmet, person + background
EPOCHS        = 20
BATCH_SIZE    = 4
LR            = 0.005
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH     = "results/faster_rcnn/best_model.pth"

print(f"🖥  Using device: {DEVICE}")

# ─────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────
class HelmetCOCODataset(Dataset):
    def __init__(self, root, split="train", transforms=None):
        self.root       = os.path.join(root, split)
        self.transforms = transforms

        ann_file = os.path.join(self.root, "_annotations.coco.json")
        with open(ann_file) as f:
            coco = json.load(f)

        # Map image_id → file_name
        self.images = {img["id"]: img for img in coco["images"]}
        self.img_ids = [img["id"] for img in coco["images"]]

        # Group annotations by image_id
        self.annotations = {}
        for ann in coco["annotations"]:
            iid = ann["image_id"]
            self.annotations.setdefault(iid, []).append(ann)

        # category_id → class index (1-based, 0 = background)
        self.cat_map = {cat["id"]: i + 1 for i, cat in enumerate(coco["categories"])}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.root, "images", img_info["file_name"])

        img = Image.open(img_path).convert("RGB")

        anns    = self.annotations.get(img_id, [])
        boxes   = []
        labels  = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_map[ann["category_id"]])

        boxes  = torch.as_tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)   if labels else torch.zeros((0,),   dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}

        if self.transforms:
            img = self.transforms(img)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
def build_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model   = fasterrcnn_resnet50_fpn(weights=weights)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model

# ─────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────
def train():
    tf = transforms.Compose([transforms.ToTensor()])

    train_ds = HelmetCOCODataset(DATASET_ROOT, "train", transforms=tf)
    valid_ds = HelmetCOCODataset(DATASET_ROOT, "valid", transforms=tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False,
                              num_workers=2, collate_fn=collate_fn)

    model = build_model(NUM_CLASSES).to(DEVICE)

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_loss   = float("inf")
    train_losses = []

    print("\n" + "=" * 50)
    print("  Faster R-CNN — Training Safety Helmet Detector")
    print("=" * 50)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        t0 = time.time()

        for imgs, targets in train_loader:
            imgs    = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss      = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step()

        print(f"  Epoch [{epoch:02d}/{EPOCHS}]  Loss: {avg_loss:.4f}  "
              f"Time: {time.time()-t0:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"    ✅ Model saved (loss improved)")

    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, EPOCHS + 1), train_losses, marker='o')
    plt.title("Faster R-CNN — Training Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("results/faster_rcnn/training_loss.png")
    print("\n📊 Loss curve saved to results/faster_rcnn/training_loss.png")
    print(f"   Best model saved to: {SAVE_PATH}")

if __name__ == "__main__":
    train()