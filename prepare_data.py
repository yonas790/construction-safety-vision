
"""
========================================
 Safety Subset Preparer & Format Converter
========================================
This script prepares a 100-image subset from a YOLOv8 dataset ZIP
and automatically generates COCO annotations for Faster R-CNN.

Steps:
1. Download ZIP from Roboflow (YOLOv8 format).
2. Rename it to 'dataset.zip'.
3. Run: python prepare_data.py
"""

import os
import zipfile
import shutil
import json
import yaml
from PIL import Image

def yolov8_to_coco(subset_dir, class_names):
    """Converts YOLO .txt labels to a COCO .json file."""
    images_dir = os.path.join(subset_dir, "images")
    labels_dir = os.path.join(subset_dir, "labels")
    
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i+1, "name": name} for i, name in enumerate(class_names)]
    }
    
    ann_id = 1
    for img_id, img_name in enumerate(os.listdir(images_dir)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(images_dir, img_name)
        with Image.open(img_path) as img:
            width, height = img.size
            
        coco["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })
        
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(labels_dir, lbl_name)
        
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5: continue
                    
                    cls_id, x_c, y_c, w, h = map(float, parts)
                    # Convert YOLO (norm center) to COCO (pixel xmin, ymin, w, h)
                    abs_w = w * width
                    abs_h = h * height
                    abs_x = (x_c * width) - (abs_w / 2)
                    abs_y = (y_c * height) - (abs_h / 2)
                    
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(cls_id) + 1, # COCO is 1-indexed (0 is background usually)
                        "bbox": [abs_x, abs_y, abs_w, abs_h],
                        "area": abs_w * abs_h,
                        "iscrowd": 0
                    })
                    ann_id += 1
                    
    with open(os.path.join(subset_dir, "_annotations.coco.json"), "w") as f:
        json.dump(coco, f, indent=2)

def main(limit=100):
    if not os.path.exists("dataset.zip"):
        print("\n❌ Error: 'dataset.zip' not found.")
        print("   Please download the YOLOv8 ZIP from Roboflow and place it in this folder.")
        return

    # Clean previous attempts
    if os.path.exists("dataset"):
        shutil.rmtree("dataset")
    
    print("\n⏳ Extracting 100-image subset from dataset.zip...")
    
    with zipfile.ZipFile("dataset.zip", 'r') as z:
        all_files = z.namelist()
        
        # 1. Read class names from data.yaml
        if 'data.yaml' not in all_files:
            print("❌ Error: 'data.yaml' missing from ZIP.")
            return
            
        z.extract('data.yaml', "dataset")
        with open("dataset/data.yaml", "r") as f:
            data_cfg = yaml.safe_load(f)
            class_names = data_cfg.get('names', [])
        
        # 2. Process Splits
        for split in ['train', 'valid', 'test']:
            print(f"   Processing {split} split...")
            os.makedirs(f"dataset/{split}/images", exist_ok=True)
            os.makedirs(f"dataset/{split}/labels", exist_ok=True)
            
            # Find images in this split
            img_prefix = f"{split}/images/"
            all_imgs = [f for f in all_files if f.startswith(img_prefix) and f.lower().endswith(('.jpg', '.png'))]
            
            # Clip to limit (e.g. 100 for train, 20 for others)
            n_to_take = limit if split == 'train' else 20
            take_imgs = all_imgs[:n_to_take]
            
            for f in take_imgs:
                # Extract image
                z.extract(f, "dataset")
                # Extract corresponding label
                lbl_f = f.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
                if lbl_f in all_files:
                    z.extract(lbl_f, "dataset")
            
            # 3. Create COCO JSON for Faster R-CNN
            yolov8_to_coco(f"dataset/{split}", class_names)

    print("\n" + "="*50)
    print("✅ SUCCESS! Dataset subset ready.")
    print(f"📍 Location: /dataset")
    print(f"📊 Contents: {limit} train images, 20 valid, 20 test.")
    print("📝 Both YOLO .txt and COCO .json labels are prepared.")
    print("="*50)
    print("\nNext steps:")
    print("1. python yolov8/train_yolo.py")
    print("2. python faster_rcnn/train_rcnn.py")

if __name__ == "__main__":
    main()
