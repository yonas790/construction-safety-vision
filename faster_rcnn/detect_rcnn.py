"""
========================================
 Safety Helmet Detection — Faster R-CNN
 Detection on Image / Video / Webcam
========================================
Run:
    python detect_rcnn.py --source path/to/image.jpg
    python detect_rcnn.py --source path/to/video.mp4
    python detect_rcnn.py --source webcam
"""

import cv2, argparse, time
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import numpy as np

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_PATH  = "results/faster_rcnn/best_model.pth"
NUM_CLASSES = 3 + 1          # helmet, no_helmet, person + background
CONF_THRESH = 0.6
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES  = {1: "helmet", 2: "no_helmet", 3: "person"}
CLASS_COLORS = {
    "helmet"    : (0, 200, 0),
    "no_helmet" : (0, 0, 220),
    "person"    : (200, 200, 0),
}

def load_model():
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model   = fasterrcnn_resnet50_fpn(weights=weights)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model

transform = T.Compose([T.ToTensor()])

def predict(model, frame_rgb):
    """Return list of (box, label, conf)."""
    img_tensor = transform(Image.fromarray(frame_rgb)).to(DEVICE)
    with torch.no_grad():
        output = model([img_tensor])[0]

    detections = []
    for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
        if score >= CONF_THRESH:
            cls_name = CLASS_NAMES.get(int(label), "unknown")
            detections.append((box.cpu().numpy(), cls_name, float(score)))
    return detections

def draw_box(frame, box, label, conf):
    x1, y1, x2, y2 = map(int, box)
    color = CLASS_COLORS.get(label, (180, 180, 180))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def run_detection(source):
    print("⏳ Loading Faster R-CNN model…")
    model = load_model()
    print("✅ Model loaded. Press Q to quit.\n")

    if source == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)

    fps_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.time()

        detections     = predict(model, frame_rgb)
        fps            = 1 / (time.time() - t0 + 1e-6)
        fps_list.append(fps)

        helmet_count    = 0
        violation_count = 0

        for box, label, conf in detections:
            draw_box(frame, box, label, conf)
            if label == "helmet":     helmet_count    += 1
            if label == "no_helmet":  violation_count += 1

        status_color = (0, 200, 0) if violation_count == 0 else (0, 0, 220)
        status_text  = "✔ COMPLIANT" if violation_count == 0 else f"⚠ {violation_count} VIOLATION(S)"

        cv2.rectangle(frame, (0, 0), (340, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Helmets: {helmet_count}  |  {status_text}",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}  |  Faster R-CNN",
                    (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Helmet Detection — Faster R-CNN", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"\n📊 Average FPS (Faster R-CNN): {avg_fps:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="webcam")
    args = parser.parse_args()
    run_detection(args.source)