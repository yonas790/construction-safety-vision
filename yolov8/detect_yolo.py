"""
========================================
 Safety Helmet Detection — YOLOv8
 Real-Time Detection (Webcam / Video)
========================================
Run:
    python detect_yolo.py --source webcam
    python detect_yolo.py --source path/to/video.mp4
    python detect_yolo.py --source path/to/image.jpg
"""

import cv2
import argparse
import time
from ultralytics import YOLO

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_PATH  = "results/yolov8/helmet_detection/weights/best.pt"
CONF_THRESH = 0.5       # confidence threshold (0–1)

# Color map: class name → BGR color
CLASS_COLORS = {
    "helmet"    : (0, 200, 0),    # green  = safe
    "no_helmet" : (0, 0, 220),    # red    = violation
    "person"    : (200, 200, 0),  # yellow = unclassified person
}

def draw_box(frame, box, label, conf, color):
    """Draw bounding box + label on frame."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def run_detection(source):
    model = YOLO(MODEL_PATH)
    class_names = model.names  # {0: 'helmet', 1: 'no_helmet', ...}

    # Open source
    if source == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Cannot open source: {source}")
        return

    fps_list = []
    print("▶  Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()

        # Run inference
        results = model(frame, conf=CONF_THRESH, verbose=False)[0]

        violation_count = 0
        helmet_count    = 0

        for det in results.boxes:
            cls_id = int(det.cls[0])
            conf   = float(det.conf[0])
            label  = class_names[cls_id]
            box    = det.xyxy[0].tolist()
            color  = CLASS_COLORS.get(label, (180, 180, 180))

            draw_box(frame, box, label, conf, color)

            if label == "helmet":
                helmet_count += 1
            elif label == "no_helmet":
                violation_count += 1

        # FPS
        fps = 1 / (time.time() - t0 + 1e-6)
        fps_list.append(fps)

        # Overlay stats
        status_color = (0, 200, 0) if violation_count == 0 else (0, 0, 220)
        status_text  = "✔ COMPLIANT" if violation_count == 0 else f"⚠ {violation_count} VIOLATION(S)"

        cv2.rectangle(frame, (0, 0), (320, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Helmets: {helmet_count}  |  {status_text}",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}  |  YOLOv8",
                    (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Helmet Detection — YOLOv8", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"\n📊 Average FPS (YOLOv8): {avg_fps:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="webcam",
                        help="'webcam' or path to video/image file")
    args = parser.parse_args()
    run_detection(args.source)