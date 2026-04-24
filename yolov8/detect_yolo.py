"""
========================================
 Safety Helmet Detection — YOLOv8
 Real-Time Detection (Webcam / Video / Image)
========================================
Run:
    python detect_yolo.py --source webcam
    python detect_yolo.py --source path/to/video.mp4
    python detect_yolo.py --source path/to/image.jpg
"""

# MUST be set before importing cv2
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"   # fix Wayland black screen on Linux
os.environ["DISPLAY"]         = ":0"    # ensure X display is set

import cv2
import argparse
import time
from pathlib import Path
from ultralytics import YOLO

# -----------------------------------------
# CONFIG
# -----------------------------------------
MODEL_PATH  = "runs/detect/results/yolov8/helmet_detection-5/weights/best.pt"
CONF_THRESH = 0.5

CLASS_COLORS = {
    "helmet"    : (0, 200, 0),    # green  = safe
    "no_helmet" : (0, 0, 220),    # red    = violation
    "head"      : (0, 165, 255),  # orange = bare head
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# -----------------------------------------
# DRAWING HELPERS
# -----------------------------------------

def draw_box(frame, box, label, conf, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def overlay_stats(frame, helmet_count, violation_count, fps):
    status_color = (0, 200, 0) if violation_count == 0 else (0, 0, 220)
    status_text  = "COMPLIANT" if violation_count == 0 else f"{violation_count} VIOLATION(S)"
    cv2.rectangle(frame, (0, 0), (420, 90), (20, 20, 20), -1)
    cv2.putText(frame, f"Helmets: {helmet_count}   {status_text}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}  |  YOLOv8n",
                (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    return frame


# -----------------------------------------
# CORE INFERENCE
# -----------------------------------------

def process_frame(frame, model):
    results = model(frame, conf=CONF_THRESH, verbose=False)[0]
    helmet_count    = 0
    violation_count = 0
    for det in results.boxes:
        cls_id = int(det.cls[0])
        conf   = float(det.conf[0])
        label  = model.names[cls_id]
        box    = det.xyxy[0].tolist()
        color  = CLASS_COLORS.get(label, (180, 180, 180))
        draw_box(frame, box, label, conf, color)
        if label == "helmet":
            helmet_count += 1
        elif label in ("no_helmet", "head"):
            violation_count += 1
    return frame, helmet_count, violation_count


# -----------------------------------------
# IMAGE MODE
# -----------------------------------------

def run_on_image(model, source):
    frame = cv2.imread(source)
    if frame is None:
        print(f"Cannot read image: {source}")
        return

    t0 = time.time()
    annotated, helmet_count, violation_count = process_frame(frame, model)
    elapsed = time.time() - t0
    overlay_stats(annotated, helmet_count, violation_count, fps=1.0 / max(elapsed, 1e-6))

    out_dir  = Path("results/yolov8/detections")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"result_{Path(source).name}"
    cv2.imwrite(str(out_path), annotated)

    print(f"Saved -> {out_path}")
    print(f"Helmets: {helmet_count}  |  Violations: {violation_count}")

    cv2.imshow("Helmet Detection - YOLOv8", annotated)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -----------------------------------------
# WEBCAM / VIDEO MODE
# -----------------------------------------

def run_on_video_or_webcam(model, source):
    if source == "webcam":
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Cannot open source: {source}")
        print("Try: ls /dev/video*  to list available cameras.")
        return

    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 20.0
    print(f"Camera: {w}x{h} @ {fps_src:.1f} fps")

    out_dir  = Path("results/yolov8/detections")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = "webcam_output.mp4" if source == "webcam" else f"result_{Path(source).name}"
    writer   = cv2.VideoWriter(
        str(out_dir / out_name),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_src, (w, h)
    )

    fps_list = []
    print("Running detection - press Q to quit\n")

    while True:
        ret, frame = cap.read()

        # skip bad/empty frames instead of crashing
        if not ret or frame is None or frame.size == 0:
            print("Empty frame - retrying...")
            time.sleep(0.05)
            continue

        t0 = time.time()
        annotated, helmet_count, violation_count = process_frame(frame, model)
        fps = 1.0 / (time.time() - t0 + 1e-6)
        fps_list.append(fps)

        overlay_stats(annotated, helmet_count, violation_count, fps)
        writer.write(annotated)
        cv2.imshow("Helmet Detection - YOLOv8", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"\nAverage FPS : {avg_fps:.2f}")
    print(f"Output saved -> {out_dir / out_name}")


# -----------------------------------------
# ENTRY POINT
# -----------------------------------------

def run_detection(source):
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Check that training completed and the path is correct.")
        return

    print(f"Loading model  : {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"Classes        : {model.names}\n")

    if source == "webcam":
        run_on_video_or_webcam(model, source)
    elif Path(source).suffix.lower() in IMAGE_EXTS:
        run_on_image(model, source)
    elif os.path.isfile(source):
        run_on_video_or_webcam(model, source)
    else:
        print(f"Source not recognised: {source}")
        print("Use 'webcam', a video path, or an image path.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Helmet Detection")
    parser.add_argument(
        "--source",
        default="webcam",
        help="'webcam' | path/to/video.mp4 | path/to/image.jpg"
    )
    args = parser.parse_args()
    run_detection(args.source)