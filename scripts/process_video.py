import cv2
import os
from ultralytics import YOLO
import shutil
from pathlib import Path

# Paths
VIDEO_PATH = "data/raw/new_training_video.mp4"
DATA_DIR = Path("grocery_cv_pipeline/data")
TRAIN_IMG_DIR = DATA_DIR / "train" / "images"
TRAIN_LBL_DIR = DATA_DIR / "train" / "labels"
VAL_IMG_DIR = DATA_DIR / "valid" / "images"
VAL_LBL_DIR = DATA_DIR / "valid" / "labels"

# Ensure directories exist
TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LBL_DIR.mkdir(parents=True, exist_ok=True)
VAL_IMG_DIR.mkdir(parents=True, exist_ok=True)
VAL_LBL_DIR.mkdir(parents=True, exist_ok=True)

def process_video():
    print(f"Processing {VIDEO_PATH}...")
    
    # Load auto-labeling model
    model = YOLO("yolov8n.pt")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Extract 1 frame every 10 frames (approx 3 fps)
        if frame_count % 10 != 0:
            continue
            
        # Run detection to get auto-labels
        results = model(frame, verbose=False)[0]
        
        # If no objects detected, skip (unless we want background samples)
        if len(results.boxes) == 0:
            continue

        # Determine split (80% train, 20% val)
        is_train = (saved_count % 5 != 0)
        img_dir = TRAIN_IMG_DIR if is_train else VAL_IMG_DIR
        lbl_dir = TRAIN_LBL_DIR if is_train else VAL_LBL_DIR
        
        # Save Image
        filename = f"video_frame_{saved_count:04d}"
        cv2.imwrite(str(img_dir / f"{filename}.jpg"), frame)
        
        # Save Label
        h, w = frame.shape[:2]
        label_lines = []
        for box in results.boxes:
            cls = int(box.cls[0])
            # Normalize coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            
        with open(lbl_dir / f"{filename}.txt", "w") as f:
            f.write("\n".join(label_lines))
            
        saved_count += 1
        print(f"Saved frame {saved_count}: {len(label_lines)} objects")

    cap.release()
    print(f"Done! Processed {saved_count} frames.")

if __name__ == "__main__":
    process_video()
