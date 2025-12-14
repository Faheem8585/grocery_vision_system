import os
import csv
import shutil
import cv2
import random
import yaml
from pathlib import Path

# Paths
RAW_DATA_DIR = Path("data/raw/grocery_dataset")
SHELF_IMAGES_DIR = RAW_DATA_DIR / "ShelfImages"
ANNOTATIONS_FILE = Path("data/raw/annotations.csv")

# Target Paths (New Pipeline)
DATA_DIR = Path("grocery_cv_pipeline/data")
IMAGES_DIR = DATA_DIR 
LABELS_DIR = DATA_DIR 

# Create directories
for split in ["train", "valid", "test"]:
    (IMAGES_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (LABELS_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

def convert_to_yolo(x1, y1, x2, y2, img_w, img_h):
    dw = 1.0 / img_w
    dh = 1.0 / img_h
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    cx *= dw
    cy *= dh
    w *= dw
    h *= dh
    return cx, cy, w, h

def main():
    print("Starting data preparation for pipeline...")
    
    if not ANNOTATIONS_FILE.exists():
        print(f"Error: Annotations file not found at {ANNOTATIONS_FILE}")
        return

    annotations = {}
    classes = set()
    
    with open(ANNOTATIONS_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            filename = row[0]
            try:
                x1, y1, x2, y2 = int(row[1]), int(row[2]), int(row[3]), int(row[4])
                class_id = int(row[5])
                classes.add(class_id)
                if filename not in annotations: annotations[filename] = []
                annotations[filename].append((x1, y1, x2, y2, class_id))
            except ValueError: continue

    existing_images = list(SHELF_IMAGES_DIR.glob("*.JPG"))
    print(f"Found {len(existing_images)} images.")
    
    random.seed(42)
    random.shuffle(existing_images)
    
    # Split: 80% train, 10% valid, 10% test
    n = len(existing_images)
    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)
    
    splits = [
        ("train", existing_images[:train_end]),
        ("valid", existing_images[train_end:valid_end]),
        ("test", existing_images[valid_end:])
    ]
    
    processed_count = 0
    
    for split_name, files in splits:
        for img_path in files:
            filename = img_path.name
            if filename not in annotations: continue
                
            img = cv2.imread(str(img_path))
            if img is None: continue
            h, w = img.shape[:2]
            
            label_lines = []
            for ann in annotations[filename]:
                x1, y1, x2, y2, class_id = ann
                cx, cy, nw, nh = convert_to_yolo(x1, y1, x2, y2, w, h)
                label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            
            if not label_lines: continue
                
            # Copy image
            shutil.copy(img_path, IMAGES_DIR / split_name / "images" / filename)
            
            # Write label
            label_filename = filename.replace(".JPG", ".txt").replace(".jpg", ".txt")
            with open(LABELS_DIR / split_name / "labels" / label_filename, "w") as f:
                f.write("\n".join(label_lines))
            
            processed_count += 1

    print(f"Processed {processed_count} images.")
    
    # Create dataset.yaml
    yaml_content = {
        "path": str(DATA_DIR.absolute()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {i: f"item_{i}" for i in sorted(list(classes))}
    }
    
    with open(DATA_DIR / "dataset.yaml", "w") as f:
        yaml.dump(yaml_content, f)
        
    print("Created dataset.yaml")

if __name__ == "__main__":
    main()
