# Developer: Faheem
from ultralytics import YOLO
import os

def train_model():
    # Load a YOLOv8 model (base)
    model = YOLO("yolov8n.pt")

    # Train with your dataset
    # Ensure data/dataset.yaml exists before running this
    if not os.path.exists("data/dataset.yaml"):
        print("Error: data/dataset.yaml not found. Please create it first.")
        return

    model.train(
        data="data/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="models",
        name="grocery_detection",
        exist_ok=True
    )

if __name__ == "__main__":
    train_model()
