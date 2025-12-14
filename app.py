# Developer: Faheem
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import json

app = Flask(__name__)

# Load model once at startup
# Load model once at startup
model = YOLO("yolov8n.pt")

# Load prices
try:
    with open("prices.json", "r") as f:
        prices = json.load(f)
except FileNotFoundError:
    prices = {}

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    file = request.files["image"].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    results = model(img)[0]
    predictions = []
    
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]
        price = prices.get(label, None)
        
        prediction = {
            "label": label, 
            "confidence": float(conf),
            "bbox": box.xyxy[0].tolist()
        }
        
        if price is not None:
            prediction["price"] = price
            
        predictions.append(prediction)

    return jsonify(predictions)

if __name__ == "__main__":
    # Run on 0.0.0.0 to be accessible
    app.run(host="0.0.0.0", port=5001, debug=True)
