# End-to-End Grocery Vision Pipeline
**Developer: Faheem**

This project implements a complete pipeline for grocery item detection and pricing, using YOLOv8 and Flask.

## Demo

### UI Screenshot
![Streamlit UI](assets/ui_screenshot.png)


## Project Structure

- `data/`: Dataset directory (train/valid/test).
- `models/`: Trained models.
- `scripts/`: Helper scripts.
- `train.py`: Script to train YOLOv8 on your custom dataset.
- `inference.py`: Real-time webcam detection with price overlay.
- `streamlit_app.py`: Streamlit web application for demo and deployment.
- `app.py`: Flask API for deploying the model.
- `prices.json`: Configuration file mapping product names to prices.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Data**:
    - Organize your dataset in `data/` following the YOLO structure.
    - Create `data/dataset.yaml` pointing to your train/val paths.

## Usage

### 1. Training
To train a custom model on your dataset:
```bash
python train.py
```
This will save the best model to `models/grocery_detection/weights/best.pt`.

### 2. Streamlit Web App
To run the interactive web application:
```bash
streamlit run streamlit_app.py
```

**Features:**
- **Webcam Capture**: Take photos using your browser's camera (cloud-compatible)
- **Video Upload**: Process pre-recorded videos
- **Real-time Detection**: Instant product identification and pricing
- **Shopping Cart**: Automatic price calculation

> **Note for Cloud Deployment (Streamlit Cloud)**: 
> The app uses `st.camera_input()` for webcam access, which is compatible with Streamlit Cloud. 
> Traditional `cv2.VideoCapture()` doesn't work on cloud servers since they don't have physical cameras.
> For local live video streaming, use `inference.py` instead.

### 3. Real-Time Inference (Local Webcam - Live Video)
To run detection on your webcam with price overlay:
```bash
python inference.py
```
*Note: Press 'ESC' to exit.*
*This requires a local setup and won't work on cloud deployments.*

### 4. Deployment (API)
To start the Flask API server:
```bash
python app.py
```
The API will be available at `http://localhost:5000/predict`.

**Test the API:**
```bash
curl -X POST -F "image=@/path/to/image.jpg" http://localhost:5000/predict
```

## Configuration
- **Prices**: Update `prices.json` to change product prices.
- **Model**: By default, scripts use `yolov8n.pt`. Update the model path in `inference.py` and `app.py` to point to your trained model (e.g., `models/grocery_detection/weights/best.pt`).

## Deployment Notes

### Streamlit Cloud
The app is optimized for Streamlit Cloud deployment:
- Uses `st.camera_input()` for browser-based camera access
- Lightweight model (`yolov8n.pt`) for faster inference
- No external dependencies beyond `requirements.txt`

### Technical Details
- **Cloud Compatibility**: `cv2.VideoCapture()` replaced with `st.camera_input()` for cloud deployment
- **Local Development**: Both live video (`inference.py`) and snapshot modes (`streamlit_app.py`) supported
- **Model**: YOLOv8-nano for optimal speed/accuracy tradeoff

