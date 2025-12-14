# Developer: Faheem
import cv2
import json
from ultralytics import YOLO

def load_prices(price_file="prices.json"):
    try:
        with open(price_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {price_file} not found. Prices will not be displayed.")
        return {}

def run_inference():
    # Load model
    # Switching back to yolov8n.pt for robust demo detection
    # The custom model was overfitted to the Pexels video and didn't generalize
    model = YOLO("yolov8n.pt") 
    
    prices = load_prices()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting inference... Press 'ESC' to exit.")

    import numpy as np

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, verbose=False)[0]

        total_price = 0.0
        detected_items = [] # List of (label, price)
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            label = model.names[cls]
            price = prices.get(label)
            
            display_text = f"{label} {conf:.2f}"
            if price:
                display_text += f" - ${price:.2f}"
                total_price += price
                detected_items.append((label, price))

            # Draw box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw label
            cv2.putText(img, display_text, (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # --- UI: Receipt Sidebar ---
        h, w = img.shape[:2]
        sidebar_w = 300
        # Create a larger canvas
        canvas = np.zeros((h, w + sidebar_w, 3), dtype=np.uint8)
        # Place video on left
        canvas[:h, :w] = img
        # Fill sidebar with white/gray
        canvas[:h, w:] = (240, 240, 240) 
        
        # Header
        cv2.putText(canvas, "SHOPPING CART", (w + 20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.line(canvas, (w + 10, 50), (w + sidebar_w - 10, 50), (0, 0, 0), 2)
        
        # List Items
        y_offset = 80
        for item_name, item_price in detected_items:
            # Item Name
            cv2.putText(canvas, f"{item_name}", (w + 20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
            # Price
            cv2.putText(canvas, f"${item_price:.2f}", (w + 200, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)
            y_offset += 30
            
        # Total
        cv2.line(canvas, (w + 10, h - 60), (w + sidebar_w - 10, h - 60), (0, 0, 0), 2)
        cv2.putText(canvas, f"Total: ${total_price:.2f}", (w + 20, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2)

        cv2.imshow("Grocery Detection", canvas)
        
        if cv2.waitKey(1) == 27: # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()
