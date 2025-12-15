# Developer: Faheem
import streamlit as st
import cv2
import numpy as np
import json
from ultralytics import YOLO
import tempfile

# Page Config
st.set_page_config(page_title="Grocery Vision System", layout="wide")

# Load Resources
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

@st.cache_data
def load_prices():
    try:
        with open("prices.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

model = load_model()
prices = load_prices()

# UI Layout
st.title("🛒 Grocery Vision System")
st.markdown("Real-time object detection and automated pricing.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed")
    source_option = st.radio("Select Input Source", ["Webcam Capture", "Video File"])
    
    run_processing = False
    cap = None
    single_frame = None
    
    if source_option == "Webcam Capture":
        st.info("📸 Click the camera button below to capture an image (Streamlit Cloud doesn't support live video)")
        camera_image = st.camera_input("Take a picture")
        if camera_image:
            # Convert to cv2 format
            bytes_data = camera_image.getvalue()
            single_frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            run_processing = True
    else:
        video_file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)
            run_processing = st.checkbox("Start Processing", value=True)

    frame_placeholder = st.empty()
    stop_button = st.button("Stop Processing") # Add Stop button

with col2:
    st.subheader("Shopping Cart")
    cart_placeholder = st.empty()
    total_placeholder = st.empty()

# Processing Loop
if run_processing:
    # Single Frame Mode (Webcam Capture)
    if single_frame is not None:
        frame = single_frame
        
        # Detection
        results = model(frame, verbose=False)[0]
        
        detected_items = []
        total_price = 0.0
        
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
                detected_items.append({"name": label, "price": price})
            
            # Draw on frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        
        # Update Cart UI
        cart_html = ""
        for item in detected_items:
            cart_html += f"""
            <div style="display: flex; justify-content: space-between; padding: 5px; border-bottom: 1px solid #eee;">
                <span>{item['name']}</span>
                <span style="font-weight: bold; color: green;">${item['price']:.2f}</span>
            </div>
            """
        
        if not detected_items:
            cart_html = "<p style='color: gray; font-style: italic;'>No items detected...</p>"
            
        cart_placeholder.markdown(cart_html, unsafe_allow_html=True)
        
        # Update Total
        total_placeholder.markdown(f"""
        <div style="margin-top: 20px; padding: 15px; background-color: #f0f2f6; border-radius: 10px;">
            <h3 style="margin: 0; display: flex; justify-content: space-between;">
                <span>Total:</span>
                <span>${total_price:.2f}</span>
            </h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Video Mode
    elif cap is not None:
        if not cap.isOpened():
            st.error("Could not open video source.")
        else:
            while run_processing:
                if stop_button: # Check stop button
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    st.info("End of video.")
                    break
                
                # Detection
                results = model(frame, verbose=False)[0]
                
                detected_items = []
                total_price = 0.0
                
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
                        detected_items.append({"name": label, "price": price})
                    
                    # Draw on frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, display_text, (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Update Video
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")
                
                # Update Cart UI
                cart_html = ""
                for item in detected_items:
                    cart_html += f"""
                    <div style="display: flex; justify-content: space-between; padding: 5px; border-bottom: 1px solid #eee;">
                        <span>{item['name']}</span>
                        <span style="font-weight: bold; color: green;">${item['price']:.2f}</span>
                    </div>
                    """
                
                if not detected_items:
                    cart_html = "<p style='color: gray; font-style: italic;'>No items detected...</p>"
                    
                cart_placeholder.markdown(cart_html, unsafe_allow_html=True)
                
                # Update Total
                total_placeholder.markdown(f"""
                <div style="margin-top: 20px; padding: 15px; background-color: #f0f2f6; border-radius: 10px;">
                    <h3 style="margin: 0; display: flex; justify-content: space-between;">
                        <span>Total:</span>
                        <span>${total_price:.2f}</span>
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
            cap.release()

