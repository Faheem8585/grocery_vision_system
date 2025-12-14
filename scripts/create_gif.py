import cv2
import imageio
import os

def convert_mp4_to_gif(video_path, gif_path, resize_scale=0.5, fps=10):
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    frames = []
    
    print(f"Reading video from {video_path}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize to reduce file size
        h, w = frame.shape[:2]
        new_dim = (int(w * resize_scale), int(h * resize_scale))
        frame = cv2.resize(frame, new_dim)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
    cap.release()
    
    if not frames:
        print("No frames extracted.")
        return

    print(f"Saving GIF to {gif_path} ({len(frames)} frames)...")
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print("Done!")

if __name__ == "__main__":
    # Ensure assets dir exists
    os.makedirs("grocery_cv_pipeline/assets", exist_ok=True)
    
    convert_mp4_to_gif(
        "grocery_cv_pipeline/data/test_video/test_run_video.mp4",
        "grocery_cv_pipeline/assets/demo.gif"
    )
