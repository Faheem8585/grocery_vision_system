import cv2
import os
import glob
import random

def create_slideshow(output_path="grocery_cv_pipeline/data/test_video/test_run_video.mp4", fps=1):
    # Get all images from the dataset
    image_paths = []
    # Look in train/images since raw is gone
    search_dir = "grocery_cv_pipeline/data/train/images" 
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print(f"No images found in {search_dir}")
        return

    print(f"Found {len(image_paths)} images. Creating slideshow...")
    random.shuffle(image_paths)
    
    # Read first image to get dimensions
    first_img = cv2.imread(image_paths[0])
    h, w = first_img.shape[:2]
    
    # Resize all to 640x480 for consistency
    target_w, target_h = 640, 480
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_w, target_h))
    
    for i, img_path in enumerate(image_paths[:50]): # Use top 50 images
        img = cv2.imread(img_path)
        if img is None: continue
        
        img = cv2.resize(img, (target_w, target_h))
        
        # Write same image multiple times to create a "pause" effect
        for _ in range(fps * 2): # Show each image for 2 seconds
            out.write(img)
            
        print(f"Added {i+1}/50: {os.path.basename(img_path)}")

    out.release()
    print(f"Saved slideshow to {output_path}")

if __name__ == "__main__":
    create_slideshow()
