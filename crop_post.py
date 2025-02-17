import cv2
from ultralytics import YOLO
import os
from concurrent.futures import ThreadPoolExecutor
import glob
import shutil

# Load the YOLOv8 model
model = YOLO("/home/cma/capture/last_1920_FB_2_26.pt")

def process_image(img_path, output_dir, no_prediction_dir):
    # Load the image
    print("here")
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    
    # Run YOLOv8 inference
    results = model.predict(img, stream=True, conf=0.6)
    
    prediction_made = False
    
    try:
        
        # Iterate through the detected objects
        for idx, result in enumerate(results):
            boxes = result.boxes
            if len(boxes) > 0:
                prediction_made = True
                for box in boxes:
                    cls = int(box.cls)
                    x1 = int(box.xyxy[0][0])
                    y1 = int(box.xyxy[0][1])
                    x2 = int(box.xyxy[0][2])
                    y2 = int(box.xyxy[0][3])
                    h = y2 - y1
                    w = x2 - x1
                    x1 = max(0, int(x1 - (w*0.55)))
                    y1 = max(0, int(y1 - (h*0.4)))
                    x2 = min(width, int(x2 + (w*0.55)))
                    y2 = min(height, int(y2 + (h*0.4)))
                    
                    # Crop the image within the modified bounding box
                    cropped_img = img[y1:y2, x1:x2]
                    
                    # Generate a unique filename for the cropped image
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    save_path = os.path.join(output_dir, f"{base_name}_cropped_{idx}.jpg")
                    cv2.imwrite(save_path, cropped_img)
                if not prediction_made:
            # Copy the original image to the no_prediction folder
                    shutil.copy(img_path, no_prediction_dir)
    
    except Exception as e:
        print(f"Failed to process {img_path}: {str(e)}")

def main():
    # input_dir = os.path.expanduser("~/capture/data_v3/goal")
   # input_dir = os.path.expanduser("~/classifier_data/test_dataset")
    input_dir = os.path.expanduser("/home/cma/classifier_data/nogoal_uncrop")
    output_dir = os.path.expanduser("/home/cma/classifier_data/no_goal_data_cropped")
    no_prediction_dir = os.path.expanduser("~/classifier_data/no_prediction")
    
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(no_prediction_dir, exist_ok=True)
    
    # Get all image files in the input directory
    image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))

    # Process images using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, img_path, output_dir, no_prediction_dir) for img_path in image_files]

        # Wait for all tasks to complete
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()