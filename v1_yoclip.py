import torch
import cv2
import os
from ultralytics import YOLO
import clip
from PIL import Image
from collections import deque
import time

# YOLO model setup
model = YOLO(os.path.expanduser("/content/drive/MyDrive/yolo_weights/last_1920_FB_2_26.pt"))

# CLIP setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load('ViT-L/14', device=device)
clip_model = clip_model.float()

# Load CLIP checkpoint
checkpoint_path = '/content/drive/MyDrive/clip_weights/clip_rf_ft_epoch_25.pt'
checkpoint = torch.load(checkpoint_path)
if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        clip_model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        clip_model = checkpoint['model']
else:
    clip_model = checkpoint
clip_model.eval()

# Text prompts for CLIP
text_prompts = [
    "football outside footballpost",
    "not a goal",
    "football inside footballpost",
    "goal",
]
text_inputs = clip.tokenize(text_prompts).to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Video setup
#vid = cv2.VideoCapture("/content/classifier_data/manc.mp4")
vid=cv2.VideoCapture("/content/drive/MyDrive/tot.mp4")
goal_counter = 0
frame_counter = 0
rows, cols = 0, 0
class1_coords = None
last_person_coords = None
last_clip_prediction_frame = 0
clip_cooldown = 180

# Frame stack
frame_stack = deque(maxlen=150)

# Video saving setup
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_directory = os.path.expanduser("/content/buffer")
os.makedirs(output_directory, exist_ok=True)
video_counter = 0

def predict(img):
    global pos, class1_coords, last_person_coords, frame_counter, last_clip_prediction_frame, rows, cols

    results = model(img, imgsz=(1920, 1080), show=False)
    
    person_detected = False
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls)
            x, y, w, h = map(int, box.xywh[0])

            if cls == 1:  # Store coordinates of class 1 object
                class1_coords = (x, y, w, h)
            if cls == 0:  # Person class
                person_detected = True
                last_person_coords = (x, y, w, h)

    # Use last known coordinates if objects are not detected
    if not person_detected and last_person_coords:
        x, y, w, h = last_person_coords
    elif not person_detected:
        x, y, w, h = cols // 2, rows // 2, 100, 200  # Default values

    # Process with CLIP if cooldown period has passed
    if frame_counter - last_clip_prediction_frame >= clip_cooldown:
        coords_to_use = class1_coords if class1_coords else (x, y, w, h)
        return process_with_clip(img, coords_to_use)
    else:
        return 2  # Return 2 if in cooldown period

def process_with_clip(img, coords):
    global last_clip_prediction_frame
    global goal_counter
    
    x, y, w, h = coords
    x1, y1 = max(0, x - int(w * 0.7)), max(0, y - int(h * 0.4))
    x2, y2 = min(cols, x + int(w * 1.7)), min(rows, y + int(h * 1.4))

    # Crop and preprocess image
    cropped_img = img[y1:y2, x1:x2]
    pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    image_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    # Get CLIP predictions
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity scores
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        scores = similarity[0].cpu().numpy()

    # Check if either "goal" or "football inside footballpost" scores are above 70%
    goal_scores = [scores[2], scores[3]]  # Indices for goal-related prompts
    is_goal = any(score > 0.6 for score in goal_scores)
    print(scores , "scoooooooooooooooooooooooooooooooooooooooooore")
    if is_goal:
        goal_counter += 1
        
        print(f"Saved goal frame: ")
        print(f"Confidence scores: inside post: {scores[2]:.2%}, goal: {scores[3]:.2%}")
        last_clip_prediction_frame = frame_counter
        return 0  # Return 0 for goal detected
    
    return 1  # Return 1 for no goal

def save_video(frames, counter):
    global rows, cols
    output_filename = f"{counter}.mkv"
    output_path = os.path.join(output_directory, output_filename)
    out = cv2.VideoWriter(output_path, fourcc, 30, (cols, rows))
    
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Saved video: {output_filename}")

if __name__ == '__main__':
    save_pending = False
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            ret, img = vid.read()
            if not ret:
                break

            frame_count += 1
            rows, cols, _ = img.shape
            frame_stack.append(img)

            frame_counter += 1
            if frame_counter % 4 == 0:
                prediction = predict(img)
                if prediction == 0:  # If CLIP predicts a goal
                    save_pending = True
                    frames_after_prediction = 0
                
            if save_pending:
                frames_after_prediction += 1
                if frames_after_prediction >= 45:
                    video_counter += 1
                    save_video(frame_stack, video_counter)
                    save_pending = False
                    frames_after_prediction = 0

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        fps = frame_count / total_time
        
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {fps:.2f}")
        
        vid.release()
        cv2.destroyAllWindows()