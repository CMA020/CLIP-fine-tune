import os
import json
import torch
import clip
from PIL import Image
from torch.cuda.amp import autocast
from tqdm import tqdm
import argparse
from datetime import datetime
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='CLIP inference on a directory of images')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the fine-tuned CLIP checkpoint')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images to process')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on (cuda/cpu)')
    return parser.parse_args()

def load_model(checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}")
    # Load the original CLIP model first
    model, preprocess = clip.load("ViT-L/14", device=device)
    model = model.float()
    
    # Load the fine-tuned state
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model = checkpoint['model']
    else:
        model = checkpoint
    
    model.eval()
    return model, preprocess

def get_image_files(directory):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [
        os.path.join(directory, f) for f in os.listdir(directory)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    return image_files

def process_batch(model, image_paths, text_candidates, preprocess, device):
    images = []
    valid_paths = []
    
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            preprocessed_image = preprocess(image)
            images.append(preprocessed_image)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    if not images:
        return []
    
    image_batch = torch.stack(images).to(device)
    text_tokens = clip.tokenize(text_candidates).to(device)
    
    with torch.no_grad(), autocast():
        logits_per_image, _ = model(image_batch, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
    results = []
    for i, path in enumerate(valid_paths):
        image_probs = probs[i]
        sorted_results = sorted(zip(text_candidates, image_probs), key=lambda x: x[1], reverse=True)
        results.append((path, sorted_results))
    
    return results

def main():
    args = parse_args()
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Define text candidates - UPDATE THIS LIST based on your use case
    text_candidates = [
        "example label 1",
        "example label 2",
        "example label 3",
        # Add more labels as needed
    ]
    
    # Load model
    model, preprocess = load_model(args.checkpoint, args.device)
    
    # Get list of images
    image_files = get_image_files(args.image_dir)
    if not image_files:
        print(f"No valid images found in {args.image_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images in batches
    all_results = []
    for i in tqdm(range(0, len(image_files), args.batch_size)):
        batch_paths = image_files[i:i + args.batch_size]
        batch_results = process_batch(model, batch_paths, text_candidates, preprocess, args.device)
        all_results.extend(batch_results)
    
    # Prepare results for saving
    results_for_csv = []
    results_for_json = {}
    
    for image_path, predictions in all_results:
        image_name = os.path.basename(image_path)
        top_prediction = predictions[0]  # Get the highest probability prediction
        
        # Prepare CSV data
        results_for_csv.append({
            'image': image_name,
            'top_prediction': top_prediction[0],
            'confidence': float(top_prediction[1]),
            **{f"{label}_prob": float(prob) for label, prob in predictions}
        })
        
        # Prepare JSON data
        results_for_json[image_name] = {
            'predictions': [{'label': label, 'probability': float(prob)} 
                           for label, prob in predictions]
        }
    
    # Save results
    csv_path = os.path.join(args.output_dir, f'results_{timestamp}.csv')
    json_path = os.path.join(args.output_dir, f'results_{timestamp}.json')
    
    # Save CSV
    pd.DataFrame(results_for_csv).to_csv(csv_path, index=False)
    
    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(results_for_json, f, indent=2)
    
    print(f"Results saved to:")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")

if __name__ == "__main__":
    main()
