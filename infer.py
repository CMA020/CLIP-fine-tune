import os
import json
import torch
from PIL import Image
import clip
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Configuration
inference_image_dir = '/home/cma/capture/no_goal_data_cropped'  # Change this to your image directory
checkpoint_path = '/path/to/your/checkpoint.pt'  # Change this to your checkpoint path
clipmodel = 'ViT-L/14'  # Make sure this matches your training configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 8  # Adjust based on your GPU memory

# Load CLIP model and preprocessing
model, preprocess = clip.load(clipmodel, device=device)
model = model.float()

# Load checkpoint
print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path)
if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model = checkpoint['model']
    print("Successfully loaded checkpoint")
else:
    model = checkpoint
    print("Loaded model from checkpoint (non-dict format)")

model.eval()

class InferenceDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [f for f in Path(image_folder).glob('*') 
                           if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = str(self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_path

def run_inference():
    # Create dataset and dataloader
    dataset = InferenceDataset(inference_image_dir, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    results = []
    
    with torch.no_grad():
        for images, image_paths in tqdm(dataloader, desc="Running inference"):
            images = images.to(device)
            
            # Get image features
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Process each image in the batch
            for i, (image_feature, image_path) in enumerate(zip(image_features, image_paths)):
                result = {
                    'image_path': image_path,
                    'features': image_feature.cpu().numpy().tolist()
                }
                results.append(result)
    
    # Save results
    output_file = 'inference_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f)
    
    print(f"Inference completed. Results saved to {output_file}")
    return results

if __name__ == "__main__":
    print(f"Running inference on images from: {inference_image_dir}")
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    
    try:
        results = run_inference()
        print(f"Successfully processed {len(results)} images")
    except Exception as e:
        print(f"Error during inference: {str(e)}")