import os
import json
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import clip
from torch.optim.lr_scheduler import OneCycleLR
import random
from colorama import Fore, Style
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
# Global lists to store metrics
training_losses = []
validation_losses = []
print("\n")

# Save paths configuration
plots_folder = 'ft-plots'
ft_checkpoints_folder = '/content/drive/MyDrive/clip_weights'
text_logs_folder = 'ft-logs'
batch_size = 8
os.makedirs(plots_folder, exist_ok=True)
os.makedirs(ft_checkpoints_folder, exist_ok=True)
os.makedirs(text_logs_folder, exist_ok=True)

if not os.access(ft_checkpoints_folder, os.W_OK):
    print(f"Warning: No write access to {ft_checkpoints_folder}")
def adjust_unfreeze_rate(epoch, adjust_after=12, increase_rate=2):
    if epoch < adjust_after:
        return 1  # Initial slower unfreeze rate
    else:
        return increase_rate  # Increased rate after initial pass

def unfreeze_layers(model, epoch, total_layers=24, unfreeze_all=False):
    if unfreeze_all:
        for param in model.parameters():
            param.requires_grad = True
    else:
        unfreeze_every_n_epochs = adjust_unfreeze_rate(epoch)
        layers_to_unfreeze = (epoch // unfreeze_every_n_epochs) % total_layers
        layers_to_unfreeze = min(layers_to_unfreeze, total_layers)
        for i, (name, param) in enumerate(model.named_parameters()):
            if i >= total_layers - layers_to_unfreeze:
                param.requires_grad = True
            else:
                param.requires_grad = False

def monitor_gradient_norms(gradient_norms, threshold=1e-5):
    alert_messages = []
    for name, norms in gradient_norms.items():
        mean_norm = sum(norms) / len(norms)
        if mean_norm < threshold:  # Vanishing gradient
            alert_messages.append(Fore.RED + f"Vanishing gradient detected in {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
        elif mean_norm > 1000:  # Exploding gradient
            alert_messages.append(Fore.RED + f"Exploding gradient detected in {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
    if alert_messages:
        for message in alert_messages:
            print(message)

def plot_gradient_norms(gradient_norms, epoch, use_log_scale=True):
    plt.figure(figsize=(20, 10))
    cmap = plt.get_cmap('Spectral')
    sorted_layers = sorted(gradient_norms.items(), key=lambda item: max(item[1]), reverse=True)
    colors = cmap(np.linspace(0, 1, len(sorted_layers)))
    
    for (layer_name, norms), color in zip(sorted_layers, colors):
        plt.plot(norms, label=layer_name, color=color)

    plt.xlabel('Batch')
    plt.ylabel('Gradient Norm')
    plt.legend(loc='upper right', fontsize='small')
    
    if use_log_scale:
        plt.yscale('log')
        plt.title(f'Gradient Norms for Epoch {epoch} - Log Scale')
        plt.savefig(f"{plots_folder}/gradient_norms_epoch_{epoch}_log.png")
    else:
        plt.savefig(f"{plots_folder}/gradient_norms_epoch_{epoch}.png")
    
    plt.close()

def plot_training_info(epoch, training_losses, validation_losses, logits_images, logits_texts):
    epochs_x = range(1, epoch + 2)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    if len(training_losses) == len(epochs_x):
        plt.plot(epochs_x, training_losses, label='Training Loss')
    if len(validation_losses) == len(epochs_x):
        plt.plot(epochs_x, validation_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    if len(logits_images) == len(epochs_x):
        plt.plot(epochs_x, logits_images, label='Average Image Logits')
    if len(logits_texts) == len(epochs_x):
        plt.plot(epochs_x, logits_texts, label='Average Text Logits')
    plt.title('Average Logits Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Logits')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_folder}/combined_plot_epoch_{epoch + 1}.png")
    plt.close()

def calculate_metrics(logits, ground_truth):
    preds = torch.argmax(logits, dim=1)
    acc = accuracy_score(ground_truth.cpu(), preds.cpu())
    f1 = f1_score(ground_truth.cpu(), preds.cpu(), average='weighted')
    return acc, f1

class ImageTextDataset(Dataset):
    def __init__(self, image_folder, annotations_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_paths = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        labels = self.annotations[self.image_paths[idx]]
        if len(labels) >= 2:
            label = random.choice([labels[0], labels[1]])
        elif labels:
            label = labels[0]
        else:
            label = ''

        text = clip.tokenize([label])
        return image, text.squeeze(0)

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits_per_image, logits_per_text):
        labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
        loss_img = self.criterion(logits_per_image, labels)
        loss_txt = self.criterion(logits_per_text, labels)
        return (loss_img + loss_txt) / 2

# ======= CONFIGURE THIS! ======= 

# Load model and preprocessing - CLIP model:
clipmodel = 'ViT-L/14'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(clipmodel, device=device)
# NEW: Add checkpoint path configuration
  # Set this to your checkpoint path to resume training
starting_epoch = 0  # Will be updated if resuming from checkpoint
unfreeze_all = True
EPOCHS = 10000000
dataset1 = ImageTextDataset("/content/classifier_data/images_ref2", "/content/classifier_data/capture_tune_refined4.json", transform=preprocess)
concatenated_dataset = ConcatDataset([dataset1])
train_dataloader = DataLoader(concatenated_dataset, batch_size=batch_size, shuffle=True)

val_dataset = ImageTextDataset("/content/classifier_data/images", "/content/classifier_data/capture_tune.json", transform=preprocess)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Calculate total steps
total_steps = len(train_dataloader) * (EPOCHS)
learning_rate = 5e-7
batch_size = 8
from adabelief_pytorch import AdaBelief
optimizer = AdaBelief(model.parameters(), lr=learning_rate, eps=1e-16, betas=(0.9, 0.995), 
                     weight_decay=1e-3, weight_decouple=False, rectify=True, print_change_log=False)
scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps, 
                       pct_start=0.1, anneal_strategy='linear')

def reset_checkpoint_epoch(checkpoint_path):
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path)
            if isinstance(checkpoint, dict):
                # Set the epoch to 0
                checkpoint['epoch'] = 0
                
                # Adjust the scheduler state if it exists
                if 'scheduler_state_dict' in checkpoint:
                    scheduler_state = checkpoint['scheduler_state_dict']
                    scheduler_state['_step_count'] = 0
                    scheduler_state['last_epoch'] = 0
                
                # Save the modified checkpoint
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint epoch reset to 0 and saved at {checkpoint_path}")
            else:
                print("Checkpoint is not in the expected dictionary format")
        except Exception as e:
            print(f"Error modifying checkpoint: {str(e)}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")

# Usage
resume_checkpoint = "/content/drive/MyDrive/clip_weights/clip_rf2_ft_epoch_15_WAY6.pt"
# reset_checkpoint_epoch(resume_checkpoint)

# Now load the modified checkpoint
if os.path.exists(resume_checkpoint):
    print(f"Loading modified checkpoint from {resume_checkpoint}")
    try:
        checkpoint = torch.load(resume_checkpoint)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                model = checkpoint['model']
            
            starting_epoch = checkpoint['epoch']  # This should now be 0
            training_losses = checkpoint.get('training_losses', [])
            validation_losses = checkpoint.get('validation_losses', [])
            
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            print(f"Successfully loaded checkpoint. Starting from epoch {starting_epoch}")
        else:
            model = checkpoint
            print("Loaded model from checkpoint (non-dict format)")
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        print("Loading fresh CLIP model instead")
        model, preprocess = clip.load(clipmodel, device=device)
else:
    print(f"No checkpoint found at {resume_checkpoint}")
    print("Starting with fresh CLIP model")

# Recalculate total steps
total_steps = len(train_dataloader) * EPOCHS

# Update the scheduler's total_steps
scheduler.total_steps = total_steps


model = model.float()

# Training parameters
unfreeze_all = True
EPOCHS = 100000000
learning_rate = 5e-7
batch_size = 8

# Dataset setup


total_steps = len(train_dataloader) * (EPOCHS )

# Optimizer setup
from adabelief_pytorch import AdaBelief
optimizer = AdaBelief(model.parameters(), lr=learning_rate, eps=1e-16, betas=(0.9, 0.995), 
                     weight_decay=1e-3, weight_decouple=False, rectify=True, print_change_log=False)

total_steps = len(train_dataloader) * (EPOCHS)

scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps, 
                      pct_start=0.1, anneal_strategy='linear')

# Load optimizer and scheduler states if available in checkpoint
if os.path.exists(resume_checkpoint):
    checkpoint = torch.load(resume_checkpoint)
    if isinstance(checkpoint, dict):
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

scaler = GradScaler()
def save_checkpoint(model, optimizer, scheduler, epoch, training_losses, validation_losses, is_final=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),  # Save state dict instead of entire model
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'training_losses': training_losses,
        'validation_losses': validation_losses
    }
    
    if is_final:
        save_path = f"{ft_checkpoints_folder}/clip_rf2_ft_epoch_{epoch+1}_final.pt"
    else:
        save_path = f"{ft_checkpoints_folder}/clip_rf2_ft_epoch_{epoch+1}_NWAY.pt  "
    torch.save(checkpoint, save_path)
    print(Fore.GREEN + f"Checkpoint saved: {save_path}" + Style.RESET_ALL)
def trainloop():
    contrastive_loss = ContrastiveLoss().to(device)
    logits_images = []
    logits_texts = []
    
    for epoch in range(starting_epoch, EPOCHS):
        gradient_norms = {}
        unfreeze_layers(model, epoch, total_layers=24, unfreeze_all=unfreeze_all)
        model.train()
        total_train_loss = 0.0
        
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                          desc=f'Epoch {epoch + 1}/{EPOCHS}', leave=True)
        
        train_accs, train_f1s, val_accs, val_f1s = [], [], [], []
        batch_logits_images = []
        batch_logits_texts = []
        
        for batch_idx, (images, texts) in progress_bar:
            images, texts = images.to(device), texts.to(device)
            
            optimizer.zero_grad()
            with autocast():
                logits_per_image, logits_per_text = model(images, texts)
                current_batch_size = images.size(0)
                ground_truth = torch.arange(current_batch_size, device=device)
                total_loss = contrastive_loss(logits_per_image, logits_per_text)
                acc, f1 = calculate_metrics(logits_per_image, ground_truth)
                train_accs.append(acc)
                train_f1s.append(f1)

            scaler.scale(total_loss).backward()
            
            # Store gradient norms
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    grad_norm = parameter.grad.norm().item()
                    gradient_norms.setdefault(name, []).append(grad_norm)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            batch_logits_images.append(logits_per_image.mean().item())
            batch_logits_texts.append(logits_per_text.mean().item())
            
            total_train_loss += total_loss.item()
            
            progress_bar.set_postfix({
                'loss': f'{total_train_loss / (batch_idx + 1):.4f}',
                'logits_img': f'{batch_logits_images[-1]:.3f}',
                'logits_txt': f'{batch_logits_texts[-1]:.3f}'
            })

        # Calculate epoch metrics
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_losses.append(avg_train_loss)
        
        epoch_avg_logits_image = sum(batch_logits_images) / len(batch_logits_images)
        epoch_avg_logits_text = sum(batch_logits_texts) / len(batch_logits_texts)
        logits_images.append(epoch_avg_logits_image)
        logits_texts.append(epoch_avg_logits_text)
        
        plot_gradient_norms(gradient_norms, epoch)

        # Validation
        # model.eval()
        # total_val_loss = 0.0
        # print("Running Validation...")
        # with torch.no_grad():
        #     for images, texts in val_dataloader:
        #         images, texts = images.to(device), texts.to(device)
        #         current_batch_size = images.size(0)
        #         ground_truth = torch.arange(current_batch_size, device=device)
                
        #         logits_per_image, logits_per_text = model(images, texts)
        #         val_loss = contrastive_loss(logits_per_image, logits_per_text)
        #         total_val_loss += val_loss.item()
                
        #         val_acc, val_f1 = calculate_metrics(logits_per_image, ground_truth)
        #         val_accs.append(val_acc)
        #         val_f1s.append(val_f1)

        # avg_val_loss = total_val_loss / len(val_dataloader)
        # validation_losses.append(avg_val_loss)
        
        # if epoch >= 1:
        #     plot_training_info(epoch, training_losses, validation_losses, logits_images, logits_texts)
        
        # epoch_val_acc = sum(val_accs) / len(val_accs)
        # epoch_val_f1 = sum(val_f1s) / len(val_f1s)
        epoch_train_acc = sum(train_accs) / len(train_accs)
        epoch_train_f1 = sum(train_f1s) / len(train_f1s)

        # Print and log epoch results
        # print(Fore.YELLOW + "======================== STATS =============================")
        # print(Fore.YELLOW + f"Epoch {epoch + 1}/{EPOCHS}")
        # print(Fore.YELLOW + f"Training - Loss: {avg_train_loss:.4f}, Acc: {epoch_train_acc:.4f}, F1: {epoch_train_f1:.4f}")
        # print(Fore.YELLOW + f"Validation - Loss: {avg_val_loss:.4f}, Acc: {epoch_val_acc:.4f}, F1: {epoch_val_f1:.4f}")
        # print(Fore.YELLOW + "============================================================" + Style.RESET_ALL)
        validation_losses=0
        # Save checkpoint
        # Save checkpoint
        if (epoch + 1) % 5 == 0:  # This will save at epochs 5, 10, 15, 20, etc.
            checkpoint = {
                'epoch': epoch,
                'model': model,
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'training_losses': training_losses,
                'validation_losses': validation_losses
            }
            # Save with unique epoch number
            model_path = f"{ft_checkpoints_folder}/clip2_ft_epoch_{epoch+1}.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, training_losses, validation_losses)
            print(Fore.GREEN + f"Checkpoint saved for epoch {epoch+1}: {model_path}" + Style.RESET_ALL)
        
        # Also save final epoch if it's not a multiple of 5
        if epoch == EPOCHS - 1 and (epoch + 1) % 5 != 0:
            checkpoint = {
                'epoch': epoch,
                'model': model,
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'training_losses': training_losses,
                'validation_losses': validation_losses
            }
            model_path = f"{ft_checkpoints_folder}/clip2_ft_epoch_{epoch+1}_final.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, training_losses, validation_losses, is_final=True)
            print(Fore.GREEN + f"Final checkpoint saved: {model_path}" + Style.RESET_ALL)

        # Training interruption checkpoint
      

# Print initial setup information
print(f"Precision: {model.dtype}")
print(f'Total batches: {len(train_dataloader)} @ Batch Size: {batch_size}')
print(f'Starting from epoch: {starting_epoch}')
print("== START == \n")

# Main execution
if __name__ == "__main__":
    try:
        trainloop()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\nTraining interrupted by user. Saving checkpoint..." + Style.RESET_ALL)
        checkpoint = {
            'epoch': starting_epoch,  # Save the last completed epoch
            'model': model,
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'training_losses': training_losses,
            'validation_losses': validation_losses
        }
        interrupt_path = f"{ft_checkpoints_folder}/clip_ft_interrupted.pt"
        torch.save(checkpoint, interrupt_path)
        print(Fore.GREEN + f"Interrupt checkpoint saved: {interrupt_path}" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"\nError occurred during training: {str(e)}" + Style.RESET_ALL)
        raise
    finally:
        print(Fore.GREEN + "\nTraining completed or interrupted. Final plots and logs saved." + Style.RESET_ALL)
