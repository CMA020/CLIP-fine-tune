# NEW: Add checkpoint path configuration
resume_checkpoint = "/content/drive/MyDrive/clip_weights/clip_ft_25.pt"  # Set this to your checkpoint path to resume training
starting_epoch = 0  # Will be updated if resuming from checkpoint

if os.path.exists(resume_checkpoint):
    print(f"Loading checkpoint from {resume_checkpoint}")
    try:
        checkpoint = torch.load(resume_checkpoint)
        print(f"Checkpoint type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print("Checkpoint keys:", checkpoint.keys())
            
            if 'model_state_dict' in checkpoint:
                # If the checkpoint contains a state dict
                model.load_state_dict(checkpoint['model_state_dict'])
                starting_epoch = checkpoint.get('epoch', 0) + 1
                training_losses = checkpoint.get('training_losses', [])
                validation_losses = checkpoint.get('validation_losses', [])
            elif 'model' in checkpoint:
                # If the checkpoint contains the entire model
                model = checkpoint['model']
                starting_epoch = checkpoint.get('epoch', 0) + 1
                training_losses = checkpoint.get('training_losses', [])
                validation_losses = checkpoint.get('validation_losses', [])
            else:
                raise KeyError("Checkpoint doesn't contain 'model' or 'model_state_dict'")
            
            print(f"Successfully loaded checkpoint. Resuming from epoch {starting_epoch}")
        else:
            # If checkpoint is just the model
            model = checkpoint
            print("Loaded model from checkpoint (non-dict format)")
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        print("Loading fresh CLIP model instead")
        model, preprocess = clip.load(clipmodel, device=device)
else:
    print(f"No checkpoint found at {resume_checkpoint}")
    print("Loading fresh CLIP model")
    model, preprocess = clip.load(clipmodel, device=device)

model = model.float()
