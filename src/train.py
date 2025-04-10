"""
train.py

This training script implements a cascaded neural network for automatic music mastering.

Improvements in this version:
- Better balanced loss function
- More robust handling of NaN values
- Improved data loading with validation split
- Better learning rate scheduling
- Training/validation loss tracking and visualization
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import librosa
import matplotlib.pyplot as plt
from datetime import datetime

from dataset import PairedAudioDataset, compute_mel_spectrogram
from models import UNet, LSTMForecasting, CascadedMastering

def collate_pad(batch):
    """Collate function that pads spectrograms to the same length in a batch."""
    # Extract input and target spectrograms
    input_specs, target_specs = zip(*batch)
    
    # Find maximum time dimension
    max_T = max(t.shape[-1] for t in input_specs)
    
    # Pad and stack input spectrograms
    padded_inputs = []
    for t in input_specs:
        # Ensure tensor is 3D: [channels=1, freq, time]
        if len(t.shape) == 2:
            t = t.unsqueeze(0)  # Add channel dim if needed
        elif len(t.shape) == 3:
            if t.shape[0] > 1:
                t = t[:1]  # Take only first channel if multiple exist
        else:
            raise ValueError(f"Unexpected tensor shape: {t.shape}")
            
        # Verify shape is [1, freq, time]
        assert t.shape[0] == 1, f"Expected 1 channel, got {t.shape[0]}"
        
        padded = F.pad(t, (0, max_T - t.shape[-1]))
        padded_inputs.append(padded)
    
    # Pad and stack target spectrograms
    padded_targets = []
    for t in target_specs:
        if len(t.shape) == 2:
            t = t.unsqueeze(0)  # Add channel dim if needed
        elif len(t.shape) == 3:
            if t.shape[0] > 1:
                t = t[:1]  # Take only first channel if multiple exist
        else:
            raise ValueError(f"Unexpected tensor shape: {t.shape}")
            
        # Verify shape is [1, freq, time]
        assert t.shape[0] == 1, f"Expected 1 channel, got {t.shape[0]}"
        
        padded = F.pad(t, (0, max_T - t.shape[-1]))
        padded_targets.append(padded)
    
    # Stack into batch tensors
    inputs = torch.stack(padded_inputs, dim=0)
    targets = torch.stack(padded_targets, dim=0)
    
    # Verify final shapes
    assert inputs.shape[1] == 1, f"Expected 1 channel in batch, got {inputs.shape[1]}"
    assert targets.shape[1] == 1, f"Expected 1 channel in batch, got {targets.shape[1]}"
    
    return inputs, targets

class AudioMasteringLoss(nn.Module):
    def __init__(self, sr=44100, n_fft=2048, hop_length=512, n_mels=128):
        super(AudioMasteringLoss, self).__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Adjusted loss weights for better balance
        self.spectrogram_weight = 0.7  # Primary focus on spectrogram match
        self.parameter_weight = 0.1    # Less weight on parameters
        self.perceptual_weight = 0.2   # Some weight on perceptual quality
        
        # Loss functions
        self.spectrogram_loss = nn.L1Loss()
        self.parameter_loss = nn.MSELoss()
    
    def forward(self, output_spec, target_spec, predicted_params, target_params):
        # Add small epsilon to prevent log(0) and NaN
        epsilon = 1e-6
        
        # Ensure spectrograms are finite
        output_spec = torch.clamp(output_spec, min=epsilon)
        target_spec = torch.clamp(target_spec, min=epsilon)
        
        # Spectrogram loss with stability checks
        if torch.isnan(output_spec).any() or torch.isnan(target_spec).any():
            spec_loss = torch.tensor(0.0, device=output_spec.device)
            print("Warning: NaN detected in spectrograms, skipping loss computation")
        else:
            spec_loss = self.spectrogram_loss(output_spec, target_spec)
        
        # Parameter loss with stability checks
        if torch.isnan(predicted_params).any() or torch.isnan(target_params).any():
            param_loss = torch.tensor(0.0, device=output_spec.device)
            print("Warning: NaN detected in parameters, skipping parameter loss")
        else:
            param_loss = self.parameter_loss(predicted_params, target_params)
        
        # Perceptual loss (direct mel spectrogram comparison)
        try:
            # Since we're already working with mel spectrograms, we can compare them directly
            perceptual_loss = self.spectrogram_loss(output_spec, target_spec)
        except Exception as e:
            print(f"Error computing perceptual loss: {e}")
            perceptual_loss = torch.tensor(0.0, device=output_spec.device)
        
        # Total loss with stability checks
        total_loss = (
            self.spectrogram_weight * spec_loss +
            self.parameter_weight * param_loss +
            self.perceptual_weight * perceptual_loss
        )
        
        # Ensure total loss is finite
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: NaN/Inf detected in total loss, resetting to 0")
            total_loss = torch.tensor(0.0, device=output_spec.device)
        
        return total_loss, {
            'spectrogram_loss': spec_loss.item(),
            'parameter_loss': param_loss.item(),
            'perceptual_loss': perceptual_loss.item()
        }

def save_loss_plot(train_losses, val_losses, filename):
    """Save a plot of training and validation losses."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs, predicted_params = model(inputs)
        
        # Ensure time dimensions match
        min_time = min(outputs.shape[-1], targets.shape[-1])
        outputs = outputs[..., :min_time]
        targets = targets[..., :min_time]
        
        # Calculate loss
        loss, _ = criterion(
            outputs, 
            targets,
            predicted_params, 
            predicted_params  # Using predicted params as target params since we don't have ground truth
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print progress every 10% of batches
        if (batch_idx + 1) % max(1, num_batches // 10) == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Batch {batch_idx + 1}/{num_batches} - Loss: {avg_loss:.4f}")
    
    return total_loss / num_batches

def train_model(model, train_loader, val_loader, device, num_epochs=50,
                learning_rate=1e-4, weight_decay=1e-5):
    """Train the cascaded mastering model with improved monitoring."""
    # Initialize log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(project_root, "logs", f"training_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize loss function
    criterion = AudioMasteringLoss()
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(),
                        lr=learning_rate,
                        weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    
    # Training loop
    best_val_loss = float('inf')
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    patience_counter = 0
    early_stopping_patience = 10
    
    print("\nStarting training...")
    print(f"Initial learning rate: {learning_rate:.2e}")
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(val_loader.dataset)} samples")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for degraded_spec, target_spec in val_loader:
                degraded_spec = degraded_spec.to(device)
                target_spec = target_spec.to(device)
                
                try:
                    output_spec, predicted_params = model(degraded_spec)
                    loss, _ = criterion(output_spec, target_spec, predicted_params, predicted_params)
                    val_loss += loss.item()
                except Exception as e:
                    print(f"Error during validation: {e}")
                    continue
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
            print("  New best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
        
        # Save latest model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(checkpoint_dir, f'cascaded_model_epoch_{epoch+1}.pt'))
        
        # Save loss plot
        save_loss_plot(
            train_losses, 
            val_losses, 
            os.path.join(log_dir, 'loss_plot.png')
        )
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    return train_losses, val_losses

def main():
    # Set project root and path
    global project_root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_root)
    print("Project root:", project_root)
    
    # Hyperparameters
    num_epochs = 30      # Increased from 10 to allow better convergence
    batch_size = 2       # Small batch size for stability
    learning_rate = 5e-5 # Adjusted learning rate
    weight_decay = 1e-5  # Regularization to prevent overfitting
    
    # Audio parameters
    sample_rate = 44100
    hop_length = 512
    n_fft = 2048
    n_mels = 128
    ir_length = 20  # IR length in time frames for reverb
    
    # Dataset location
    dataset_dir = os.path.join(project_root, "experiments", "output_full", "output_audio")
    dataset_dir = os.path.abspath(dataset_dir)
    print("Dataset directory:", dataset_dir)
    
    # Creating the Dataset
    dataset = PairedAudioDataset(
        audio_dir=dataset_dir,
        sr=sample_rate,
        transform=compute_mel_spectrogram,
        mode="spectrogram"
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Split dataset into training and validation (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_pad
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pad
    )
    
    # Initialize models
    unet = UNet(in_channels=1, out_channels=1)
    lstm = LSTMForecasting(input_size=1, hidden_size=32, num_layers=1)
    
    # Initialize weights properly
    for name, param in lstm.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)
    
    # Create the cascaded model
    model = CascadedMastering(
        unet, lstm, sr=sample_rate, hop_length=hop_length, ir_length=ir_length
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Using device:", device)
    
    # Train the model
    print("Starting training...")
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    print("Training complete.")

if __name__ == "__main__":
    main()
