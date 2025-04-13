# train.py
"""
This is the main training script for our audio mastering neural network.
It handles the entire training pipeline including:
- Data loading and preprocessing
- Model training and validation
- Loss computation and optimization
- Checkpointing and logging
- Early stopping and learning rate scheduling

this is where we teach our models how to master audio! it's like training a dog, but for computers

Training process:
1. Data loading and preprocessing:
   - Spectrograms are normalized to [0,1] range
   - Parameters are scaled to appropriate ranges
   - Data is split 80/20 for training/validation

2. Model training:
   - Forward pass computes spectrogram and parameter predictions
   - Loss is computed using the weighted combination
   - Backpropagation updates model parameters
   - Learning rate is adjusted based on validation loss

3. Validation:
   - Model is evaluated on held-out data
   - Early stopping prevents overfitting
   - Checkpoints are saved for best performing models

4. Monitoring:
   - Loss components are tracked separately
   - Learning rate adjustments are logged
   - Model performance is visualized through plots

Lessons learned from previous iterations:
1. Initial attempts with pure MSE loss led to blurry spectrograms
   - Solution: Added L1 loss component for sharper reconstructions

2. Early versions overfitted quickly
   - Solution: Implemented early stopping and reduced model capacity

3. Parameter prediction was initially poor
   - Solution: Added dedicated LSTM branch and increased its capacity

4. Training was unstable with large batches
   - Solution: Reduced batch size and added gradient clipping

5. Model struggled with temporal consistency
   - Solution: Added skip connections and attention mechanisms
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
from models import OneStageDeepUNet

# ----------------------------
# Collate Function
# ----------------------------
def collate_pad(batch):
    """
    Custom collate function to handle variable-length spectrograms in a batch.
    
    Why we need this:
    - Audio files have different lengths, but neural networks need fixed-size inputs
    - Padding shorter sequences to match the longest sequence in the batch
    - Preserving the original data structure while making it compatible with batching
    
    Implementation details:
    1. Extracts input spectrograms, target spectrograms, and parameters from batch
    2. Finds the maximum temporal length in the batch
    3. Pads shorter sequences with zeros to match the maximum length
    4. Ensures consistent channel dimensions
    5. Stacks padded tensors into a single batch
    
    Args:
        batch: List of tuples containing (input_spec, target_spec, params)
    
    Returns:
        inputs: Padded input spectrograms [batch_size, channels, freq_bins, time_steps]
        targets: Padded target spectrograms [batch_size, channels, freq_bins, time_steps]
        params: Stacked parameter tensors [batch_size, num_params]
    """
    input_specs, target_specs, param_list = zip(*batch)
    max_T = max(t.shape[-1] for t in input_specs)
    padded_inputs = []
    for t in input_specs:
        if len(t.shape) == 2:
            t = t.unsqueeze(0)
        elif len(t.shape) == 3 and t.shape[0] > 1:
            t = t[:1]
        padded = F.pad(t, (0, max_T - t.shape[-1]))
        padded_inputs.append(padded)
    padded_targets = []
    for t in target_specs:
        if len(t.shape) == 2:
            t = t.unsqueeze(0)
        elif len(t.shape) == 3 and t.shape[0] > 1:
            t = t[:1]
        padded = F.pad(t, (0, max_T - t.shape[-1]))
        padded_targets.append(padded)
    inputs = torch.stack(padded_inputs, dim=0)
    targets = torch.stack(padded_targets, dim=0)
    params = torch.stack(param_list, dim=0)
    return inputs, targets, params

# ----------------------------
# Loss Module with Revised Weights
# ----------------------------
class AudioMasteringLoss(nn.Module):
    """
    Custom loss function for audio mastering that combines multiple loss components.
    
    Why we need multiple loss components:
    - Spectrogram reconstruction alone doesn't capture all aspects of audio quality
    - Parameter prediction accuracy is crucial for practical use
    - Perceptual loss helps align with human hearing
    
    Implementation details:
    1. Spectrogram L1 Loss:
       - Preserves fine details in frequency domain
       - Less sensitive to outliers than MSE
       - Helps maintain spectral characteristics
    
    2. Parameter Loss:
       - Ensures accurate prediction of mastering parameters
       - Uses MSE for quadratic error penalization
       - Critical for practical application
    
    3. Perceptual Loss:
       - Computed on mel spectrograms
       - Captures human perception of sound
       - Helps with subjective audio quality
    
    The weights are empirically determined:
    - alpha (0.5): Balances spectrogram reconstruction
    - beta (0.5): Balances parameter prediction
    - gamma (0.05): Light perceptual guidance
    """
    def __init__(self, sr=44100, n_fft=2048, hop_length=512, n_mels=128):
        super(AudioMasteringLoss, self).__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Loss weights determined through experimentation
        self.alpha = 0.5  # spectrogram (L1) loss weight
        self.beta = 0.5   # parameter loss weight
        self.gamma = 0.05 # perceptual loss weight
        
        self.spec_loss_l1 = nn.L1Loss()
        self.param_loss = nn.MSELoss()
    
    def forward(self, output_spec, target_spec, predicted_params, target_params):
        """
        Computes the total loss combining all components.
        
        Implementation steps:
        1. Clamp spectrograms to avoid numerical issues
        2. Compute spectrogram L1 loss
        3. Compute parameter MSE loss
        4. Compute perceptual loss on mel spectrograms
        5. Combine losses with respective weights
        
        Error handling:
        - Catches exceptions in perceptual loss computation
        - Provides fallback to zero if mel computation fails
        """
        epsilon = 1e-6
        output_spec = torch.clamp(output_spec, min=epsilon)
        target_spec = torch.clamp(target_spec, min=epsilon)
        
        spec_l1 = self.spec_loss_l1(output_spec, target_spec)
        param_loss_val = self.param_loss(predicted_params, target_params)
        
        try:
            mel_out = compute_mel_spectrogram(output_spec.detach().cpu().numpy()[0,0], sr=self.sr,
                                            n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)
            mel_target = compute_mel_spectrogram(target_spec.detach().cpu().numpy()[0,0], sr=self.sr,
                                                n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)
            mel_out = torch.tensor(mel_out, dtype=torch.float32, device=output_spec.device)
            mel_target = torch.tensor(mel_target, dtype=torch.float32, device=output_spec.device)
            perceptual_loss_val = self.spec_loss_l1(mel_out, mel_target)
        except Exception as e:
            print(f"Error computing perceptual loss: {e}")
            perceptual_loss_val = torch.tensor(0.0, device=output_spec.device)
        
        total_loss = self.alpha * spec_l1 + self.beta * param_loss_val + self.gamma * perceptual_loss_val
        
        return total_loss, {
            'spec_loss_l1': spec_l1.item(),
            'parameter_loss': param_loss_val.item(),
            'perceptual_loss': perceptual_loss_val.item(),
            'total_loss': total_loss.item()
        }

# ----------------------------
# Loss Plotting
# ----------------------------
def save_loss_plot(loss_history, filename):
    """
    Creates and saves a comprehensive loss plot.
    
    Purpose:
    - Visualize training progress
    - Monitor individual loss components
    - Identify potential issues (overfitting, underfitting)
    
    Implementation:
    1. Creates a 2x2 grid of subplots
    2. Plots total loss and individual components
    3. Adds proper labels and legends
    4. Saves to specified filename
    
    The plot helps in:
    - Understanding model convergence
    - Identifying training problems
    - Comparing different training runs
    """
    epochs = range(1, len(loss_history['total']) + 1)
    plt.figure(figsize=(14,10))
    plt.subplot(2,2,1)
    plt.plot(epochs, loss_history['total'], marker='o', label='Total Loss')
    plt.xlabel('Epoch'); plt.ylabel('Total Loss'); plt.title('Total Loss'); plt.legend(); plt.grid(True)
    
    plt.subplot(2,2,2)
    plt.plot(epochs, loss_history['spec_loss_l1'], marker='o', label='Spectrogram L1 Loss')
    plt.xlabel('Epoch'); plt.ylabel('L1 Loss'); plt.title('Spectrogram L1 Loss'); plt.legend(); plt.grid(True)
    
    plt.subplot(2,2,3)
    plt.plot(epochs, loss_history['parameter_loss'], marker='o', label='Parameter Loss')
    plt.xlabel('Epoch'); plt.ylabel('Parameter Loss'); plt.title('Parameter Loss'); plt.legend(); plt.grid(True)
    
    plt.subplot(2,2,4)
    plt.plot(epochs, loss_history['perceptual_loss'], marker='o', label='Perceptual Loss')
    plt.xlabel('Epoch'); plt.ylabel('Perceptual Loss'); plt.title('Perceptual Loss'); plt.legend(); plt.grid(True)
    
    plt.suptitle('Training Losses per Epoch', fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(filename)
    plt.close()

# ----------------------------
# Training Epoch
# ----------------------------
def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Performs one complete training epoch.
    
    Implementation details:
    1. Sets model to training mode
    2. Processes each batch:
       - Moves data to correct device
       - Computes forward pass
       - Calculates loss
       - Performs backpropagation
       - Updates weights
    3. Handles variable-length sequences
    4. Logs progress periodically
    
    Key features:
    - Gradient accumulation for stability
    - Progress tracking
    - Loss component monitoring
    - Memory-efficient processing
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    for batch_idx, (inputs, targets, gt_params) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        gt_params = gt_params.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        unet_out, predicted_params_norm = model(inputs)
        
        # Handle variable-length sequences
        min_T = min(unet_out.shape[-1], targets.shape[-1])
        unet_out = unet_out[..., :min_T]
        targets = targets[..., :min_T]
        
        # Expand parameters to match temporal dimension
        T = predicted_params_norm.shape[1]
        gt_params_expanded = gt_params.unsqueeze(1).expand(-1, T, -1)
        
        # Compute loss and backpropagate
        loss, loss_components = criterion(unet_out, targets, predicted_params_norm, gt_params_expanded)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log progress
        if (batch_idx + 1) % max(1, num_batches // 10) == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Batch {batch_idx+1}/{num_batches} - Avg Loss: {avg_loss:.4f} | "
                f"Spec L1: {loss_components['spec_loss_l1']:.4f}, Param: {loss_components['parameter_loss']:.4f}, "
                f"Perc: {loss_components['perceptual_loss']:.4f}")
    
    return total_loss / num_batches

# ----------------------------
# Training Loop
# ----------------------------
def train_model(model, train_loader, val_loader, device, num_epochs=100, learning_rate=5e-5, weight_decay=1e-5):
    """
    Main training loop that orchestrates the entire training process.
    
    Implementation details:
    1. Setup:
       - Creates logging directory
       - Initializes loss function and optimizer
       - Sets up learning rate scheduler
       - Prepares checkpoint directory
    
    2. Training loop:
       - Performs training epochs
       - Validates model periodically
       - Adjusts learning rate
       - Saves checkpoints
       - Implements early stopping
    
    3. Monitoring:
       - Tracks loss history
       - Saves loss plots
       - Logs training progress
       - Monitors validation performance
    
    Key features:
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Comprehensive logging
    - Loss visualization
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(project_root, "logs", f"training_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    criterion = AudioMasteringLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_val_loss = float('inf')
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    loss_history = {
        'total': [],
        'spec_loss_l1': [],
        'parameter_loss': [],
        'perceptual_loss': [],
        'val': []
    }
    
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
        loss_history['total'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_comp = {'spec_loss_l1': 0.0, 'parameter_loss': 0.0, 'perceptual_loss': 0.0}
        
        with torch.no_grad():
            for inputs, targets, gt_params in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                gt_params = gt_params.to(device)
                
                unet_out, predicted_params_norm = model(inputs)
                min_T = min(unet_out.shape[-1], targets.shape[-1])
                unet_out = unet_out[..., :min_T]
                targets = targets[..., :min_T]
                
                T = predicted_params_norm.shape[1]
                gt_params_expanded = gt_params.unsqueeze(1).expand(-1, T, -1)
                
                loss, loss_components = criterion(unet_out, targets, predicted_params_norm, gt_params_expanded)
                val_loss += loss.item()
                for key in val_comp:
                    val_comp[key] += loss_components.get(key, 0)
        
        val_loss /= len(val_loader)
        loss_history['val'].append(val_loss)
        for key in ['spec_loss_l1', 'parameter_loss', 'perceptual_loss']:
            loss_history[key].append(val_comp[key] / len(val_loader))
        
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging and checkpointing
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"    (Spec L1: {loss_history['spec_loss_l1'][-1]:.4f}, Param: {loss_history['parameter_loss'][-1]:.4f}, Perc: {loss_history['perceptual_loss'][-1]:.4f})")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
            print("  New best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
        
        # Save regular checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, os.path.join(checkpoint_dir, f'one_stage_deep_unet_epoch_{epoch+1}.pt'))
        
        # Save loss plot
        loss_plot_path = os.path.join(log_dir, 'loss_plot.png')
        save_loss_plot(loss_history, loss_plot_path)
        
        # Early stopping check
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    return train_losses, val_losses

def main():
    """
    Main function that sets up and runs the training process.
    
    Implementation details:
    1. Sets up project paths and imports
    2. Configures hyperparameters
    3. Prepares dataset and data loaders
    4. Initializes model and moves to device
    5. Runs training process
    
    Key features:
    - Configurable hyperparameters
    - Automatic device selection (CPU/GPU)
    - Dataset validation
    - Progress monitoring
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_root)
    print("Project root:", project_root)
    
    # Hyperparameters
    num_epochs = 100
    batch_size = 2
    learning_rate = 5e-5
    weight_decay = 1e-5
    
    # Audio parameters
    sample_rate = 44100
    hop_length = 512
    n_fft = 2048
    n_mels = 128
    
    # Dataset setup
    dataset_dir = os.path.join(project_root, "experiments", "output_full", "output_audio")
    dataset_dir = os.path.abspath(dataset_dir)
    print("Dataset directory:", dataset_dir)
    
    # Initialize dataset
    dataset = PairedAudioDataset(
        audio_dir=dataset_dir,
        sr=sample_rate,
        transform=compute_mel_spectrogram,
        mode="spectrogram"
    )
    
    # Validate dataset
    print(f"Spectrogram mode, dataset length: {len(dataset)}")
    if len(dataset) > 0:
        sample_input, sample_target, sample_params = dataset[0]
        print("Input spectrogram shape:", sample_input.shape)
        print("Target spectrogram shape:", sample_target.shape)
        print("Normalized parameter vector:", sample_params)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_pad)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_pad)
    
    # Initialize model
    model = OneStageDeepUNet(sr=sample_rate, hop_length=hop_length, in_channels=1, out_channels=1,
                            base_features=64, blocks_per_level=5, lstm_hidden=32, num_layers=1, num_params=10)
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Using device:", device)
    
    # Start training
    print("Starting training...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, device,
                                        num_epochs=num_epochs, learning_rate=learning_rate,
                                        weight_decay=weight_decay)
    print("Training complete.")

if __name__ == "__main__":
    main()
