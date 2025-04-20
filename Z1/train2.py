#!/usr/bin/env python3
"""
train.py

This training script sets up data loading, the training loop, and logging for our audio mastering model.
It uses the new OneStageDeepUNet architecture which returns both a "restored" spectrogram and predicted parameters.
The loss function consists of:
• Spectrogram reconstruction L1 loss (α weight),
• Parameter MSE (L2) loss (β weight) left unscaled,
• Perceptual loss (L1 on mel spectrograms) that is scaled by 0.01 and weighted with γ.
A ReduceLROnPlateau scheduler (with verbose disabled) is used to adjust the learning rate.
Anomaly detection is enabled to help catch in-place modifications.
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
import argparse
import logging

# Enable anomaly detection so that errors like in-place modifications are flagged with more detail.
torch.autograd.set_detect_anomaly(True)

from dataset import PairedAudioDataset, compute_mel_spectrogram
from models import OneStageDeepUNet

# ----------------------------
# Logger Setup
# ----------------------------
logger = logging.getLogger("TrainLogger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ----------------------------
# Collate Function
# ----------------------------
def collate_pad(batch):
    """
    Custom collate function to pad variable-length spectrograms.
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
# Custom Loss Module
# ----------------------------
class AudioMasteringLoss(nn.Module):
    """
    Loss function combining:
    - Spectrogram L1 loss (reconstruction loss),
    - Parameter MSE (L2) loss, and
    - Perceptual loss computed on mel spectrograms.
    
    We use weights:
    α = 0.60, β = 0.30, γ = 0.10.
    The perceptual loss is scaled by 0.01.
    """
    def __init__(self, sr=44100, n_fft=2048, hop_length=512, n_mels=128, scale_percept_loss=0.01):
        super(AudioMasteringLoss, self).__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        self.alpha = 0.60  # weight for spectrogram L1 loss
        self.beta = 0.30   # weight for parameter MSE loss
        self.gamma = 0.10  # weight for perceptual loss
        
        self.scale_percept_loss = scale_percept_loss
        
        self.spec_loss_l1 = nn.L1Loss()
        self.param_loss = nn.MSELoss()
        
    def forward(self, output_spec, target_spec, predicted_params, target_params):
        epsilon = 1e-6
        output_spec = torch.clamp(output_spec, min=epsilon)
        target_spec = torch.clamp(target_spec, min=epsilon)
        
        spec_l1 = self.spec_loss_l1(output_spec, target_spec)
        param_loss_val = self.param_loss(predicted_params, target_params)
        
        try:
            mel_out = compute_mel_spectrogram(output_spec.detach().cpu().numpy()[0,0],
                                            sr=self.sr, n_mels=self.n_mels,
                                            n_fft=self.n_fft, hop_length=self.hop_length)
            mel_target = compute_mel_spectrogram(target_spec.detach().cpu().numpy()[0,0],
                                                sr=self.sr, n_mels=self.n_mels,
                                                n_fft=self.n_fft, hop_length=self.hop_length)
            mel_out = torch.tensor(mel_out, dtype=torch.float32, device=output_spec.device)
            mel_target = torch.tensor(mel_target, dtype=torch.float32, device=output_spec.device)
            perceptual_loss_val = self.spec_loss_l1(mel_out, mel_target) * self.scale_percept_loss
        except Exception as e:
            logger.warning(f"Error computing perceptual loss: {e}")
            perceptual_loss_val = torch.tensor(0.0, device=output_spec.device)
        
        total_loss = self.alpha * spec_l1 + self.beta * param_loss_val + self.gamma * perceptual_loss_val
        return total_loss, {
            'spec_loss_l1': spec_l1.item(),
            'parameter_loss': param_loss_val.item(),
            'perceptual_loss': perceptual_loss_val.item(),
            'total_loss': total_loss.item()
        }

# ----------------------------
# Loss Plotting Function
# ----------------------------
def save_loss_plot(loss_history, filename):
    """
    Plots and saves total and component losses per epoch.
    """
    epochs = range(1, len(loss_history['total']) + 1)
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2,2,1)
    plt.plot(epochs, loss_history['total'], marker='o', label='Train Total')
    plt.plot(epochs, loss_history['val'], marker='s', label='Val Total')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2,2,2)
    plt.plot(epochs, loss_history['spec_loss_l1'], marker='o', label='Train L1')
    plt.plot(epochs, loss_history['val_spec_loss_l1'], marker='s', label='Val L1')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.title('Spectrogram L1 Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2,2,3)
    plt.plot(epochs, loss_history['parameter_loss'], marker='o', label='Train Param MSE')
    plt.plot(epochs, loss_history['val_parameter_loss'], marker='s', label='Val Param MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Parameter Loss')
    plt.title('Parameter MSE Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2,2,4)
    plt.plot(epochs, loss_history['perceptual_loss'], marker='o', label='Train Perceptual')
    plt.plot(epochs, loss_history['val_perceptual_loss'], marker='s', label='Val Perceptual')
    plt.xlabel('Epoch')
    plt.ylabel('Perceptual Loss (Scaled)')
    plt.title('Perceptual Loss')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle('Loss Components per Epoch', fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(filename)
    plt.close()

# ----------------------------
# Training Epoch Function
# ----------------------------
def train_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs):
    """
    Trains model for one epoch, logs and returns average total and component losses.
    """
    model.train()
    total_loss = 0.0
    total_spec_loss = 0.0
    total_param_loss = 0.0
    total_percept_loss = 0.0
    num_batches = len(train_loader)
    log_interval = max(1, num_batches // 20)
    
    for batch_idx, (inputs, targets, gt_params) in enumerate(train_loader):
        global_batch = epoch * num_batches + batch_idx + 1
        inputs, targets = inputs.to(device), targets.to(device)
        gt_params = gt_params.to(device)
        optimizer.zero_grad()
        
        restored_spec, predicted_params_norm = model(inputs)
        min_T = min(restored_spec.shape[-1], targets.shape[-1])
        restored_spec = restored_spec[..., :min_T]
        targets = targets[..., :min_T]
        
        T = predicted_params_norm.shape[1]
        gt_params_expanded = gt_params.unsqueeze(1).expand(-1, T, -1)
        
        loss, loss_components = criterion(restored_spec, targets, predicted_params_norm, gt_params_expanded)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_spec_loss += loss_components['spec_loss_l1']
        total_param_loss += loss_components['parameter_loss']
        total_percept_loss += loss_components['perceptual_loss']
        
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{batch_idx+1}/{num_batches}] (Global Batch {global_batch}) - "
                f"Avg Loss: {avg_loss:.4f} | L1: {loss_components['spec_loss_l1']:.4f}, "
                f"Param: {loss_components['parameter_loss']:.4f}, Perc: {loss_components['perceptual_loss']:.4f}")
    
    avg_total = total_loss / num_batches
    avg_components = {
        'spec_loss_l1': total_spec_loss / num_batches,
        'parameter_loss': total_param_loss / num_batches,
        'perceptual_loss': total_percept_loss / num_batches
    }
    return avg_total, avg_components

# ----------------------------
# Main Training Loop
# ----------------------------
def train_model(model, train_loader, val_loader, device, num_epochs=100, learning_rate=5e-5,
                weight_decay=1e-5, start_epoch=0, checkpoint_data=None):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(project_root, "logs", f"training_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Remove deprecated verbose parameter from scheduler.
    criterion = AudioMasteringLoss(scale_percept_loss=0.01)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    loss_history = {
        'total': [], 'val': [],
        'spec_loss_l1': [], 'val_spec_loss_l1': [],
        'parameter_loss': [], 'val_parameter_loss': [],
        'perceptual_loss': [], 'val_perceptual_loss': []
    }
    
    if checkpoint_data is not None:
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        best_val_loss = checkpoint_data['val_loss']
        if 'loss_history' in checkpoint_data:
            loss_history = checkpoint_data['loss_history']
        print(f"\nCheckpoint loaded from epoch {checkpoint_data['epoch']}")
    else:
        best_val_loss = float('inf')
        print("\nStarting new training run")
    
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if start_epoch == 0:
        print("\nStarting training...")
        print(f"Epoch {start_epoch+1} start with LR {learning_rate:.2e}")
        print(f"Training on {len(train_loader.dataset)} samples; Validating on {len(val_loader.dataset)} samples")
    
    patience_counter = 0
    early_stopping_patience = 10
    
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_comp = train_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        loss_history['total'].append(train_loss)
        loss_history['spec_loss_l1'].append(train_comp['spec_loss_l1'])
        loss_history['parameter_loss'].append(train_comp['parameter_loss'])
        loss_history['perceptual_loss'].append(train_comp['perceptual_loss'])
        
        model.eval()
        val_loss = 0.0
        val_spec_loss = 0.0
        val_param_loss = 0.0
        val_percept_loss = 0.0
        with torch.no_grad():
            for inputs, targets, gt_params in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                gt_params = gt_params.to(device)
                restored_spec, predicted_params_norm = model(inputs)
                min_T = min(restored_spec.shape[-1], targets.shape[-1])
                restored_spec = restored_spec[..., :min_T]
                targets = targets[..., :min_T]
                T = predicted_params_norm.shape[1]
                gt_params_expanded = gt_params.unsqueeze(1).expand(-1, T, -1)
                loss, batch_losses = criterion(restored_spec, targets, predicted_params_norm, gt_params_expanded)
                val_loss += loss.item()
                val_spec_loss += batch_losses['spec_loss_l1']
                val_param_loss += batch_losses['parameter_loss']
                val_percept_loss += batch_losses['perceptual_loss']
        val_loss /= len(val_loader)
        loss_history['val'].append(val_loss)
        loss_history['val_spec_loss_l1'].append(val_spec_loss / len(val_loader))
        loss_history['val_parameter_loss'].append(val_param_loss / len(val_loader))
        loss_history['val_perceptual_loss'].append(val_percept_loss / len(val_loader))
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {current_lr:.2e}")
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'loss_history': loss_history
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, 'best_model.pt'))
            print("  Best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
        
        torch.save(checkpoint_data, os.path.join(checkpoint_dir, f'one_stage_deep_unet_epoch_{epoch+1}.pt'))
        loss_plot_path = os.path.join(log_dir, 'loss_plot.png')
        save_loss_plot(loss_history, loss_plot_path)
        
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

# ----------------------------
# Main Function
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description='Train or resume training of the audio mastering model')
    parser.add_argument('--resume', type=str, help='Path to checkpoint file to resume training from')
    args = parser.parse_args()
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_root)
    logger.info(f"Project root: {project_root}")
    
    num_epochs = 100
    batch_size = 2
    learning_rate = 5e-5
    weight_decay = 1e-5
    
    sample_rate = 44100
    hop_length = 512
    n_fft = 2048
    n_mels = 128
    
    dataset_dir = os.path.join(project_root, "experiments", "output_full", "output_audio")
    dataset_dir = os.path.abspath(dataset_dir)
    logger.info(f"Dataset directory: {dataset_dir}")
    
    dataset = PairedAudioDataset(
        audio_dir=dataset_dir,
        sr=sample_rate,
        transform=compute_mel_spectrogram,
        mode="spectrogram"
    )
    
    logger.info(f"Spectrogram mode, dataset length: {len(dataset)}")
    if len(dataset) > 0:
        sample_input, sample_target, sample_params = dataset[0]
        logger.info(f"Input spectrogram shape: {sample_input.shape}")
        logger.info(f"Target spectrogram shape: {sample_target.shape}")
        logger.info(f"Normalized parameter vector: {sample_params}")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_pad)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_pad)
    
    model = OneStageDeepUNet(sr=sample_rate, hop_length=hop_length, in_channels=1, out_channels=1,
                             base_features=64, blocks_per_level=3, lstm_hidden=32, num_layers=1, num_params=10)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Using device: {device}")
    
    start_epoch = 0
    checkpoint_data = None
    if args.resume:
        logger.info(f"Loading checkpoint: {args.resume}")
        checkpoint_path = args.resume
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(project_root, "checkpoints", checkpoint_path)
        if os.path.exists(checkpoint_path):
            checkpoint_data = torch.load(checkpoint_path, map_location=device)
            start_epoch = checkpoint_data['epoch'] + 1
            logger.info(f"Resuming from epoch {start_epoch}")
        else:
            logger.warning(f"Checkpoint file {checkpoint_path} not found. Starting from scratch.")
    
    logger.info("Starting training...")
    train_model(model, train_loader, val_loader, device,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                start_epoch=start_epoch,
                checkpoint_data=checkpoint_data)
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
