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
from models import TwoStageUNet

# ----------------------------
# Collate Function (no changes needed)
# ----------------------------
def collate_pad(batch):
    # Each sample in the batch is (input_spec, target_spec, param_vector)
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
    params = torch.stack(param_list, dim=0)  # shape: [batch, 10]
    return inputs, targets, params

# ----------------------------
# Loss Module
# ----------------------------
class AudioMasteringLoss(nn.Module):
    """
    Loss function with three components:
      - Spectrogram Loss: Emphasize L1 (MAE) very heavily to preserve fine details.
      - Parameter Loss: Critical for our work — weighted high.
      - Perceptual Loss: Computed on mel spectrograms.
    The weights have been revised so that:
      spectrogram_weight = 0.3, parameter_weight = 0.5, perceptual_weight = 0.2 (sum=1.0).
    """
    def __init__(self, sr=44100, n_fft=2048, hop_length=512, n_mels=128):
        super(AudioMasteringLoss, self).__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.spectrogram_weight = 0.3
        self.parameter_weight = 0.5
        self.perceptual_weight = 0.2
        
        self.spec_loss_l1 = nn.L1Loss()
        self.spec_loss_l2 = nn.MSELoss()
        self.param_loss = nn.MSELoss()
    
    def forward(self, output_spec, target_spec, predicted_params, target_params):
        epsilon = 1e-6
        
        # Clip spectrograms to avoid log(0)
        output_spec = torch.clamp(output_spec, min=epsilon)
        target_spec = torch.clamp(target_spec, min=epsilon)
        
        # Compute L1 and L2 spectrogram losses.
        spec_l1 = self.spec_loss_l1(output_spec, target_spec)
        spec_l2 = self.spec_loss_l2(output_spec, target_spec)
        
        # Parameter loss (the ground-truth parameters are replicated for each time step)
        param_loss_val = self.param_loss(predicted_params, target_params)
        
        # Perceptual loss: compute mel spectrograms.
        # Use detach() to avoid grad issues.
        try:
            mel_out = compute_mel_spectrogram(output_spec.detach().cpu().numpy()[0,0], sr=self.sr,
                                              n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)
            mel_target = compute_mel_spectrogram(target_spec.detach().cpu().numpy()[0,0], sr=self.sr,
                                                 n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)
            # Convert back to tensors for loss computation.
            mel_out = torch.tensor(mel_out, dtype=torch.float32, device=output_spec.device)
            mel_target = torch.tensor(mel_target, dtype=torch.float32, device=output_spec.device)
            perceptual_loss_val = self.spec_loss_l1(mel_out, mel_target)
        except Exception as e:
            print(f"Error computing perceptual loss: {e}")
            perceptual_loss_val = torch.tensor(0.0, device=output_spec.device)
        
        total_loss = (self.spectrogram_weight * spec_l2 +
                      self.parameter_weight * param_loss_val +
                      self.perceptual_weight * perceptual_loss_val)
        
        return total_loss, {
            'spec_loss_l1': spec_l1.item(),
            'spec_loss_l2': spec_l2.item(),
            'parameter_loss': param_loss_val.item(),
            'perceptual_loss': perceptual_loss_val.item(),
            'total_loss': total_loss.item()
        }

# ----------------------------
# Loss Plotting Helper
# ----------------------------
def save_loss_plot(loss_history, filename):
    epochs = range(1, len(loss_history['total']) + 1)
    plt.figure(figsize=(14,10))
    plt.subplot(2,2,1)
    plt.plot(epochs, loss_history['total'], label='Total Loss', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Total Loss'); plt.title('Total Loss'); plt.legend(); plt.grid(True)
    plt.subplot(2,2,2)
    plt.plot(epochs, loss_history['spec_loss_l2'], label='Spec L2 Loss', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('L2 Loss'); plt.title('Spectrogram L2 Loss'); plt.legend(); plt.grid(True)
    plt.subplot(2,2,3)
    plt.plot(epochs, loss_history['parameter_loss'], label='Parameter Loss', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Parameter Loss'); plt.title('Parameter Loss'); plt.legend(); plt.grid(True)
    plt.subplot(2,2,4)
    plt.plot(epochs, loss_history['perceptual_loss'], label='Perceptual Loss', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Perceptual Loss'); plt.title('Perceptual Loss'); plt.legend(); plt.grid(True)
    plt.suptitle('Training Losses per Epoch', fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(filename)
    plt.close()

# ----------------------------
# Training Epoch Function
# ----------------------------
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    for batch_idx, (inputs, targets, gt_params) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        gt_params = gt_params.to(device)  # shape: [batch, 10]
        optimizer.zero_grad()
        # Forward pass: our model returns refined output, stage1 output, and predicted parameters.
        refined_out, stage1_out, predicted_params_norm = model(inputs)
        # Force refined_out to match targets (using cropping if necessary)
        min_T = min(refined_out.shape[-1], targets.shape[-1])
        refined_out = refined_out[..., :min_T]
        targets = targets[..., :min_T]
        
        # Expand ground truth parameters (normalized) to match time steps
        T = predicted_params_norm.shape[1]
        gt_params_expanded = gt_params.unsqueeze(1).expand(-1, T, -1)
        
        loss, loss_components = criterion(refined_out, targets, predicted_params_norm, gt_params_expanded)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (batch_idx + 1) % max(1, num_batches // 10) == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Batch {batch_idx+1}/{num_batches} - Avg Loss: {avg_loss:.4f} | " +
                  f"Spec L1: {loss_components['spec_loss_l1']:.4f}, " +
                  f"Spec L2: {loss_components['spec_loss_l2']:.4f}, " +
                  f"Param: {loss_components['parameter_loss']:.4f}, " +
                  f"Perc: {loss_components['perceptual_loss']:.4f}")
    return total_loss / num_batches

# ----------------------------
# Training Loop
# ----------------------------
def train_model(model, train_loader, val_loader, device, num_epochs=100,
                learning_rate=5e-5, weight_decay=1e-5):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "logs", f"training_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    criterion = AudioMasteringLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    best_val_loss = float('inf')
    checkpoint_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    loss_history = {
        'total': [],
        'spec_loss_l1': [],
        'spec_loss_l2': [],
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
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        loss_history['total'].append(train_loss)
        
        model.eval()
        val_loss = 0.0
        val_comp = {'spec_loss_l1': 0.0, 'spec_loss_l2': 0.0, 'parameter_loss': 0.0, 'perceptual_loss': 0.0}
        with torch.no_grad():
            for inputs, targets, gt_params in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                gt_params = gt_params.to(device)
                refined_out, stage1_out, predicted_params_norm = model(inputs)
                min_T = min(refined_out.shape[-1], targets.shape[-1])
                refined_out = refined_out[..., :min_T]
                targets = targets[..., :min_T]
                T = predicted_params_norm.shape[1]
                gt_params_expanded = gt_params.unsqueeze(1).expand(-1, T, -1)
                loss, loss_components = criterion(refined_out, targets, predicted_params_norm, gt_params_expanded)
                val_loss += loss.item()
                for key in val_comp:
                    val_comp[key] += loss_components.get(key, 0)
        val_loss /= len(val_loader)
        loss_history['val'].append(val_loss)
        for key in ['spec_loss_l1', 'spec_loss_l2', 'parameter_loss', 'perceptual_loss']:
            loss_history[key].append(val_comp[key] / len(val_loader))
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"    (Spec L1: {loss_history['spec_loss_l1'][-1]:.4f}, Spec L2: {loss_history['spec_loss_l2'][-1]:.4f}, " +
              f"Param: {loss_history['parameter_loss'][-1]:.4f}, Perc: {loss_history['perceptual_loss'][-1]:.4f})")
        print(f"  Learning Rate: {current_lr:.2e}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss},
                       os.path.join(checkpoint_dir, 'best_model.pt'))
            print("  New best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss},
                   os.path.join(checkpoint_dir, f'cascaded_model_epoch_{epoch+1}.pt'))
        loss_plot_path = os.path.join(log_dir, 'loss_plot.png')
        save_loss_plot(loss_history, loss_plot_path)
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    return train_losses, val_losses

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_root)
    print("Project root:", project_root)
    
    num_epochs = 100  # Increase epochs for overnight training.
    batch_size = 2
    learning_rate = 5e-5
    weight_decay = 1e-5
    sample_rate = 44100
    hop_length = 512
    n_fft = 2048
    n_mels = 128
    ir_length = 20
    
    dataset_dir = os.path.join(project_root, "experiments", "output_full", "output_audio")
    dataset_dir = os.path.abspath(dataset_dir)
    print("Dataset directory:", dataset_dir)
    
    dataset = PairedAudioDataset(
        audio_dir=dataset_dir,
        sr=sample_rate,
        transform=compute_mel_spectrogram,
        mode="spectrogram"
    )
    print(f"Spectrogram mode, dataset length: {len(dataset)}")
    if len(dataset) > 0:
        sample_input, sample_target, sample_params = dataset[0]
        print("Input spectrogram shape:", sample_input.shape)
        print("Target spectrogram shape:", sample_target.shape)
        print("Normalized parameter vector:", sample_params)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_pad)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_pad)
    
    model = TwoStageUNet(sr=sample_rate, hop_length=hop_length, in_channels=1, out_channels=1, base_features=64,
                          lstm_hidden=32, num_layers=1, num_params=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Using device:", device)
    print("Starting training...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, device,
                                           num_epochs=num_epochs, learning_rate=learning_rate,
                                           weight_decay=weight_decay)
    print("Training complete.")

if __name__ == "__main__":
    main()
