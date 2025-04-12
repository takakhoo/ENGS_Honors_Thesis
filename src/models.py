# models.py
"""
This module contains the neural network architectures for our audio mastering system.
It includes:
- Basic building blocks (ResidualBlock, AttentionBlock)
- Deep UNet architecture for spectrogram processing
- LSTM module for parameter prediction
- Combined model that integrates both components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

##############################
# Basic Building Blocks
##############################
class ResidualBlock(nn.Module):
    """
    A residual block with skip connections to help with gradient flow and feature reuse.
    Each block contains:
    - Two convolutional layers with batch normalization
    - ReLU activation
    - Optional shortcut connection for dimension matching
    
    The skip connection helps preserve important features and makes training deeper networks easier.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                            kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    """
    Attention mechanism for the UNet decoder that helps focus on relevant features.
    Implements a gating mechanism that:
    1. Processes both the upsampled feature map and skip connection
    2. Computes attention weights using a sigmoid activation
    3. Applies the weights to the skip connection features
    
    This helps the model focus on important spectral features during reconstruction.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

def make_residual_stack(num_blocks, in_channels, out_channels, stride=1):
    """
    Creates a stack of residual blocks for deeper feature extraction.
    
    Args:
        num_blocks: Number of residual blocks in the stack
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        stride: Stride for the first block (others use stride=1)
    
    Returns:
        Sequential container of residual blocks
    """
    layers = []
    # First block takes care of potential dimension change
    layers.append(ResidualBlock(in_channels, out_channels, stride))
    for _ in range(1, num_blocks):
        layers.append(ResidualBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)

##############################
# Deep UNet Architecture
##############################
class DeepUNet(nn.Module):
    """
    A deep UNet architecture with residual blocks and attention mechanisms.
    Key features:
    - 4-level encoder-decoder structure
    - Multiple residual blocks per level
    - Attention gates in decoder
    - Skip connections with feature concatenation
    
    The architecture is designed to:
    1. Extract hierarchical features in the encoder
    2. Preserve fine details through skip connections
    3. Focus on relevant features using attention
    4. Reconstruct the spectrogram in the decoder
    """
    def __init__(self, in_channels=1, out_channels=1, base_features=64, blocks_per_level=5):
        super(DeepUNet, self).__init__()
        # Encoder: 4 levels
        self.enc1 = make_residual_stack(blocks_per_level, in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = make_residual_stack(blocks_per_level, base_features, base_features*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = make_residual_stack(blocks_per_level, base_features*2, base_features*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = make_residual_stack(blocks_per_level, base_features*4, base_features*8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = make_residual_stack(blocks_per_level, base_features*8, base_features*16)
        
        # Decoder: mirror encoder
        self.up4 = nn.ConvTranspose2d(base_features*16, base_features*8, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=base_features*8, F_l=base_features*8, F_int=base_features*4)
        self.dec4 = make_residual_stack(blocks_per_level, base_features*16, base_features*8)
        
        self.up3 = nn.ConvTranspose2d(base_features*8, base_features*4, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=base_features*4, F_l=base_features*4, F_int=base_features*2)
        self.dec3 = make_residual_stack(blocks_per_level, base_features*8, base_features*4)
        
        self.up2 = nn.ConvTranspose2d(base_features*4, base_features*2, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=base_features*2, F_l=base_features*2, F_int=base_features)
        self.dec2 = make_residual_stack(blocks_per_level, base_features*4, base_features*2)
        
        self.up1 = nn.ConvTranspose2d(base_features*2, base_features, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=base_features, F_l=base_features, F_int=base_features//2)
        self.dec1 = make_residual_stack(blocks_per_level, base_features*2, base_features)
        
        self.conv_final = nn.Conv2d(base_features, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                 # [B, base_features, H, W]
        e2 = self.enc2(self.pool1(e1))      # [B, base_features*2, H/2, W/2]
        e3 = self.enc3(self.pool2(e2))      # [B, base_features*4, H/4, W/4]
        e4 = self.enc4(self.pool3(e3))      # [B, base_features*8, H/8, W/8]
        b = self.bottleneck(self.pool4(e4)) # [B, base_features*16, H/16, W/16]
        
        # Decoder with attention and skip connections
        d4 = self.up4(b)
        if d4.shape[-2:] != e4.shape[-2:]:
            d4 = F.interpolate(d4, size=e4.shape[-2:], mode='bilinear', align_corners=False)
        a4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, a4], dim=1))
        
        d3 = self.up3(d4)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(d3, size=e3.shape[-2:], mode='bilinear', align_corners=False)
        a3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, a3], dim=1))
        
        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        a2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, a2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        a1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, a1], dim=1))
        
        out = self.conv_final(d1)
        # Force output size to match input if needed
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return out

##############################
# LSTM for Parameter Prediction
##############################
class LSTMForecasting(nn.Module):
    """
    LSTM network for predicting audio processing parameters over time.
    Features:
    - Single or multi-layer LSTM
    - Xavier/orthogonal initialization
    - Sigmoid output for normalized parameters
    
    The LSTM processes the temporal evolution of spectral features
    to predict appropriate processing parameters for each time frame.
    """
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, num_params=10):
        super(LSTMForecasting, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        self.fc = nn.Linear(hidden_size, num_params)
    
    def forward(self, x):
        # x shape: [batch, time, 1]
        lstm_out, _ = self.lstm(x)
        params_norm = torch.sigmoid(self.fc(lstm_out))
        return params_norm

##############################
# One Stage Deep UNet with LSTM Forecasting
##############################
class OneStageDeepUNet(nn.Module):
    """
    The complete model that combines spectrogram processing and parameter prediction.
    
    Architecture:
    1. Deep UNet processes the input spectrogram
    2. Average energy is extracted along frequency axis
    3. LSTM predicts processing parameters for each time frame
    
    This unified approach allows the model to:
    - Process the spectrogram while preserving details
    - Predict appropriate processing parameters
    - Maintain temporal consistency in parameter predictions
    """
    def __init__(self, sr, hop_length, in_channels=1, out_channels=1, base_features=64,
                blocks_per_level=5, lstm_hidden=32, num_layers=1, num_params=10):
        super(OneStageDeepUNet, self).__init__()
        self.deep_unet = DeepUNet(in_channels, out_channels, base_features, blocks_per_level)
        self.lstm = LSTMForecasting(input_size=1, hidden_size=lstm_hidden, num_layers=num_layers, num_params=num_params)
        self.sr = sr
        self.hop_length = hop_length
        
    def forward(self, x):
        # Pass input through the deep UNet
        unet_out = self.deep_unet(x)
        if unet_out.shape[-2:] != x.shape[-2:]:
            unet_out = F.interpolate(unet_out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        # Extract average energy along the frequency axis for each time frame
        avg_energy = unet_out.mean(dim=2)  # shape: [batch, 1, time]
        avg_energy = avg_energy.transpose(1, 2)  # shape: [batch, time, 1]
        predicted_params_norm = self.lstm(avg_energy)  # shape: [batch, time, num_params]
        return unet_out, predicted_params_norm

if __name__ == '__main__':
    # For testing: simulate a random input spectrogram of size [batch, channel, 128, 2582]
    x = torch.randn(1, 1, 128, 2582)
    model = OneStageDeepUNet(sr=44100, hop_length=512, blocks_per_level=5)
    unet_output, params = model(x)
    print("Deep UNet output shape:", unet_output.shape)
    print("Predicted normalized parameters shape:", params.shape)
