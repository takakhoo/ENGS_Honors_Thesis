#!/usr/bin/env python3
"""
models.py

This module contains the neural network architectures for our audio mastering system.

Features:
- Basic building blocks (ResidualBlock, AttentionBlock)
- Deep UNet for spectrogram processing
- LSTM for forecasting processing parameters over time
- A unified OneStageDeepUNet that integrates both branches and applies an inverse correction:
    • Inverts degradation effects such as gain, EQ, echo, and reverb based on the predicted parameters.
    • Combines an explicit inverse correction and the UNet residual correction to restore the target spectrogram.

Inverse corrections:
1. Gain and EQ Inversion:
- Gain: unnormalized predicted gain is used to compute inv_gain = 1/(1 + gain_db).
- EQ: predicted EQ center, Q, and EQ gain form a frequency-dependent Gaussian‐like mask, then inverted.
2. Echo Inversion:
- Uses the average predicted echo delay and attenuation to subtract a delayed copy from the spectrogram.
3. Reverb Inversion:
- Computes a scalar inverse factor from the predicted reverb decay.

This integrated approach lets the network share restoration work between the inverse correction and the UNet’s learned residual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6

##############################
# Basic Building Blocks
##############################

class ResidualBlock(nn.Module):
    """
    A residual block with skip connections to ease gradient flow.
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
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
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
    An attention block for focusing on salient features in skip connections.
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
    Creates a sequential stack of residual blocks.
    """
    layers = []
    layers.append(ResidualBlock(in_channels, out_channels, stride))
    for _ in range(1, num_blocks):
        layers.append(ResidualBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)

##############################
# Deep UNet Architecture
##############################

class DeepUNet(nn.Module):
    """
    Deep UNet with an encoder-decoder structure, residual blocks and attention gates.
    A bypass connection is added so that the network learns only the residual correction.
    """
    def __init__(self, in_channels=1, out_channels=1, base_features=64, blocks_per_level=3):
        super(DeepUNet, self).__init__()
        # Encoder
        self.enc1 = make_residual_stack(blocks_per_level, in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = make_residual_stack(blocks_per_level, base_features, base_features*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = make_residual_stack(blocks_per_level, base_features*2, base_features*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = make_residual_stack(blocks_per_level, base_features*4, base_features*8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = make_residual_stack(blocks_per_level, base_features*8, base_features*16)
        # Decoder
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
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
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
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        # Bypass: add the input so that the network learns only the residual correction.
        out = out + x
        return out

##############################
# LSTM for Parameter Prediction
##############################

class LSTMForecasting(nn.Module):
    """
    LSTM network to forecast processing parameters for each time frame.
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
        # x shape: [B, T, 1]
        lstm_out, _ = self.lstm(x)
        params_norm = torch.sigmoid(self.fc(lstm_out))
        return params_norm

##############################
# One Stage Deep UNet with Inverse Correction
##############################

class OneStageDeepUNet(nn.Module):
    """
    Combined model which:
    1. Uses a deep UNet to compute a residual correction.
    2. Extracts average energy per time frame to feed an LSTM that predicts processing parameters.
    3. Applies inverse corrections to the input spectrogram by inverting:
        - Gain and EQ (via frequency-dependent masks)
        - Echo (by subtracting a delayed copy)
        - Reverb (using a scalar inverse factor)
    4. Combines the inverse-corrected input with the UNet correction to produce the restored spectrogram.
    """
    def __init__(self, sr, hop_length, in_channels=1, out_channels=1,
                base_features=64, blocks_per_level=3, lstm_hidden=32,
                num_layers=1, num_params=10):
        super(OneStageDeepUNet, self).__init__()
        self.deep_unet = DeepUNet(in_channels, out_channels, base_features, blocks_per_level)
        self.lstm = LSTMForecasting(input_size=1, hidden_size=lstm_hidden, num_layers=num_layers, num_params=num_params)
        self.sr = sr
        self.hop_length = hop_length
    
    def _apply_inverse_correction(self, x, predicted_params_norm):
        """
        Applies an inverse correction to the input spectrogram x using the forecasted parameters.

        For echo inversion, instead of modifying x in place,
        we accumulate echo corrections in a new tensor and then subtract them from x.
        This avoids in‑place modifications that interfere with gradient computation.

        Args:
            x: Input spectrogram tensor of shape [B, 1, F, T].
            predicted_params_norm: Predicted normalized parameters from the LSTM, shape [B, T, num_params].

        Returns:
            Inverse-corrected spectrogram of shape [B, 1, F, T].
        """
        eps = 1e-6
        B, C, F, T = x.shape

        # Unnormalize parameters for gain and EQ inversion (parameters 0-3)
        gain_db = predicted_params_norm[:, :, 0] * 2 - 1         # approx in [-1, 1]
        eq_center = predicted_params_norm[:, :, 1] * (self.sr / 2)  # in Hz
        eq_Q = predicted_params_norm[:, :, 2] * 9.9 + 0.1           # Q-factor
        eq_gain = predicted_params_norm[:, :, 3] * 20 - 10          # in dB

        # Compute inverse gain mask
        inv_gain = 1.0 / (1.0 + gain_db + eps)                     # shape [B, T]
        inv_gain = inv_gain.unsqueeze(1).unsqueeze(2)              # reshape to [B, 1, 1, T]

        # Prepare frequency grid for EQ inversion
        freqs = torch.linspace(0, self.sr / 2, steps=F, device=x.device).reshape(1, 1, F, 1)
        eq_center = eq_center.unsqueeze(1).unsqueeze(2)            # shape [B, 1, 1, T]
        eq_Q = eq_Q.unsqueeze(1).unsqueeze(2)                      # shape [B, 1, 1, T]
        eq_gain = eq_gain.unsqueeze(1).unsqueeze(2)                # shape [B, 1, 1, T]

        bandwidth = eq_center / (eq_Q + eps)
        response = 1.0 + eq_gain * torch.exp(-((freqs - eq_center) ** 2) / (2 * (bandwidth ** 2) + eps))
        inv_eq = 1.0 / (response + eps)                            # shape [B, 1, F, T]

        # Combine inverse gain and inverse EQ masks
        inverse_mask = inv_gain * inv_eq

        # Start with a clone of x for corrections
        x_corr = x.clone()

        # Echo inversion:
        # Unnormalize echo parameters: here, parameters 8 and 9.
        echo_delay_sec = predicted_params_norm[:, :, 8] * 100      # in seconds
        echo_atten = predicted_params_norm[:, :, 9]                # as is
        # Compute average echo parameters per sample
        echo_delay_avg = torch.mean(echo_delay_sec, dim=1)         # [B]
        echo_atten_avg = torch.mean(echo_atten, dim=1)               # [B]
        # Convert echo delay (seconds) to delay in frames
        delay_frames = (echo_delay_avg * self.sr / self.hop_length).round().long()  # [B]

        # Instead of in-place subtraction, accumulate echo corrections in a separate tensor.
        echo_correction = torch.zeros_like(x_corr)
        for b in range(B):
            d_frames = delay_frames[b].item()
            if d_frames < 1 or d_frames >= T:
                continue
            for t in range(d_frames, T):
                echo_correction[b, :, :, t] = echo_atten_avg[b] * x_corr[b, :, :, t - d_frames]
        # Subtract the accumulated echo correction out-of-place.
        x_corr = x_corr - echo_correction

        # Reverb inversion:
        # Unnormalize the reverb decay (parameter 7)
        reverb_decay = predicted_params_norm[:, :, 7] * 9.9 + 0.1
        reverb_decay_avg = torch.mean(reverb_decay, dim=1)         # [B]
        inv_reverb = 1.0 / (1.0 + reverb_decay_avg + eps)            # [B]
        inv_reverb = inv_reverb.view(B, 1, 1, 1)                     # reshape for broadcasting
        x_corr = x_corr * inv_reverb

        # Finally, apply the inverse gain/EQ mask to x_corr to get the corrected spectrogram.
        corrected = x_corr * inverse_mask
        return corrected

    
    def forward(self, x):
        """
        Forward pass:
        1. Compute a UNet residual correction.
        2. Extract average energy over frequency for each time frame.
        3. Use the LSTM to predict processing parameters.
        4. Compute an inverse correction on the input using these predicted parameters.
        5. Combine the inverse-corrected input with the UNet correction.
        """
        # Obtain UNet residual correction.
        unet_correction = self.deep_unet(x)
        if unet_correction.shape[-2:] != x.shape[-2:]:
            unet_correction = F.interpolate(unet_correction, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # Extract average energy per time frame: [B, T, 1]
        avg_energy = unet_correction.mean(dim=2)   # [B, 1, T]
        avg_energy = avg_energy.transpose(1, 2)      # [B, T, 1]
        
        # LSTM predicts per-frame parameters.
        predicted_params_norm = self.lstm(avg_energy)  # [B, T, num_params]
        
        # Apply inverse correction to the input spectrogram.
        inverse_spec = self._apply_inverse_correction(x, predicted_params_norm)
        
        # Final output is the sum of inverse-corrected input and UNet residual correction.
        final_output = inverse_spec + unet_correction
        
        return final_output, predicted_params_norm

##############################
# End of Module
##############################

if __name__ == '__main__':
    # Quick test: simulate a random input spectrogram [batch, 1, freq, time]
    x = torch.randn(1, 1, 128, 2582)
    model = OneStageDeepUNet(sr=44100, hop_length=512, blocks_per_level=3)
    restored_spec, params = model(x)
    print("Restored spectrogram shape:", restored_spec.shape)
    print("Predicted normalized parameters shape:", params.shape)
