import torch
import torch.nn as nn
import torch.nn.functional as F

##############################
# Basic Building Blocks
##############################

class ResidualBlock(nn.Module):
    """A standard residual block with two convolutional layers and a skip connection."""
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
            # Adjust dimensions of skip connection
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
    """Attention module for feature gating in the decoder."""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        # F_g: gating signal (from decoder); F_l: encoder features; F_int: intermediate channel number
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
        # g: gating signal; x: encoder feature map
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

##############################
# Modified UNet Definition
##############################

class ModifiedUNet(nn.Module):
    """
    A UNet with residual blocks and attention in the decoder.
    It uses interpolation to force the output size to match the input.
    """
    def __init__(self, in_channels=1, out_channels=1, base_features=64):
        super(ModifiedUNet, self).__init__()
        # Encoder layers
        self.enc1 = ResidualBlock(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualBlock(base_features * 4, base_features * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(base_features * 8, base_features * 16)
        # Decoder layers with attention
        self.up4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=base_features * 8, F_l=base_features * 8, F_int=base_features * 4)
        self.dec4 = ResidualBlock(base_features * 16, base_features * 8)
        
        self.up3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=base_features * 4, F_l=base_features * 4, F_int=base_features * 2)
        self.dec3 = ResidualBlock(base_features * 8, base_features * 4)
        
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=base_features * 2, F_l=base_features * 2, F_int=base_features)
        self.dec2 = ResidualBlock(base_features * 4, base_features * 2)
        
        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=base_features, F_l=base_features, F_int=base_features // 2)
        self.dec1 = ResidualBlock(base_features * 2, base_features)
        
        self.conv_final = nn.Conv2d(base_features, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                      # size: same as x
        e2 = self.enc2(self.pool1(e1))           # down by 2
        e3 = self.enc3(self.pool2(e2))           # down by 4
        e4 = self.enc4(self.pool3(e3))           # down by 8
        b = self.bottleneck(self.pool4(e4))      # down by 16
        
        # Decoder with attention and skip connections:
        d4 = self.up4(b)
        # Force d4 size to be same as e4
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
        # To be safe, force the output size to match input using interpolation:
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return out

##############################
# LSTM for Parameter Prediction
##############################
class LSTMForecasting(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, num_params=10):
        super(LSTMForecasting, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Initialize weights
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
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch, time, hidden_size]
        # Use sigmoid to produce outputs in [0,1]
        params_norm = torch.sigmoid(self.fc(lstm_out))
        return params_norm

##############################
# Two-Stage UNet (Full Model)
##############################
class TwoStageUNet(nn.Module):
    """
    TwoStageUNet processes the spectrogram in two stages:
      Stage 1: A ModifiedUNet to get an initial reconstruction (for diagnosis)
      Stage 2: A second ModifiedUNet that further refines the output.
    Then, average energy of the refined output feeds into an LSTM to predict effect parameters.
    Returns:
      - refined_output: final refined spectrogram (should match input size)
      - stage1_output: intermediate output from the first UNet (for diagnostic visualization)
      - predicted_params_norm: LSTM predicted normalized parameter tensor with shape [batch, T, 10]
    """
    def __init__(self, sr, hop_length, in_channels=1, out_channels=1, base_features=64, lstm_hidden=32, num_layers=1, num_params=10):
        super(TwoStageUNet, self).__init__()
        self.unet1 = ModifiedUNet(in_channels, out_channels, base_features)
        self.unet2 = ModifiedUNet(in_channels, out_channels, base_features)
        self.lstm = LSTMForecasting(input_size=1, hidden_size=lstm_hidden, num_layers=num_layers, num_params=num_params)
        self.sr = sr
        self.hop_length = hop_length
        
    def forward(self, x):
        # Stage 1: initial UNet
        stage1_out = self.unet1(x)
        # Force stage1 output to match input size if necessary
        if stage1_out.shape[-2:] != x.shape[-2:]:
            stage1_out = F.interpolate(stage1_out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        # Stage 2: refine using second UNet
        refined_out = self.unet2(stage1_out)
        if refined_out.shape[-2:] != x.shape[-2:]:
            refined_out = F.interpolate(refined_out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        # Extract average energy per time frame for LSTM input
        avg_energy = refined_out.mean(dim=2)  # [batch, 1, time]
        avg_energy = avg_energy.transpose(1, 2)  # [batch, time, 1]
        predicted_params_norm = self.lstm(avg_energy)  # [batch, time, num_params]
        return refined_out, stage1_out, predicted_params_norm

if __name__ == '__main__':
    # Test run of the entire model:
    x = torch.randn(1, 1, 128, 2582)
    model = TwoStageUNet(sr=44100, hop_length=512)
    refined_out, stage1_out, params = model(x)
    print("Refined output shape:", refined_out.shape)
    print("Stage1 output shape:", stage1_out.shape)
    print("Predicted normalized parameters shape:", params.shape)
