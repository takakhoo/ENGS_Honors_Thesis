import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

""" 
This is going to implement the cascaded architecture NN, 2 separate stages
Stage 1: U Net module to restore degraded spectrogram
Stage 2: LSTM module to predict a vector of effect paramters for each time frame

Current Assumptions: 
For each time frame, forecasting module outputs 10 parameters
-Gain adjustment (1 scalar)
-EQ adjustment (3 scalars): center frequency, Q factor, and EQ gain
-Compression parameters (3 scalars): threshold, ratio, makeup gain
-Reverb Parameter (1 scalar): decay
-Echo Parameter (2 scalars): delay and attenuation
"""

#Stage 1: U Net for Spectrogram Restoration
"""
Learns to map a degraded spectro to a restored spectrogram
Encoder Decoder w/ skip connections, preserving fine spectral details during reconstruction
"""
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=64):
        super(UNet, self).__init__()
        features = init_features
        
        #Encoder blocks where each block applies 2 conv layers
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = UNet._block(in_channels, features, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = UNet._block(in_channels, features, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = UNet._block(in_channels, features, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        #Bottleneck captures depeest representation
        self.bottleneck = UNet._block(features*8, features*16, name="bottleneck")
        
        #Decoder Blocks: upconv upsamples & concatenates w/ corr encoder
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")
        
        #Final 1x1 convolution to map to desired ouput channels
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)
    
    def forward(self, x):
        
        #Encoder path - comp intermediate representations
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        #Decoder pathway with skip connection for fine details preserv
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        #2 Successive conv layers 3x3 w/ BatchNorm and ReLU
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
    
#Stage 2: LSTM FOrecasting for MultiPara Prediction
"""
LSTM Forecasts effect paramters from a time series of energy
Predict a vector of 10 paramters per time frame:
[gain, EQ_center, EQ_Q, EQ_gain, comp_threshold, comp_ratio, comp_makeup, reverb_decay, echo_delay, echo_attenuation]
"""

class LSTMForecasting(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, num_params=10):
        """
        Args:
            input_size (int): Dimension of input at each time step (e.g., 1 if using a single RMS value).
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of LSTM layers.
            num_params (int): Number of effect parameters to predict per time frame.
        """
        super(LSTMForecasting, self).__init()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_params) #Predict a vector of effect params
        
    def forward(self, x):
        #X [batch, time, input_size]
        lstm_out, _ = self.lstm(x)
        #Each time step gets a prediction vector w length num_params
        params = self.fc(lstm_out) #[batch, time, num_params]
        return params
    
#Differentiable Effect Modules to simulate traditional signal processing effects in a differentiable way

#1. Diffferentiable EQ
#Approximate a parametric EQ filter using Gaussian-esque response
#EQ_response(f, t) = 1 + EQ_gain(t) * exp( -((f - f_c(t))^2) / (2*(f_c(t)/Q(t))^2) )

def differentiable_eq(spectrogram, eq_params, freqs):
    """
    Apply a differentiable parametric EQ.
    
    Args:
        spectrogram: Tensor [batch, 1, freq, time].
        eq_params:Tensor [batch, time, 3] containing [center_frequency, Q, EQ_gain].
                These values should be normalized appropriately.
        freqs: 1D tensor of frequency bin centers.
    
    Returns:
        Adjusted spectrogram.
    """
    batch, _, num_freq, num_time = spectrogram.shape
    #Expanding freq axis for broadcasting [1,1,num_freq,1]
    freqs = freqs.view(1,1,num_freq,1)
    
    # Reshape eq_params: current shape [batch, time, 3] → [batch, 3, time] for easier handling.
    eq_params = eq_params.transpose(1, 2)  #Now shape: [batch, 3, time]
    center_freq = eq_params[:, 0, :].unsqueeze(1).unsqueeze(2)  #[batch, 1, 1, time]
    Q = eq_params[:, 1, :].unsqueeze(1).unsqueeze(2) #[batch, 1, 1, time]
    eq_gain = eq_params[:, 2, :].unsqueeze(1).unsqueeze(2) #[batch, 1, 1, time]
    
    #Bandwidth estimation to avoid division by zero with epsilon.
    epsilon = 1e-6
    bandwidth = center_freq / (Q + epsilon)
    # Gaussian-like EQ response for each frequency bin and time frame.
    response = 1 + eq_gain * torch.exp(-((freqs - center_freq) ** 2) / (2 * (bandwidth ** 2) + epsilon))
    
    return spectrogram * response

#2 Differentiable Compression
#Based on compression law compressed_level = threshold + (L - threshold)/ratio
#Apply soft thresholding operation for differentiability

def differentiable_compression(spectrogram, comp_params):
    """
    Apply a differentiable compression effect.
    
    Args:
        spectrogram: Tensor [batch, 1, freq, time].
        comp_params: Tensor [batch, time, 3] containing [threshold, ratio, makeup_gain] (in dB for threshold and makeup).
    
    Returns:
        Compressed spectrogram.
    """
    batch, _, num_freq, num_time = spectrogram.shape
    
    #Compute mean level over freq (in dB) per time frame
    spec_mean = spectrogram.mea(dim=2, keepdim=True) #[batch,1,1,time]
    spec_db = 20 * torch.log10(spec_mean + 1e-6)
    
    #Reshape compression parameters: [batch, time, 3] -> [batch, 3, time]
    comp_params = comp_params.transpose(1, 2)
    threshold = comp_params[:, 0, :].unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, time]
    ratio = comp_params[:, 1, :].unsqueeze(1).unsqueeze(1)      # [batch, 1, 1, time]
    makeup_gain = comp_params[:, 2, :].unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, time]
    
    #Compute the difference from the threshold.
    diff = spec_db - threshold
    #A smooth transition (using sigmoid) to simulate soft-knee compression.
    smooth_factor = torch.sigmoid(diff)
    gain_reduction = smooth_factor * (1 - 1/ratio) * diff
    new_level_db = spec_db - gain_reduction
    #Calculate the target gain to be applied.
    target_gain = 10 ** ((new_level_db - spec_db + makeup_gain) / 20.0)
    
    return spectrogram * target_gain

#3. Differentiable reverb:
#Simulate by convolving along the time axis with exponentially decaying impulse resp
def differentiable_reverb(spectrogram, reverb_decay, sr, hop_length, ir_length):
    """
    Apply differentiable reverb by convolving along the time axis.
    
    Args:
        spectrogram: Tensor [batch, 1, freq, time].
        reverb_decay: Scalar or tensor for decay rate.
        sr: Sampling rate.
        hop_length: Hop length of the STFT.
        ir_length: Number of time frames for the impulse response.
    
    Returns:
        Reverberated spectrogram.
    """
    #Creating time vect (s) for impulse resp
    t = torch.linspace(0, (ir_length-1)*hop_length/sr, steps=ir_length, device=spectrogram.device)
    ir = torch.exp(-reverb_decay * t)
    #Normalize the impulse response to preserve energy.
    ir = ir / (ir.sum() + 1e-6)
    #Reshape IR to [1, 1, 1, ir_length] to use as a convolution kernel.
    ir = ir.view(1, 1, 1, ir_length)
    #Pad the spectrogram along the time axis (simulate causal reverb).
    pad = (ir_length - 1, 0)
    spec_padded = F.pad(spectrogram, pad)
    #Convolve along the time axis.
    spec_reverb = F.conv2d(spec_padded, ir)
    return spec_reverb

#4: Differentiable Echo
#Simulate by delaying signal along time axis and adding attenuated copy

def differentiable_echo(spectrogram, echo_delay, echo_attenuation):
    """
    Apply a differentiable echo effect.
    
    Args:
        spectrogram: Tensor [batch, 1, freq, time].
        echo_delay: Delay in time frames.
        echo_attenuation: Attenuation factor (between 0 and 1) for the echoed signal.
    
    Returns:
        Spectrogram with echo applied.
    """
    batch, channels, freq, time = spectrogram.shape
    echo = torch.zeros_like(spectrogram)
    if echo_delay >= time:
        return spectrogram #No echo if delay is longer than time dimension
    echo[:, :, :, echo_delay:] = spectrogram[:, :, :, :-echo_delay] * echo_attenuation
    return spectrogram + echo

#########################################################################################
#Cascaded Mastering Model for all parameters
"""
1. U Net restoration to generate cleaned spectrogram
2. LSTM forecasting to predict a vector of 10 params / time frame
3. App of diff effect modules above
    - Gain adjustment: S1(f,t) = S_restored(f,t) * (1 + G(t))
    - EQ adjustment: S2(f,t) = S1(f,t) * EQ_response(f,t)
    - Compression: S3(f,t) = Compression(S2(f,t))
    - Reverb: S4(f,t) = Reverb(S3(f,t))
    - Echo:  S_mastered(f,t) = Echo(S4(f,t))
"""
class CascadedMastering(nn.Module):
    def __init__(self, unet, forecasting_model, sr, hop_length, ir_length):
        super(CascadedMastering, self).__init__()
        self.unet = unet
        self.forecasting_model = forecasting_model
        self.sr = sr
        self.hop_length = hop_length
        self.ir_length = ir_length
        
    def forward(self, x):
        """
        Args:
            x: Degraded spectrogram of shape [batch, 1, freq, time].
        Returns:
            mastered_spec: Final mastered spectrogram.
            predicted_params: Predicted effect parameters per time frame.
        """
        #1 U Net Restoration
        restored_spec = self.unet(x) #[batch,1,fre,time]
        
        #2 Extra Time series of avg energy over frequency
        avg_energy = restored_spec.mean(dim=2, keepdim=True) #[batch,1,time]
        avg_energy = avg_energy.transpose(1,2) #[batch,time,1]
        
        #3 Forecast effect params
        predicted_params = self.forecasting_model(avg_energy)
        
        #Unpack predicted params
        gain = predicted_params[:,:,0]
        eq_params = predicted_params[:,:,1:4]
        comp_params = predicted_params[:,:,4:7]
        reverb_decay = predicted_params[:,:,7]
        echo_params = predicted_params[:,:,8:10]
        
        #4 Applying gain
        gain = gain.unsqueeze(1) #[batch,1,time]
        spec_gain  = restored_spec * (1+gain)
        
        #5 Applying EQ
        #Prep freq bins for EQ processing
        batch, channels, num_freq, num_time = spec_gain.shape
        nyquist = self.sr / 2.0
        freqs = torch.linspace(0, nyquist, steps = num_freq, device=x.device)
        spec_eq = differentiable_eq(spec_gain, eq_params, freqs)
        
        #6 Apply Compression
        spec_comp = differentiable_compression(spec_eq, comp_params)
        
        #7 Applying reverb using mean predicted reverb across time
        mean_reverb_decay = reverb_decay.mean(dim=1).unsqueeze(10) #[batch,1]
        spec_reverb = differentiable_reverb(spec_comp, mean_reverb_decay, self.sr, self.hop_length, self.ir_length)
        
        #8 Applying Echo using the mean predicted delay and attenuation
        mean_echo_delay = echo_params[:,:,0].menas().item() #Scalar in time frames
        mean_echo_attenuation = echo_params[:,:,1].mean().item() #Scalar
        spec_echo = differentiable_echo(spec_reverb, int(mean_echo_delay), mean_echo_attenuation)
        
        mastered_spec = spec_echo
        
        return mastered_spec, predicted_params
    
#Tester
if __name__ == "__main__":
    # Test each module individually:
    # 1. Test U-Net restoration.
    unet_model = UNet(in_channels=1, out_channels=1, init_features=64)
    print("U-Net Model:")
    print(unet_model)
    x = torch.randn(1, 1, 128, 2582)  # Example degraded spectrogram.
    restored = unet_model(x)
    print("Restored spectrogram shape:", restored.shape)

    # 2. Test LSTM forecasting for multi-parameter prediction.
    lstm_model = LSTMForecasting(input_size=1, hidden_size=32, num_layers=1, num_params=10)
    time_series = torch.randn(1, 2582, 1)  # Simulated time series (e.g., average energy per time frame).
    params_pred = lstm_model(time_series)
    print("Predicted parameters shape:", params_pred.shape)  # Expected: [1, 2582, 10]

    # 3. Test the cascaded model.
    sr = 44100
    hop_length = 512
    ir_length = 20  # Example IR length (in time frames) for reverb.
    cascaded_model = CascadedMastering(unet_model, lstm_model, sr, hop_length, ir_length)
    mastered_spec, effect_params = cascaded_model(x)
    print("Mastered spectrogram shape:", mastered_spec.shape)
    print("Predicted effect parameters shape:", effect_params.shape)  # Expected: [batch, time, 10]        