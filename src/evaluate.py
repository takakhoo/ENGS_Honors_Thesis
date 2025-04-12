# evaluate.py
"""
This script handles the evaluation of our trained audio mastering model.
It provides comprehensive tools for:
- Loading and processing audio files
- Running model inference
- Visualizing results
- Analyzing parameter predictions
- Converting spectrograms back to audio
"""

import os
import sys
import shutil
import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm

from models import OneStageDeepUNet, LSTMForecasting  # We use our new model only

# Set project root (assumes evaluate.py is in src/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print("Project root added to path:", project_root)
sys.path.append(project_root)

###############################
# Utility Functions
###############################
def get_most_recent_checkpoint(checkpoints_dir):
    """
    Finds the most recent or best model checkpoint to use for evaluation.
    Prioritizes 'best_model.pt' if available, otherwise uses the most recent epoch checkpoint.
    
    Args:
        checkpoints_dir: Directory containing model checkpoints
    
    Returns:
        Path to the selected checkpoint file
    """
    best_checkpoint = os.path.join(checkpoints_dir, "best_model.pt")
    if os.path.exists(best_checkpoint):
        print(f"Using best_model.pt: {best_checkpoint}")
        return best_checkpoint
    checkpoint_files = [f for f in os.listdir(checkpoints_dir)
                        if f.startswith("one_stage_deep_unet_epoch_") and f.endswith(".pt")]
    if not checkpoint_files:
        raise Exception("No checkpoint files found in the checkpoints directory!")
    checkpoint_files.sort(key=lambda x: int(x.split("epoch_")[1].split(".")[0]), reverse=True)
    most_recent = os.path.join(checkpoints_dir, checkpoint_files[0])
    print(f"Using most recent checkpoint: {most_recent}")
    return most_recent

def copy_modified_audio(source_dir, target_dir, num_files=5):
    """
    Copies a specified number of modified audio files for evaluation.
    Useful for batch processing multiple test files.
    
    Args:
        source_dir: Directory containing original audio files
        target_dir: Directory to copy files to
        num_files: Maximum number of files to copy
    
    Returns:
        List of paths to copied files
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    files = [f for f in os.listdir(source_dir) if "modified" in f.lower() and f.endswith(".wav")]
    files.sort()
    if not files:
        print(f"No modified audio files found in {source_dir}")
        return []
    selected = files[:num_files]
    copied_paths = []
    for f in selected:
        src = os.path.join(source_dir, f)
        dst = os.path.join(target_dir, f)
        shutil.copy2(src, dst)
        copied_paths.append(dst)
        print(f"Copied {src} -> {dst}")
    return copied_paths

def process_audio_file(audio_path, sr=44100, n_fft=2048, hop_length=512, n_mels=128):
    """
    Processes an audio file into a normalized mel spectrogram tensor.
    Handles loading, preprocessing, and normalization of audio data.
    
    Args:
        audio_path: Path to the audio file
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bands
    
    Returns:
        Tuple of (normalized_mel_tensor, original_mel_db, min_val, max_val, raw_audio)
    """
    raw_audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    raw_audio = np.clip(raw_audio, -1.0, 1.0)
    mel_spec = librosa.feature.melspectrogram(y=raw_audio, sr=sr, n_fft=n_fft,
                                            hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec + 1e-6, ref=np.max)
    min_val = mel_spec_db.min()
    max_val = mel_spec_db.max()
    normalized = (mel_spec_db - min_val) / (max_val - min_val + 1e-6)
    normalized_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return normalized_tensor, mel_spec_db, min_val, max_val, raw_audio

def denormalize_spectrogram(normalized_spec, min_val, max_val):
    """
    Converts a normalized spectrogram back to its original dB scale.
    
    Args:
        normalized_spec: Normalized spectrogram (0-1 range)
        min_val: Original minimum value
        max_val: Original maximum value
    
    Returns:
        Denormalized spectrogram in dB
    """
    return normalized_spec * (max_val - min_val) + min_val

def mel_to_audio(mel_spec_db, sr=44100, n_fft=2048, hop_length=512, n_iter=64):
    """
    Converts a mel spectrogram back to audio using the Griffin-Lim algorithm.
    This is an approximate inverse transform that estimates the phase information.
    
    Args:
        mel_spec_db: Mel spectrogram in dB
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_iter: Number of iterations for Griffin-Lim
    
    Returns:
        Reconstructed audio signal
    """
    mel_spec_power = librosa.db_to_power(mel_spec_db)
    mel_spec_power = np.maximum(mel_spec_power, 1e-10)
    stft_magnitude = librosa.feature.inverse.mel_to_stft(mel_spec_power, sr=sr, n_fft=n_fft, power=2.0)
    stft_magnitude = np.maximum(stft_magnitude, 1e-10)
    audio = librosa.griffinlim(stft_magnitude, hop_length=hop_length, win_length=n_fft, n_iter=n_iter)
    audio = librosa.util.normalize(audio)
    audio = np.clip(audio, -1.0, 1.0)
    return audio

def save_spectrogram(spectrogram, filename, title="Spectrogram", sr=44100, hop_length=512):
    """
    Saves a spectrogram visualization as a PNG file.
    Includes proper scaling and colorbar for dB values.
    
    Args:
        spectrogram: Spectrogram data to visualize
        filename: Output file path
        title: Plot title
        sr: Sample rate
        hop_length: Hop length for time axis
    """
    plt.figure(figsize=(10,6))
    librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved spectrogram image: {filename}")

def save_comparison_plot(input_mel, output_mel, filename, sr=44100, hop_length=512):
    """
    Creates a side-by-side comparison of input and output spectrograms.
    Useful for visually assessing the model's transformations.
    
    Args:
        input_mel: Input mel spectrogram
        output_mel: Output mel spectrogram
        filename: Output file path
        sr: Sample rate
        hop_length: Hop length for time axis
    """
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    img1 = librosa.display.specshow(input_mel, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(img1, format="%+2.0f dB")
    plt.title("Input Mel-Spectrogram")
    plt.subplot(1,2,2)
    img2 = librosa.display.specshow(output_mel, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(img2, format="%+2.0f dB")
    plt.title("Output Mel-Spectrogram")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved comparison plot: {filename}")

def save_parameters_info(predicted_params_norm, unnorm_params, filename):
    """
    Saves detailed statistics about predicted parameters to a text file.
    Includes both normalized and unnormalized parameter values.
    
    Args:
        predicted_params_norm: Normalized parameter predictions
        unnorm_params: Unnormalized parameter values
        filename: Output file path
    """
    param_names = ["Gain", "EQ Center", "EQ Q", "EQ Gain",
                "Comp Threshold", "Comp Ratio", "Comp Makeup",
                "Reverb Decay", "Echo Delay", "Echo Attenuation"]
    with open(filename, "w") as f:
        f.write("Predicted Normalized Parameters:\n")
        for i, name in enumerate(param_names):
            vals = predicted_params_norm[:, :, i].detach().cpu().numpy()
            f.write(f"{name} (Norm): Mean={np.mean(vals):.4f}, Std={np.std(vals):.4f}, Min={np.min(vals):.4f}, Max={np.max(vals):.4f}\n")
        f.write("\nPredicted Unnormalized Parameters:\n")
        for i, name in enumerate(param_names):
            vals = unnorm_params[:, :, i].detach().cpu().numpy()
            f.write(f"{name}: Mean={np.mean(vals):.4f}, Std={np.std(vals):.4f}, Min={np.min(vals):.4f}, Max={np.max(vals):.4f}\n")
    print(f"Saved parameter info: {filename}")

def unnormalize_parameters(predicted_params_norm, sr=44100):
    """
    Converts normalized parameters (0-1 range) back to their original scales.
    Each parameter has its own specific scaling range based on typical audio processing values.
    
    Args:
        predicted_params_norm: Normalized parameters
        sr: Sample rate (used for frequency-related parameters)
    
    Returns:
        Unnormalized parameters with proper scaling
    """
    gain = predicted_params_norm[..., 0] * 2 - 1
    eq_center = predicted_params_norm[..., 1] * (sr / 2)
    eq_Q = predicted_params_norm[..., 2] * 9.9 + 0.1
    eq_gain = predicted_params_norm[..., 3] * 20 - 10
    comp_thresh = predicted_params_norm[..., 4] * 60 - 60
    comp_ratio = predicted_params_norm[..., 5] * 19 + 1
    comp_makeup = predicted_params_norm[..., 6] * 20
    reverb_decay = predicted_params_norm[..., 7] * 9.9 + 0.1
    echo_delay = predicted_params_norm[..., 8] * 100
    echo_atten = predicted_params_norm[..., 9]
    unnorm_params = torch.stack([gain, eq_center, eq_Q, eq_gain,
                                comp_thresh, comp_ratio, comp_makeup,
                                reverb_decay, echo_delay, echo_atten], dim=-1)
    return unnorm_params

###############################
# Evaluation Function
###############################
def evaluate_model(checkpoint_path, audio_path, output_dir, sr=44100, n_fft=2048,
                hop_length=512, n_mels=128, n_iter=64):
    """
    Main evaluation function that processes an audio file through the trained model.
    
    Steps:
    1. Loads the model and checkpoint
    2. Processes the input audio
    3. Runs model inference
    4. Generates visualizations
    5. Saves results and parameters
    6. Optionally performs bypass reversal
    
    Args:
        checkpoint_path: Path to model checkpoint
        audio_path: Path to input audio file
        output_dir: Directory to save results
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bands
        n_iter: Number of iterations for audio reconstruction
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the new model architecture
    model = OneStageDeepUNet(sr=sr, hop_length=hop_length, in_channels=1, out_channels=1,
                            base_features=64, blocks_per_level=5, lstm_hidden=32, num_layers=1, num_params=10)
    model = model.to(device)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        print(f"Loaded model from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        epoch = 0
        print("Loaded model state dict directly")
    model.eval()
    
    print(f"Processing audio file: {audio_path}")
    normalized_mel, original_mel_db, min_val, max_val, raw_audio = process_audio_file(audio_path, sr, n_fft, hop_length, n_mels)
    print(f"Normalized mel spectrogram shape: {normalized_mel.shape}")
    normalized_mel = normalized_mel.to(device)
    
    # Set up output directories
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    audio_dir = os.path.join(output_dir, "audio", f"epoch_{epoch:04d}")
    spectro_dir = os.path.join(output_dir, "spectrograms", f"epoch_{epoch:04d}")
    params_dir = os.path.join(output_dir, "parameters", f"epoch_{epoch:04d}")
    debug_dir = os.path.join(output_dir, "debug", f"epoch_{epoch:04d}")
    bypassed_dir = os.path.join(output_dir, "bypassed", f"epoch_{epoch:04d}")
    for d in [audio_dir, spectro_dir, params_dir, debug_dir, bypassed_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Save input spectrograms for debugging
    input_debug_path = os.path.join(debug_dir, f"{base_name}_input_mel.png")
    save_spectrogram(original_mel_db, input_debug_path, title="Input Mel-Spectrogram (dB)")
    norm_debug_path = os.path.join(debug_dir, f"{base_name}_normalized_mel.png")
    save_spectrogram(normalized_mel.squeeze(0).squeeze(0).cpu().numpy(), norm_debug_path, title="Normalized Input Mel-Spectrogram (0-1)")
    
    with torch.no_grad():
        print("Running model inference...")
        unet_out, predicted_params_norm = model(normalized_mel)
        output_spec = torch.clamp(unet_out, 0.0, 1.0)
        print(f"UNet output shape: {unet_out.shape}")
        print(f"Predicted normalized parameters shape: {predicted_params_norm.shape}")
        
        stage1_debug_path = os.path.join(debug_dir, f"{base_name}_unet_output.png")
        save_spectrogram(unet_out.squeeze(0).squeeze(0).cpu().numpy(), stage1_debug_path, title="Deep UNet Output")
        
        if output_spec.shape[-1] != normalized_mel.shape[-1]:
            print("Adjusting output spectrogram dimensions to match input.")
            output_spec = output_spec[..., :normalized_mel.shape[-1]]
        
        output_spec_np = output_spec.squeeze(0).squeeze(0).cpu().numpy()
        output_spec_db = denormalize_spectrogram(output_spec_np, min_val, max_val)
        output_proc_debug_path = os.path.join(debug_dir, f"{base_name}_output_processed.png")
        save_spectrogram(output_spec_db, output_proc_debug_path, title="Processed Output Mel-Spectrogram (dB)")
        
        comparison_path = os.path.join(spectro_dir, f"{base_name}_comparison.png")
        save_comparison_plot(original_mel_db, output_spec_db, comparison_path, sr, hop_length)
        
        unnorm_params = unnormalize_parameters(predicted_params_norm, sr)
        params_path = os.path.join(params_dir, f"{base_name}_parameters.txt")
        save_parameters_info(predicted_params_norm, unnorm_params, params_path)
        
        # Optionally, perform bypass reversal (if applicable)
        from dataset import PairedAudioDataset
        def read_ground_zero_params(audio_path, sr=44100):
            song_id = os.path.basename(audio_path).split("_")[0]
            param_dir = os.path.join(project_root, "experiments", "output_full", "output_txt")
            param_file = os.path.join(param_dir, f"{song_id}_params.txt")
            if not os.path.exists(param_file):
                print(f"Ground zero parameter file not found: {param_file}")
                return None
            dataset_temp = PairedAudioDataset(audio_dir="dummy", sr=sr, mode="spectrogram")
            gt_params_norm = dataset_temp._parse_parameter_file(param_file)
            print(f"Ground-zero normalized parameters for {song_id}: {gt_params_norm}")
            return gt_params_norm
        
        gt_params_norm = read_ground_zero_params(audio_path, sr=sr)
        if gt_params_norm is None:
            gt_params_norm = torch.zeros(10, dtype=torch.float32, device=device)
            print("No ground-zero parameters found; using zeros.")
        else:
            gt_params_norm = gt_params_norm.to(device)
        mod_spec_norm = normalized_mel.squeeze(0).squeeze(0)
        def unapply_ground_zero(mod_spec_norm, gt_params_norm, sr=44100):
            gt_params = unnormalize_parameters(gt_params_norm.unsqueeze(0), sr=sr).squeeze(0)
            gain = gt_params[0].item()
            eq_center = gt_params[1].item()
            eq_Q = gt_params[2].item()
            eq_gain = gt_params[3].item()
            inv_gain = 1.0 / (1.0 + gain + 1e-6)
            n_mels = mod_spec_norm.shape[0]
            freqs = torch.linspace(0, sr/2, steps=n_mels, device=mod_spec_norm.device)
            epsilon = 1e-6
            bandwidth = eq_center / (eq_Q + epsilon)
            response = 1.0 + eq_gain * torch.exp(-((freqs - eq_center)**2) / (2 * (bandwidth**2) + epsilon))
            inv_eq = 1.0 / (response + epsilon)
            inv_eq = inv_eq.unsqueeze(1)
            bypass_spec_norm = mod_spec_norm * inv_gain * inv_eq
            return bypass_spec_norm
        bypass_spec_norm = unapply_ground_zero(mod_spec_norm, gt_params_norm, sr)
        bypass_spec_db = denormalize_spectrogram(bypass_spec_norm.cpu().numpy(), min_val, max_val)
        bypass_debug_path = os.path.join(debug_dir, f"{base_name}_bypassed_output.png")
        save_spectrogram(bypass_spec_db, bypass_debug_path, title="Bypassed Output (Ground Zero Inversion) (dB)")
        
        reconstructed_audio = mel_to_audio(output_spec_db, sr, n_fft, hop_length, n_iter)
        bypassed_audio = mel_to_audio(bypass_spec_db, sr, n_fft, hop_length, n_iter)
        
        input_audio_path = os.path.join(audio_dir, f"{base_name}_input.wav")
        output_audio_path = os.path.join(audio_dir, f"{base_name}_output.wav")
        bypass_audio_path = os.path.join(bypassed_dir, f"{base_name}_bypassed.wav")
        print(f"Saving audio files to: {audio_dir} and bypassed audio to: {bypassed_dir}")
        sf.write(input_audio_path, raw_audio, sr)
        sf.write(output_audio_path, reconstructed_audio, sr)
        sf.write(bypass_audio_path, bypassed_audio, sr)
    
    print(f"Evaluation complete for {base_name}")
    return reconstructed_audio, output_spec_db, predicted_params_norm, bypassed_audio

def main():
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    source_audio_dir = os.path.join(project_root, "experiments", "output_full", "output_audio")
    target_audio_dir = os.path.join(project_root, "audio", "internal_demastered")
    output_dir = os.path.join(project_root, "runs")
    
    os.makedirs(target_audio_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    sr = 44100
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    n_iter = 64
    
    print("Finding the best checkpoint...")
    checkpoint_path = get_most_recent_checkpoint(os.path.join(project_root, "checkpoints"))
    
    print(f"Copying modified audio files from {source_audio_dir} to {target_audio_dir}")
    audio_files = copy_modified_audio(source_audio_dir, target_audio_dir, num_files=5)
    if not audio_files:
        print("No audio files to evaluate.")
        return
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        try:
            evaluate_model(checkpoint_path, audio_path, output_dir, sr=sr, n_fft=n_fft,
                        hop_length=hop_length, n_mels=n_mels, n_iter=n_iter)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            import traceback
            traceback.print_exc()
    print("Evaluation complete!")

if __name__ == "__main__":
    main()
