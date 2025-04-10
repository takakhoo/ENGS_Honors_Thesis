"""
evaluate.py

This evaluation script loads the trained cascaded mastering model checkpoint,
processes audio files using the model, and reconstructs the audio output.
Key improvements in this version:
- Uses the exact same mel spectrogram computation and normalization as in dataset.py.
- Clamps (or squashes) the model’s raw output to [0, 1] before denormalization.
- Denormalizes using the input file’s original min/max dB values.
- Uses librosa’s inverse functions and Griffin-Lim for better phase reconstruction.
- Copies five modified files from experiments/output_full/output_audio into
    audio/internal_demastered, and saves outputs (including debug plots) in runs/.
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

# Import model components from models.py
from models import UNet, LSTMForecasting, CascadedMastering

# Set project root (assume evaluate.py is in src/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print("Project root added to path:", project_root)
sys.path.append(project_root)


###############################
# UTILITY FUNCTIONS
###############################

def get_most_recent_checkpoint(checkpoints_dir):
    """Return the path to the checkpoint with the highest epoch number."""
    checkpoint_files = [
        f for f in os.listdir(checkpoints_dir)
        if f.startswith("cascaded_model_epoch_") and f.endswith(".pt")
    ]
    if not checkpoint_files:
        if os.path.exists(os.path.join(checkpoints_dir, "best_model.pt")):
            return os.path.join(checkpoints_dir, "best_model.pt")
        raise Exception("No checkpoint files found in the checkpoints directory!")
    checkpoint_files.sort(key=lambda x: int(x.split("epoch_")[1].split(".")[0]), reverse=True)
    most_recent = os.path.join(checkpoints_dir, checkpoint_files[0])
    print(f"Most recent checkpoint found: {most_recent}")
    return most_recent

def copy_modified_audio(source_dir, target_dir, num_files=5):
    """
    Copy the first num_files containing 'modified' (case-insensitive) from source_dir
    to target_dir.
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
    Process the audio file exactly as in dataset.py:
    - Loads the audio (mono)
    - Computes a mel spectrogram using librosa.feature.melspectrogram
    - Converts to dB (with a small offset) using librosa.power_to_db
    - Normalizes the result to [0, 1]
    Returns:
    normalized_tensor, original_mel_db, min_value, max_value, raw_audio
    """
    # Load and clip
    raw_audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    raw_audio = np.clip(raw_audio, -1.0, 1.0)
    # Compute mel spectrogram (same parameters as in dataset.py)
    mel_spec = librosa.feature.melspectrogram(
        y=raw_audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    # Convert to dB scale with a small offset to avoid log(0)
    mel_spec_db = librosa.power_to_db(mel_spec + 1e-6, ref=np.max)
    # Save the min and max for later denormalization
    min_value = mel_spec_db.min()
    max_value = mel_spec_db.max()
    normalized = (mel_spec_db - min_value) / (max_value - min_value + 1e-6)
    normalized_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return normalized_tensor, mel_spec_db, min_value, max_value, raw_audio

def denormalize_spectrogram(normalized_spec, min_value, max_value):
    """Denormalize a spectrogram from [0,1] back to its original dB range."""
    return normalized_spec * (max_value - min_value) + min_value

def mel_to_audio(mel_spec_db, sr=44100, n_fft=2048, hop_length=512, n_iter=64):
    """
    Convert a mel spectrogram in dB scale back to an audio signal.
    The process:
    1. Convert from dB to power.
    2. Invert the mel filter bank to recover an STFT magnitude.
    3. Use Griffin-Lim for phase reconstruction.
    4. Normalize and clip final audio.
    """
    # dB to power conversion
    mel_spec_power = librosa.db_to_power(mel_spec_db)
    mel_spec_power = np.maximum(mel_spec_power, 1e-10)
    
    # Invert mel spectrogram to STFT magnitude
    stft_magnitude = librosa.feature.inverse.mel_to_stft(
        mel_spec_power,
        sr=sr,
        n_fft=n_fft,
        power=2.0
    )
    stft_magnitude = np.maximum(stft_magnitude, 1e-10)
    
    # Griffin-Lim reconstruction (more iterations for quality)
    audio = librosa.griffinlim(
        stft_magnitude,
        hop_length=hop_length,
        win_length=n_fft,
        n_iter=n_iter
    )
    audio = librosa.util.normalize(audio)
    audio = np.clip(audio, -1.0, 1.0)
    return audio

def save_spectrogram(spectrogram, filename, title="Spectrogram", sr=44100, hop_length=512):
    """
    Save a spectrogram visualization.
    Expects spectrogram as a 2D array (frequency x time).
    """
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(
        spectrogram,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_comparison_plot(input_mel, output_mel, filename, sr=44100, hop_length=512):
    """Save a side-by-side comparison of input and output mel spectrograms."""
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    img1 = librosa.display.specshow(
        input_mel,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel'
    )
    plt.colorbar(img1, format="%+2.0f dB")
    plt.title("Input Mel-Spectrogram")
    
    plt.subplot(1, 2, 2)
    img2 = librosa.display.specshow(
        output_mel,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel'
    )
    plt.colorbar(img2, format="%+2.0f dB")
    plt.title("Output Mel-Spectrogram")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_parameters_info(predicted_params, filename):
    """Save predicted effect parameters' statistics to a text file."""
    param_names = [
        "Gain", "EQ Center", "EQ Q", "EQ Gain",
        "Comp Threshold", "Comp Ratio", "Comp Makeup",
        "Reverb Decay", "Echo Delay", "Echo Attenuation"
    ]
    with open(filename, "w") as f:
        f.write("Predicted Effect Parameters (mean, std, min, max):\n")
        for i, name in enumerate(param_names):
            param_values = predicted_params[:, :, i].detach().cpu().numpy()
            f.write(f"{name}:\n")
            f.write(f" Mean: {np.mean(param_values):.4f}\n")
            f.write(f" Std: {np.std(param_values):.4f}\n")
            f.write(f" Min: {np.min(param_values):.4f}\n")
            f.write(f" Max: {np.max(param_values):.4f}\n")
            f.write("\n")

def evaluate_model(checkpoint_path, audio_path, output_dir, sr=44100, n_fft=2048,
                hop_length=512, n_mels=128, n_iter=64):
    """
    Evaluate a trained cascaded mastering model on a single audio file.
    Steps:
    1. Load and process the audio exactly as in dataset.py.
    2. Run model inference.
    3. Clamp the raw model output to [0, 1] and then denormalize
        using the same min/max as the input.
    4. Convert the denormalized mel spectrogram back to audio.
    5. Save intermediate spectrogram images, parameter info, and audio files.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model architecture
    unet = UNet(in_channels=1, out_channels=1)
    lstm = LSTMForecasting(input_size=1, hidden_size=32, num_layers=1)
    model = CascadedMastering(unet, lstm, sr=sr, hop_length=hop_length, ir_length=20)
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
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
    
    # Process input audio the same way as dataset.py
    print(f"Processing audio file: {audio_path}")
    normalized_mel, original_mel_db, min_value, max_value, raw_audio = process_audio_file(
        audio_path, sr, n_fft, hop_length, n_mels
    )
    normalized_mel = normalized_mel.to(device)
    
    # Create output directories
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    audio_dir = os.path.join(output_dir, "audio", f"epoch_{epoch:04d}")
    spectro_dir = os.path.join(output_dir, "spectrograms", f"epoch_{epoch:04d}")
    params_dir = os.path.join(output_dir, "parameters", f"epoch_{epoch:04d}")
    debug_dir = os.path.join(output_dir, "debug", f"epoch_{epoch:04d}")
    for d in [audio_dir, spectro_dir, params_dir, debug_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Save input spectrogram (dB scale) for debugging
    input_debug_path = os.path.join(debug_dir, f"{base_name}_input_mel.png")
    save_spectrogram(original_mel_db, input_debug_path, "Input Mel-Spectrogram")
    
    # Save normalized input spectrogram (0-1 range)
    norm_debug_path = os.path.join(debug_dir, f"{base_name}_normalized_mel.png")
    save_spectrogram(
        normalized_mel.squeeze(0).squeeze(0).cpu().numpy(),
        norm_debug_path,
        "Normalized Input Mel-Spectrogram (0-1 range)"
    )
    
    # Model inference
    with torch.no_grad():
        print("Running model inference...")
        output_spec, predicted_params = model(normalized_mel)
        # Clamp raw output to enforce [0,1] range
        output_spec = torch.clamp(output_spec, 0.0, 1.0)
        # Save raw model output for debugging
        output_raw_debug_path = os.path.join(debug_dir, f"{base_name}_model_output_raw.png")
        save_spectrogram(
            output_spec.squeeze(0).squeeze(0).cpu().numpy(),
            output_raw_debug_path,
            "Raw Model Output (clamped 0-1 range)"
        )
        # Denormalize the output spectrogram back to dB scale using input's min/max
        output_spec_np = output_spec.squeeze(0).squeeze(0).cpu().numpy()
        output_spec_db = denormalize_spectrogram(output_spec_np, min_value, max_value)
        # Save processed output spectrogram (dB scale) for debugging
        output_proc_debug_path = os.path.join(debug_dir, f"{base_name}_output_processed.png")
        save_spectrogram(
            output_spec_db,
            output_proc_debug_path,
            "Processed Output Mel-Spectrogram (dB scale)"
        )
        # Save side-by-side comparison of input vs. output
        comparison_path = os.path.join(spectro_dir, f"{base_name}_comparison.png")
        save_comparison_plot(original_mel_db, output_spec_db, comparison_path)
        # Save predicted parameters
        params_path = os.path.join(params_dir, f"{base_name}_parameters.txt")
        save_parameters_info(predicted_params, params_path)
        
        # Convert the denormalized mel spectrogram back to audio
        print("Converting mel spectrogram to audio...")
        reconstructed_audio = mel_to_audio(output_spec_db, sr, n_fft, hop_length, n_iter)
        
        # Save original and reconstructed audio
        input_audio_path = os.path.join(audio_dir, f"{base_name}_input.wav")
        output_audio_path = os.path.join(audio_dir, f"{base_name}_output.wav")
        print(f"Saving audio files to {audio_dir}")
        sf.write(input_audio_path, raw_audio, sr)
        sf.write(output_audio_path, reconstructed_audio, sr)
    
    print(f"Evaluation complete for {base_name}")
    return reconstructed_audio, output_spec_db, predicted_params

def main():
    """Main function to evaluate the model on multiple audio files."""
    # Define directories relative to project root
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    source_audio_dir = os.path.join(project_root, "experiments", "output_full", "output_audio")
    target_audio_dir = os.path.join(project_root, "audio", "internal_demastered")
    output_dir = os.path.join(project_root, "runs")
    
    os.makedirs(target_audio_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Audio processing parameters (match dataset.py)
    sr = 44100
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    n_iter = 64
    
    print("Finding most recent checkpoint...")
    checkpoint_path = get_most_recent_checkpoint(checkpoints_dir)
    
    print(f"Copying modified audio files from {source_audio_dir} to {target_audio_dir}")
    audio_files = copy_modified_audio(source_audio_dir, target_audio_dir, num_files=5)
    if not audio_files:
        print("No audio files to evaluate.")
        return
    
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        try:
            evaluate_model(
                checkpoint_path,
                audio_path,
                output_dir,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                n_iter=n_iter
            )
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()
