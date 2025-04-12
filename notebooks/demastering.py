#!/usr/bin/env python3
"""
Demastering a Song from the FMA Dataset

This script demonstrates a simple "demastering" process by:
- Loading full tracks from the FMA dataset.
- Applying several degradation functions to the entire track using random parameters:
      * EQ (peaking filter)
      * Gain adjustment
      * Echo
      * Reverb
      * Compression
- Displaying and saving the STFT (spectrogram) of both the original and degraded tracks.
- Saving processed audio and associated parameter data for future use.

Note:
- Update the folder paths as needed to point to your valid FMA dataset files.
- This version uses float32 to minimize memory usage and explicitly frees memory after processing.
"""

import os
import glob
import random
import gc
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import torch
import torchaudio
import scipy
from scipy.signal import iirpeak, lfilter, fftconvolve

def print_dependency_versions():
    print("PyTorch version:", torch.__version__)
    print("torchaudio version:", torchaudio.__version__)
    print("Librosa version:", librosa.__version__)
    print("NumPy version:", np.__version__)
    print("Pandas version:", pd.__version__)
    print("Matplotlib version:", plt.matplotlib.__version__)
    print("SciPy version:", scipy.__version__)

def plot_stft(audio, sr, title, extra_info=None):
    """
    Compute the Short-Time Fourier Transform (STFT) of audio and plot the
    spectrogram, with an optional annotation for extra information.
    """
    D = librosa.stft(audio)  # Compute STFT
    DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Convert amplitude to dB
    
    fig, ax = plt.subplots(figsize=(12, 6))
    img = librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='hz', ax=ax)
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.set_label("Amplitude (dB)", fontsize=12)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Frequency (Hz)", fontsize=14)
    
    if extra_info:
        ax.text(0.01, 0.01, extra_info, transform=ax.transAxes,
                fontsize=12, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5))
    plt.tight_layout()
    return fig

def apply_eq(audio, sr, fc, Q, gain_db):
    """
    Apply a peaking EQ filter to simulate an unmastered version.

    Parameters:
    audio (np.array): Input audio waveform.
    sr (int): Sampling rate.
    fc (float): Center frequency in Hz.
    Q (float): Q-factor.
    gain_db (float): EQ gain adjustment in dB.

    Returns:
    np.array: EQ processed audio.
    """
    w0 = fc / (sr / 2)  # Normalize center frequency (w0 in [0,1], where 1 = Nyquist Frequency)
    b, a = iirpeak(w0, Q)
    gain_linear = 10 ** (gain_db / 20.0)
    b = b * gain_linear  # Scale filter numerator to apply gain
    return lfilter(b, a, audio)

def apply_gain(audio, gain_db):
    """Apply a gain adjustment in dB to the audio."""
    gain_linear = 10 ** (gain_db / 20.0)
    return audio * gain_linear

def apply_echo(audio, sr, delay_seconds, attenuation):
    """
    Apply echo effect by adding a delayed, attenuated copy of the audio.

    Parameters:
    audio (np.array): Input audio.
    sr (int): Sampling rate.
    delay_seconds (float): Delay in seconds.
    attenuation (float): Attenuation factor for the echo.

    Returns:
    np.array: Audio with echo applied.
    """
    delay_samples = int(sr * delay_seconds)
    echo = np.zeros_like(audio)
    if len(audio) > delay_samples:
        echo[delay_samples:] = audio[:-delay_samples] * attenuation
    return audio + echo

def apply_reverb(audio, sr, decay, ir_length):
    """
    Apply a simple reverb effect by convolving the audio with a synthetic impulse response.

    Parameters:
    audio (np.array): Input audio.
    sr (int): Sampling rate.
    decay (float): Decay rate for the exponential impulse response.
    ir_length (int): Length of the impulse response in samples.

    Returns:
    np.array: Audio with reverb applied.
    """
    ir_length = int(ir_length)
    print("Inside apply_reverb: received ir_length =", ir_length)
    if ir_length <= 0:
        raise ValueError("ir_length must be positive, but got {}".format(ir_length))
    t = np.linspace(0, ir_length / sr, ir_length)
    ir = np.exp(-decay * t)
    try:
        result = fftconvolve(audio, ir, mode='same')
        if result.size == 0:
            raise ValueError("fftconvolve returned an empty array")
    except Exception as e:
        print(f"fftconvolve failed ({e}); using np.convolve instead.")
        result = np.convolve(audio, ir, mode='same')
    if result.size == 0:
        print("Warning: apply_reverb resulted in an empty array; returning original audio.")
        return audio
    return result

def apply_compression(audio, threshold_db, ratio, makeup_gain_db):
    """
    Apply dynamic range compression to the audio.

    Parameters:
    audio (np.array): Input audio.
    threshold_db (float): Threshold in dB above which compression occurs.
    ratio (float): Compression ratio.
    makeup_gain_db (float): Makeup gain in dB to apply after compression.

    Returns:
    np.array: Compressed audio.
    """
    threshold_linear = 10 ** (threshold_db / 20.0)
    abs_audio = np.abs(audio)
    compressed = np.where(abs_audio > threshold_linear,
                        threshold_linear + (abs_audio - threshold_linear) / ratio,
                        abs_audio)
    compressed = np.sign(audio) * compressed
    makeup_gain = 10 ** (makeup_gain_db / 20.0)
    return compressed * makeup_gain

def main():
    # Print dependency versions.
    print_dependency_versions()

    print("\nDemastering a Song from the FMA Dataset\n")
    print("This script demonstrates a simple 'demastering' process by:")
    print("  - Loading full tracks from the FMA dataset.")
    print("  - Applying several degradation functions (EQ, gain, echo, reverb, compression).")
    print("  - Displaying and saving the STFT of both the original and degraded tracks.")
    print("  - Saving the processed audio for dataset creation.\n")
    
    print("Current working directory:", os.getcwd())
    
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define output directories using absolute paths
    base_output_dir = os.path.join(project_root, "experiments", "output_full")
    output_audio_dir = os.path.join(base_output_dir, "output_audio")
    output_spectrogram_dir = os.path.join(base_output_dir, "output_spectrograms")
    output_txt_dir = os.path.join(base_output_dir, "output_txt")
    
    # Create output directories if they don't exist
    for folder in [base_output_dir, output_audio_dir, output_spectrogram_dir, output_txt_dir]:
        os.makedirs(folder, exist_ok=True)
    
    # Define input directories using absolute paths
    base_input_folder = os.path.join(project_root, "data", "raw", "fma_small")
    
    # Check if input directory exists
    if not os.path.exists(base_input_folder):
        raise FileNotFoundError(f"Input directory not found: {base_input_folder}")
    
    subfolders = [name for name in os.listdir(base_input_folder)
            if os.path.isdir(os.path.join(base_input_folder, name))]
    
    # Process each MP3 file in the subfolders
    for sub in subfolders:
        folder_path = os.path.join(base_input_folder, sub)
        mp3_files = glob.glob(os.path.join(folder_path, "*.mp3"))
        for song_path in mp3_files:
            print(f"\nProcessing song: {song_path}")
            
            # Load audio and ensure it is float32
            audio, sr = librosa.load(song_path, sr=None)
            if len(audio) == 0:
                print(f"Warning: Audio from {song_path} is empty. Skipping.")
                continue
            audio = audio.astype(np.float32)
            
            # Generate a unique identifier based on the file name
            song_id = os.path.splitext(os.path.basename(song_path))[0]
            
            # Save original spectrogram
            fig_orig = plot_stft(audio, sr, f"Original Audio STFT - {song_id}")
            orig_spec_path = os.path.join(output_spectrogram_dir, f"{song_id}_input.png")
            fig_orig.savefig(orig_spec_path)
            plt.close(fig_orig)
            
            # Randomly generate degradation parameters
            # EQ parameters:
            fc = random.uniform(200, 8000)         # Center frequency in Hz
            Q = random.uniform(0.7, 2.5)             # Q-factor
            eq_gain_db = random.uniform(-2, 2)       # EQ gain adjustment in dB
            
            # Gain adjustment:
            gain_db = random.uniform(-1, 1)
            
            # Echo:
            delay_seconds = random.uniform(0.3, 1.0)
            attenuation = random.uniform(0.3, 0.5)
            
            # Reverb:
            decay = random.uniform(0.2, 1.0)
            lower_ir = max(int(0.1 * sr), 1)
            upper_ir = max(int(0.3 * sr), lower_ir + 1)
            ir_length = random.randint(lower_ir, upper_ir)
            
            # Compression:
            threshold_db = random.uniform(-20, -12)
            ratio = random.uniform(2, 3)
            makeup_gain_db = random.uniform(0, 0.5)
            
            # Save parameters to a text file
            txt_path = os.path.join(output_txt_dir, f"{song_id}_params.txt")
            with open(txt_path, "w") as f:
                f.write(f"EQ: fc={fc:.2f} Hz, Q={Q:.2f}, eq_gain_db={eq_gain_db:.2f}\n")
                f.write(f"Gain: gain_db={gain_db:.2f}\n")
                f.write(f"Echo: delay_seconds={delay_seconds:.2f}, attenuation={attenuation:.2f}\n")
                f.write(f"Reverb: decay={decay:.2f}, ir_length={ir_length}\n")
                f.write(f"Compression: threshold_db={threshold_db:.2f}, ratio={ratio:.2f}, makeup_gain_db={makeup_gain_db:.2f}\n")
            print(f"Parameters for {song_id} saved to {txt_path}")
            
            # Apply degradation effects sequentially
            modified_audio = apply_eq(audio, sr, fc, Q, eq_gain_db)
            modified_audio = apply_gain(modified_audio, gain_db)
            modified_audio = apply_echo(modified_audio, sr, delay_seconds, attenuation)
            modified_audio = apply_reverb(modified_audio, sr, decay, ir_length)
            modified_audio = apply_compression(modified_audio, threshold_db, ratio, makeup_gain_db)
            
            # Save modified spectrogram with extra info annotation
            extra = (
                f"EQ: fc={fc:.2f} Hz, Q={Q:.2f}, gain_db={eq_gain_db:.2f}\n"
                f"Gain: {gain_db:.2f} dB\n"
                f"Echo: delay={delay_seconds:.2f}s, att={attenuation:.2f}\n"
                f"Reverb: decay={decay:.2f}, IR_len={ir_length}\n"
                f"Comp: thr={threshold_db:.2f} dB, ratio={ratio:.2f}, makeup={makeup_gain_db:.2f} dB"
            )
            fig_mod = plot_stft(modified_audio, sr, f"Modified Audio STFT - {song_id}", extra_info=extra)
            mod_spec_path = os.path.join(output_spectrogram_dir, f"{song_id}_output.png")
            fig_mod.savefig(mod_spec_path)
            plt.close(fig_mod)
            
            # Save original and modified audio files
            orig_audio_path = os.path.join(output_audio_dir, f"{song_id}_original.wav")
            mod_audio_path = os.path.join(output_audio_dir, f"{song_id}_modified.wav")
            sf.write(orig_audio_path, audio, sr)
            sf.write(mod_audio_path, modified_audio, sr)
            
            print(f"Saved audio for {song_id} to {output_audio_dir}\n")
            
            # Free memory for these variables (this deletes the in-memory arrays, not your saved files)
            del audio, modified_audio, fig_orig, fig_mod
            gc.collect()

if __name__ == "__main__":
    main()
