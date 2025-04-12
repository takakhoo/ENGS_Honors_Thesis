"""
dataset.py

This file handles all the data loading and preprocessing for our audio mastering model.
Think of it like a recipe book - it tells us how to prepare our audio data before feeding it to the model.

Key concepts:
- Mel spectrograms: A way to represent audio that's more similar to how humans hear sound
- RMS (Root Mean Square): A measure of audio energy over time
- Normalization: Scaling our data to make it easier for the model to learn
- Parameter parsing: Reading and normalizing audio processing parameters
"""

import os
import glob
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

# Append project root for module imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def compute_mel_spectrogram(audio, sr, n_mels=128, n_fft=2048, hop_length=512):
    """
    Convert raw audio into a mel spectrogram.
    Why mel? Because our ears don't hear frequencies linearly - we're more sensitive to 
    changes in lower frequencies. Mel scale mimics this human perception.
    
    n_mels: Number of mel bands (like frequency bins, but warped to match human hearing)
    n_fft: Size of the FFT window (bigger = better frequency resolution, but worse time resolution)
    hop_length: How much we slide the window each time (smaller = more time resolution)
    """
    # First compute the regular spectrogram, then convert to mel scale
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, 
                                                hop_length=hop_length, n_mels=n_mels)
    # Convert to dB scale because our ears perceive sound logarithmically
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def compute_rms(audio, frame_length=2048, hop_length=512):
    """
    Compute the Root Mean Square (RMS) energy of the audio signal.
    This is like measuring how "loud" the audio is over time.
    
    frame_length: How many samples to look at at once
    hop_length: How much to slide the window each time
    """
    # Compute RMS using librosa's built-in function
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    # Convert to dB scale (logarithmic, like human hearing)
    rms_db = 10 * np.log10(rms + 1e-10)  # Add small number to avoid log(0)
    return rms_db

def compute_segmented_rms(audio, sr, segment_duration=0.1):
    """
    Compute RMS energy in fixed-duration segments.
    This is like measuring the average loudness every 100ms.
    
    segment_duration: How long each segment should be in seconds
    """
    # Convert duration to samples
    segment_length = int(segment_duration * sr)
    rms_vals = []
    
    # Process audio in chunks
    for i in range(0, len(audio), segment_length):
        segment = audio[i:i+segment_length]
        if len(segment) == 0:
            continue
        # Compute RMS for this segment
        rms = np.sqrt(np.mean(segment**2))
        # Convert to dB
        rms_db = 10 * np.log10(rms + 1e-10)
        rms_vals.append(rms_db)
    
    return np.array(rms_vals)

class PairedAudioDataset(Dataset):
    """
    This is where the magic happens! This class handles loading pairs of audio files:
    - Original (clean) audio
    - Modified (processed) audio
    
    It's like having before/after photos, but for sound.
    """
    def __init__(self, audio_dir, sr=None, transform=None, mode="spectrogram", segment_duration=None):
        """
        Initialize our dataset.
        audio_dir: Where to find our audio files
        sr: Sample rate (how many samples per second). None = use original file's rate
        transform: Any extra processing we want to do
        mode: How to represent the audio ("spectrogram" or "rms")
        segment_duration: For segmented RMS mode, how long each segment should be
        """
        self.audio_dir = audio_dir
        # Find all original audio files (they should end with _original.wav)
        self.original_files = glob.glob(os.path.join(audio_dir, "*_original.wav"))
        print("Found original files:", self.original_files)  # Debug print
        self.sr = sr
        self.transform = transform
        self.mode = mode
        self.segment_duration = segment_duration
        
        # The parameter files are expected in experiments/output_full/output_txt
        self.param_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "output_full", "output_txt")
        
    def __len__(self):
        """How many audio files do we have?"""
        return len(self.original_files)
    
    def __getitem__(self, idx):
        """
        Get one pair of audio files and their parameters.
        This is what PyTorch calls when it needs a training example.
        """
        # Get paths to original and modified audio files
        orig_path = self.original_files[idx]
        song_id = os.path.basename(orig_path).split("_")[0]
        mod_path = os.path.join(self.audio_dir, f"{song_id}_modified.wav")
        
        try:
            # Load both audio files
            original, sr_orig = librosa.load(orig_path, sr=self.sr, mono=True)
            modified, sr_mod = librosa.load(mod_path, sr=self.sr, mono=True)
        except Exception as e:
            print(f"Error loading {orig_path}: {e}")
            return self._empty_tensor(), self._empty_tensor(), torch.zeros(10, dtype=torch.float32)
        
        # Check for invalid values
        if np.isnan(original).any() or np.isinf(original).any():
            print(f"NaN/Inf in {orig_path}; using zeros.")
            return self._empty_tensor(), self._empty_tensor(), torch.zeros(10, dtype=torch.float32)
        if np.isnan(modified).any() or np.isinf(modified).any():
            print(f"NaN/Inf in {mod_path}; using zeros.")
            return self._empty_tensor(), self._empty_tensor(), torch.zeros(10, dtype=torch.float32)
        
        # Make sure audio values are between -1 and 1
        original = np.clip(original, -1.0, 1.0)
        modified = np.clip(modified, -1.0, 1.0)
        
        # Convert to our chosen representation (spectrogram or RMS)
        if self.mode == "spectrogram":
            original_spec = self._process_spectrogram(original, sr_orig)
            modified_spec = self._process_spectrogram(modified, sr_mod)
        elif self.mode == "rms":
            original_spec = compute_rms(original)
            modified_spec = compute_rms(modified)
        elif self.mode == "rms_segmented":
            if not self.segment_duration:
                raise ValueError("Mode must be 'spectrogram' or 'rms'")
            original_spec = compute_segmented_rms(original, sr_orig, self.segment_duration)
            modified_spec = compute_segmented_rms(modified, sr_mod, self.segment_duration)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        # Parse and normalize the parameter vector
        param_file = os.path.join(self.param_dir, f"{song_id}_params.txt")
        if not os.path.exists(param_file):
            print(f"Warning: Parameter file {param_file} not found. Using zeros.")
            param_vector = torch.zeros(10, dtype=torch.float32)
        else:
            param_vector = self._parse_parameter_file(param_file)
        
        return modified_spec, original_spec, param_vector
    
    def _process_spectrogram(self, audio, sr):
        """
        Process audio into a mel spectrogram and normalize it.
        This is like converting a photo to black and white and adjusting the contrast.
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        # Convert to dB scale
        mel_spec = librosa.power_to_db(mel_spec + 1e-6, ref=np.max)
        # Convert to PyTorch tensor
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
        # Add channel dimension if needed
        if len(mel_spec.shape) == 2:
            mel_spec = mel_spec.unsqueeze(0)
        # Normalize to [0,1] range
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)
        return mel_spec
    
    def _parse_parameter_file(self, filepath):
        """
        Read and normalize audio processing parameters from a text file.
        This is like reading a recipe and converting all measurements to the same units.
        """
        # Read the parameter file
        with open(filepath, "r") as f:
            lines = f.read().splitlines()
        
        # Parse different types of parameters
        params = {}
        for line in lines:
            if line.startswith("EQ:"):
                # Parse equalizer parameters
                parts = line.split(":")[1].strip().split(",")
                for p in parts:
                    key_val = p.strip().split("=")
                    key = key_val[0].strip()
                    val = float(key_val[1].split()[0].strip())
                    params[f"eq_{key}"] = val
            elif line.startswith("Gain:"):
                # Parse gain parameter
                p = line.split(":")[1].strip()
                key_val = p.split("=")
                params["gain_db"] = float(key_val[1].strip())
            elif line.startswith("Echo:"):
                # Parse echo parameters
                parts = line.split(":")[1].strip().split(",")
                for p in parts:
                    key_val = p.strip().split("=")
                    params[key_val[0].strip()] = float(key_val[1].strip())
            elif line.startswith("Reverb:"):
                # Parse reverb parameters
                parts = line.split(":")[1].strip().split(",")
                for p in parts:
                    key_val = p.strip().split("=")
                    if key_val[0].strip() == "decay":
                        params["decay"] = float(key_val[1].strip())
            elif line.startswith("Compression:"):
                # Parse compression parameters
                parts = line.split(":")[1].strip().split(",")
                for p in parts:
                    key_val = p.strip().split("=")
                    params[key_val[0].strip()] = float(key_val[1].strip())
        
        # Get sample rate (default to 44.1kHz if not specified)
        sr = self.sr if self.sr else 44100
        
        # Extract all parameters with defaults
        gain = params.get("gain_db", 0.0)
        eq_center = params.get("eq_fc", 0.0)
        eq_Q = params.get("eq_Q", 0.0)
        eq_gain = params.get("eq_gain_db", 0.0)
        comp_thresh = params.get("threshold_db", 0.0)
        comp_ratio = params.get("ratio", 0.0)
        comp_makeup = params.get("makeup_gain_db", 0.0)
        reverb_decay = params.get("decay", 0.0)
        echo_delay = params.get("delay_seconds", 0.0)
        echo_atten = params.get("attenuation", 0.0)
        
        # Normalize all parameters to [0,1] range
        # This makes it easier for the neural network to learn
        norm_gain = (gain + 1) / 2.0  # [-1,1] -> [0,1]
        norm_eq_center = eq_center / (sr/2)  # [0,sr/2] -> [0,1]
        norm_eq_Q = (eq_Q - 0.1)/9.9  # [0.1,10] -> [0,1]
        norm_eq_gain = (eq_gain + 10)/20.0  # [-10,10] -> [0,1]
        norm_comp_thresh = (comp_thresh + 60) / 60.0  # [-60,0] -> [0,1]
        norm_comp_ratio = (comp_ratio - 1) / 19.0  # [1,20] -> [0,1]
        norm_comp_makeup = comp_makeup/20.0  # [0,20] -> [0,1]
        norm_reverb_decay = (reverb_decay - 0.1)/9.9  # [0.1,10] -> [0,1]
        norm_echo_delay = echo_delay/100.0  # [0,100] -> [0,1]
        norm_echo_atten = echo_atten  # Already in [0,1]
        
        # Combine all normalized parameters into a vector
        vector = [norm_gain, norm_eq_center, norm_eq_Q, norm_eq_gain,
                norm_comp_thresh, norm_comp_ratio, norm_comp_makeup,
                norm_reverb_decay, norm_echo_delay, norm_echo_atten]
        
        return torch.tensor(vector, dtype=torch.float32)
    
    def _empty_tensor(self):
        """Return an empty tensor when we can't load the audio."""
        return torch.zeros(1, dtype=torch.float32)

if __name__ == "__main__":
    # This part is just for testing - it shows us what our processed data looks like
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    AUDIO_DIR = os.path.join(CURRENT_DIR, "..", "experiments", "output_full", "output_audio")
    
    # Try loading some data in spectrogram mode
    dataset_spec = PairedAudioDataset(
        audio_dir=AUDIO_DIR, 
        sr=44100, 
        transform=compute_mel_spectrogram, 
        mode="spectrogram"
    )
    print(f"Spectrogram mode, dataset length: {len(dataset_spec)}")
    if len(dataset_spec) > 0:
        sample_input, sample_target, sample_params = dataset_spec[0]
        print("Input spectrogram shape:", sample_input.shape)
        print("Target spectrogram shape:", sample_target.shape)
        print("Normalized parameter vector:", sample_params)
    
    # Try loading some data in segmented RMS mode
    dataset_rms_seg = PairedAudioDataset(
        audio_dir="experiments/output_full/output_audio", 
        sr=44100, 
        mode="rms_segmented", 
        segment_duration=0.1  # 100 ms segments
    )
    print(f"Segmented RMS mode, dataset length: {len(dataset_rms_seg)}")
    if len(dataset_rms_seg) > 0:
        rms_input, rms_target, rms_params = dataset_rms_seg[0]
        print("Input RMS sequence shape:", rms_input.shape)
        print("Target RMS sequence shape:", rms_target.shape)
        print("Normalized parameter vector:", rms_params)
