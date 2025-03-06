import sys
import os
import glob
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset  # Capital "D" is required
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

print("Python executable:", sys.executable)
def compute_mel_spectrogram(audio, sr, n_mels=128, n_fft=2048, hop_length=512):
    # Compute the mel spectrogram and convert to dB scale.
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, 
                                            hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def compute_rms(audio, frame_length=2048, hop_length=512):
    # Compute RMS values for each frame and convert to dB.
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = 10 * np.log10(rms + 1e-10)
    return rms_db

class PairedAudioDataset(Dataset):
    def __init__(self, audio_dir, sr=None, transform=None, mode="spectrogram"):
        """
        Args:
        audio_dir (str): Directory where paired audio files are stored.
                        Expects files named like <song_id>_original.wav and <song_id>_modified.wav.
        sr (int or None): Sampling rate to resample audio; if None, the original sample rate is used.
        transform (callable, optional): A function to compute features (e.g., mel-spectrogram).
        mode (str): "spectrogram" for a time-frequency representation,
                    "rms" for a time series of RMS values.
        """
        self.audio_dir = audio_dir
        # Use a wildcard pattern to match any file ending in _original.wav.
        self.original_files = glob.glob(os.path.join(audio_dir, "*_original.wav"))
        self.sr = sr
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.original_files)
    
    def __getitem__(self, idx):
        # Get the original file path.
        orig_path = self.original_files[idx]
        # Derive the corresponding modified file (demastered) path.
        song_id = os.path.basename(orig_path).split("_")[0]
        mod_path = os.path.join(self.audio_dir, f"{song_id}_modified.wav")
        
        # Load audio using librosa.
        original, sr_orig = librosa.load(orig_path, sr=self.sr)
        modified, sr_mod = librosa.load(mod_path, sr=self.sr)
        
        if self.mode == "spectrogram":
            # Compute spectrogram features.
            if self.transform:
                original = self.transform(original, sr_orig)
                modified = self.transform(modified, sr_mod)
            else:
                original = compute_mel_spectrogram(original, sr_orig)
                modified = compute_mel_spectrogram(modified, sr_mod)
        elif self.mode == "rms":
            # Compute RMS level features.
            original = compute_rms(original)
            modified = compute_rms(modified)
        else:
            raise ValueError("Mode must be 'spectrogram' or 'rms'")
        
        # Convert features to torch tensors.
        if self.mode == "spectrogram":
            # Add a channel dimension for 2D spectrograms: [channel, frequency, time]
            original = torch.tensor(original, dtype=torch.float32).unsqueeze(0)
            modified = torch.tensor(modified, dtype=torch.float32).unsqueeze(0)
        else:
            original = torch.tensor(original, dtype=torch.float32)
            modified = torch.tensor(modified, dtype=torch.float32)
        
        # Return a tuple (input, target): degraded input and mastered target.
        return modified, original

# Test usage:
if __name__ == "__main__":
    # Adjust the path as needed; ensure you're running this with your thesis_env activated.
    dataset = PairedAudioDataset(audio_dir="../experiments/output_full/output_audio", 
                                sr=44100, transform=compute_mel_spectrogram, mode="spectrogram")
    print(f"Dataset length: {len(dataset)}")
    sample_input, sample_target = dataset[0]
    print("Input feature shape:", sample_input.shape)
    print("Target feature shape:", sample_target.shape)
