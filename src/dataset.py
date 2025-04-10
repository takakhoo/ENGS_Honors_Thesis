import sys
import os
import glob
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset  # Use the correct, capitalized Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def compute_mel_spectrogram(audio, sr, n_mels=128, n_fft=2048, hop_length=512):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, 
                                                hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def compute_rms(audio, frame_length=2048, hop_length=512):
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = 10 * np.log10(rms + 1e-10)
    return rms_db

def compute_segmented_rms(audio, sr, segment_duration=0.1):
    #This will compute RMS values in dB for short segments of the AUDIO
    segment_length = int(segment_duration *sr)
    #Using a hop length equal to seg length for non-overlapping segments
    rms_vals=[]
    for i in range(0, len(audio), segment_length):
        segment = audio[i : i+segment_length]
        if len(segment)==0:
            continue
        rms = np.sqrt(np.mean(segment**2))
        rms_db = 10 * np.log10(rms + 1e-10)
        rms_vals.append(rms_db)
    return np.array(rms_vals)

class PairedAudioDataset(Dataset):
    def __init__(self, audio_dir, sr=None, transform=None, mode="spectrogram", segment_duration=None):
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
        print("Found original files:", self.original_files)  # Debug print
        self.sr = sr
        self.transform = transform
        self.mode = mode
        self.segment_duration = segment_duration
        
    def __len__(self):
        return len(self.original_files)
    
    def __getitem__(self, idx):
        orig_path = self.original_files[idx]
        song_id = os.path.basename(orig_path).split("_")[0]
        mod_path = os.path.join(self.audio_dir, f"{song_id}_modified.wav")
        
        try:
            #Loading Audio
            original, sr_orig = librosa.load(orig_path, sr=self.sr, mono=True)
            modified, sr_mod = librosa.load(mod_path, sr=self.sr, mono=True)
        except Exception as e:
            print(f"Error loading {orig_path}: {e}")
            return self._empty_tensor()
        
        if np.isnan(original).any() or np.isinf(original).any():
            print(f"NaN/Inf detected in {orig_path}, using zeros")
            return self._empty_tensor()
        
        if np.isnan(modified).any() or np.isinf(modified).any():
            print(f"NaN/Inf detected in {orig_path}, using zeros")
            return self._empty_tensor()
        
        #Clipping audio to valid range
        original = np.clip(original, -1.0, 1.0)
        modified = np.clip(modified, -1.0, 1.0)
        
        #Feature extraction
        if self.mode == "spectrogram":
            original = self._process_spectrogram(original, sr_orig)
            modified = self._process_spectrogram(modified, sr_mod)
        elif self.mode == "rms":
            original = compute_rms(original)
            modified = compute_rms(modified)
        elif self.mode == "rms_segmented":
            if not self.segment_duration:
                raise ValueError("Mode must be 'spectrogram' or 'rms'")
            original = compute_segmented_rms(original, sr_orig, self.segment_duration)
            modified = compute_segmented_rms(modified, sr_mod, self.segment_duration)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        

        return modified, original
        
    def _process_spectrogram(self, audio, sr):
        """Process audio into a mel spectrogram with correct shape and normalization."""
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        
        # Convert to dB scale
        mel_spec = librosa.power_to_db(mel_spec + 1e-6, ref=np.max)
        
        # Convert to tensor and ensure shape [1, freq, time]
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
        if len(mel_spec.shape) == 2:
            mel_spec = mel_spec.unsqueeze(0)  # Add channel dimension
        
        # Normalize to [0, 1] range
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)
        
        return mel_spec
        
# Test usage:
if __name__ == "__main__":
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Go two levels up to the project root, then into experiments/output_full/output_audio
    AUDIO_DIR = os.path.join(CURRENT_DIR, "..", "experiments", "output_full", "output_audio")
    
    # To test spectrogram mode:
    dataset_spec = PairedAudioDataset(
        audio_dir=AUDIO_DIR, 
        sr=44100, 
        transform=compute_mel_spectrogram, 
        mode="spectrogram"
    )
    print(f"Spectrogram mode, dataset length: {len(dataset_spec)}")
    if len(dataset_spec) > 0:
        sample_input, sample_target = dataset_spec[0]
        print("Input spectrogram shape:", sample_input.shape)
        print("Target spectrogram shape:", sample_target.shape)
    
    # To test segmented RMS mode:
    dataset_rms_seg = PairedAudioDataset(
        audio_dir="experiments/output_full/output_audio", 
        sr=44100, 
        mode="rms_segmented", 
        segment_duration=0.1  # 100 ms segments
    )
    print(f"Segmented RMS mode, dataset length: {len(dataset_rms_seg)}")
    if len(dataset_rms_seg) > 0:
        rms_input, rms_target = dataset_rms_seg[0]
        print("Input RMS sequence shape:", rms_input.shape)
        print("Target RMS sequence shape:", rms_target.shape)
