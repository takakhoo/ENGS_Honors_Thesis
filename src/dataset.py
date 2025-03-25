import sys
import os
import glob
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset  # Use the correct, capitalized Dataset
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

print("Python executable:", sys.executable)

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
        
        #Loading Audio
        original, sr_orig = librosa.load(orig_path, sr=self.sr)
        modified, sr_mod = librosa.load(mod_path, sr=self.sr)
        
        if self.mode == "spectrogram":
            #Computing melspectrogram features
            if self.transform:
                original = self.transform(original, sr_orig)
                modified = self.transform(modified, sr_mod)
            else:
                original = compute_mel_spectrogram(original, sr_orig)
                modified = compute_mel_spectrogram(modified, sr_mod)
        elif self.mode == "rms":
            original = compute_rms(original)
            modified = compute_rms(modified)
        elif self.mode == "rms_segmented":
            if self.segment_duration is None:
                raise ValueError("segment_duration must be provided for 'rms_segmented' mode")
            original = compute_segmented_rms(original, sr_orig, segment_duration=self.segment_duration)
            modified = compute_segmented_rms(modified, sr_mod, segment_duration=self.segment_duration)
        else:
            raise ValueError("Mode must be 'spectrogram' or 'rms'")
        
        #Converting to torch tensors
        if self.mode == "spectrogram": #To add channel dimension ]channel, freq, time]
            original = torch.tensor(original, dtype=torch.float32).unsqueeze(0)
            modified = torch.tensor(modified, dtype=torch.float32).unsqueeze(0)
        else:
            original = torch.tensor(original, dtype=torch.float32)
            modified = torch.tensor(modified, dtype=torch.float32)
        
        #Return the tuple (input, target)
        return modified, original

# Test usage:
if __name__ == "__main__":
    # To test spectrogram mode:
    dataset_spec = PairedAudioDataset(
        audio_dir="experiments/output_full/output_audio", 
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
