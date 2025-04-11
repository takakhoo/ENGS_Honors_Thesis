"""
dataset.py

This file handles all the data loading and preprocessing for our audio mastering model.
It loads paired audio files (original and modified), converts them to mel spectrograms (or RMS as needed),
and also parses associated parameter files.
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
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft,
                                                hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def compute_rms(audio, frame_length=2048, hop_length=512):
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = 10 * np.log10(rms + 1e-10)
    return rms_db

def compute_segmented_rms(audio, sr, segment_duration=0.1):
    segment_length = int(segment_duration * sr)
    rms_vals = []
    for i in range(0, len(audio), segment_length):
        segment = audio[i:i+segment_length]
        if len(segment) == 0:
            continue
        rms = np.sqrt(np.mean(segment**2))
        rms_db = 10 * np.log10(rms + 1e-10)
        rms_vals.append(rms_db)
    return np.array(rms_vals)

class PairedAudioDataset(Dataset):
    def __init__(self, audio_dir, sr=None, transform=None, mode="spectrogram", segment_duration=None):
        """
        Args:
            audio_dir (str): Directory with paired audio files (_original.wav and _modified.wav)
            sr (int or None): Sample rate to load audio. If None, original sampling rate is used.
            transform: Function to compute features (e.g., compute_mel_spectrogram)
            mode (str): "spectrogram" or "rms" or "rms_segmented"
            segment_duration: Duration in seconds for segmented RMS mode.
        """
        self.audio_dir = audio_dir
        self.original_files = glob.glob(os.path.join(audio_dir, "*_original.wav"))
        print("Found original files:", self.original_files)
        self.sr = sr
        self.transform = transform
        self.mode = mode
        self.segment_duration = segment_duration
        
        # Parameter files directory
        self.param_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "output_full", "output_txt")
        
    def __len__(self):
        return len(self.original_files)
    
    def __getitem__(self, idx):
        orig_path = self.original_files[idx]
        song_id = os.path.basename(orig_path).split("_")[0]
        mod_path = os.path.join(self.audio_dir, f"{song_id}_modified.wav")
        try:
            original, sr_orig = librosa.load(orig_path, sr=self.sr, mono=True)
            modified, sr_mod = librosa.load(mod_path, sr=self.sr, mono=True)
        except Exception as e:
            print(f"Error loading {orig_path}: {e}")
            return self._empty_tensor(), self._empty_tensor(), torch.zeros(10, dtype=torch.float32)
        
        if np.isnan(original).any() or np.isinf(original).any():
            print(f"NaN/Inf in {orig_path}; using zeros.")
            return self._empty_tensor(), self._empty_tensor(), torch.zeros(10, dtype=torch.float32)
        if np.isnan(modified).any() or np.isinf(modified).any():
            print(f"NaN/Inf in {mod_path}; using zeros.")
            return self._empty_tensor(), self._empty_tensor(), torch.zeros(10, dtype=torch.float32)
        
        original = np.clip(original, -1.0, 1.0)
        modified = np.clip(modified, -1.0, 1.0)
        
        if self.mode == "spectrogram":
            original_spec = self._process_spectrogram(original, sr_orig)
            modified_spec = self._process_spectrogram(modified, sr_mod)
        elif self.mode == "rms":
            original_spec = compute_rms(original)
            modified_spec = compute_rms(modified)
        elif self.mode == "rms_segmented":
            if not self.segment_duration:
                raise ValueError("Set segment_duration for rms_segmented mode.")
            original_spec = compute_segmented_rms(original, sr_orig, self.segment_duration)
            modified_spec = compute_segmented_rms(modified, sr_mod, self.segment_duration)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        param_file = os.path.join(self.param_dir, f"{song_id}_params.txt")
        if not os.path.exists(param_file):
            print(f"Parameter file missing: {param_file}. Using zeros.")
            param_vector = torch.zeros(10, dtype=torch.float32)
        else:
            param_vector = self._parse_parameter_file(param_file)
        return modified_spec, original_spec, param_vector
    
    def _process_spectrogram(self, audio, sr):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mel_spec = librosa.power_to_db(mel_spec + 1e-6, ref=np.max)
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
        if len(mel_spec.shape) == 2:
            mel_spec = mel_spec.unsqueeze(0)
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)
        return mel_spec
    
    def _parse_parameter_file(self, filepath):
        with open(filepath, "r") as f:
            lines = f.read().splitlines()
        params = {}
        for line in lines:
            if line.startswith("EQ:"):
                parts = line.split(":")[1].strip().split(",")
                for p in parts:
                    key_val = p.strip().split("=")
                    key = key_val[0].strip()
                    val = float(key_val[1].split()[0].strip())
                    params[f"eq_{key}"] = val
            elif line.startswith("Gain:"):
                p = line.split(":")[1].strip()
                key_val = p.split("=")
                params["gain_db"] = float(key_val[1].strip())
            elif line.startswith("Echo:"):
                parts = line.split(":")[1].strip().split(",")
                for p in parts:
                    key_val = p.strip().split("=")
                    params[key_val[0].strip()] = float(key_val[1].strip())
            elif line.startswith("Reverb:"):
                parts = line.split(":")[1].strip().split(",")
                for p in parts:
                    key_val = p.strip().split("=")
                    if key_val[0].strip() == "decay":
                        params["decay"] = float(key_val[1].strip())
            elif line.startswith("Compression:"):
                parts = line.split(":")[1].strip().split(",")
                for p in parts:
                    key_val = p.strip().split("=")
                    params[key_val[0].strip()] = float(key_val[1].strip())
        
        sr = self.sr if self.sr else 44100
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
        
        norm_gain = (gain + 1) / 2.0
        norm_eq_center = eq_center / (sr/2)
        norm_eq_Q = (eq_Q - 0.1)/9.9
        norm_eq_gain = (eq_gain + 10)/20.0
        norm_comp_thresh = (comp_thresh + 60) / 60.0
        norm_comp_ratio = (comp_ratio - 1) / 19.0
        norm_comp_makeup = comp_makeup/20.0
        norm_reverb_decay = (reverb_decay - 0.1)/9.9
        norm_echo_delay = echo_delay/100.0
        norm_echo_atten = echo_atten
        vector = [norm_gain, norm_eq_center, norm_eq_Q, norm_eq_gain,
                  norm_comp_thresh, norm_comp_ratio, norm_comp_makeup,
                  norm_reverb_decay, norm_echo_delay, norm_echo_atten]
        return torch.tensor(vector, dtype=torch.float32)
    
    def _empty_tensor(self):
        return torch.zeros(1, dtype=torch.float32)

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    AUDIO_DIR = os.path.join(CURRENT_DIR, "..", "experiments", "output_full", "output_audio")
    dataset_spec = PairedAudioDataset(audio_dir=AUDIO_DIR, sr=44100, transform=compute_mel_spectrogram, mode="spectrogram")
    print(f"Spectrogram mode, dataset length: {len(dataset_spec)}")
    if len(dataset_spec) > 0:
        sample_input, sample_target, sample_params = dataset_spec[0]
        print("Input spectrogram shape:", sample_input.shape)
        print("Target spectrogram shape:", sample_target.shape)
        print("Normalized parameter vector:", sample_params)
