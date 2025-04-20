"""
dataset.py

This file handles all the data loading and preprocessing for our audio mastering model.
Think of it like a recipe book - it tells us how to prepare our audio data before feeding it to the model.

Key concepts:
- Mel spectrograms: A way to represent audio that's more similar to how humans hear sound
- RMS (Root Mean Square): A measure of audio energy over time
- Normalization: Scaling our data to make it easier for the model to learn
- Parameter parsing: Reading and normalizing audio processing parameters

we handle:
1. loading pairs of audio files (before/after mastering)
2. turning audio into pictures (spectrograms) or volume measurements (RMS)
3. making all our numbers behave (normalization)
4. reading the mastering "recipes" (parameter files)
"""

import os
import glob
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

# gotta add the project root so we can import stuff from other folders
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def compute_mel_spectrogram(audio, sr, n_mels=128, n_fft=2048, hop_length=512):
    """
    turns raw audio into a mel spectrogram - think of it like taking a photo of sound!
    
    why mel scale? our ears are weird - we hear low notes better than high notes
    mel scale warps the frequencies to match how we actually hear stuff
    
    Args:
        audio: Raw audio signal
        sr: Sample rate
        n_mels: Number of mel bands (like frequency bins, but warped to match human hearing)
        n_fft: Size of the FFT window (bigger = better frequency resolution, but worse time resolution)
        hop_length: How much we slide the window each time (smaller = more time resolution)
    
    Returns:
        a mel spectrogram in dB (decibels) - basically a picture of the sound
    """
    # first we make a regular spectrogram, then warp it to match human hearing
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, 
                                                hop_length=hop_length, n_mels=n_mels)
    # convert to dB because our ears hear sound logarithmically (like how we measure earthquakes)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def compute_rms(audio, frame_length=2048, hop_length=512):
    """
    Compute the Root Mean Square (RMS) energy of the audio signal.
    This is like measuring how "loud" the audio is over time.
    
    Args:
        audio: the raw sound wave
        frame_length: how many samples to look at at once (like taking a snapshot)
        hop_length: how much to slide the window each time (like frames in a video)
    
    Returns:
        RMS values in dB - basically a list of how loud the sound is at each moment
    """
    # compute RMS using librosa's built-in function (saves us from doing the math)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    # convert to dB (logarithmic scale) because that's how we hear loudness
    # adding 1e-10 to avoid log(0) which would make the computer cry
    rms_db = 10 * np.log10(rms + 1e-10)
    return rms_db

def compute_segmented_rms(audio, sr, segment_duration=0.1):
    """
    measures the average loudness every 100ms - like checking the volume every few seconds
    
    Args:
        audio: the raw sound wave
        sr: samples per second
        segment_duration: how long each chunk should be in seconds
    
    Returns:
        array of RMS values for each time chunk
    """
    # convert seconds to samples (like converting minutes to seconds)
    segment_length = int(segment_duration * sr)
    rms_vals = []
    
    # chop the audio into pieces and measure each one
    for i in range(0, len(audio), segment_length):
        segment = audio[i:i+segment_length]
        if len(segment) == 0:
            continue
        # RMS formula: sqrt(mean(samples²)) - basically average of squared values, then square root
        rms = np.sqrt(np.mean(segment**2))
        # Convert to dB
        rms_db = 10 * np.log10(rms + 1e-10)
        rms_vals.append(rms_db)
    
    return np.array(rms_vals)

class PairedAudioDataset(Dataset):
    """
    this is where the magic happens! we load pairs of audio files:
    - original (clean) audio
    - modified (processed) audio
    
    It's like having before/after photos, but for sound.
    
    we provide:
    1. spectrogram/RMS representations (like photos of the sound)
    2. normalized processing parameters (the "recipe" for how to process the audio)
    3. proper batching for training (like preparing multiple meals at once)
    """
    def __init__(self, audio_dir, sr=None, transform=None, mode="spectrogram", segment_duration=None):
        """
        Initialize our dataset.
        
        Args:
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
        """Returns the number of audio pairs in the dataset."""
        return len(self.original_files)
    
    def __getitem__(self, idx):
        """
        Get one pair of audio files and their parameters.
        This is what PyTorch calls when it needs a training example.
        
        Args:
            idx: Index of the audio pair to retrieve
        
        Returns:
            Tuple of (modified_spec, original_spec, param_vector)
            - modified_spec: Spectrogram/RMS of processed audio
            - original_spec: Spectrogram/RMS of original audio
            - param_vector: Normalized processing parameters
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
        turns audio into a mel spectrogram and normalizes it
        like converting a photo to black and white and adjusting the contrast
        
        Steps:
        1. compute mel spectrogram (like taking a photo)
        2. convert to dB scale (like adjusting brightness)
        3. convert to tensor (like saving the photo)
        4. add channel dimension (like adding color channels)
        5. normalize to [0,1] range (like adjusting contrast)
        
        Args:
            audio: raw audio signal
            sr: sample rate
        
        Returns:
            Normalized mel spectrogram tensor
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        # convert to dB scale (logarithmic, like how we hear)
        mel_spec = librosa.power_to_db(mel_spec + 1e-6, ref=np.max)
        # Convert to PyTorch tensor
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
        # Add channel dimension if needed
        if len(mel_spec.shape) == 2:
            mel_spec = mel_spec.unsqueeze(0)
        # normalize to [0,1] range (makes training easier)
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)
        return mel_spec
    
    def _parse_parameter_file(self, filepath):
        """
        Read and normalize audio processing parameters from a text file.
        This is like reading a recipe and converting all measurements to the same units.
        
        The parameters include:
        - Gain: overall volume adjustment (like turning up/down the volume)
        - EQ: equalization settings (like bass/treble controls)
        - Compression: dynamic range control (like auto-volume)
        - Reverb: room simulation (like adding echo)
        - Echo: delay effects (like repeating the sound)
        
        Args:
            filepath: Path to the parameter file
        
        Returns:
            Normalized parameter vector (10 values in [0,1] range)
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
                # parse gain parameter (like volume knob)
                p = line.split(":")[1].strip()
                key_val = p.split("=")
                params["gain_db"] = float(key_val[1].strip())
            elif line.startswith("Echo:"):
                # parse echo parameters (like delay settings)
                parts = line.split(":")[1].strip().split(",")
                for p in parts:
                    key_val = p.strip().split("=")
                    params[key_val[0].strip()] = float(key_val[1].strip())
            elif line.startswith("Reverb:"):
                # parse reverb parameters (like room size)
                parts = line.split(":")[1].strip().split(",")
                for p in parts:
                    key_val = p.strip().split("=")
                    if key_val[0].strip() == "decay":
                        params["decay"] = float(key_val[1].strip())
            elif line.startswith("Compression:"):
                # parse compression parameters (like auto-volume settings)
                parts = line.split(":")[1].strip().split(",")
                for p in parts:
                    key_val = p.strip().split("=")
                    params[key_val[0].strip()] = float(key_val[1].strip())
        
        # get sample rate (default to 44.1kHz if not specified)
        sr = self.sr if self.sr else 44100
        
        # extract all parameters with defaults
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
        
        # normalize all parameters to [0,1] range (makes training easier)
        # this is like converting all measurements to the same units
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
        """
        Creates an empty tensor for error cases.
        Used when audio loading fails or contains invalid values.
        
        Returns:
            zero tensor with appropriate shape
        """
        return torch.zeros(1, dtype=torch.float32)

if __name__ == "__main__":
    # This part is just for testing - it shows us what our processed data looks like
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    AUDIO_DIR = os.path.join(CURRENT_DIR, "..", "experiments", "output_full", "output_audio")
    
    # try loading some data in spectrogram mode
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
    
    # try loading some data in segmented RMS mode
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
