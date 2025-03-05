import os
import glob
import librosa
import torch
import torch.utils.data 
import Dataset

class PairedAudioDataset(Dataset):
    def __init__(self, audio_dir, sr=None, transform=None):
        """
        audio_dir: Directory where paired audio files are stored.
        Expects files named like <song_id>_original.wav and <song_id>_modified.wav.
        sr: Optional sampling rate to resample audio.
        transform: A callable transformation (e.g., compute spectrogram) applied to both audio files.
        """
        self.audio_dir = audio_dir
        self.original_files = glob.glob(os.path.join(audio_dir, "_original.wav"))
        self.sr=sr
        self.transform = transform
        
    def __len__(self):
        return len(self.original_files)
    
    def __getitem__(self, idx):
        orig_path = self.original_files[idx] #Getting OG File Path
        #Derive the corr modified (demastered) file path
        song_id = os.path.basename(orig_path).split("_")[0]
        mod_path = os.path.join(self.audio_dir, f"{song_id}_modified.wav")
        
        #Load audio using librosa
        original, sr_orig = librosa.load(orig_path, sr=self.sr)
        modified, sr_mod = librosa.load(mod_path, sr=self.sr)
        
        #Optionally used, we will see conv raw audio to mel spectro
        if self.transform:
            original = self.transform(original, sr_orig)
            modified = self.transform(modified, sr_mod)
        
        #Returning our tuple (input, target)
        return torch.tensor(modified, dtype=torch.float32), torch.tensor(original, dtype=torch.float32)
    
    def compute_mel_spectrogram(audio, sr, n_mels=128, n_fft=2048, hop_length=512):
        import librosa
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        #Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
