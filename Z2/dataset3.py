#!/usr/bin/env python3
"""
dataset.py  –  Data loading & preprocessing (22 kHz • 80‑mel • hop 256)

Think of this as the recipe book for our mastering network:

  1. load *paired* WAV files  ( *_original.wav  +  *_modified.wav )
  2. turn them into pictures of sound (mel‑spectrograms) **or** RMS curves
  3. scale everything to friendly ranges
  4. read the “recipe cards” (10‑value parameter text files) and normalise them

All of the explanatory comments and parameter‑handling logic from the
previous 44 kHz / 128‑mel version are retained – only the signal settings
have changed so the data lines up with the 22 kHz HiFi‑GAN cloned.

Key DSP choices
---------------
* **Sample‑rate**……………… 22 050 Hz   (exactly half of 44 100 Hz)
* **FFT size**………………… 2 048 samples  (92 ms analysis window)
* **Hop size**…………………   256 samples  (11.6 ms stride)   ← matches HiFi‑GAN
* **Mel‑bins**…………………      80 bands  (same as HiFi‑GAN)
"""

import os, glob, sys
from typing import Tuple, List, Optional

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

# ───────────────────────────── constants ────────────────────────────── #

DEFAULT_SR          = 22_050         # 22 kHz
DEFAULT_N_FFT       = 2_048
DEFAULT_HOP         =   256
DEFAULT_N_MELS      =    80
DEFAULT_FRAME_LEN   = 2_048          # for RMS if needed
PARAM_VECTOR_LEN    =     10

# add project root to path so downstream imports work when testing
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ─────────────────── helper: compute mel spectrogram ────────────────── #

def compute_mel_spectrogram(
        audio: np.ndarray,
        sr: int             = DEFAULT_SR,
        n_mels: int         = DEFAULT_N_MELS,
        n_fft:  int         = DEFAULT_N_FFT,
        hop_length: int     = DEFAULT_HOP
    ) -> np.ndarray:
    """
    Turn raw PCM → mel power → dB    (shape [mel, T])

    *Human‑centric*: the mel filter‑bank warps frequency so low notes get
    finer spacing – closer to how we actually hear.

    Returned values are **linear dB** (≈‑80 dB … 0 dB), not yet normalised.
    """
    mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
          )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

# ─────────────────── helper: compute RMS loudness curve ─────────────── #

def compute_rms(
        audio: np.ndarray,
        frame_length: int = DEFAULT_FRAME_LEN,
        hop_length:   int = DEFAULT_HOP
    ) -> np.ndarray:
    """
    “How loud is the signal?” – classic RMS envelope in decibels.
    """
    rms = librosa.feature.rms(y=audio,
                              frame_length=frame_length,
                              hop_length=hop_length)[0]
    return 10.0 * np.log10(rms + 1e-10)          # dB scale

def compute_segmented_rms(
        audio: np.ndarray,
        sr: int,
        segment_duration: float = 0.10
    ) -> np.ndarray:
    """
    Average RMS every *segment_duration* seconds (default 100 ms).
    """
    seg_len = int(segment_duration * sr)
    rms_vals: List[float] = []
    for i in range(0, len(audio), seg_len):
        seg = audio[i:i+seg_len]
        if seg.size == 0:
            continue
        rms = np.sqrt(np.mean(np.square(seg)))
        rms_vals.append(10.0 * np.log10(rms + 1e-10))
    return np.asarray(rms_vals, dtype=np.float32)

# ───────────────────────── dataset class ─────────────────────────────── #

class PairedAudioDataset(Dataset):
    """
    Loads *paired* files:

        123456_original.wav     clean
        123456_modified.wav     degraded / “demastered”

    plus  ➜  123456_params.txt  (10 processing parameters)

    Returns
    -------
    modified_tensor : torch.FloatTensor  [1, 80, T]   (or RMS curve)
    original_tensor : torch.FloatTensor  [1, 80, T]
    param_vector    : torch.FloatTensor  [10]
    """

    def __init__(self,
                 audio_dir: str,
                 sr: int                = DEFAULT_SR,
                 n_fft: int             = DEFAULT_N_FFT,
                 hop_length: int        = DEFAULT_HOP,
                 n_mels: int            = DEFAULT_N_MELS,
                 mode: str              = "spectrogram",
                 segment_duration: Optional[float] = None):
        super().__init__()

        self.audio_dir       = audio_dir
        self.sr              = sr
        self.n_fft           = n_fft
        self.hop_length      = hop_length
        self.n_mels          = n_mels
        self.mode            = mode
        self.segment_duration= segment_duration

        self.original_files  = sorted(glob.glob(os.path.join(audio_dir, "*_original.wav")))
        print(f"[Dataset] {len(self.original_files)} pairs found in {audio_dir}")

        # parameter files live in   …/experiments/output_full/output_txt/
        self.param_dir = os.path.join(os.path.dirname(__file__),
                                      "..", "experiments", "output_full", "output_txt")

    # ───────────────────────── Dataset protocol ────────────────────── #

    def __len__(self) -> int:
        return len(self.original_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # --------------------------------------------------------------
        # 1) Locate the  pair of wavs  +  parameter file
        # --------------------------------------------------------------
        orig_path = self.original_files[idx]
        song_id   = os.path.basename(orig_path).split("_")[0]
        mod_path  = os.path.join(self.audio_dir, f"{song_id}_modified.wav")
        param_path= os.path.join(self.param_dir,  f"{song_id}_params.txt")

        # --------------------------------------------------------------
        # 2) Load audio (mono)  •  clip ⇒ [-1,1]
        # --------------------------------------------------------------
        try:
            original, _ = librosa.load(orig_path, sr=self.sr, mono=True)
            modified, _ = librosa.load(mod_path, sr=self.sr, mono=True)
        except Exception as e:
            print(f"[Dataset] ⚠️  load error for {song_id}: {e}")
            return self._zero(), self._zero(), torch.zeros(PARAM_VECTOR_LEN)

        original = np.clip(original, -1.0, 1.0).astype(np.float32)
        modified = np.clip(modified, -1.0, 1.0).astype(np.float32)

        # --------------------------------------------------------------
        # 3) Representation  (spectrogram or RMS)
        # --------------------------------------------------------------
        if self.mode == "spectrogram":
            orig_rep = self._process_mel(original, tag=f"{song_id}-orig")
            mod_rep  = self._process_mel(modified, tag=f"{song_id}-mod")
        elif self.mode == "rms":
            orig_rep = torch.from_numpy(compute_rms(original)).unsqueeze(0)
            mod_rep  = torch.from_numpy(compute_rms(modified )).unsqueeze(0)
        elif self.mode == "rms_segmented":
            if self.segment_duration is None:
                raise ValueError("segment_duration must be set for rms_segmented mode")
            orig_rep = torch.from_numpy(compute_segmented_rms(original, self.sr,
                                                              self.segment_duration)).unsqueeze(0)
            mod_rep  = torch.from_numpy(compute_segmented_rms(modified, self.sr,
                                                              self.segment_duration)).unsqueeze(0)
        else:
            raise ValueError(f"Invalid mode '{self.mode}'")

        # --------------------------------------------------------------
        # 4) Parameter vector  (10 values, each ∈ [0,1])
        # --------------------------------------------------------------
        if not os.path.exists(param_path):
            print(f"[Dataset] ⚠️  {param_path} missing – using zeros.")
            params = torch.zeros(PARAM_VECTOR_LEN)
        else:
            params = self._parse_params(param_path)

        return mod_rep, orig_rep, params

    # ─────────────────────── helpers / internals ───────────────────── #

    def _process_mel(self, y: np.ndarray, *, tag: str) -> torch.Tensor:
        """
        PCM → 80‑mel dB → 0‑to‑1 → torch[1,80,T]
        """
        mel_db = compute_mel_spectrogram(
                    y, sr=self.sr,
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                 )
        lo, hi = mel_db.min(), mel_db.max()
        norm   = (mel_db - lo) / (hi - lo + 1e-6)
        t = torch.from_numpy(norm).float().unsqueeze(0)

        # Debug prints (keep short so the console isn’t spammed every batch)
        print(f"[Dataset] {tag:>12s}  mel→shape {t.shape}  dB‑range ({lo:.1f}, {hi:.1f})")

        return t

    def _parse_params(self, path: str) -> torch.Tensor:
        """
        Read a *_params.txt file and squash every value into [0,1].

        The mapping ranges are identical to the previous 44 kHz version,
        so anything learned about parameter statistics continues to apply.
        """
        raw: dict[str, float] = {}
        with open(path, "r", encoding="utf‑8") as f:
            for line in f:
                if not line.strip():
                    continue
                head, body = line.split(":", 1)
                items = [p.strip() for p in body.split(",")]
                for item in items:
                    k, v = item.split("=")
                    key  = k.strip()
                    val  = float(v.strip().split()[0])
                    raw[key] = val

        # ---- pull values (fall back to 0) ----
        gain_db      = raw.get("gain_db",           0.0)
        eq_fc        = raw.get("fc",                0.0)   # centre Hz
        eq_Q         = raw.get("Q",                 0.0)
        eq_gain_db   = raw.get("eq_gain_db",        0.0)
        comp_thresh  = raw.get("threshold_db",      0.0)
        comp_ratio   = raw.get("ratio",             0.0)
        comp_makeup  = raw.get("makeup_gain_db",    0.0)
        rev_decay    = raw.get("decay",             0.0)
        echo_delay   = raw.get("delay_seconds",     0.0)
        echo_atten   = raw.get("attenuation",       0.0)

        # ---- same normalisation rules as before ----
        sr = self.sr or DEFAULT_SR
        vec = [
            (gain_db + 1.0) / 2.0,             # [-1,1] dB  → [0,1]
            eq_fc / (sr/2),                    #   0‥Nyquist → [0,1]
            (eq_Q - 0.1)   / 9.9,              #   0.1‥10    → [0,1]
            (eq_gain_db+10)/20.0,              # [-10,10]    → [0,1]
            (comp_thresh+60)/60.0,             # [-60,0]     → [0,1]
            (comp_ratio -1)/19.0,              #   1‥20      → [0,1]
            comp_makeup/20.0,                  #   0‥20 dB   → [0,1]
            (rev_decay  -0.1)/9.9,             #   0.1‥10    → [0,1]
            echo_delay/100.0,                  #   0‥100 s   → [0,1]
            echo_atten                         # already 0‥1
        ]
        v = torch.tensor(vec, dtype=torch.float32)
        assert v.shape[0] == PARAM_VECTOR_LEN
        return v

    @staticmethod
    def _zero() -> torch.Tensor:
        return torch.zeros(1, dtype=torch.float32)

# ──────────────────────────── quick test ────────────────────────────── #

if __name__ == "__main__":
    """
    Run > python dataset.py  to sanity‑check that shapes look right.
    """
    PRJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    AUDIO_DIR = os.path.join(PRJ, "experiments", "output_full", "output_audio")

    ds = PairedAudioDataset(
            audio_dir=AUDIO_DIR,
            sr=DEFAULT_SR,
            mode="spectrogram"
         )
    print(f"\n[Test] Dataset length = {len(ds)}")
    if len(ds):
        mod, orig, params = ds[0]
        print(f"[Test] mod  shape {mod.shape}")
        print(f"[Test] orig shape {orig.shape}")
        print(f"[Test] params → {params}")
