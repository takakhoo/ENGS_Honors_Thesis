#!/usr/bin/env python3
"""
evaluate.py – MasterNet + HiFi‑GAN end‑to‑end evaluator
=======================================================

Runs the first 10 *_modified.wav through MasterNet + HiFi‑GAN (or
Griffin‑Lim), saving spectrogram images, parameter stats, and
signed 16‑bit PCM WAVs via Python's wave module.
"""

import sys
import os
import json
import shutil
import argparse
import logging
import wave
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from dataset import compute_mel_spectrogram
from models  import MasterNet

# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("Evaluate")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-5s: %(message)s",
                                       datefmt="%H:%M:%S"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MasterNet + HiFi‑GAN")
    p.add_argument("--input_dir",
                   default=str(PROJECT_ROOT/"experiments"/"output_full"/"output_audio"),
                   help="Where *_modified.wav lives")
    # Use specific epoch checkpoint
    p.add_argument("--unet_ckpt", 
                   default=str(PROJECT_ROOT/"MasterNetTraining"/"2025-04-18_04-16-28"/"checkpoints"/"epoch_112.pt"),
                   help="MasterNet checkpoint")
    p.add_argument("--hifi_clone", default=str(PROJECT_ROOT/"vocoder"/"hifi-gan"),
                   help="HiFi‑GAN repo root")
    p.add_argument("--hifi_weights",
                   default=str(PROJECT_ROOT/"vocoder"/"checkpoints"/"generator_universal_v1.pth"),
                   help="HiFi‑GAN universal weights")
    p.add_argument("--output_root", default=str(PROJECT_ROOT/"MasterNetEval"),
                   help="Where to dump results")
    p.add_argument("--sr",     type=int, default=22050)
    p.add_argument("--n_fft",  type=int, default=2048)
    p.add_argument("--hop",    type=int, default=256)
    p.add_argument("--n_mels", type=int, default=80)
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
def setup_dirs(base: Path):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = base / ts
    dirs = {k: root / k for k in ("checkpoints","images","audio","params")}
    dirs["root"] = root
    for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)
    return dirs

# ─────────────────────────────────────────────────────────────────────────────
def mel_from_wav(path, sr, n_fft, hop, n_mels):
    y,_ = librosa.load(path, sr=sr, mono=True)
    y   = np.clip(y, -1, 1)
    mel = librosa.feature.melspectrogram(y=y, sr=sr,
                                         n_fft=n_fft,
                                         hop_length=hop,
                                         n_mels=n_mels)
    mel_db = librosa.power_to_db(mel+1e-6, ref=np.max)
    lo, hi = mel_db.min(), mel_db.max()
    mel_norm = (mel_db - lo)/(hi-lo+1e-6)
    return mel_norm, mel_db, lo, hi, y

def denorm(norm, lo, hi):
    return norm*(hi-lo)+lo

def save_spec(spec, path:Path, title, sr, hop):
    plt.figure(figsize=(8,3))
    librosa.display.specshow(spec, sr=sr, hop_length=hop,
                             y_axis='mel', x_axis='time')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

def save_cmp(a,b,path:Path,sr,hop):
    plt.figure(figsize=(12,4))
    for i,(m,ttl) in enumerate(((a,"Input"),(b,"Restored"))):
        plt.subplot(1,2,i+1)
        librosa.display.specshow(m, sr=sr, hop_length=hop,
                                 y_axis='mel', x_axis='time')
        plt.title(ttl)
        plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

def save_params(p_norm: torch.Tensor, path: Path):
    names = ["Gain","EQ‑Ctr","EQ‑Q","EQ‑Gain",
             "C‑Thresh","C‑Ratio","C‑MakeUp",
             "Rev‑Decay","Echo‑Delay","Echo‑Atten"]
    arr = p_norm.cpu().numpy()[0]  # [T,10]
    with path.open("w", encoding="utf-8") as f:  # <- force UTF-8 encoding
        for i, n in enumerate(names):
            col = arr[:, i]
            f.write(f"{n:>10s}: μ={col.mean():.4f} σ={col.std():.4f} "
                    f"min={col.min():.4f} max={col.max():.4f}\n")
    logger.info(f"Saved params → {path}")


def save_audio(path:Path, audio:np.ndarray, sr:int):
    # signed 16‑bit PCM via wave
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.clip(audio, -1,1)
    data = (a*32767).astype(np.int16)
    with wave.open(str(path),'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())

def load_hifigan(device, clone:Path, weights:Path):
    if not clone.is_dir():
        logger.warning("HiFi‑GAN clone missing")
        return None
    sys.path.insert(0,str(clone))
    cfg = json.loads((clone/"config_v1.json").read_text())
    from types import SimpleNamespace
    cfg = SimpleNamespace(**cfg)
    import importlib.util
    spec = importlib.util.spec_from_file_location("hifi_m", str(clone/"models.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    gen = m.Generator(cfg).to(device)
    if weights.exists():
        sd = torch.load(weights, map_location=device)
        sd = sd.get("generator", sd)
        gen.load_state_dict(sd)
        gen.remove_weight_norm()
    gen.eval()
    logger.info(f"Loaded HiFi‑GAN from {weights}")
    return gen

def prepare_hifigan_mel(out_norm: torch.Tensor, lo: float, hi: float, device):
    """
    Transform UNet out_norm into HiFi-GAN mel: log(mel_amp)
    """
    db = out_norm.squeeze(0).squeeze(0) * (hi - lo) + lo
    # Convert dB to power
    power = librosa.db_to_power(db.cpu().numpy().astype(np.float32))
    mel_amp = np.sqrt(power)  # Amplitude
    mel_log = np.log(np.clip(mel_amp, 1e-5, None))  # log-mel amplitude
    mel = torch.from_numpy(mel_log).unsqueeze(0).to(device)  # [1, 80, T]
    return mel


def load_unet(ckpt:Path, device):
    model = MasterNet(sr=22050, hop=256).to(device).eval()
    ck     = torch.load(ckpt, map_location=device)
    state  = ck.get("model", ck)
    model.load_state_dict(state)
    logger.info(f"Loaded MasterNet from {ckpt}")
    return model

def evaluate_file(wav:Path, unet, hifi, device, dirs, sr,n_fft,hop,n_mels):
    base = wav.stem.replace("_modified","")
    mn,md,lo,hi,raw = mel_from_wav(str(wav),sr,n_fft,hop,n_mels)

    dbg = dirs["images"]/ "debug"; dbg.mkdir(exist_ok=True)
    save_spec(md,  dbg/f"{base}_in_db.png",    "Input dB",   sr,hop)
    save_spec(mn,  dbg/f"{base}_in_norm.png", "Input 0–1", sr,hop)

    x = torch.from_numpy(mn)[None,None].to(device)
    with torch.no_grad():
        out_norm,p_norm = unet(x)
    out_norm = out_norm.clamp(0,1)
    out_np   = out_norm.cpu().squeeze().numpy()
    out_db   = denorm(out_np,lo,hi)

    save_spec(out_np, dirs["images"]/f"{base}_out_norm.png","Out 0–1",sr,hop)
    save_spec(out_db, dirs["images"]/f"{base}_out_db.png",    "Out dB",  sr,hop)
    save_cmp(md,out_db, dirs["images"]/f"{base}_cmp.png",     sr,hop)
    save_params(p_norm, dirs["params"]/f"{base}_params.txt")

    # Griffin-Lim reconstruction
    # Convert mel spectrogram back to linear spectrogram
    mel_power = librosa.db_to_power(out_db)
    S = librosa.feature.inverse.mel_to_stft(mel_power, sr=sr, n_fft=n_fft)
    gl = librosa.griffinlim(S, hop_length=hop, win_length=n_fft, n_iter=512)
    gl = librosa.util.normalize(gl)
    save_audio(dirs["audio"]/f"{base}_griffinlim.wav", gl, sr)

    # HiFi-GAN reconstruction (if available)
    if hifi:
        # Prepare mel spectrogram specifically for HiFi-GAN
        mel_in = prepare_hifigan_mel(out_norm, lo, hi, device)
        with torch.no_grad():
            wav_out = hifi(mel_in)[0].cpu().numpy()
        save_audio(dirs["audio"]/f"{base}_hifigan.wav", wav_out, sr)

    save_audio(dirs["audio"]/f"{base}_input.wav", raw, sr)
    logger.info(f"Done {wav.name}")

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    dirs = setup_dirs(Path(args.output_root))
    shutil.copy2(args.unet_ckpt,    dirs["checkpoints"]/ "masternet.ckpt")
    shutil.copy2(args.hifi_weights, dirs["checkpoints"]/ "hifi_gen.pth")

    unet = load_unet(Path(args.unet_ckpt), device)
    hifi = load_hifigan(device, Path(args.hifi_clone), Path(args.hifi_weights))

    wavs = sorted(Path(args.input_dir).glob("*_modified.wav"))[:10]
    if not wavs:
        logger.error("No '*_modified.wav' files found."); sys.exit(1)

    logger.info(f"Processing {len(wavs)} files…")
    for w in tqdm(wavs, desc="Evaluating"):
        try:
            evaluate_file(w,unet,hifi,device,
                          dirs,args.sr,args.n_fft,args.hop,args.n_mels)
        except Exception as e:
            logger.error(f"Error on {w.name}: {e}")

    logger.info(f"All done!  Outputs in {dirs['root']}")

if __name__=="__main__":
    main()
