#!/usr/bin/env python3
"""
evaluate.py – One‑file evaluator for OneStageDeepUNet + local HiFi‑GAN
======================================================================
• Loads newest UNet checkpoint in <project>/checkpoints/
• Converts *_modified.wav → 128‑bin mel (44 100 Hz, hop 512)
• Runs the UNet, saves restored spectrograms & parameter stats
• Vocodes with HiFi‑GAN from your local clone (hop 256, 80 mels)
• Falls back to Griffin‑Lim if HiFi‑GAN fails
"""

import os
import sys
import json
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from types import SimpleNamespace
import importlib.util

# ── repo root ──────────────────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# point at your src folder first so we import *your* models/dataset*
sys.path.insert(0, project_root)

from models import OneStageDeepUNet
from dataset import PairedAudioDataset

print(f"CUDA  : {torch.cuda.is_available()}   |   PyTorch {torch.__version__}")
if torch.cuda.is_available():
    print("GPU   :", torch.cuda.get_device_name(0))

# ─────────────────── Utilities ────────────────────────────────────────────────
def newest_ckpt(folder: str) -> str:
    best = os.path.join(folder, "best_model.pt")
    if os.path.exists(best):
        return best
    cpts = [f for f in os.listdir(folder)
            if f.startswith("one_stage_deep_unet_epoch_") and f.endswith(".pt")]
    if not cpts:
        raise FileNotFoundError(f"No checkpoints in {folder}")
    cpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]), reverse=True)
    return os.path.join(folder, cpts[0])

def copy_sources(src: str, dst: str, keep=5):
    os.makedirs(dst, exist_ok=True)
    files = sorted(f for f in os.listdir(src) if f.lower().endswith("_modified.wav"))
    chosen = files[:keep]
    out = []
    for f in chosen:
        s, d = os.path.join(src, f), os.path.join(dst, f)
        shutil.copy2(s, d)
        out.append(d)
        print("Copied", s, "→", d)
    return out

def mel_from_wav(path, *, sr=44100, n_fft=2048, hop=512, n_mels=128):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = np.clip(y, -1, 1)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel + 1e-6, ref=np.max)
    lo, hi = mel_db.min(), mel_db.max()
    mel_norm = (mel_db - lo) / (hi - lo + 1e-6)
    return mel_norm, mel_db, lo, hi, y

def denorm(norm, lo, hi):
    return norm * (hi - lo) + lo

def save_spec(m, fn, title, *, sr=44100, hop=512):
    plt.figure(figsize=(10,4))
    librosa.display.specshow(m, sr=sr, hop_length=hop,
                             x_axis='time', y_axis='mel')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(fn); plt.close()
    print("Saved", fn)

def save_cmp(a, b, fn, *, sr=44100, hop=512):
    plt.figure(figsize=(12,4))
    for i,(m,t) in enumerate([(a,"Input"),(b,"Output")]):
        plt.subplot(1,2,i+1)
        librosa.display.specshow(m, sr=sr, hop_length=hop,
                                 x_axis='time', y_axis='mel')
        plt.title(t); plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(fn); plt.close()
    print("Saved", fn)

def save_param_stats(pn, pu, fn):
    names = ["Gain","EQ‑Ctr","EQ‑Q","EQ‑Gain",
             "C‑Thresh","C‑Ratio","C‑MakeUp",
             "Rev‑Decay","Echo‑Delay","Echo‑Atten"]
    n = pn.cpu().numpy(); u = pu.cpu().numpy()
    with open(fn, "w", encoding="utf-8") as f:
        f.write("Normalized\n")
        for i,nm in enumerate(names):
            x = n[:,:,i].ravel()
            f.write(f"{nm}: μ={x.mean():.4f} σ={x.std():.4f} "
                    f"min={x.min():.4f} max={x.max():.4f}\n")
        f.write("\nUn‑normalized\n")
        for i,nm in enumerate(names):
            x = u[:,:,i].ravel()
            f.write(f"{nm}: μ={x.mean():.4f} σ={x.std():.4f} "
                    f"min={x.min():.4f} max={x.max():.4f}\n")
    print("Saved", fn)

# ─────────────────── HiFi‑GAN loader (local clone + universal weights) ──────────
def load_hifigan(device):
    repo = os.path.join(project_root, "vocoder", "hifi-gan")
    if not os.path.isdir(repo):
        print("⚠ HiFi‑GAN clone not found at", repo)
        return None, None

    # 1) import hifi-gan's utils.py under name 'utils'
    utils_py = os.path.join(repo, "utils.py")
    spec_u = importlib.util.spec_from_file_location("utils", utils_py)
    hifi_utils = importlib.util.module_from_spec(spec_u)
    spec_u.loader.exec_module(hifi_utils)
    sys.modules['utils'] = hifi_utils

    # 2) import hifi-gan's models.py under name 'hifi_models'
    models_py = os.path.join(repo, "models.py")
    spec_m = importlib.util.spec_from_file_location("hifi_models", models_py)
    hifi_models = importlib.util.module_from_spec(spec_m)
    spec_m.loader.exec_module(hifi_models)

    # cleanup
    del sys.modules['utils']

    # 3) load config_v1.json from the clone
    cfg_path = os.path.join(repo, "config_v1.json")
    if not os.path.isfile(cfg_path):
        print("⚠ config_v1.json not found in", repo)
        return None, None
    cfg = json.load(open(cfg_path))
    h = SimpleNamespace(**cfg)

    # 4) instantiate Generator
    gen = hifi_models.Generator(h).to(device)

    # 5) load Universal v1 weights from vocoder/checkpoints
    ckpt_w = os.path.join(project_root, "vocoder", "checkpoints", "generator_universal_v1.pth")
    if not os.path.isfile(ckpt_w):
        print("⚠ generator_universal_v1.pth not found in vocoder/checkpoints")
        return None, None
    sd = torch.load(ckpt_w, map_location=device)
    sd = sd.get("generator", sd)
    gen.load_state_dict(sd)
    gen.remove_weight_norm()
    gen.eval()
    print("✓ Loaded local HiFi‑GAN universal v1 from", ckpt_w)

    return gen, None  # this clone has no separate denoiser

# ─────────────────── mel prep for HiFi‑GAN ───────────────────────────────────
def prepare_hifigan_mel(out_norm: torch.Tensor,
                        lo: float, hi: float,
                        device: torch.device) -> torch.Tensor:
    """
    UNet out_norm [1,1,128,T1] → HiFi‑GAN [1,80,T2] natural‑log.
    T2 = T1 * 2 (hop512→hop256)
    """
    # 1) de‑norm to dB
    db = out_norm.squeeze(0).squeeze(0) * (hi - lo) + lo        # [128,T1]
    # 2) dB→power→amp
    power = librosa.db_to_power(db.cpu().numpy().astype(np.float32))
    amp   = np.sqrt(power)
    mel128 = torch.from_numpy(amp)[None,None].to(device)        # [1,1,128,T1]
    # 3) up‑sample time ×2
    T2 = mel128.shape[-1] * 2
    mel128 = F.interpolate(mel128, size=(128, T2),
                           mode="bilinear", align_corners=False)
    # 4) down‑sample freq 128→80
    mel80 = F.interpolate(mel128, size=(80, T2),
                          mode="bilinear", align_corners=False)
    # 5) natural-log
    mel80 = torch.log(torch.clamp(mel80, min=1e-5))             # [1,1,80,T2]
    return mel80.squeeze(1)                                      # [1,80,T2]

# ───────────────────────── Evaluate one file ──────────────────────────────────
def run_one(wav, model, ckpt, gen, den, device,
            *, sr=44100, n_fft=2048, hop=512, n_mels=128, n_iter=128):
    # load UNet checkpoint
    ck = torch.load(ckpt, map_location=device)
    model.load_state_dict(ck.get("model_state_dict", ck))
    model.eval()

    epoch = ck.get("epoch", 0)
    root = os.path.join(project_root, "runs")
    dirs = {k: os.path.join(root, k, f"epoch_{epoch:04d}")
            for k in ("audio","spectrograms","parameters","debug")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    base = os.path.splitext(os.path.basename(wav))[0]

    # --- UNet preprocessing & forward
    norm, db_in, lo, hi, raw = mel_from_wav(wav, sr=sr, n_fft=n_fft, hop=hop, n_mels=n_mels)
    save_spec(db_in,  os.path.join(dirs["debug"],   f"{base}_in_db.png"),   "Input dB",  sr=sr, hop=hop)
    save_spec(norm,   os.path.join(dirs["debug"],   f"{base}_in_norm.png"), "Input 0–1",sr=sr, hop=hop)

    x = torch.from_numpy(norm)[None,None].to(device)
    with torch.no_grad():
        out, p_norm = model(x)
    out = out.clamp(0,1)

    out_np = out.squeeze().cpu().numpy()
    save_spec(out_np, os.path.join(dirs["debug"],   f"{base}_out_norm.png"), "Out norm", sr=sr, hop=hop)
    db_out = denorm(out_np, lo, hi)
    save_spec(db_out, os.path.join(dirs["debug"],   f"{base}_out_db.png"),   "Out dB",   sr=sr, hop=hop)
    save_cmp(db_in, db_out,
             os.path.join(dirs["spectrograms"], f"{base}_cmp.png"), sr=sr, hop=hop)

    save_param_stats(p_norm, p_norm,
                     os.path.join(dirs["parameters"], f"{base}_params.txt"))

    # --- vocoder
    if gen is not None:
        mel80 = prepare_hifigan_mel(out, lo, hi, device)
        with torch.no_grad():
            wav_out = gen(mel80).squeeze()
        wav_np = wav_out.clamp(-1,1).cpu().numpy()
        sf.write(os.path.join(dirs["audio"], f"{base}_hifigan.wav"), wav_np, sr)
        print("Saved HiFi‑GAN audio")
    else:
        wav_gl = librosa.griffinlim(
            librosa.feature.inverse.mel_to_stft(librosa.db_to_power(db_out),
                                                sr=sr, n_fft=n_fft, power=2.0),
            hop_length=hop, win_length=n_fft, n_iter=n_iter)
        wav_gl = librosa.util.normalize(wav_gl)
        sf.write(os.path.join(dirs["audio"], f"{base}_griffinlim.wav"), wav_gl, sr)
        print("Saved Griffin‑Lim audio")

    sf.write(os.path.join(dirs["audio"], f"{base}_input.wav"), raw, sr)
    print("✓ Finished", base)

# ────────────────────────────────── main ──────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = OneStageDeepUNet(sr=44100, hop_length=512,
                           in_channels=1, out_channels=1,
                           base_features=64, blocks_per_level=3,
                           lstm_hidden=32, num_layers=1, num_params=10).to(device)

    ckpt = newest_ckpt(os.path.join(project_root, "checkpoints"))
    print("Using UNet checkpoint:", ckpt)

    gen, den = load_hifigan(device)

    src = os.path.join(project_root, "experiments", "output_full", "output_audio")
    dst = os.path.join(project_root, "audio", "internal_demastered")
    wavs = copy_sources(src, dst, keep=5)
    if not wavs:
        print("No *_modified.wav files found."); return

    print("\n── Evaluating files ──")
    for w in tqdm(wavs, desc="Evaluating"):
        try:
            run_one(w, net, ckpt, gen, den, device)
        except Exception as e:
            print("❌ Error on", w, ":", e)

    print("\nAll done!")

if __name__ == "__main__":
    main()
