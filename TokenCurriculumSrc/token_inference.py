#!/usr/bin/env python3
"""
token_inference.py – Inference for Token-UNet on precomputed EnCodec tokens

- Loads a trained TokenUNet checkpoint
- Processes all .pt token pairs in a given stage's output_tokens/
- Decodes both original and model-restored audio using EnCodec
- Saves audio, mel/chroma images, and CSV metrics to out_dir
"""

import argparse, sys, csv
from pathlib import Path
from tqdm import tqdm
import torch, torchaudio
import numpy as np
import librosa, librosa.display, matplotlib.pyplot as plt

from encodec import EncodecModel
from encodec.utils import convert_audio
from token_unet import TokenUNet

# ──────────────── Helper Functions ────────────────

def save_wav(wav, sr, path):
    wav = np.clip(wav, -1, 1)
    torchaudio.save(str(path), torch.from_numpy(wav).unsqueeze(0), sr)

def save_mel(wav, sr, path, title):
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    dB  = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(dB, sr=sr, hop_length=512, x_axis="time", y_axis="mel", cmap="magma")
    plt.colorbar(format='%+2.0f dB'); plt.title(title); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()

def save_chroma(wav, sr, path, title):
    chr_ = librosa.feature.chroma_stft(y=wav, sr=sr, n_fft=2048, hop_length=512)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chr_, x_axis="time", y_axis="chroma", cmap="coolwarm")
    plt.colorbar(); plt.title(title); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()

def metrics(orig: np.ndarray, rest: np.ndarray, sr_orig: int):
    """Return SNR, PESQ, STOI (resample to 16 kHz if needed)."""
    from pesq import pesq
    from pystoi import stoi
    if orig.ndim > 1:  orig  = orig.mean(0)
    if rest.ndim > 1:  rest  = rest.mean(0)
    orig /= np.max(np.abs(orig) + 1e-9)
    rest /= np.max(np.abs(rest) + 1e-9)
    snr = 10 * np.log10(np.sum(orig**2) / np.sum((orig-rest)**2) + 1e-9)
    if sr_orig != 16_000:
        orig_rs = librosa.resample(orig, sr_orig, 16_000)
        rest_rs = librosa.resample(rest, sr_orig, 16_000)
        sr_m    = 16_000
    else:
        orig_rs, rest_rs, sr_m = orig, rest, sr_orig
    pesq_s = pesq(sr_m, orig_rs, rest_rs, 'wb')
    stoi_s = stoi(orig_rs, rest_rs, sr_m, extended=False)
    return snr, pesq_s, stoi_s

# Helper to ensure mono audio for metrics
def to_mono(wav):
    return wav.mean(axis=0) if wav.ndim > 1 else wav

# Spectral convergence and log-MSE metrics
def spectral_convergence(S_ref, S_est):
    # S_ref, S_est: [freq, time] (magnitude)
    return np.linalg.norm(S_ref - S_est, 'fro') / (np.linalg.norm(S_ref, 'fro') + 1e-8)

def log_mse(S_ref, S_est):
    # S_ref, S_est: [freq, time] (magnitude)
    log_S_ref = np.log(np.abs(S_ref) + 1e-8)
    log_S_est = np.log(np.abs(S_est) + 1e-8)
    return np.mean((log_S_ref - log_S_est) ** 2)

# ──────────────── Main Inference ────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path, help="Path to model checkpoint (.pt)")
    ap.add_argument("--token_dir", required=True, type=Path, help="Directory with .pt token files (output_tokens/)")
    ap.add_argument("--out_dir", required=True, type=Path, help="Output directory for results")
    ap.add_argument("--codec", choices=("24khz","48khz"), default="48khz")
    ap.add_argument("--bandwd", type=float, default=24.0)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = EncodecModel.encodec_model_48khz() if args.codec == "48khz" else EncodecModel.encodec_model_24khz()
    codec.set_target_bandwidth(args.bandwd)
    codec = codec.to(dev).eval()
    sr = codec.sample_rate
    print(f"✓ EnCodec {args.codec} {args.bandwd} kbps  sr={sr}")

    # Load model
    ckpt = torch.load(args.ckpt, map_location=dev)
    if "model" not in ckpt:
        raise KeyError("Checkpoint is missing 'model' state_dict")
    n_q = ckpt["model"]["heads.0.weight"].shape[1]
    net = TokenUNet(n_q).to(dev)
    net.load_state_dict(ckpt["model"])
    net.eval()
    print(f"✓ Loaded checkpoint {args.ckpt}")

    # Prepare output dirs
    audio_b = args.out_dir / "audio" / "before"
    audio_a = args.out_dir / "audio" / "after"
    mel_b   = args.out_dir / "images" / "mel_before"
    mel_a   = args.out_dir / "images" / "mel_after"
    chr_b   = args.out_dir / "images" / "chroma_before"
    chr_a   = args.out_dir / "images" / "chroma_after"
    for d in (audio_b, audio_a, mel_b, mel_a, chr_b, chr_a): d.mkdir(parents=True, exist_ok=True)

    # Gather all .pt files
    pt_files = sorted(args.token_dir.glob("*.pt"))
    if not pt_files:
        sys.exit(f"❌ No .pt files found in {args.token_dir}")

    stats, qual = [("file","tokens","changed","%")], [("file","snr","pesq","stoi","spec_conv","log_mse")]
    spec_conv_vals, log_mse_vals, snr_vals, pesq_vals, stoi_vals = [], [], [], [], []

    # Helper: decode EnCodec tokens to audio
    def decode_audio_from_tokens(tokens):
        # tokens: [n_q, T]
        return codec.decode([(tokens.unsqueeze(0), None)])[0].cpu().numpy()

    for pt_file in tqdm(pt_files, desc="Token inference"):
        stem = pt_file.stem
        try:
            pair = torch.load(pt_file, map_location=dev)
            X, Y = pair["X"].to(dev), pair["Y"].to(dev)
        except Exception as e:
            print(f"⚠️  Failed to load {pt_file}: {e}")
            continue
        # Model inference
        with torch.no_grad():
            logits = net(X.unsqueeze(0))  # [1, K, n_q, T]
            codes  = logits.argmax(1).squeeze(0)  # [n_q, T]
        # Decode to audio
        wav_rest = decode_audio_from_tokens(codes)
        wav_orig = decode_audio_from_tokens(Y)
        # Clamp to min length to avoid length mismatch
        L = min(wav_rest.shape[-1], wav_orig.shape[-1])
        wav_rest = wav_rest[..., :L]
        wav_orig = wav_orig[..., :L]
        # Save audio (enforce float32)
        save_wav(wav_orig.astype(np.float32), sr, audio_b / f"{stem}_before.wav")
        save_wav(wav_rest.astype(np.float32), sr, audio_a / f"{stem}_after.wav")
        # Save images
        save_mel(wav_orig[0], sr, mel_b / f"{stem}_mel_before.png", f"{stem} – mel BEFORE")
        save_mel(wav_rest[0], sr, mel_a / f"{stem}_mel_after.png",  f"{stem} – mel AFTER")
        save_chroma(wav_orig[0], sr, chr_b / f"{stem}_chroma_before.png", f"{stem} – chroma BEFORE")
        save_chroma(wav_rest[0], sr, chr_a / f"{stem}_chroma_after.png",  f"{stem} – chroma AFTER")
        # Token change stats
        changed = (Y != codes).sum().item()
        total   = Y.numel()
        stats.append((stem, total, changed, f"{100*changed/total:.2f}"))
        # Quality metrics (use mono)
        try:
            # Compute mel spectrograms for spectral metrics
            mel_orig = librosa.feature.melspectrogram(y=to_mono(wav_orig), sr=sr, n_fft=2048, hop_length=512, n_mels=128)
            mel_rest = librosa.feature.melspectrogram(y=to_mono(wav_rest), sr=sr, n_fft=2048, hop_length=512, n_mels=128)
            spec_conv = spectral_convergence(mel_orig, mel_rest)
            logmse = log_mse(mel_orig, mel_rest)
            snr, pesq_s, stoi_s = metrics(to_mono(wav_orig), to_mono(wav_rest), sr)
            qual.append((stem, f"{snr:.2f}", f"{pesq_s:.2f}", f"{stoi_s:.2f}", f"{spec_conv:.4f}", f"{logmse:.4f}"))
            print(f"   {stem}: SNR {snr:.2f} dB  PESQ {pesq_s:.2f}  STOI {stoi_s:.2f}  SC {spec_conv:.4f}  logMSE {logmse:.4f}")
            # Collect for summary plot
            snr_vals.append(snr)
            pesq_vals.append(pesq_s)
            stoi_vals.append(stoi_s)
            spec_conv_vals.append(spec_conv)
            log_mse_vals.append(logmse)
        except Exception as e:
            print(f"   {stem}: Metric error: {e}")

    # Write CSVs with utf-8 encoding
    with open(args.out_dir / "token_stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerows(stats)
    with open(args.out_dir / "quality_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerows(qual)
    # Generate summary bar plot
    metrics_names = ["SNR", "PESQ", "STOI", "Spectral Convergence", "log-MSE"]
    means = [
        np.mean(snr_vals) if snr_vals else 0,
        np.mean(pesq_vals) if pesq_vals else 0,
        np.mean(stoi_vals) if stoi_vals else 0,
        np.mean(spec_conv_vals) if spec_conv_vals else 0,
        np.mean(log_mse_vals) if log_mse_vals else 0,
    ]
    plt.figure(figsize=(8,5))
    plt.bar(metrics_names, means, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"])
    plt.ylabel("Average Value")
    plt.title("Average Metrics Across All Samples")
    plt.tight_layout()
    plt.savefig(args.out_dir / "summary_metrics.png", dpi=120)
    plt.close()
    print(f"\n✓ Results and summary plot saved in {args.out_dir}\n")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
