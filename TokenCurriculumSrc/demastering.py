#!/usr/bin/env python3
"""
demastering.py – build curriculum datasets (stages 0‑4) from FMA‑Medium

Stage map
  0  identity                (0 effects)         – 1000 tracks
  1  single                  (1 mild effect)     – 2000 tracks
  2  double                  (2 mild effects)    – 3000 tracks
  3  triple                  (3 mild effects)    – 4000 tracks
  4  full                    (1–5 mild effects)  – 5000 tracks

Run examples:
  python src/demastering.py --stage 0 --seed 42
  python src/demastering.py --stage 3 --stronger --seed 42
"""

from __future__ import annotations
import argparse, random, shutil, gc, glob, textwrap, warnings
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt, librosa, librosa.display, soundfile as sf
from scipy.signal import iirpeak, lfilter, fftconvolve

# ─────────────────────────────── CLI ────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--stage",     type=int, required=True, choices=range(0,5),
                help="0‑identity … 4‑full‑random")
ap.add_argument("--stronger",  action="store_true",
                help="widen parameter ranges (+/‑50 %) – useful for stage≥3")
ap.add_argument("--seed",      type=int, default=None,
                help="random seed for reproducibility")
args = ap.parse_args()

# Set random seed if provided
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)

# ─────────────────────────── paths & housekeeping ───────────────────
PRJ = Path(__file__).resolve().parents[1]
RAW = PRJ / "data" / "raw" / "fma_medium" / "fma_medium"
STAGE_NAME = {0:"stage0_identity", 1:"stage1_single",
              2:"stage2_double",  3:"stage3_triple",
              4:"stage4_full"}[args.stage]
# Add _stronger suffix if using stronger parameters
if args.stronger:
    STAGE_NAME += "_stronger"
OUT = PRJ / "experiments" / "curriculums" / STAGE_NAME

if OUT.exists(): shutil.rmtree(OUT)
(OUT/"output_audio").mkdir(parents=True)
(OUT/"output_txt").mkdir()
(OUT/"output_spectrograms").mkdir()

# ───────────────────────────  constants  ─────────────────────────────
EFFECTS = ("eq","gain","echo","reverb","compression")
STAGE_LIMITS = [1000, 3000, 8000, 15000, 25000]  # Progressive increase in dataset size
NUM_TRACKS = STAGE_LIMITS[args.stage]

# Milder base parameters
BASE = dict(
    eq   = dict(fc=(500,2000), Q=(0.9,1.2), gain=(-0.8,0.8)),  # Reduced frequency range and gain
    gain = dict(db=(-1,1)),  # Reduced gain range
    echo = dict(delay=(0.1,0.2), att=(0.2,0.3)),  # Shorter delays, less attenuation
    reverb=dict(decay=(0.2,0.4), ir=(0.05,0.15)),  # Shorter decay and IR
    comp = dict(thr=(-12,-8), ratio=(1.5,2.5), makeup=(0,0.5)),  # Less aggressive compression
)

if args.stronger:
    for d in BASE.values():
        for k,(lo,hi) in d.items():
            span = hi-lo
            # For EQ frequency, ensure we don't exceed Nyquist frequency
            if k == 'fc':
                d[k] = (lo-0.3*span, min(hi+0.3*span, 20000))  # Cap at 20kHz
            else:
                d[k] = (lo-0.5*span, hi+0.5*span)

def choose_effects():
    if args.stage == 0: return []
    if args.stage == 1: return random.sample(EFFECTS, 1)
    if args.stage == 2: return random.sample(EFFECTS, 2)
    if args.stage == 3: return random.sample(EFFECTS, 3)
    return random.sample(EFFECTS, random.randint(1, 5))

# ─────────────────────── degradation effects ────────────────────────
def apply_eq(x, sr, fc, Q, gain_db):
    w0 = fc / (sr/2)
    b, a = iirpeak(w0, Q)
    b    = b * 10**(gain_db/20)
    return lfilter(b, a, x)

def apply_gain(x, gain_db):
    return x * 10**(gain_db/20)

def apply_echo(x, sr, delay, attenuation):
    d = int(sr * delay)
    echo = np.zeros_like(x)
    if len(x) > d:
        echo[d:] = x[:-d] * attenuation
    return x + echo

def apply_reverb(x, sr, decay, ir_len):
    t  = np.linspace(0, ir_len/sr, ir_len)
    ir = np.exp(-decay * t)
    out = fftconvolve(x, ir, mode="same")
    return out.astype(np.float32)

def apply_compression(x, thr_db, ratio, makeup_db):
    thr_lin = 10**(thr_db/20)
    absx    = np.abs(x)
    cmp     = np.where(absx > thr_lin,
                       thr_lin + (absx-thr_lin)/ratio,
                       absx)
    cmp     = np.sign(x) * cmp
    return cmp * 10**(makeup_db/20)

# ────────────────────── mel spectrogram plot ────────────────────────
def plot_mel(wav: np.ndarray, sr:int, title:str, subtitle:str=""):
    M  = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=2048,
                                        hop_length=512, n_mels=128)
    dB = librosa.power_to_db(M, ref=np.max)
    plt.figure(figsize=(10,4))
    librosa.display.specshow(dB, sr=sr, hop_length=512,
                             x_axis="time", y_axis="mel", cmap="magma")
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    if subtitle:
        plt.annotate(subtitle, xy=(0.99,0.01), xycoords="axes fraction",
                     ha="right", va="bottom",
                     bbox=dict(fc="yellow", alpha=0.4, pad=2), fontsize=8)
    plt.tight_layout(pad=0.2)
    fig = plt.gcf()
    plt.close(fig)
    return fig

# ───────────────────────────  main loop  ────────────────────────────
mp3_files = sorted(RAW.rglob("*.mp3"))
random.shuffle(mp3_files)
print(f"Found {len(mp3_files)} mp3 files in FMA‑medium")
processed = 0

for mp3 in mp3_files:
    if processed >= NUM_TRACKS:
        break

    try:
        wav, sr = librosa.load(mp3, sr=None, mono=True)
    except Exception as e:
        warnings.warn(f"{mp3.name} failed to load: {e}")
        continue

    if wav is None or wav.size == 0:
        warnings.warn(f"{mp3.name} empty – skipped")
        continue

    if not np.isfinite(wav).all():
        warnings.warn(f"{mp3.name} contains NaNs or Infs – skipped")
        continue


    sid = mp3.stem
    y = wav.copy()
    log = []

    for eff in choose_effects():
        if eff == "eq":
            fc  = random.uniform(*BASE["eq"]["fc"])
            Q   = random.uniform(*BASE["eq"]["Q"])
            g   = random.uniform(*BASE["eq"]["gain"])
            y   = apply_eq(y, sr, fc, Q, g)
            log.append(f"EQ fc={fc:.0f}Hz Q={Q:.2f} g={g:+.1f}dB")
        elif eff == "gain":
            db = random.uniform(*BASE["gain"]["db"])
            y = apply_gain(y, db)
            log.append(f"GAIN {db:+.1f}dB")
        elif eff == "echo":
            d = random.uniform(*BASE["echo"]["delay"])
            a = random.uniform(*BASE["echo"]["att"])
            y = apply_echo(y, sr, d, a)
            log.append(f"ECHO {d:.2f}s ×{a:.2f}")
        elif eff == "reverb":
            decay = random.uniform(*BASE["reverb"]["decay"])
            ir_s  = random.uniform(*BASE["reverb"]["ir"])
            y = apply_reverb(y, sr, decay, int(ir_s * sr))
            log.append(f"REV dec={decay:.2f} ir={ir_s:.2f}s")
        elif eff == "compression":
            thr   = random.uniform(*BASE["comp"]["thr"])
            ratio = random.uniform(*BASE["comp"]["ratio"])
            mg    = random.uniform(*BASE["comp"]["makeup"])
            y = apply_compression(y, thr, ratio, mg)
            log.append(f"COMP thr={thr:.0f} ratio={ratio:.1f} mg={mg:.1f}")

    if not log:
        log.append("identity")

    sf.write(OUT/"output_audio"/f"{sid}_original.wav", wav, sr)
    sf.write(OUT/"output_audio"/f"{sid}_modified.wav", y,   sr)
    (OUT/"output_txt"/f"{sid}.txt").write_text("\n".join(log))

    fig = plot_mel(y, sr, f"{sid}  (stage {args.stage})",
                   "\n".join(textwrap.wrap(", ".join(log), 50)))
    fig.savefig(OUT/"output_spectrograms"/f"{sid}_mel_after.png", dpi=120)
    plt.close(fig)

    fig = plot_mel(wav, sr, f"{sid}  (ORIGINAL)", "")
    fig.savefig(OUT/"output_spectrograms"/f"{sid}_mel_before.png", dpi=120)
    plt.close(fig)

    processed += 1
    del wav, y, fig
    gc.collect()

print(f"✓  Stage‑{args.stage} dataset written to {OUT}  ({processed} tracks)")
