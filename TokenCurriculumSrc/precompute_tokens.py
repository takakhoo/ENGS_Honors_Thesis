#!/usr/bin/env python3
"""
Precompute EnCodec tokens and save to .pt files for faster training.

Usage:
    python src/precompute_tokens.py

This script processes all curriculum stages in experiments/curriculums/*,
converts each pair of *_original.wav and *_modified.wav to EnCodec tokens,
and saves them as {song_id}.pt in output_tokens/.
"""
import os, torch, torchaudio, warnings
from pathlib import Path
from encodec import EncodecModel
from encodec.utils import convert_audio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import datetime

def tokenize_pair(stage_dir, codec):
    audio_dir = stage_dir / "output_audio"
    token_dir = stage_dir / "output_tokens"
    token_dir.mkdir(exist_ok=True)
    log_path = stage_dir / "tokens_log.txt"
    log_lines = []
    orig_files = sorted(audio_dir.glob("*_original.wav"))
    def process_one(orig_path):
        sid = orig_path.name.split("_")[0]
        mod_path = audio_dir / f"{sid}_modified.wav"
        out_path = token_dir / f"{sid}.pt"
        if out_path.exists():
            return None  # Already cached
        try:
            wav_orig, sr = torchaudio.load(orig_path)
            wav_mod,  _  = torchaudio.load(mod_path)
        except Exception as e:
            warnings.warn(f"Failed to load {sid}: {e}")
            return f"{datetime.datetime.now()}  WARN: Failed to load {sid}: {e}"
        wav_orig = convert_audio(wav_orig, sr, codec.sample_rate, codec.channels).unsqueeze(0)
        wav_mod  = convert_audio(wav_mod, sr, codec.sample_rate, codec.channels).unsqueeze(0)
        with torch.no_grad():
            codes_orig = torch.cat([f[0] for f in codec.encode(wav_orig)], dim=-1).squeeze(0).long()
            codes_mod  = torch.cat([f[0] for f in codec.encode(wav_mod)],  dim=-1).squeeze(0).long()
        torch.save({"X": codes_mod, "Y": codes_orig}, out_path)
        msg = f"{datetime.datetime.now()}  OK: {out_path.relative_to(stage_dir.parent.parent)}"
        print(f"[✓] {out_path.relative_to(stage_dir.parent.parent)}")
        return msg
    # Use ThreadPoolExecutor for parallel tokenization
    with ThreadPoolExecutor() as executor:
        for result in tqdm(executor.map(process_one, orig_files), total=len(orig_files), desc=f"Tokenizing {stage_dir.name}"):
            if result:
                log_lines.append(result)
    # Write log file
    with open(log_path, "a") as f:
        f.write(f"\n{datetime.datetime.now()}  Processed {len(orig_files)} files\n")
        for line in log_lines:
            f.write(line + "\n")

if __name__ == "__main__":
    SRC = Path(__file__).resolve().parents[1]
    CURR = SRC / "experiments" / "curriculums"
    codec = EncodecModel.encodec_model_48khz()
    codec.set_target_bandwidth(24.0)
    for stage in sorted(CURR.iterdir()):
        if not stage.is_dir() or not (stage / "output_audio").exists():
            continue
        print(f"\n🔁 Processing {stage.name}...")
        tokenize_pair(stage, codec)
    print("\n✅ Token precomputation complete.") 