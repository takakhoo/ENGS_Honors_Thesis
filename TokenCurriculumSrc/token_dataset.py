# src/token_dataset.py  –  paired EnCodec token dataset
# --------------------------------------------------------------------------------
# This file implements a PyTorch Dataset for handling paired audio data
# where each sample consists of a degraded audio and its original version.
# The dataset converts audio waveforms into EnCodec tokens for training.

from __future__ import annotations
import pathlib, typing as tp, torch, torchaudio
from torch.utils.data import Dataset
from encodec import EncodecModel
from encodec.utils import convert_audio
import random

# ──────────────────────────────────────────────────────────────────────
def _load_codec(model_type: str, bandwidth: float) -> EncodecModel:
    """Initialize and configure the EnCodec model for audio encoding/decoding.
    
    Args:
        model_type: Either "24khz" or "48khz" to specify the model variant
        bandwidth: Target bandwidth in kbps (must be valid for the model type)
    
    Returns:
        Configured EnCodec model instance
    
    Raises:
        ValueError: If model_type or bandwidth is invalid
    """
    if model_type == "24khz":
        codec = EncodecModel.encodec_model_24khz()
        allowed = {1.5, 3, 6, 12, 24}
    elif model_type == "48khz":
        codec = EncodecModel.encodec_model_48khz()
        allowed = {3, 6, 12, 24}
    else:
        raise ValueError("model_type must be '24khz' or '48khz'")
    if bandwidth not in allowed:
        raise ValueError(f"{bandwidth} kbps invalid for {model_type}")
    codec.set_target_bandwidth(bandwidth)
    return codec

# ──────────────────────────────────────────────────────────────────────
class TokenPairDataset(Dataset):
    """Dataset for paired audio data (degraded + original) using EnCodec tokens.
    
    This dataset loads pairs of audio files from curriculum stages where:
    - The degraded version has "_modified.wav" suffix
    - The original version has "_original.wav" suffix
    
    The audio is converted to EnCodec tokens for training the Token-UNet model.
    """
    
    def __init__(self,
                base_dir    : str | pathlib.Path,
                stages      : list[str] | None = None,
                model_type  : str   = "48khz",
                bandwidth   : float = 24.0,
                max_debug   : int   = 3):
        """Initialize the dataset.
        
        Args:
            base_dir: Base directory containing curriculum stages
            stages: List of stage names to include (e.g., ["stage0_identity", "stage1_single"])
                   If None, uses all available stages
            model_type: EnCodec model type ("24khz" or "48khz")
            bandwidth: Target bandwidth in kbps
            max_debug: Number of samples to print debug info for
        """
        self.base_dir = pathlib.Path(base_dir)
        if not self.base_dir.exists():
            raise RuntimeError(f"Base directory {self.base_dir} not found")

        # Find all available stages if none specified
        if stages is None:
            self.stages = [d.name for d in self.base_dir.iterdir() 
                         if d.is_dir() and d.name.startswith("stage")]
        else:
            self.stages = stages

        # Check if precomputed tokens exist for the first stage
        self.use_tokens = False
        self.token_files = []
        self.clean_wav = []
        for stage in self.stages:
            stage_dir = self.base_dir / stage
            token_dir = stage_dir / "output_tokens"
            audio_dir = stage_dir / "output_audio"
            if token_dir.exists() and any(token_dir.glob("*.pt")):
                self.use_tokens = True
                stage_token_files = sorted(token_dir.glob("*.pt"))
                self.token_files.extend(stage_token_files)
                print(f"[TokenDataset] Using precomputed tokens for {stage} ({len(stage_token_files)} files)")
            elif audio_dir.exists():
                stage_files = sorted(audio_dir.glob("*_original.wav"))
                self.clean_wav.extend(stage_files)
                print(f"[TokenDataset] Using audio for {stage} ({len(stage_files)} files)")
            else:
                print(f"Warning: Stage directory {stage_dir} missing output_audio and output_tokens")

        if self.use_tokens:
            if not self.token_files:
                raise RuntimeError(f"No .pt token files found in any of {self.stages}")
            print(f"[TokenDataset] Total token pairs: {len(self.token_files):,}\n")
        else:
            if not self.clean_wav:
                raise RuntimeError(f"No *_original.wav found in any of {self.stages}")
            print(f"[TokenDataset] Total audio pairs: {len(self.clean_wav):,}\n")

        # Only initialize EnCodec if not using tokens
        if not self.use_tokens:
            self.codec     = _load_codec(model_type, bandwidth)
            self.sr        = self.codec.sample_rate
            self.channels  = self.codec.channels
            self.n_q       = self.codec.quantizer.n_q      # Number of codebooks
        else:
            # Load shape info from first token file
            sample = torch.load(self.token_files[0])
            self.n_q = sample["X"].shape[0]
            self.sr = 48000 if model_type == "48khz" else 24000
            self.channels = 1  # EnCodec tokens are mono
        self._dbg_left = int(max_debug)

        print(f"[TokenDataset] model {model_type}  • {bandwidth} kbps")
        print(f"[TokenDataset] {self.sr} Hz  {self.channels}-ch  "
            f"{self.n_q} codebooks")
        print(f"[TokenDataset] Using stages: {', '.join(self.stages)}")

        if self.use_tokens:
            random.shuffle(self.token_files)  # Shuffle for random_split
        else:
            random.shuffle(self.clean_wav)    # Shuffle for random_split

    # -----------------------------------------------------------------
    def _wav2codes(self, path: pathlib.Path) -> torch.LongTensor:
        """Convert a waveform file to EnCodec tokens.
        
        Args:
            path: Path to the audio file
            
        Returns:
            Tensor of shape [n_q, T] containing the token codes
        """
        # Load and preprocess audio
        wav, sr = torchaudio.load(path)
        wav = convert_audio(wav, sr, self.sr, self.channels).unsqueeze(0)
        
        # Encode to tokens
        with torch.no_grad():
            frames = self.codec.encode(wav)            # list[(codes,scale)]
        codes = torch.cat([c for c, _ in frames], dim=-1)   # [1,n_q,T]
        return codes.squeeze(0).long()                      # [n_q,T]

    # -----------------------------------------------------------------
    def __getitem__(self, idx: int):
        """Get a pair of degraded and original audio tokens.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (degraded_tokens, original_tokens)
        """
        if self.use_tokens:
            tok_file = self.token_files[idx]
            tok = torch.load(tok_file)
            X, Y = tok["X"], tok["Y"]
            # Token integrity check
            assert X.shape[0] == self.n_q and Y.shape[0] == self.n_q, f"Token shape mismatch in {tok_file}"
            if self._dbg_left > 0:
                self._dbg_left -= 1
                print(f"[{idx}] {tok_file.name} (tokens) → X {tuple(X.shape)} Y {tuple(Y.shape)}")
            return X, Y
        # Get file paths
        clean = self.clean_wav[idx]
        stem  = clean.stem.split("_")[0]
        deg   = clean.parent / f"{stem}_modified.wav"
        if not deg.exists():
            raise FileNotFoundError(f"Missing degraded {deg}")

        # Convert both files to tokens
        X, Y = self._wav2codes(deg), self._wav2codes(clean)

        # Debug output for first few samples
        if self._dbg_left > 0:
            self._dbg_left -= 1
            stage = clean.parent.parent.name
            print(f"[{idx}] {stage}/{deg.name} → {tuple(X.shape)} "
                f"min {X.min().item()}  max {X.max().item()}")
            print(f"     {stage}/{clean.name} → {tuple(Y.shape)} "
                f"min {Y.min().item()}  max {Y.max().item()}\n")
        return X, Y

    def __len__(self):
        # Return correct length for both token and audio modes
        return len(self.token_files) if self.use_tokens else len(self.clean_wav)

# ---------------------------------------------------------------------
def pad_collate(batch: tp.List[tp.Tuple[torch.Tensor, torch.Tensor]],
                pad_val: int = -100):
    """Collate function for DataLoader to handle variable length sequences.
    
    This function pads all sequences in a batch to the length of the longest one.
    
    Args:
        batch: List of (degraded_tokens, original_tokens) pairs
        pad_val: Value to use for padding
        
    Returns:
        Tuple of (padded_degraded_tokens, padded_original_tokens)
    """
    xs, ys = zip(*batch)
    T = max(t.shape[-1] for t in xs)
    pad = lambda t: torch.nn.functional.pad(t, (0, T - t.shape[-1]),
                                            value=pad_val)
    return torch.stack([pad(x) for x in xs]), torch.stack([pad(y) for y in ys])

# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Quick test of the dataset functionality.
    
    Note: For actual curriculum training, you should use one stage at a time,
    progressing from stage0 to stage4. The ability to load multiple stages
    is mainly for testing and debugging purposes.
    """
    import sys, torch
    from torch.utils.data import DataLoader
    
    # Default to curriculum directory
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments/curriculums"
    
    # Test each stage individually
    test_cases = [
        "stage0_identity",  # No effects
        "stage1_single",    # Single effects
        "stage1_single_stronger",  # Stronger single effects
        "stage2_double",    # Double effects
        "stage3_triple",    # Triple effects
        "stage3_triple_stronger",  # Stronger triple effects
        "stage4_full",      # Full random
        "stage4_full_stronger"     # Stronger full random
    ]
    
    for stage in test_cases:
        print(f"\n{'='*80}")
        print(f"Testing stage: {stage}")
        print(f"{'='*80}\n")
        
        try:
            ds = TokenPairDataset(base_dir, stages=[stage], model_type="48khz", 
                                bandwidth=24.0, max_debug=2)
            dl = DataLoader(ds, 2, shuffle=True, collate_fn=pad_collate, num_workers=8)
            X, Y = next(iter(dl))
            print(f"[batch] X {tuple(X.shape)}   Y {tuple(Y.shape)}")
            print(f"Total samples in stage: {len(ds)}")
        except Exception as e:
            print(f"Error: {e}")
