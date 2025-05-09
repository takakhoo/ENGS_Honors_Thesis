# ------------------------------------------------------------
# src/token_unet.py – 1-D U-Net for EnCodec tokens (Curriculum Ready)
# ------------------------------------------------------------
# Redesigned for curriculum learning: memory-efficient, robust, and modular.
# Includes CBAM, FiLM, learnable skip gating, and transposed conv upsampling.
#
# Refinements:
#   - Optional bottleneck (1x1 Conv1d) after mid block (use_bottleneck)
#   - Configurable dropout rate (dropout)
#   - set_dropout() method for dynamic adjustment
#
# When to use options:
#   - use_bottleneck=True: Only if you see poor generalization or loss spikes in stage 4/4-stronger (set from training script)
#   - dropout > 0.10: If you see overfitting in later stages, especially stage 4 (set from training script or via set_dropout)
#   - Gradient norm printing: Should be handled in the training script, not here
#   - Checkpointing: Leave off unless you hit OOM (set from training script)
#
# Control all options from the training script for flexibility.

from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange

# ─── Model hyper-parameters ──────────────────────────────────
BASE_DIM, DEPTH, K = 384, 4, 1024  # Base channels, depth, vocab size
PAD = -100  # Padding value

# ─── CBAM: Convolutional Block Attention Module ─────────────
class CBAM(nn.Module):
    """Lightweight attention: channel + spatial attention."""
    def __init__(self, ch, reduction=16, kernel=7):
        super().__init__()
        # Channel attention
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(ch, ch // reduction, 1), nn.ReLU(),
            nn.Conv1d(ch // reduction, ch, 1)
        )
        # Spatial attention
        self.conv = nn.Conv1d(2, 1, kernel, padding=kernel//2)
    def forward(self, x):
        # Channel attention
        w = torch.sigmoid(self.mlp(x))
        x = x * w
        # Spatial attention
        max_out, _ = torch.max(x, 1, keepdim=True)
        avg_out = torch.mean(x, 1, keepdim=True)
        s = torch.cat([max_out, avg_out], dim=1)
        s = torch.sigmoid(self.conv(s))
        return x * s

# ─── FiLM: Feature-wise Linear Modulation ───────────────────
class FiLM(nn.Module):
    """Learnable per-channel scale and shift."""
    def __init__(self, ch):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, ch, 1))
        self.beta = nn.Parameter(torch.zeros(1, ch, 1))
    def forward(self, x):
        return x * self.gamma + self.beta

# ─── Residual Block ─────────────────────────────────────────
class ResBlock(nn.Module):
    """Residual block: conv → norm → GELU → dropout → conv → norm → GELU."""
    def __init__(self, ch, dropout=0.10):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
            nn.GELU()
        )
    def forward(self, x):
        return x + self.body(x)

# ─── U-Net main class ───────────────────────────────────────
class TokenUNet(nn.Module):
    """
    1D U-Net for EnCodec tokens, curriculum-ready.
    - 4 encoder/decoder blocks, base_dim=384, doubling channels
    - CBAM and FiLM for attention and modulation
    - ConvTranspose1d for upsampling
    - Learnable skip gating
    - Optional bottleneck after mid block
    - Configurable dropout rate
    - Robust to different curriculum stages
    """
    def __init__(self, n_q: int, k: int = K, base_dim: int = BASE_DIM, depth: int = DEPTH,
                 checkpointing: bool = False, use_bottleneck: bool = False, dropout: float = 0.10):
        super().__init__()
        self.n_q, self.k, self.depth = n_q, k, depth
        self.checkpointing = checkpointing
        self.use_bottleneck = use_bottleneck
        self.dropout = dropout
        # Token embedding: [B, n_q, T] → [B, n_q*emb_q, T] → [B, base_dim, T]
        emb_q = base_dim // n_q
        self.emb = nn.Embedding(n_q * k, emb_q)
        self.inp = nn.Conv1d(n_q * emb_q, base_dim, 1)
        # Encoder: 2x ResBlock + CBAM + downsample (Conv1d)
        ch, enc = base_dim, nn.ModuleList()
        self.skip_gates = nn.ParameterList()
        for _ in range(depth):
            enc.append(nn.Sequential(
                ResBlock(ch, dropout=dropout),
                CBAM(ch),
                ResBlock(ch, dropout=dropout),
                CBAM(ch),
                nn.Conv1d(ch, ch*2, 4, stride=2, padding=1)
            ))
            self.skip_gates.append(nn.Parameter(torch.tensor(1.0)))  # Learnable skip gate
            ch *= 2
        self.enc = enc
        # Mid block: ResBlock + Dropout + FiLM
        self.mid = nn.Sequential(
            ResBlock(ch, dropout=dropout),
            nn.Dropout(dropout),
            FiLM(ch)
        )
        # Optional bottleneck after mid block
        if use_bottleneck:
            self.bottleneck = nn.Conv1d(ch, ch, 1)
        else:
            self.bottleneck = None
        # Decoder: upsample (ConvTranspose1d) + 2x ResBlock + CBAM
        dec = nn.ModuleList()
        for _ in range(depth):
            dec.append(nn.Sequential(
                nn.ConvTranspose1d(ch, ch//2, 4, stride=2, padding=1),
                ResBlock(ch//2, dropout=dropout),
                CBAM(ch//2),
                ResBlock(ch//2, dropout=dropout),
                CBAM(ch//2)
            ))
            ch //= 2
        self.dec = dec
        # Prediction heads: one per codebook
        self.heads = nn.ModuleList([
            nn.Conv1d(base_dim, k, 1) for _ in range(n_q)
        ])
        self.apply(self._init_w)

    @staticmethod
    def _init_w(m: nn.Module):
        """Kaiming initialization for conv/linear layers."""
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_uniform_(m.weight, a=.2, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None: nn.init.zeros_(m.bias)

    @staticmethod
    def _crop(t: torch.Tensor, tgt: int):
        """Center crop for skip connections if needed."""
        diff = t.size(-1) - tgt
        if diff <= 0: return t
        lo = diff // 2
        return t[..., lo:lo + tgt]

    def set_dropout(self, new_dropout: float):
        """Dynamically set dropout rate for all ResBlocks and mid Dropout."""
        self.dropout = new_dropout
        # Update encoder
        for blk in self.enc:
            for layer in blk:
                if isinstance(layer, ResBlock):
                    for m in layer.body:
                        if isinstance(m, nn.Dropout):
                            m.p = new_dropout
        # Update mid block
        for m in self.mid:
            if isinstance(m, nn.Dropout):
                m.p = new_dropout
            if isinstance(m, ResBlock):
                for mm in m.body:
                    if isinstance(mm, nn.Dropout):
                        mm.p = new_dropout
        # Update decoder
        for blk in self.dec:
            for layer in blk:
                if isinstance(layer, ResBlock):
                    for m in layer.body:
                        if isinstance(m, nn.Dropout):
                            m.p = new_dropout

    def forward(self, tok: torch.LongTensor):
        """
        Args:
            tok: [B, n_q, T] EnCodec token indices
        Returns:
            logits: [B, K, n_q, T] unnormalized probabilities
        """
        B, n_q, T = tok.shape
        pad_mask = tok.eq(PAD)
        tok_safe = tok.masked_fill(pad_mask, 0)
        # Token embedding
        offset = (torch.arange(n_q, device=tok.device) * self.k)[None, :, None]
        idx = offset + tok_safe
        x = rearrange(self.emb(idx), 'b q t d -> b (q d) t')
        x = self.inp(x)
        # Encoder
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            skips.append(x)
        # Mid block
        x = self.mid(x)
        # Optional bottleneck
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        # Decoder
        for i, blk in enumerate(self.dec):
            skip = skips.pop()
            gate = torch.sigmoid(self.skip_gates[-(i+1)])  # Learnable skip gate (0-1)
            x = blk(x + gate * self._crop(skip, x.size(-1)))
        # Ensure output length matches input
        if x.size(-1) != T:
            x = F.interpolate(x, size=T, mode='nearest')
        # Prediction heads (one per codebook)
        B, C, T = x.shape
        K = self.k
        logits = torch.empty(B, K, self.n_q, T, device=x.device, dtype=x.dtype)
        for qi, head in enumerate(self.heads):
            logits[:, :, qi, :] = head(x)
        return logits

def print_model_stats(model):
    """Prints parameter count and MACs (if ptflops is available)."""
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model Stats] Total parameters: {n_params:,}")
    try:
        from ptflops import get_model_complexity_info
        # Example input: (n_q, T) with T=4096
        macs, params = get_model_complexity_info(model, (model.n_q, 4096), as_strings=True, print_per_layer_stat=False)
        print(f"[Model Stats] MACs: {macs}")
    except ImportError:
        print("[Model Stats] ptflops not installed; skipping MACs.")

# ────── Self-test: run "python src/token_unet.py" ───────────
if __name__ == "__main__":
    """
    Quick test of the model's shape transformations and block connections.
    Loads a small batch from TokenPairDataset and runs a forward pass.
    To test with bottleneck or different dropout, edit below:
        net = TokenUNet(ds.n_q, use_bottleneck=True, dropout=0.15)
    """
    from token_dataset import TokenPairDataset, pad_collate
    from torch.utils.data import DataLoader
    import sys
    print("▶ quick shape-check (curriculum-ready TokenUNet)")
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "thesis_project/experiments/curriculums"
    ds = TokenPairDataset(base_dir, stages=["stage0_identity"], model_type="48khz", bandwidth=24.0, max_debug=1)
    dl = DataLoader(ds, 2, collate_fn=pad_collate, num_workers=8)
    x, _ = next(iter(dl))
    # Edit here to test with/without bottleneck and different dropout
    net = TokenUNet(ds.n_q, use_bottleneck=False, dropout=0.10)
    print_model_stats(net)
    with torch.no_grad():
        logits = net(x)
        print(f"[test] input: {tuple(x.shape)}  logits: {tuple(logits.shape)}")
        print("Model test completed. If no errors, block connections are stable.")
