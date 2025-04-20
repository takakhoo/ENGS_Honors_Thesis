#!/usr/bin/env python3
"""
train.py – MasterNet trainer with full logging to MasterNetTraining/
###################################################################

Folders produced (one per run)
└── MasterNetTraining/YYYY‑MM‑DD_HH‑MM‑SS/
    ├── checkpoints/     epoch_000.pt , best.pt …
    ├── images/          val_spec_E005.png , train_spec_E005.png …
    ├── plots/           loss_curve.png   (updated each epoch)
    └── tb/              TensorBoard event files
"""

import os, sys, math, random, argparse
from pathlib import Path
from datetime import datetime

import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data      import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

# (optional) if anomaly detection is giving you extra noise, comment it out:
torch.autograd.set_detect_anomaly(True)

# ─── repo imports ──────────────────────────────────────────────────── #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dataset import PairedAudioDataset
from models  import MasterNet

# ─────────────────────── loss definitions ─────────────────────────── #
class MultiResSTFT(nn.Module):
    """
    Multi‑resolution STFT with improved stability and full spectral range preservation.
    Input : x,y ∈ [B,1,80,T]
    """
    def __init__(self, ffts=(512, 1024, 2048), hop=256):
        super().__init__()
        self.ffts, self.hop = ffts, hop
        for f in ffts:
            win = torch.hann_window(f)
            self.register_buffer(f"win_{f}", win, persistent=False)

    def forward(self, x, y):
        import torch.nn.functional as Fnn
        B, _, C, T = x.shape
        
        # Add small epsilon and clamp while preserving range
        x = torch.clamp(x, min=1e-8, max=1.0)
        y = torch.clamp(y, min=1e-8, max=1.0)
        
        # Flatten mel bins
        x_flat = x.squeeze(1).reshape(B * C, T).contiguous()
        y_flat = y.squeeze(1).reshape(B * C, T).contiguous()

        loss = 0.0
        for f in self.ffts:
            win = getattr(self, f"win_{f}")
            if win.device != x_flat.device:
                win = win.to(x_flat.device)

            # Compute STFTs with normalized windows
            X = torch.stft(
                x_flat, n_fft=f,
                hop_length=self.hop,
                win_length=f,
                window=win/win.sum(),  # Normalize window
                return_complex=True
            )
            Y = torch.stft(
                y_flat, n_fft=f,
                hop_length=self.hop,
                win_length=f,
                window=win/win.sum(),  # Normalize window
                return_complex=True
            )
            
            # Compute magnitude with better numerical stability
            X_mag = torch.sqrt(X.real.pow(2) + X.imag.pow(2) + 1e-8)
            Y_mag = torch.sqrt(Y.real.pow(2) + Y.imag.pow(2) + 1e-8)
            
            # Log-scale magnitudes while preserving full range
            X_log = torch.log(X_mag + 1e-8)
            Y_log = torch.log(Y_mag + 1e-8)
            
            # Compute loss with gradient clipping
            diff = torch.clamp(torch.abs(X_log - Y_log), max=10.0)
            loss += diff.mean()

        return loss / len(self.ffts)


class FullLoss(nn.Module):
    """
    Rebalanced loss components with improved stability while preserving full dynamic range:
    α·L1 mel + β·MS‑STFT + γ·param‑MSE + δ·log‑mel perceptual
    """
    def __init__(self, α=0.35, β=0.15, γ=0.35, δ=0.15,
                sr=22_050, hop=256):
        super().__init__()
        self.α, self.β, self.γ, self.δ = α,β,γ,δ
        self.l1     = nn.L1Loss()
        self.stft   = MultiResSTFT(hop=hop)
        self.mse    = nn.MSELoss()
        
    @staticmethod
    def _logmel(spec: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized output [0–1] back to dB with improved stability
        while preserving full -80dB to 0dB range for HiFi-GAN compatibility
        """
        # Clamp input for stability but preserve full range
        spec = torch.clamp(spec, min=1e-8, max=1.0)
        # Map to full -80dB to 0dB range
        db = spec * 80.0 - 80.0
        # detach before numpy():
        db_np = db.detach().cpu().numpy()
        # Improved amplitude computation while preserving range
        amp = np.clip(librosa.db_to_amplitude(db_np), 1e-8, None)
        # Stable log computation that maintains dynamic range
        logmel = np.log(amp + 1e-8)
        return torch.from_numpy(logmel).to(spec.device)

    def forward(self, out, tgt, p_pred, p_tgt):
        # Add stability clamps while preserving range
        out = torch.clamp(out, min=1e-8, max=1.0)
        tgt = torch.clamp(tgt, min=1e-8, max=1.0)
        
        # Compute losses with better numerical stability
        l1   = self.l1(out, tgt)
        stft = torch.clamp(self.stft(out, tgt), max=10.0)
        mse  = self.mse(p_pred, p_tgt)
        perc = self.l1(self._logmel(out), self._logmel(tgt))
        
        # Dynamic loss scaling based on magnitudes
        l1_scale = torch.exp(-l1).detach()
        stft_scale = torch.exp(-stft).detach()
        mse_scale = torch.exp(-mse).detach()
        perc_scale = torch.exp(-perc).detach()
        
        # Weighted sum with dynamic scaling
        total = (self.α * l1 * l1_scale + 
                self.β * stft * stft_scale + 
                self.γ * mse * mse_scale + 
                self.δ * perc * perc_scale)
        
        return total, dict(l1=l1.item(), stft=stft.item(),
                         mse=mse.item(), perc=perc.item(),
                         total=total.item())

# ─────────────── pad‑to‑max‑T collate for variable length ──────────── #
def collate_pad(batch):
    xs, ys, ps = zip(*batch)
    T = max(t.shape[-1] for t in xs)
    pad = lambda t: F.pad(t,(0,T-t.shape[-1]))
    return (torch.stack([pad(t) for t in xs]),
            torch.stack([pad(t) for t in ys]),
            torch.stack(ps))

# ───────────────────── epoch runner (train/val) ────────────────────── #
def run_epoch(model, loader, opt, scaler, loss_fn, device,
              epoch, epochs, mode, writer, img_dir):
    is_train = mode=="train"
    model.train() if is_train else model.eval()
    meters = {k:0.0 for k in ["l1","stft","mse","perc","total"]}
    batch_losses = {k:[] for k in meters}

    pbar = tqdm(loader, ncols=110,
                desc=f"{mode:5} {epoch:03}/{epochs}")
    for i,(x,y,p) in enumerate(pbar):
        x,y,p = x.to(device), y.to(device), p.to(device)
        
        # Zero gradients at start of loop for better stability
        if is_train:
            opt.zero_grad(set_to_none=True)
            
        with autocast(device_type=device.type, enabled=(device.type=="cuda")):
            out,p_hat = model(x)
            T = p_hat.shape[1]
            loss, parts = loss_fn(out, y,
                                p_hat,
                                p.unsqueeze(1).expand(-1,T,-1))
        if is_train:
            scaler.scale(loss).backward()
            
            # Unscale before clipping
            scaler.unscale_(opt)
            
            # Add gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(opt)
            scaler.update()

        for k in meters:
            meters[k] += parts[k]
            batch_losses[k].append(parts[k])
        if i % max(1,len(loader)//20)==0:
            pbar.set_postfix({k:f"{meters[k]/(i+1):.3f}" for k in meters})

    # epoch means
    for k in meters: meters[k]/=len(loader)
    for k,v in meters.items(): writer.add_scalar(f"{mode}/{k}", v, epoch)

    # save example spectrogram with better formatting
    img = out[0,0].detach().cpu().numpy()
    plt.figure(figsize=(12, 4))
    plt.imshow(img, aspect='auto', origin='lower', cmap='magma')
    cbar = plt.colorbar(label='Magnitude (dB)')
    cbar.ax.tick_params(labelsize=10)
    plt.title(f'{mode.capitalize()} Spectrogram - Epoch {epoch}', 
              fontsize=12, pad=15)
    plt.xlabel('Time (frames)', fontsize=11)
    plt.ylabel('Mel Frequency Bin', fontsize=11)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    plt.savefig(img_dir/f"{mode}_spec_E{epoch:03}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return meters["total"], batch_losses

# ────────────────── update & save loss curve PNG ────────────────────── #
def plot_curve(hist, batch_hist, png_dir):
    """Plot detailed loss curves for each component"""
    components = {
        "l1": "L1 Spectrogram Loss",
        "stft": "Multi-Resolution STFT Loss",
        "mse": "Parameter MSE Loss",
        "perc": "Perceptual Log-Mel Loss",
        "total": "Total Combined Loss"
    }
    
    # Define a professional color palette
    colors = {
        'train': '#1f77b4',  # Deep blue
        'val': '#d62728',    # Deep red
        'grid': '#e0e0e0',   # Light gray
        'text': '#2c3e50',   # Dark blue-gray
        'background': '#f8f9fa'  # Off-white
    }
    
    for component, title in components.items():
        plt.figure(figsize=(12, 7))
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Set background color
        plt.gca().set_facecolor(colors['background'])
        
        # Get the data for this component
        train_vals = hist["train"][component]
        val_vals = hist["val"][component]
        
        # Ensure we have data to plot
        if not train_vals or not val_vals:
            print(f"Warning: No data to plot for {component}")
            plt.close()
            continue
            
        # Create x-axis values based on the length of the data
        epochs = range(1, len(train_vals) + 1)
        
        # Plot training loss
        plt.plot(epochs, train_vals, '-', 
                color=colors['train'], 
                label='Training Loss', 
                linewidth=2.5,
                alpha=0.8)
        
        # Plot validation loss
        plt.plot(epochs, val_vals, '-', 
                color=colors['val'], 
                label='Validation Loss', 
                linewidth=2.5,
                alpha=0.8)
        
        # Add batch-level points for training
        if component in batch_hist["train"]:
            for epoch, batch_losses in enumerate(batch_hist["train"][component], 1):
                if batch_losses:  # Check if we have batch losses
                    plt.scatter([epoch] * len(batch_losses), batch_losses, 
                              c=colors['train'], alpha=0.15, s=15)
        
        # Add batch-level points for validation
        if component in batch_hist["val"]:
            for epoch, batch_losses in enumerate(batch_hist["val"][component], 1):
                if batch_losses:  # Check if we have batch losses
                    plt.scatter([epoch] * len(batch_losses), batch_losses, 
                              c=colors['val'], alpha=0.15, s=15)
        
        # Add statistics
        min_train = min(train_vals)
        min_val = min(val_vals)
        final_train = train_vals[-1]
        final_val = val_vals[-1]
        
        stats_text = (f"Minimum Training Loss: {min_train:.4f}\n"
                     f"Minimum Validation Loss: {min_val:.4f}\n"
                     f"Final Training Loss: {final_train:.4f}\n"
                     f"Final Validation Loss: {final_val:.4f}")
        
        plt.text(0.02, 0.98, stats_text, 
                transform=plt.gca().transAxes,
                verticalalignment='top',
                fontsize=10,
                color=colors['text'],
                bbox=dict(boxstyle='round', 
                         facecolor='white', 
                         alpha=0.8,
                         edgecolor=colors['grid']))
        
        # Customize the plot
        plt.title(title, fontsize=14, pad=20, color=colors['text'])
        plt.xlabel('Training Epoch', fontsize=12, color=colors['text'])
        plt.ylabel('Loss Value', fontsize=12, color=colors['text'])
        
        # Customize grid
        plt.grid(True, color=colors['grid'], linestyle='--', alpha=0.7)
        
        # Customize ticks
        plt.xticks(fontsize=10, color=colors['text'])
        plt.yticks(fontsize=10, color=colors['text'])
        
        # Customize legend
        legend = plt.legend(fontsize=11, framealpha=0.9)
        legend.get_frame().set_edgecolor(colors['grid'])
        
        # Add a subtle border
        for spine in plt.gca().spines.values():
            spine.set_edgecolor(colors['grid'])
            spine.set_linewidth(1.5)
        
        plt.tight_layout()
        
        # Save with high quality
        plt.savefig(png_dir/f"{component}_loss.png", 
                   dpi=300, 
                   bbox_inches='tight',
                   facecolor=colors['background'])
        plt.close()

# ──────────────────────────── main() ────────────────────────────────── #
def main():
    # ----- paths & CLI ------------------------------------------------
    default_data = PROJECT_ROOT/"experiments"/"output_full"/"output_audio"
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(default_data),
        help=f"path containing *_original/_modified wavs (default: {default_data})")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch",  type=int, default=4)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--resume", type=str,
        help="path to checkpoint file to resume training from")
    args = ap.parse_args()

    try:
        data_dir = Path(args.data_dir).expanduser().resolve()
        if not data_dir.is_dir():
            raise FileNotFoundError(f"--data_dir {data_dir} does not exist")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("▶ device:", device)

        # ----- dataset ----------------------------------------------------
        ds = PairedAudioDataset(data_dir, sr=22_050,
                                n_fft=2048, hop_length=256, n_mels=80)
        tr_len = int(0.9*len(ds)); va_len = len(ds)-tr_len
        tr_ds, va_ds = random_split(ds, [tr_len, va_len],
                                    generator=torch.Generator().manual_seed(42))
        tr = DataLoader(tr_ds, args.batch, True,  collate_fn=collate_pad)
        va = DataLoader(va_ds, args.batch, False, collate_fn=collate_pad)
        print(f"▶ dataset  total={len(ds)}  train={len(tr_ds)}  val={len(va_ds)}")

        # ----- run folders ------------------------------------------------
        if args.resume:
            # If resuming, use the same run directory as the checkpoint
            run_root = Path(args.resume).parent.parent
            print(f"▶ resuming in existing run directory: {run_root}")
        else:
            run_root = (PROJECT_ROOT/"MasterNetTraining"/
                        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            print(f"▶ starting new run in directory: {run_root}")
            
        ckpt_dir = run_root/"checkpoints"; img_dir = run_root/"images"
        plot_dir = run_root/"plots"; tb_dir = run_root/"tb"
        for d in (ckpt_dir,img_dir,plot_dir,tb_dir): d.mkdir(parents=True, exist_ok=True)

        # ----- model / optim / sched -------------------------------------
        model = MasterNet().to(device)
        
        # Improved optimizer settings
        opt = optim.AdamW(model.parameters(), 
                          lr=args.lr,
                          weight_decay=2e-4,  # Increased weight decay
                          betas=(0.9, 0.999),
                          eps=1e-8)
        
        # Better learning rate scheduling
        sched = optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(tr),
            pct_start=0.1,  # Warm up for 10% of training
            div_factor=25,  # Initial lr = max_lr/25
            final_div_factor=1000,  # Final lr = max_lr/25000
            anneal_strategy='cos'
        )
        
        scaler = GradScaler(enabled=(device.type=="cuda"))
        loss_fn = FullLoss()  # Using the improved loss function

        start_ep, best = 1, math.inf
        if args.resume:
            try:
                ck = torch.load(args.resume, map_location=device)
                model.load_state_dict(ck["model"])
                opt.load_state_dict(ck["optim"])
                scaler.load_state_dict(ck["scaler"])
                start_ep = ck["epoch"]+1; best = ck["best"]
                
                # Create scheduler first, then load its state
                sched = optim.lr_scheduler.OneCycleLR(
                    opt,
                    max_lr=args.lr,
                    epochs=args.epochs,
                    steps_per_epoch=len(tr),
                    pct_start=0.1,
                    div_factor=25,
                    final_div_factor=1000,
                    anneal_strategy='cos'
                )
                
                # Load scheduler state if it exists in checkpoint
                if "scheduler" in ck:
                    sched.load_state_dict(ck["scheduler"])
                
                print(f"▶ successfully resumed from {args.resume} @ epoch {start_ep}")
                print(f"▶ previous best validation loss: {best:.4f}")
            except Exception as e:
                print(f"❌ Error loading checkpoint {args.resume}: {str(e)}")
                print("Starting fresh training run...")
                start_ep, best = 1, math.inf

        writer = SummaryWriter(str(tb_dir))
        hist = {"train": {k:[] for k in ["l1","stft","mse","perc","total"]},
                "val": {k:[] for k in ["l1","stft","mse","perc","total"]}}
        batch_hist = {"train": {k:[] for k in ["l1","stft","mse","perc","total"]},
                     "val": {k:[] for k in ["l1","stft","mse","perc","total"]}}

        # ----------------------- training loop ----------------------------
        try:
            for ep in range(start_ep, args.epochs+1):
                try:
                    tr_loss, tr_batch_losses = run_epoch(model,tr,opt,scaler,loss_fn,device,
                                        ep,args.epochs,"train",writer,img_dir)
                    with torch.no_grad():
                        va_loss, va_batch_losses = run_epoch(model,va,opt,scaler,loss_fn,device,
                                            ep,args.epochs,"val",writer,img_dir)
                    sched.step()
                    
                    # Update history
                    for k in hist["train"]:
                        hist["train"][k].append(tr_batch_losses[k][-1])
                        hist["val"][k].append(va_batch_losses[k][-1])
                        batch_hist["train"][k].append(tr_batch_losses[k])
                        batch_hist["val"][k].append(va_batch_losses[k])
                    
                    # Save checkpoint before plotting to ensure we don't lose progress
                    ck = {"epoch":ep,"model":model.state_dict(),"optim":opt.state_dict(),
                          "scaler":scaler.state_dict(),"best":best,
                          "scheduler":sched.state_dict()}  # Add scheduler state
                    torch.save(ck, ckpt_dir/f"epoch_{ep:03}.pt")
                    if va_loss < best:
                        best = va_loss
                        torch.save(ck, ckpt_dir/"best.pt")
                        print(f"  ★ new best validation loss: {best:.4f}")
                    
                    # Plot after saving checkpoint
                    plot_curve(hist, batch_hist, plot_dir)

                    print(f"E{ep:03}  train {tr_loss:.3f}  val {va_loss:.3f}  lr {opt.param_groups[0]['lr']:.2e}")
                    
                except Exception as e:
                    print(f"❌ Error in epoch {ep}: {str(e)}")
                    print("Saving emergency checkpoint...")
                    torch.save({
                        "epoch": ep,
                        "model": model.state_dict(),
                        "optim": opt.state_dict(),
                        "scaler": scaler.state_dict(),
                        "best": best,
                        "scheduler": sched.state_dict(),  # Add scheduler state
                        "error": str(e)
                    }, ckpt_dir/f"emergency_epoch_{ep:03}.pt")
                    raise  # Re-raise the exception after saving

        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
            print("Saving emergency checkpoint...")
            torch.save({
                "epoch": ep,
                "model": model.state_dict(),
                "optim": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "best": best,
                "scheduler": sched.state_dict(),  # Add scheduler state
                "interrupted": True
            }, ckpt_dir/f"interrupted_epoch_{ep:03}.pt")
            print("Emergency checkpoint saved. You can resume later with --resume")
            return

        print("✔ training finished – artefacts saved to", run_root)

    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        return

# ────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    main()
