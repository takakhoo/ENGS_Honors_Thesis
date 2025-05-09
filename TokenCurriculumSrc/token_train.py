#!/usr/bin/env python3
"""
Curriculum-aware Token-UNet Trainer
==================================
- Trains TokenUNet on curriculum stages, advancing only when validation loss plateaus or training loss stagnates.
- Dynamically adjusts dropout, bottleneck, learning rate, batch size, and gradient clipping per stage.
- Robust to NaNs, OOMs, and resumes from checkpoint.
- Enhanced plotting and logging for deep debugging and analysis.
- Stores outputs in CurriculumTraining/stageX_name/ckpts, imgs, logs.
- 48kHz, 24kbps EnCodec enforced throughout.

Stage Advancement Rules:
- Advance when:
    - Validation loss plateaus (3-epoch moving average, abs(ema[-1] - ema[-6]) < 0.05 for 5 epochs, and only after min_epochs_per_stage)
    - Or training loss stagnates for 8+ epochs
- min_epochs_per_stage = max(10, n_train // 100)
- Apply higher dropout (0.15) in stage 4+
- Switch use_bottleneck=True in stage4_full_stronger
- Shrink LR by 25–50% each stage (start 5e-5, down to 1e-5 in later stages)
- OneCycleLR with pct_start=0.1 early, 0.02 in final stages
- AdamW (foreach=False, fused=False)
- Validation/test split (80/10/10), fixed seed (min 1 for val/test)
- Dynamic batch size: increase if memory allows
- Adaptive gradient clipping: adjust based on grad norm stats
- Minimal but critical prints for debugging, warnings, and stage transitions
- Save/restore all state for robust resume
- Save best model globally as CurriculumTraining/best_overall.pt
- Save per-stage logs as training_log.csv in stage_dir/logs/
- --until_stage CLI flag to stop at an intermediate stage
- (Future: attention memory optimization if needed)
"""
from __future__ import annotations
import argparse, math, random, time, warnings, traceback, csv
from datetime import datetime
from pathlib import Path
import os, gc
import torch, torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from token_dataset import TokenPairDataset, pad_collate
from token_unet    import TokenUNet, PAD

# ──────────────── Hyper-params and Stage Config ────────────────
STAGES = [
    ("stage0_identity", 0.10, False, 5e-5, 0.1),
    ("stage1_single",   0.10, False, 4e-5, 0.1),
    ("stage1_single_stronger", 0.10, False, 3e-5, 0.1),
    ("stage2_double",   0.10, False, 2.5e-5, 0.08),
    ("stage3_triple",   0.10, False, 2e-5, 0.05),
    ("stage3_triple_stronger", 0.10, False, 1.5e-5, 0.05),
    ("stage4_full",     0.15, False, 1.2e-5, 0.02),
    ("stage4_full_stronger", 0.15, True, 1e-5, 0.02),
]
VAL_FRAC, TEST_FRAC = 0.10, 0.10
MAX_PATIENCE = 5
TRAIN_PLATEAU_EPOCHS = 8
GRAD_NORM_WARN = 1000
SEED = 42
BATCH_START = 2
BATCH_MAX = 4
BATCH_INC_EPOCHS = 3  # Try to increase batch size every N epochs if no OOM
CLIP_BASE = 2.0
CLIP_MIN = 1.0
CLIP_MAX = 4.0
CLIP_WINDOW = 10  # Number of epochs to track grad norm for adaptive clipping

# ──────────────── Utility Functions ────────────────
def pretty(t): m,s=divmod(int(t),60); h,m=divmod(m,60); return f"{h}:{m:02d}:{s:02d}"
def set_seed(s): random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def plot_curves(logs, outdir):
    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1); plt.plot(logs['train_loss'], label='train'); plt.plot(logs['val_loss'], label='val'); plt.legend(); plt.title('Loss'); plt.grid()
    plt.subplot(2,2,2); plt.plot(logs['grad_norm'], label='grad norm'); plt.axhline(GRAD_NORM_WARN, color='r', ls='--'); plt.title('Grad Norm'); plt.grid()
    plt.subplot(2,2,3); plt.plot(logs['lr'], label='lr'); plt.title('Learning Rate'); plt.grid()
    plt.subplot(2,2,4); plt.plot(logs['stage'], label='stage'); plt.title('Stage'); plt.grid()
    plt.tight_layout(); plt.savefig(outdir/'training_curves.png', dpi=120); plt.close()

def print_gpu_stats(where):
    print(f"[GPU:{where}] {torch.cuda.memory_summary()}")
    objs = gc.get_objects()
    tensors = [obj for obj in objs if torch.is_tensor(obj) and obj.is_cuda]
    total = sum(t.numel() * t.element_size() for t in tensors)
    print(f"[MEMCHK:{where}] {len(tensors)} CUDA tensors, total {total/1024/1024:.2f} MB")

def grad_norm(model):
    total_norm = 0.0
    max_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            max_norm = max(max_norm, param_norm.item())
    return total_norm ** 0.5, max_norm

def moving_average(seq, window=3):
    if len(seq) < window:
        return seq[:]
    return [sum(seq[max(0,i-window+1):i+1])/min(window,i+1) for i in range(len(seq))]

# ──────────────── Main Curriculum Training Loop ────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--until_stage', type=int, default=None, help='Stop at this stage index (0-based, inclusive)')
    args = parser.parse_args()
    set_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    run_root = Path("CurriculumTraining")
    run_root.mkdir(exist_ok=True)
    logs = {'train_loss':[], 'val_loss':[], 'grad_norm':[], 'lr':[], 'stage':[]}
    stage_idx = 0
    best_val = float('inf')
    best_overall = None
    batch_size = BATCH_START
    clip_val = CLIP_BASE
    grad_norm_hist = []
    resume_epoch = 1
    # Resume logic
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=dev)
        stage_idx = resume_ckpt['stage_idx']
        best_val = resume_ckpt.get('best_val', float('inf'))
        resume_epoch = resume_ckpt['epoch'] + 1
        print(f"▶ Resumed from {args.resume}, stage {stage_idx}, best val {best_val:.4f}, epoch {resume_epoch}")
    while stage_idx < len(STAGES):
        if args.until_stage is not None and stage_idx > args.until_stage:
            print(f"[EXIT] Stopping at stage {stage_idx-1} as requested by --until_stage.")
            break
        stage_name, dropout, use_bottleneck, max_lr, pct_start = STAGES[stage_idx]
        print(f"\n=== Starting {stage_name} (dropout={dropout}, bottleneck={use_bottleneck}, max_lr={max_lr}, batch={batch_size}, clip={clip_val}) ===")
        stage_dir = run_root / stage_name
        (stage_dir/'ckpts').mkdir(parents=True, exist_ok=True)
        (stage_dir/'imgs').mkdir(exist_ok=True)
        (stage_dir/'logs').mkdir(exist_ok=True)
        # DATASET ---------------------------------------------------
        ds = TokenPairDataset("experiments/curriculums", stages=[stage_name], model_type="48khz", bandwidth=24.0)
        n = len(ds)
        if n < 10:
            print(f"[SKIP] Stage {stage_name} has <10 samples, skipping.")
            stage_idx += 1
            continue
        n_val = int(n * VAL_FRAC)
        n_test = int(n * TEST_FRAC)
        if n_test == 0: n_test = 1
        if n_val == 0: n_val = 1
        n_train = n - n_val - n_test
        train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(SEED))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate, num_workers=8)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, num_workers=8)
        testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=pad_collate)
        # MODEL -----------------------------------------------------
        net = TokenUNet(ds.n_q, dropout=dropout, use_bottleneck=use_bottleneck).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=max_lr, betas=(0.9,0.95), weight_decay=1e-2, foreach=False, fused=False)
        sched = OneCycleLR(opt, max_lr=max_lr, total_steps=100*len(train_loader), pct_start=pct_start, div_factor=30, final_div_factor=300)
        scaler = GradScaler(enabled=False)
        patience, plateau_epochs = 0, 0
        val_loss_hist, train_loss_hist = [], []
        # Resume model/opt/sched if resuming
        if args.resume:
            net.load_state_dict(resume_ckpt['model'])
            opt.load_state_dict(resume_ckpt['opt'])
            # Re-init scheduler if max_lr or batch_size changed
            sched = OneCycleLR(opt, max_lr=max_lr, total_steps=100*len(train_loader), pct_start=pct_start, div_factor=30, final_div_factor=300)
            sched.load_state_dict(resume_ckpt['sched'])
        # Per-stage log file
        log_path = stage_dir/'logs'/'training_log.csv'
        with open(log_path, 'w', newline='') as logf:
            logwriter = csv.writer(logf)
            logwriter.writerow(['epoch','train_loss','val_loss','grad_norm','lr','batch','clip','time'])
            min_epochs_per_stage = max(10, n_train // 100)
            for epoch in range(resume_epoch, 1001):
                net.train(); t0 = time.time(); running = 0.
                opt.zero_grad(set_to_none=True)
                print_gpu_stats(f"{stage_name}_epoch{epoch}_start")
                oom_flag = False
                for step, (x, y) in enumerate(train_loader, 1):
                    try:
                        x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
                        with autocast(enabled=False, device_type=dev.type):
                            logits = net(x)
                            loss = F.cross_entropy(logits, y, ignore_index=PAD, label_smoothing=0.05)
                        if not torch.isfinite(loss):
                            print(f"[NaN] Loss is not finite at step {step}, skipping batch.")
                            print(traceback.format_exc())
                            opt.zero_grad(set_to_none=True)
                            torch.cuda.empty_cache()
                            continue
                        loss.backward()
                        tnorm, maxnorm = grad_norm(net)
                        grad_norm_hist.append(tnorm)
                        if tnorm > GRAD_NORM_WARN:
                            print(f"[WARN] Grad norm spike: {tnorm:.1f} (max {maxnorm:.1f}) at step {step}")
                        # Adaptive gradient clipping
                        if len(grad_norm_hist) > CLIP_WINDOW:
                            recent = grad_norm_hist[-CLIP_WINDOW:]
                            std = torch.std(torch.tensor(recent)).item()
                            mean = torch.mean(torch.tensor(recent)).item()
                            if mean < 1.5: clip_val = max(CLIP_MIN, clip_val - 0.1)
                            elif mean > 3.0: clip_val = min(CLIP_MAX, clip_val + 0.1)
                        if step % 2 == 0:
                            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_val)
                            try:
                                opt.step(); opt.zero_grad(set_to_none=True); sched.step()
                            except RuntimeError as e:
                                if 'out of memory' in str(e):
                                    print(f"[OOM] Out of memory at step {step}, reducing batch size.")
                                    batch_size = max(BATCH_START, batch_size - 1)
                                    oom_flag = True
                                    torch.cuda.empty_cache()
                                    break
                                else:
                                    raise
                            torch.cuda.empty_cache()
                        running += loss.item()
                    except Exception as e:
                        print(f"[EXC] Exception at step {step}: {e}")
                        print(traceback.format_exc())
                        opt.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        continue
                if oom_flag:
                    print(f"[ADAPT] Restarting epoch {epoch} with batch size {batch_size}")
                    break  # Restart epoch with new batch size
                avg_train = running / len(train_loader)
                train_loss_hist.append(avg_train)
                # Try to increase batch size every BATCH_INC_EPOCHS if no OOM
                if epoch % BATCH_INC_EPOCHS == 0 and batch_size < BATCH_MAX:
                    batch_size += 1
                    print(f"[ADAPT] Increased batch size to {batch_size}")
                # VALIDATION -------------------------------------------
                net.eval(); vloss = 0.
                with torch.no_grad():
                    for x, y in val_loader:
                        x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
                        logits = net(x)
                        loss = F.cross_entropy(logits, y, ignore_index=PAD, label_smoothing=0.05)
                        vloss += loss.item()
                avg_val = vloss / len(val_loader)
                val_loss_hist.append(avg_val)
                logs['train_loss'].append(avg_train)
                logs['val_loss'].append(avg_val)
                logs['grad_norm'].append(tnorm)
                logs['lr'].append(sched.get_last_lr()[0])
                logs['stage'].append(stage_idx)
                # Write per-epoch log row
                logwriter.writerow([epoch, avg_train, avg_val, tnorm, sched.get_last_lr()[0], batch_size, clip_val, pretty(time.time()-t0)])
                print(f"[ep {epoch:03d}] train {avg_train:.4f}  val {avg_val:.4f}  grad_norm {tnorm:.1f}  lr {sched.get_last_lr()[0]:.2e}  batch {batch_size}  clip {clip_val:.2f}")
                print_gpu_stats(f"{stage_name}_epoch{epoch}_end")
                # Plateau detection with moving average and min_epochs_per_stage
                val_ema = moving_average(val_loss_hist, window=3)
                can_advance = (epoch >= min_epochs_per_stage)
                plateau = (len(val_ema) > 6 and abs(val_ema[-1] - val_ema[-6]) < 0.05)
                if plateau and can_advance:
                    patience += 1
                    print(f"[PLATEAU] Validation loss plateaued for {patience} epochs.")
                else:
                    patience = 0
                if len(train_loss_hist) > TRAIN_PLATEAU_EPOCHS and abs(train_loss_hist[-1] - train_loss_hist[-TRAIN_PLATEAU_EPOCHS]) < 0.05:
                    plateau_epochs += 1
                    print(f"[PLATEAU] Training loss stagnated for {plateau_epochs} epochs.")
                else:
                    plateau_epochs = 0
                # Save best checkpoint (per-stage)
                if avg_val < best_val:
                    best_val = avg_val
                    torch.save({'epoch': epoch, 'model': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'best_val': best_val, 'stage_idx': stage_idx}, stage_dir/'ckpts'/'best.pt')
                # Save best overall checkpoint
                if best_overall is None or avg_val < best_overall['val']:
                    best_overall = {'epoch': epoch, 'model': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'val': avg_val, 'stage_idx': stage_idx, 'stage_name': stage_name}
                    torch.save(best_overall, run_root/'best_overall.pt')
                # Save every epoch
                torch.save({'epoch': epoch, 'model': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'best_val': best_val, 'stage_idx': stage_idx}, stage_dir/'ckpts'/f'epoch_{epoch:03d}.pt')
                plot_curves(logs, stage_dir/'imgs')
                # Stage advancement
                if (patience >= MAX_PATIENCE or plateau_epochs >= TRAIN_PLATEAU_EPOCHS) and can_advance:
                    print(f"[ADVANCE] Advancing to next stage after epoch {epoch}")
                    break
        # TEST SET EVALUATION --------------------------------------
        net.eval(); test_loss = 0.
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                logits = net(x)
                test_loss += F.cross_entropy(logits, y, ignore_index=PAD, label_smoothing=0.05).item()
        print(f"[TEST] Stage {stage_name} final test loss: {test_loss / len(testloader):.4f}")
        stage_idx += 1
        resume_epoch = 1  # Reset for next stage
        # Adjust LR, dropout, bottleneck, batch size for next stage automatically
        # (Handled by STAGES table and dynamic logic)
        # (Future: attention memory optimization if needed)
    print("✅ Curriculum training complete.")
    if best_overall is not None:
        print(f"🏁 Best model was in {best_overall['stage_name']} with val {best_overall['val']:.4f}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
