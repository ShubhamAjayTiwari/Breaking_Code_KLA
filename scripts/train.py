"""
Training script for the KLA restoration baseline.

Usage:
    python scripts/train.py --data_root /path/to/kla_dataset --epochs 30 --scale 2

Point --data_root at KLA's real dataset once released (expects ground_truth/
and degraded/ subfolders — see data/dataset.py for the layout, adjust the
loader if KLA ships a different structure e.g. a manifest CSV).

For now, running without --data_root will auto-generate a small synthetic
dataset so you can confirm the whole pipeline (data -> model -> loss ->
backward -> checkpoint) runs end-to-end before real data arrives.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.dataset import PairedRestorationDataset, make_synthetic_dataset
from models.unet_restoration import RestorationUNet
from models.losses import CombinedRestorationLoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=None,
                    help="Path to dataset with ground_truth/ and degraded/ folders. "
                         "If omitted, a synthetic dataset is generated for pipeline testing.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=128, help="degraded-image crop size")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base_ch", type=int, default=48)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--use_lpips", action="store_true", default=True,
                    help="Set --no_lpips to disable (e.g. if no internet for weights)")
    p.add_argument("--no_lpips", dest="use_lpips", action="store_false")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.data_root is None:
        print("No --data_root given -> generating a small synthetic dataset for a pipeline smoke test.")
        args.data_root = "synthetic_data"
        make_synthetic_dataset(args.data_root, n=32, gt_size=256, scale=args.scale)

    full_ds = PairedRestorationDataset(
        args.data_root, patch_size=args.patch_size, scale=args.scale, augment=True
    )
    n_val = max(1, int(len(full_ds) * args.val_frac))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = RestorationUNet(in_ch=1, base_ch=args.base_ch, scale=args.scale).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    if args.use_lpips:
        loss_fn = CombinedRestorationLoss(w_l1=1.0, w_ssim=0.5, w_lpips=0.1).to(device)
    else:
        # fallback: L1 + SSIM only (no internet / no LPIPS weights available)
        from models.losses import CombinedRestorationLoss as _L
        loss_fn = _L(w_l1=1.0, w_ssim=0.5, w_lpips=0.0).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = {"l1": 0.0, "ssim_loss": 0.0, "lpips": 0.0, "total": 0.0}

        for deg, gt in train_loader:
            deg, gt = deg.to(device), gt.to(device)
            optimizer.zero_grad()
            pred = model(deg)
            loss, parts = loss_fn(pred, gt)
            loss.backward()
            optimizer.step()
            for k in running:
                running[k] += parts[k]

        scheduler.step()
        n_batches = len(train_loader)
        train_msg = " | ".join(f"{k}={v/n_batches:.4f}" for k, v in running.items())
        print(f"[Epoch {epoch}/{args.epochs}] train: {train_msg} | time={time.time()-t0:.1f}s")

        # validation
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for deg, gt in val_loader:
                deg, gt = deg.to(device), gt.to(device)
                pred = model(deg)
                loss, _ = loss_fn(pred, gt)
                val_total += loss.item()
        val_avg = val_total / max(1, len(val_loader))
        print(f"           val_loss={val_avg:.4f}")

        if val_avg < best_val:
            best_val = val_avg
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"           -> saved new best checkpoint (val_loss={val_avg:.4f})")

    torch.save(model.state_dict(), out_dir / "last_model.pt")
    print(f"Training done. Best val_loss={best_val:.4f}. Checkpoints in {out_dir}/")


if __name__ == "__main__":
    main()
