"""
Self-evaluation: computes the same metrics KLA will score you on
(SSIM, pSNR, LPIPS) between your restored outputs and ground truth,
so you can check quality BEFORE submitting.

Usage:
    python scripts/compute_metrics.py --restored_dir test_outputs \
        --gt_dir synthetic_data/ground_truth --use_lpips
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import ssim


def load(path):
    # KLA's data is raw float32 .npy — loaded directly, no 8-bit rescaling.
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]  # if saved as (1,H,W), squeeze to (H,W)
    return torch.from_numpy(arr)[None, None, ...]  # (1,1,H,W)


def psnr(pred, target, data_range=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(data_range ** 2 / mse).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--restored_dir", required=True)
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--use_lpips", action="store_true")
    args = p.parse_args()

    restored_dir = Path(args.restored_dir)
    gt_dir = Path(args.gt_dir)

    restored_files = {f.stem: f for f in restored_dir.glob("*.npy")}
    gt_files = {f.stem: f for f in gt_dir.glob("*.npy")}
    common = sorted(set(restored_files) & set(gt_files))
    if not common:
        print("No matching .npy filenames between restored and ground-truth dirs.")
        return

    lpips_fn = None
    if args.use_lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex")

    ssim_scores, psnr_scores, lpips_scores = [], [], []
    for stem in common:
        pred = load(restored_files[stem])
        target = load(gt_files[stem])
        if pred.shape != target.shape:
            print(f"Skipping {stem}: shape mismatch {tuple(pred.shape)} vs {tuple(target.shape)}")
            continue

        ssim_scores.append(ssim(pred, target, data_range=1.0).item())
        psnr_scores.append(psnr(pred, target))

        if lpips_fn is not None:
            pred3 = pred.repeat(1, 3, 1, 1) * 2 - 1
            target3 = target.repeat(1, 3, 1, 1) * 2 - 1
            with torch.no_grad():
                lpips_scores.append(lpips_fn(pred3, target3).item())

    print(f"Evaluated {len(ssim_scores)} image pairs")
    print(f"SSIM:  mean={np.mean(ssim_scores):.4f}  std={np.std(ssim_scores):.4f}")
    finite_psnr = [x for x in psnr_scores if np.isfinite(x)]
    print(f"pSNR:  mean={np.mean(finite_psnr):.2f} dB  std={np.std(finite_psnr):.2f} dB")
    if lpips_scores:
        print(f"LPIPS: mean={np.mean(lpips_scores):.4f}  std={np.std(lpips_scores):.4f}  (lower is better)")


if __name__ == "__main__":
    main()
