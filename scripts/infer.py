"""
Standalone evaluation script — REQUIRED SUBMISSION FORMAT per KLA's rules:
  - Plain Python script (not a notebook)
  - Accepts: path to test images directory, path to output directory
  - Loads the trained model, runs inference on all inputs, writes outputs
  - Must run with zero manual edits (this is what gets benchmarked on the H100)

Usage:
    python scripts/infer.py --input_dir /path/to/test_images --output_dir /path/to/outputs \
        --checkpoint checkpoints/best_model.pt --scale 2

Everything not required for inference (data augmentation, loss functions, etc.)
is deliberately excluded here to keep startup time and import overhead minimal
-- inference time is part of the score.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.unet_restoration import RestorationUNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True,
                    help="Directory of degraded test .npy files")
    p.add_argument("--output_dir", type=str, required=True,
                    help="Directory to write restored .npy files to")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--base_ch", type=int, default=48)
    p.add_argument("--batch_size", type=int, default=8,
                    help="Batch inference for throughput; files are grouped by "
                         "matching resolution since the model needs fixed-size batches.")
    return p.parse_args()


def load_array(path):
    # KLA's data is raw float32 .npy — no rescaling needed, unlike 8-bit images.
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]  # (H, W) -> (1, H, W)
    return arr


def save_array(arr, path):
    # arr: (1, H, W) float32. Save as .npy to match KLA's data format exactly
    # (their benchmark likely compares .npy to .npy, not rendered images).
    np.save(path, arr[0].astype(np.float32))


def main():
    args = parse_args()
    t_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RestorationUNet(in_ch=1, base_ch=args.base_ch, scale=args.scale)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.npy"))
    if not files:
        print(f"No .npy files found in {in_dir}")
        return

    t_setup_done = time.time()
    print(f"Setup (model load): {t_setup_done - t_start:.2f}s | {len(files)} files to process")

    # Group files by input resolution so we can batch (fixed-size tensors per batch).
    groups = {}
    for f in files:
        arr = np.load(f, mmap_mode="r")  # just peek at shape, don't load fully yet
        groups.setdefault(arr.shape, []).append(f)

    n_done = 0
    with torch.no_grad():
        for shape, group_files in groups.items():
            for i in range(0, len(group_files), args.batch_size):
                batch_files = group_files[i:i + args.batch_size]
                batch = np.stack([load_array(f) for f in batch_files])  # (B,1,H,W)
                batch_t = torch.from_numpy(batch).to(device)

                out = model(batch_t).cpu().numpy()

                for f, restored in zip(batch_files, out):
                    save_array(restored, out_dir / f.name)
                n_done += len(batch_files)

    t_end = time.time()
    print(f"Inference + I/O: {t_end - t_setup_done:.2f}s for {n_done} images "
          f"({(t_end - t_setup_done)/max(1,n_done)*1000:.1f} ms/image)")
    print(f"Total end-to-end: {t_end - t_start:.2f}s")
    print(f"Restored images written to: {out_dir}")


if __name__ == "__main__":
    main()
