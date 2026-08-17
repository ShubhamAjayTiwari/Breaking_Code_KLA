r"""
Quick visual sanity check: side-by-side degraded input vs restored output
for a handful of test samples. Since the real test set has no ground truth,
this is your main way to eyeball whether the model is actually restoring
detail (sharper edges, less speckle) or just producing a blurry average.

Supports two modes:
  - 2-column: Degraded | Restored  (original behavior)
  - 3-column: Degraded | Restored (no LPIPS) | Restored (with LPIPS)
    (when --restored_nolpips_dir is provided for A/B comparison)

Usage:
    # Basic 2-column mode:
    python visualize.py --degraded_dir "path\to\Test_NoisyLR\NoisyLR" \
        --restored_dir outputs --n 4

    # 3-column A/B comparison:
    python visualize.py --degraded_dir "path\to\Test_NoisyLR\NoisyLR" \
        --restored_dir outputs --restored_nolpips_dir outputs_no_lpips \
        --n 6 --out comparison_lpips_vs_nolpips.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--degraded_dir", required=True)
    p.add_argument("--restored_dir", required=True,
                   help="Primary restored outputs (e.g. with-LPIPS)")
    p.add_argument("--restored_nolpips_dir", default=None,
                   help="Optional: no-LPIPS restored outputs for A/B comparison")
    p.add_argument("--n", type=int, default=4, help="number of samples to show")
    p.add_argument("--out", type=str, default=None,
                   help="output filename (default: comparison.png or "
                        "comparison_lpips_vs_nolpips.png)")
    p.add_argument("--indices", type=int, nargs="*", default=None,
                   help="specific file indices to show (e.g. --indices 5 42 100)")
    args = p.parse_args()

    deg_dir = Path(args.degraded_dir)
    restored_dir = Path(args.restored_dir)
    nolpips_dir = Path(args.restored_nolpips_dir) if args.restored_nolpips_dir else None

    # Determine which mode we're in
    three_col = nolpips_dir is not None
    ncols = 3 if three_col else 2

    # Pick files
    all_restored = sorted(restored_dir.glob("*.npy"))
    if not all_restored:
        print(f"No .npy files found in {restored_dir}")
        return

    if args.indices is not None:
        # Pick specific files by index into the sorted list
        restored_files = [all_restored[i] for i in args.indices if i < len(all_restored)]
    else:
        # Spread samples evenly across the dataset for diversity
        step = max(1, len(all_restored) // args.n)
        restored_files = all_restored[::step][:args.n]

    nrows = len(restored_files)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    if nrows == 1:
        axes = axes[None, :]

    for i, rpath in enumerate(restored_files):
        # Load degraded
        deg = np.load(deg_dir / rpath.name)
        restored = np.load(rpath)

        # Column 0: Degraded
        axes[i, 0].imshow(deg, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(
            f"{rpath.stem} — Degraded ({deg.shape[0]}×{deg.shape[1]})",
            fontsize=10, fontweight="bold"
        )
        axes[i, 0].axis("off")

        if three_col:
            # Column 1: No-LPIPS restored
            nolpips_path = nolpips_dir / rpath.name
            if nolpips_path.exists():
                nolpips = np.load(nolpips_path)
            else:
                nolpips = np.zeros_like(restored)
            axes[i, 1].imshow(nolpips, cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title(
                f"Restored — no LPIPS ({nolpips.shape[0]}×{nolpips.shape[1]})",
                fontsize=10
            )
            axes[i, 1].axis("off")

            # Column 2: With-LPIPS restored
            axes[i, 2].imshow(restored, cmap="gray", vmin=0, vmax=1)
            axes[i, 2].set_title(
                f"Restored — with LPIPS ({restored.shape[0]}×{restored.shape[1]})",
                fontsize=10, color="green"
            )
            axes[i, 2].axis("off")
        else:
            # Column 1: Restored (original 2-column mode)
            axes[i, 1].imshow(restored, cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title(
                f"{rpath.stem} — Restored ({restored.shape[0]}×{restored.shape[1]})",
                fontsize=10
            )
            axes[i, 1].axis("off")

    plt.tight_layout()

    # Determine output filename
    if args.out:
        out_path = args.out
    elif three_col:
        out_path = "comparison_lpips_vs_nolpips.png"
    else:
        out_path = "comparison.png"

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
