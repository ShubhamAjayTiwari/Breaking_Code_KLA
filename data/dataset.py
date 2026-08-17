"""
Paired dataset loader for KLA's degraded/ground-truth image pairs.

CONFIRMED ACTUAL FOLDER LAYOUT (KLA's real dataset, as of Aug 2026):

    train/train/GT/          000000.npy, 000001.npy, ...   (256x256 float32, [0,1])
    train/train/NoisyLR/     000000.npy, 000001.npy, ...   (128x128 float32, can exceed [0,1])
    Test_NoisyLR/NoisyLR/    (degraded-only, no GT — this is the actual test set)

Files are raw NumPy arrays (.npy), NOT image files (png/jpg) — loaded with
np.load(), not PIL. Filenames match 1:1 between GT and NoisyLR by stem.

`gt_subdir`/`deg_subdir` are configurable in case KLA later renames folders
or the test set structure differs slightly.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _list_pairs(root, gt_subdir="GT", deg_subdir="NoisyLR"):
    gt_dir = Path(root) / gt_subdir
    deg_dir = Path(root) / deg_subdir
    gt_files = {p.stem: p for p in gt_dir.glob("*.npy")}
    deg_files = {p.stem: p for p in deg_dir.glob("*.npy")}
    common = sorted(set(gt_files) & set(deg_files))
    if not common:
        raise RuntimeError(
            f"No matching .npy filename pairs found between {gt_dir} and {deg_dir}. "
            f"Check the folder names and that files share the same stems (e.g. 000000.npy)."
        )
    return [(deg_files[k], gt_files[k]) for k in common]


class PairedRestorationDataset(Dataset):
    def __init__(self, root, patch_size=64, scale=2, augment=True,
                 gt_subdir="GT", deg_subdir="NoisyLR"):
        """
        Args:
            root: dataset root folder, e.g. ".../train/train" (contains GT/ and NoisyLR/)
            patch_size: size of the *degraded* (low-res) patch to crop for training.
                        The corresponding GT crop is patch_size * scale.
                        Default 64 -> 128x128 GT crops, matching the real data's
                        128->256 resolution pair. Training on random crops (not full
                        images) multiplies effective dataset size and keeps memory low.
            scale: upsampling factor — 2 for this dataset (128 -> 256)
            augment: random flips/rotations for generalization (helps OOD test set)
            gt_subdir / deg_subdir: folder names under `root` — defaults match
                                     KLA's actual release ("GT", "NoisyLR")
        """
        self.pairs = _list_pairs(root, gt_subdir, deg_subdir)
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def _load(self, path):
        # KLA's files are raw float32 .npy arrays, already normalized
        # (GT in [0,1]; NoisyLR can go slightly outside due to speckle noise).
        # No division by 255 needed — that would be wrong here, unlike for
        # standard 8-bit image files.
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]  # (H, W) -> (1, H, W)
        elif arr.ndim == 3 and arr.shape[0] not in (1, 3):
            arr = arr.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W), just in case
        return arr

    def __getitem__(self, idx):
        deg_path, gt_path = self.pairs[idx]
        deg = self._load(deg_path)
        gt = self._load(gt_path)

        ps = self.patch_size
        _, dh, dw = deg.shape
        _, gh, gw = gt.shape

        # sanity: GT should be exactly `scale`x the degraded image
        assert gh == dh * self.scale and gw == dw * self.scale, (
            f"Size mismatch for {deg_path.name}: degraded {dh}x{dw}, "
            f"ground_truth {gh}x{gw}, expected GT = {self.scale}x degraded"
        )

        if dh >= ps and dw >= ps:
            top = random.randint(0, dh - ps)
            left = random.randint(0, dw - ps)
        else:
            top, left = 0, 0
            ps = min(dh, dw)

        deg_crop = deg[:, top:top + ps, left:left + ps]
        gt_crop = gt[:, top * self.scale: (top + ps) * self.scale,
                        left * self.scale: (left + ps) * self.scale]

        if self.augment:
            if random.random() < 0.5:
                deg_crop = np.ascontiguousarray(deg_crop[:, :, ::-1])
                gt_crop = np.ascontiguousarray(gt_crop[:, :, ::-1])
            if random.random() < 0.5:
                deg_crop = np.ascontiguousarray(deg_crop[:, ::-1, :])
                gt_crop = np.ascontiguousarray(gt_crop[:, ::-1, :])
            if random.random() < 0.5:
                deg_crop = np.ascontiguousarray(deg_crop.transpose(0, 2, 1))
                gt_crop = np.ascontiguousarray(gt_crop.transpose(0, 2, 1))

        return torch.from_numpy(deg_crop.copy()), torch.from_numpy(gt_crop.copy())


def make_synthetic_dataset(root, n=8, gt_size=512, scale=2, seed=0):
    """
    Generates a tiny synthetic paired dataset ONLY for locally testing this
    pipeline before KLA's real dataset is available. Simulates:
      - resolution loss (downsampling by `scale`)
      - multiplicative speckle noise (gamma-distributed, mean 1 — matches the
        real statistical model, not naive additive noise)
      - additive Gaussian noise
    Delete/ignore this once you have the real KLA dataset — swap in the
    provided files instead.
    """
    rng = np.random.default_rng(seed)
    gt_dir = Path(root) / "ground_truth"
    deg_dir = Path(root) / "degraded"
    gt_dir.mkdir(parents=True, exist_ok=True)
    deg_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        # simple synthetic "inspection-like" texture: blobs + lines
        yy, xx = np.mgrid[0:gt_size, 0:gt_size]
        img = 0.5 + 0.3 * np.sin(xx / 15.0) * np.cos(yy / 15.0)
        n_blobs = rng.integers(5, 15)
        for _ in range(n_blobs):
            cx, cy = rng.integers(0, gt_size, size=2)
            r = rng.integers(10, 40)
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
            img[mask] += rng.uniform(-0.3, 0.3)
        img = np.clip(img, 0, 1).astype(np.float32)

        gt_img = Image.fromarray((img * 255).astype(np.uint8), mode="L")
        gt_img.save(gt_dir / f"{i:04d}.png")

        # downsample for resolution loss
        deg_size = gt_size // scale
        deg = np.asarray(
            gt_img.resize((deg_size, deg_size), Image.BICUBIC), dtype=np.float32
        ) / 255.0

        # multiplicative speckle: multiply by Gamma(L, 1/L) field, mean 1
        L = 4.0
        speckle = rng.gamma(shape=L, scale=1.0 / L, size=deg.shape)
        deg = deg * speckle

        # additive Gaussian noise on top
        deg = deg + rng.normal(0, 0.03, size=deg.shape)

        deg = np.clip(deg, 0, None)  # allow >1 briefly then rescale like real speckle behavior
        deg_img = Image.fromarray(
            np.clip(deg * 255, 0, 255).astype(np.uint8), mode="L"
        )
        deg_img.save(deg_dir / f"{i:04d}.png")

    return str(root)


if __name__ == "__main__":
    import tempfile
    tmp = tempfile.mkdtemp()
    make_synthetic_dataset(tmp, n=4, gt_size=256, scale=2)
    ds = PairedRestorationDataset(tmp, patch_size=64, scale=2)
    deg, gt = ds[0]
    print(f"Dataset size: {len(ds)}")
    print(f"Degraded patch: {tuple(deg.shape)}  GT patch: {tuple(gt.shape)}")
