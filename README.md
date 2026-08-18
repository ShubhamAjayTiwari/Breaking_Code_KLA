# KLA Image Restoration — Baseline Pipeline

Joint denoising + super-resolution baseline for KLA's "AI-Based Restoration of
Degraded Images" hackathon problem.

## Repository structure

```
kla_restoration/
├── models/
│   ├── unet_restoration.py        # U-Net + PixelShuffle SR head, residual learning
│   ├── losses.py                  # Combined L1 + SSIM + LPIPS loss
│   └── best_model.pt              # Trained model weights (copy for run.py)
├── data/
│   └── dataset.py                 # Paired .npy dataset loader + synthetic data generator
├── scripts/
│   ├── train.py                   # Training loop (AdamW + CosineAnnealing)
│   ├── infer.py                   # Batch inference with timing breakdown
│   └── compute_metrics.py         # Self-scoring: SSIM / pSNR / LPIPS vs ground truth
├── checkpoints/
│   ├── best_model.pt              # Best validation-loss checkpoint
│   ├── best_model_no_lpips.pt     # Best checkpoint (L1 + SSIM only, no LPIPS)
│   └── last_model.pt              # Last-epoch checkpoint
├── outputs/                       # Restored .npy files (with-LPIPS model)
├── outputs_no_lpips/              # Restored .npy files (no-LPIPS model)
├── run.py                         # ⭐ SUBMISSION ENTRY POINT: python run.py <in> <out>
├── visualize.py                   # Side-by-side comparison plots (2- or 3-column)
├── comparison.png                 # Sample visual comparison output
├── comparison_lpips_vs_nolpips.png # LPIPS vs no-LPIPS A/B comparison
├── requirements.txt
└── README.md
```

## Architecture summary

```
 Degraded Input (.npy)                              Restored Output (.npy)
  128×128 grayscale                                   256×256 grayscale
        │                                                    ▲
        │                                                    │
        ▼                                              ┌─────┴─────┐
  ┌───────────┐                                        │   clamp   │
  │   Stem    │ ConvBlock(1 → base_ch)                 │  [0, 1]   │
  │           │                                        └─────┬─────┘
  └─────┬─────┘                                              │
        │ x0 (skip ─────────────────────────────┐      ┌─────┴─────┐
        ▼                   connection)         │      │     +     │ residual add
  ┌───────────┐                                 │      │  (base +  │
  │  Down 1   │ Strided Conv → ConvBlock        │      │ residual) │
  │ base → 2x │                                 │      └─────┬─────┘
  └─────┬─────┘                                 │        ▲         ▲
        │ x1 (skip ──────────────────┐          │        │         │
        ▼                            │          │   ┌────┘    ┌────┘
  ┌───────────┐                      │          │   │         │
  │  Down 2   │                      │          │   │    Bicubic Upsample
  │ 2x → 4x  │                      │          │   │    (input × scale)
  └─────┬─────┘                      │          │   │
        │ x2 (skip ───────┐         │          │   │
        ▼                  │         │          │   │
  ┌───────────┐            │         │          │   │
  │  Down 3   │            │         │          │   │
  │ 4x → 8x  │            │         │          │   │
  └─────┬─────┘            │         │          │   │
        │                  │         │          │   │
        ▼                  │         │          │   │
  ┌───────────┐            │         │          │   │
  │ Bottleneck│            │         │          │   │
  │ 8x → 8x  │            │         │          │   │
  └─────┬─────┘            │         │          │   │
        │                  │         │          │   │
        ▼                  ▼         │          │   │
  ┌───────────┐     ┌──────────┐    │          │   │
  │   Up 3    │◄────│ concat   │    │          │   │
  │ 8x → 4x  │     │ (x2)     │    │          │   │
  └─────┬─────┘     └──────────┘    │          │   │
        │                           ▼          │   │
        ▼                    ┌──────────┐      │   │
  ┌───────────┐              │ concat   │      │   │
  │   Up 2    │◄─────────────│ (x1)     │      │   │
  │ 4x → 2x  │              └──────────┘      │   │
  └─────┬─────┘                                │   │
        │                                      ▼   │
        ▼                               ┌─────────┐
  ┌───────────┐                          │ concat  │
  │   Up 1    │◄─────────────────────────│ (x0)    │
  │ 2x → base│                          └─────────┘
  └─────┬─────┘
        │
        ▼
  ┌─────────────────────┐
  │   SR Head            │
  │ Conv → PixelShuffle  │  base_ch → base_ch×scale² → rearrange → base_ch
  │ → GELU → Conv(→1ch)  │
  └──────────┬──────────┘
             │ (residual)
             └────────────────────────────────────► (+) → output
```

### Key design decisions

| Decision | Rationale |
|---|---|
| **U-Net encoder-decoder** | Combines local and contextual (down/upsampled) features for denoising — single-scale networks miss the global context needed to distinguish noise from structure. |
| **PixelShuffle SR head** | Learned upsampling (rather than fixed bilinear/bicubic interpolation) lets the network synthesize high-frequency detail the degradation destroyed. |
| **Residual learning** | The bicubic-upsampled input is added back at the end, so the network only needs to learn the *correction* (noise removal + detail synthesis), not reconstruct low-frequency content from scratch — trains faster and more stably. |
| **GroupNorm (not BatchNorm)** | Restoration models train with small batch sizes (large images eat VRAM); BatchNorm becomes noisy/unstable at small batch sizes. |
| **GELU activation** | Smoother than ReLU, avoids dead neurons in the correction-learning regime. |
| **Learned downsampling** | `Down` blocks use a strided Conv2d(k=4, s=2, p=1) instead of max-pool — preserves more spatial information for the skip connections. |
| **~1-4M parameters** | Lightweight by design — inference time on the H100 is part of the KLA score, so this is deliberately not a heavy transformer. Controlled via `--base_ch` (default 48). |

### Loss function (`models/losses.py`)

Combined loss: **L1** (pixel fidelity) + **SSIM** (structural similarity) + **LPIPS** (perceptual quality).

| Component | Default weight | Purpose |
|---|---|---|
| L1 | 1.0 | Drives pSNR — pixel-accurate reconstruction |
| 1 − SSIM | 0.5 | Structural / contrast fidelity |
| LPIPS (AlexNet) | 0.1 | Learned perceptual metric — captures texture realism that pixel-wise losses miss |

> LPIPS expects 3-channel input in [-1, 1]; the loss auto-converts grayscale → 3-ch by
> repeating the single channel. Use `--no_lpips` during training if you lack internet
> access for the pretrained AlexNet weight download.

## Data format (KLA's actual dataset)

```
train/train/
├── GT/         000000.npy, 000001.npy, ...   (256×256, float32, [0,1])
└── NoisyLR/    000000.npy, 000001.npy, ...   (128×128, float32, may exceed [0,1])

Test_NoisyLR/
└── NoisyLR/    (degraded-only, no GT — the actual test set)
```

- Files are **raw NumPy arrays** (`.npy`), not images — loaded with `np.load()`, not PIL.
- Filenames match 1:1 between GT and NoisyLR by stem.
- Scale factor: **×2** (128 → 256).
- NoisyLR values can exceed `[0, 1]` due to multiplicative speckle noise.

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
python scripts/train.py --data_root path/to/train/train --epochs 50 \
    --batch_size 16 --patch_size 64 --scale 2 --base_ch 48
```

Key flags:
- `--patch_size 64` — trains on random 64×64 degraded crops (→ 128×128 GT crops), multiplying effective dataset size
- `--no_lpips` — skip LPIPS (faster, no internet needed for weight download)
- `--val_frac 0.1` — 10% validation split (default)
- Checkpoints are saved to `checkpoints/` (best + last)

### 3. Run inference (submission format)

```bash
# This is exactly what KLA will benchmark on the H100:
python run.py path/to/Test_NoisyLR/NoisyLR outputs
```

`run.py` loads the model from `models/best_model.pt`, batches inputs by resolution, and writes restored `.npy` files to the output directory.

For more detailed inference with timing breakdown:

```bash
python scripts/infer.py --input_dir path/to/Test_NoisyLR/NoisyLR \
    --output_dir outputs --checkpoint checkpoints/best_model.pt
```

### 4. Self-score before submitting

```bash
python scripts/compute_metrics.py --restored_dir outputs \
    --gt_dir path/to/train/train/GT --use_lpips
```

Reports per-image SSIM, pSNR, and LPIPS statistics (mean ± std).

### 5. Visualize results

```bash
# 2-column: Degraded | Restored
python visualize.py --degraded_dir "path/to/Test_NoisyLR/NoisyLR" \
    --restored_dir outputs --n 4

# 3-column A/B comparison: Degraded | No-LPIPS | With-LPIPS
python visualize.py --degraded_dir "path/to/Test_NoisyLR/NoisyLR" \
    --restored_dir outputs --restored_nolpips_dir outputs_no_lpips \
    --n 6 --out comparison_lpips_vs_nolpips.png
```

Use `--indices 5 42 100` to pick specific samples.

## What's verified working

- ✅ Model forward pass: correct input→output shapes (128×128 → 256×256), ~1-4M params
- ✅ Dataset loader reads KLA's real `.npy` format (GT/ and NoisyLR/ subdirs)
- ✅ Combined loss — L1 + SSIM confirmed correct; LPIPS with AlexNet backbone working
- ✅ Full training loop — loss decreases across epochs, best/last checkpointing works
- ✅ `run.py` submission script — correct batched upsampling, `.npy` output
- ✅ Metrics script — SSIM / pSNR / LPIPS computed correctly against ground truth
- ✅ Visualization — 2-column and 3-column (LPIPS vs no-LPIPS) comparison plots
- ✅ Trained checkpoints included — `best_model.pt`, `best_model_no_lpips.pt`, `last_model.pt`

## Next steps / things to tune

- **Loss weights** (`w_l1`, `w_ssim`, `w_lpips`) — start with the defaults
  (1.0 / 0.5 / 0.1), then adjust based on which metric is weakest on your validation set.
- **`base_ch`** — try 32/48/64 and check the quality-vs-inference-time tradeoff,
  since the score rewards fast pipelines at comparable quality.
- **Augmentation strength** — increase if you see a gap between in-distribution
  and out-of-distribution validation performance.
- **Mixed precision (`torch.cuda.amp`)** — add this to `run.py` / `infer.py` for a likely
  free speedup on the H100 benchmark once you're confident in output quality.
- **Larger patch size** — currently training on 64×64 crops; try 96 or 128 if VRAM allows,
  which gives the model more context per gradient step.
- **Learning rate warmup** — the current CosineAnnealing starts at full LR; adding
  a short warmup (5-10% of epochs) may stabilize early training.
