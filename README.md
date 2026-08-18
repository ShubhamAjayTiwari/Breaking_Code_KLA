# KLA Image Restoration — Baseline Pipeline

Joint denoising + super-resolution baseline for KLA's "AI-Based Restoration of
Degraded Images" hackathon problem.

## What's here

```
kla_restoration/
├── models/
│   ├── unet_restoration.py   # U-Net + PixelShuffle SR head, residual learning
│   └── losses.py              # Combined L1 + SSIM + LPIPS loss
├── data/
│   └── dataset.py              # Paired dataset loader + synthetic data generator
├── scripts/
│   ├── train.py                 # Training loop
│   ├── infer.py                 # REQUIRED submission format: standalone eval script
│   └── compute_metrics.py       # Self-scoring: SSIM/pSNR/LPIPS vs ground truth
├── requirements.txt
└── README.md
```

## Architecture summary

- **Input:** degraded grayscale image (e.g. 256×256, noisy + downsampled)
- **Output:** restored image at ground-truth resolution (e.g. 512×512)
- U-Net encoder-decoder handles denoising context; a PixelShuffle head does the
  learned upsampling. The bicubic-upsampled input is added back as a residual
  connection, so the network only has to learn the *correction* rather than
  reconstructing everything from scratch — this trains faster and more stably.
- ~1-4M parameters depending on `--base_ch` (lightweight on purpose — inference
  time on the H100 is part of the score, so this is not a heavy transformer).

## Getting started on your actual GPU

1. **Once KLA releases the real dataset**, place it so you have:
   ```
   your_dataset/
       ground_truth/   0001.png, 0002.png, ...
       degraded/        0001.png, 0002.png, ...   (matching filenames)
   ```
   If KLA ships a different structure (e.g. a manifest CSV, or 256↔128 pairs
   instead of 512↔256), adjust `data/dataset.py`'s `_list_pairs()` — everything
   else stays the same.

2. **Install dependencies** (on your 4070/3050 machine, with internet access
   for the LPIPS pretrained weights):
   ```
   pip install -r requirements.txt
   ```

3. **Train:**
   ```
   python scripts/train.py --data_root /path/to/your_dataset --epochs 50 \
       --batch_size 16 --patch_size 128 --scale 2 --base_ch 48
   ```
   (Use `--scale 4` if KLA's degradation is 512→128 rather than 512→256.)
   Add `--no_lpips` for a quick first run if you want to skip the LPIPS
   weight download / speed up early iteration.

4. **Run inference exactly as KLA will benchmark it:**
   ```
   python run.py /path/to/test_images outputs
   ```

5. **Self-score before submitting:**
   ```
   python scripts/compute_metrics.py --restored_dir outputs \
       --gt_dir /path/to/test_ground_truth --use_lpips
   ```

## What's already verified working (tested in sandbox, CPU-only)

- Model forward pass: correct input→output shapes, ~1-4M params ✓
- Dataset loader + synthetic data generator (multiplicative speckle model) ✓
- Combined loss (L1 + SSIM confirmed correct; LPIPS needs internet for weights) ✓
- Full training loop: loss decreases across epochs, checkpointing works ✓
- Full inference script: correct upsampling, timing breakdown reported ✓
- Metrics script: SSIM/pSNR computed correctly against ground truth ✓

## Next steps / things to tune once you have real data and a GPU

- **Loss weights** (`w_l1`, `w_ssim`, `w_lpips` in `models/losses.py`) — start
  with the defaults (1.0 / 0.5 / 0.1), then adjust based on which metric is
  weakest on your validation set.
- **`base_ch`** — try 32/48/64 and check the quality-vs-inference-time tradeoff,
  since the score rewards fast pipelines at comparable quality.
- **Augmentation strength** — increase if you see a gap between in-distribution
  and out-of-distribution validation performance.
- **Mixed precision (`torch.cuda.amp`)** — add this to `infer.py` for a likely
  free speedup on the H100 benchmark once you're confident in output quality.
- **Batching in `infer.py`** currently groups by exact input resolution — fine
  if KLA's test images are all one size per split; revisit if sizes vary more.
