import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add current directory to path if needed to import from models
sys.path.append(str(Path(__file__).resolve().parent))
from models.unet_restoration import RestorationUNet

def load_array(path):
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]  # (H, W) -> (1, H, W)
    return arr

def save_array(arr, path):
    # Ensure [0, 1] and no nan/inf
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    
    # Save as (H, W) or (H, W, 1), depending on what's preferred.
    # The requirement says "(H, W) or (H, W, 1)". arr[0] gives (H, W).
    np.save(path, arr[0].astype(np.float32))

def main():
    if len(sys.argv) < 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    # The default checkpoint is assumed to be in models/ or checkpoints/
    # The rules say it should contain all required model weights.
    checkpoint_path = Path(__file__).resolve().parent / "models" / "best_model.pt"
    
    # Scale and base_ch should match training.
    # infer.py used scale=2, base_ch=48
    scale = 2
    base_ch = 48
    batch_size = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RestorationUNet(in_ch=1, base_ch=base_ch, scale=scale)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}.")
        
    model.to(device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.npy"))
    if not files:
        print(f"No .npy files found in {input_dir}")
        return

    # Group files by input resolution so we can batch
    groups = {}
    for f in files:
        arr = np.load(f, mmap_mode="r") 
        groups.setdefault(arr.shape, []).append(f)

    with torch.no_grad():
        for shape, group_files in groups.items():
            for i in range(0, len(group_files), batch_size):
                batch_files = group_files[i:i + batch_size]
                batch = np.stack([load_array(f) for f in batch_files])  # (B,1,H,W)
                batch_t = torch.from_numpy(batch).to(device)

                out = model(batch_t).cpu().numpy()

                for f, restored in zip(batch_files, out):
                    save_array(restored, output_dir / f.name)

if __name__ == "__main__":
    main()
