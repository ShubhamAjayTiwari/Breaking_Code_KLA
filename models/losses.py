"""
Combined restoration loss: L1 (pixel fidelity) + SSIM (structure) + LPIPS (perceptual).

Why combine all three instead of just one:
- L1 alone -> correlates well with pSNR, but tends to produce slightly blurry,
  "safe" outputs (regresses to the mean over plausible textures).
- SSIM term -> pushes for structural/contrast fidelity, complements L1.
- LPIPS -> a learned perceptual metric (deep features), captures texture realism
  that pixel-wise losses miss entirely. This is what the KLA slides explicitly
  ask for ("combine perceptual loss (LPIPS) with pixel-level metrics").

LPIPS expects 3-channel inputs normalized to roughly [-1, 1] and a specific
backbone (default: AlexNet, lightweight and fast). We convert grayscale ->
3-channel by repeating the single channel, since KLA's images are single-channel.
"""

import lpips
import torch
import torch.nn as nn
from pytorch_msssim import ssim


class CombinedRestorationLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_ssim=0.5, w_lpips=0.1, lpips_net="alex"):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_lpips = w_lpips
        self.l1 = nn.L1Loss()
        # Only load the LPIPS backbone (downloads pretrained weights on first use)
        # if it's actually going to be used. Lets you train with --no_lpips when
        # you don't have internet access for the weight download.
        if self.w_lpips > 0:
            self.lpips_fn = lpips.LPIPS(net=lpips_net)
            for p in self.lpips_fn.parameters():
                p.requires_grad = False
        else:
            self.lpips_fn = None

    def _to_lpips_input(self, x):
        # x is (B, 1, H, W) in [0, 1] -> LPIPS wants (B, 3, H, W) in [-1, 1]
        x3 = x.repeat(1, 3, 1, 1)
        return x3 * 2 - 1

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)

        # SSIM returns similarity (1 = identical), so loss is (1 - ssim)
        ssim_val = ssim(pred, target, data_range=1.0, size_average=True)
        ssim_loss = 1.0 - ssim_val

        if self.lpips_fn is not None:
            lpips_loss = self.lpips_fn(
                self._to_lpips_input(pred), self._to_lpips_input(target)
            ).mean()
        else:
            lpips_loss = torch.tensor(0.0, device=pred.device)

        total = (
            self.w_l1 * l1_loss
            + self.w_ssim * ssim_loss
            + self.w_lpips * lpips_loss
        )

        return total, {
            "l1": l1_loss.item(),
            "ssim_loss": ssim_loss.item(),
            "lpips": lpips_loss.item(),
            "total": total.item(),
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    loss_fn = CombinedRestorationLoss()
    pred = torch.rand(2, 1, 128, 128)
    target = torch.rand(2, 1, 128, 128)
    total, parts = loss_fn(pred, target)
    print("Loss breakdown:", parts)
