"""
Baseline restoration model for the KLA challenge: joint denoising + super-resolution.

Design notes (why it's built this way):
- Input: degraded image, e.g. 256x256 (noisy + low-res)
- Output: restored image at ground-truth resolution, e.g. 512x512 (denoised + upsampled)
- Architecture: U-Net encoder-decoder for denoising context, with a PixelShuffle
  upsampling head at the end to handle the resolution-restoration part.
- This follows the "in-network" joint denoising+SR pattern: the encoder-decoder
  denoises using both local and contextual (down/upsampled) features, and the
  final PixelShuffle stage does the learned upsampling — rather than chaining a
  separate denoiser and a separate SR model, which risks losing SR-relevant
  detail during denoising.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two conv layers + GroupNorm + GELU. GroupNorm is used instead of BatchNorm
    because restoration models often train with small batch sizes (large images
    eat VRAM), and BatchNorm gets noisy/unstable at small batch sizes."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )
        # residual projection if channel count changes
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.Conv2d(in_ch, in_ch, 4, stride=2, padding=1)  # learned downsample
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # handle any off-by-one size mismatch from odd input dims
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class RestorationUNet(nn.Module):
    """
    Args:
        in_ch: input channels (1 for grayscale SEM/inspection images)
        base_ch: base channel width, doubled at each encoder stage
        scale: upsampling factor for the resolution-restoration part (2 or 4)
    """

    def __init__(self, in_ch=1, base_ch=48, scale=2):
        super().__init__()
        self.scale = scale

        self.stem = ConvBlock(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)

        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8)

        self.up3 = Up(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = Up(base_ch * 2, base_ch, base_ch)

        # Super-resolution head: PixelShuffle upsampling.
        # We predict scale^2 times the channels, then rearrange into spatial resolution.
        # This is preferred over plain bilinear+conv because PixelShuffle lets the
        # network learn the upsampling kernel rather than relying on a fixed interpolation.
        self.sr_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.GELU(),
            nn.Conv2d(base_ch, in_ch, 3, padding=1),
        )

        # Long skip: bicubic-upsampled input added back at the end (residual learning).
        # This lets the network focus on learning the *correction* (noise removal +
        # detail synthesis) rather than having to relearn how to reproduce the
        # input's low-frequency content from scratch.
        self.register_buffer("_dummy", torch.zeros(1))  # placeholder, no-op

    def forward(self, x):
        base = nn.functional.interpolate(
            x, scale_factor=self.scale, mode="bicubic", align_corners=False
        )

        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        b = self.bottleneck(x3)

        u3 = self.up3(b, x2)
        u2 = self.up2(u3, x1)
        u1 = self.up1(u2, x0)

        residual = self.sr_head(u1)
        out = base + residual
        return torch.clamp(out, 0.0, 1.0)


if __name__ == "__main__":
    # quick sanity check: does the forward pass run and produce the right shape?
    model = RestorationUNet(in_ch=1, base_ch=32, scale=2)
    dummy_input = torch.rand(2, 1, 256, 256)
    out = model(dummy_input)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input:  {tuple(dummy_input.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"Params: {n_params:,}")
