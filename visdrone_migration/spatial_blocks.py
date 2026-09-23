# SPDX-License-Identifier: AGPL-3.0-only
"""Wavelet and dynamic spatial operators for the extension experiments.

HaarWTConv adapts the MIT-licensed ECCV 2024 WTConv implementation; see
licenses/WTConv-LICENSE and docs/NEW_MODULES.md. LSConv is an independent
implementation of the CVPR 2025 large-perception/small-aggregation algorithm,
using ordinary PyTorch operations instead of the authors' Triton kernels.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def haar_filters(channels: int) -> torch.Tensor:
    low = torch.tensor([math.sqrt(0.5), math.sqrt(0.5)], dtype=torch.float32)
    high = low * torch.tensor([1.0, -1.0])
    bank = torch.stack([torch.outer(low, low), torch.outer(high, low),
                        torch.outer(low, high), torch.outer(high, high)])
    return bank[:, None].repeat(channels, 1, 1, 1)


class HaarWTConv(nn.Module):
    """Two-level db1 WTConv with immutable analysis/synthesis filters.

Only the Haar case used in this experiment is implemented. Fixed filters are
buffers, not counted as trainable parameters. Odd dimensions are zero-padded
at each level and cropped during reconstruction, as in the reference.
"""
    def __init__(self, channels: int, levels: int = 2, kernel_size: int = 5):
        super().__init__()
        if channels < 1 or levels < 1 or kernel_size % 2 != 1:
            raise ValueError("positive channels/levels and odd kernel size required")
        self.channels, self.levels = channels, levels
        self.register_buffer("analysis_filter", haar_filters(channels))
        self.register_buffer("synthesis_filter", haar_filters(channels))
        self.base = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2,
                              groups=channels, bias=True)
        self.base_scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bands = nn.ModuleList(nn.Conv2d(4 * channels, 4 * channels, kernel_size,
                                  padding=kernel_size // 2, groups=4 * channels,
                                  bias=False) for _ in range(levels))
        self.band_scales = nn.ParameterList(nn.Parameter(torch.full((1, 4 * channels, 1, 1), 0.1))
                                           for _ in range(levels))

    def forward(self, x):
        low = x
        history = []
        for convolution, scale in zip(self.bands, self.band_scales):
            n, c, h, w = low.shape
            padded = F.pad(low, (0, w % 2, 0, h % 2))
            analysis = F.conv2d(padded, self.analysis_filter, stride=2, groups=c)
            height, width = analysis.shape[-2:]
            low = analysis.reshape(n, c, 4, height, width)[:, :, 0]
            processed = (convolution(analysis) * scale).reshape(n, c, 4, height, width)
            history.append((processed, h, w))
        reconstruction = 0
        for processed, height, width in reversed(history):
            combined = torch.cat(((processed[:, :, 0] + reconstruction).unsqueeze(2),
                                  processed[:, :, 1:]), dim=2)
            reconstruction = F.conv_transpose2d(combined.flatten(1, 2),
                self.synthesis_filter, stride=2, groups=self.channels)[..., :height, :width]
        return self.base(x) * self.base_scale + reconstruction


class WTBlock(nn.Module):
    """WTConv spatial mixer in a residual, expansion-2 pointwise FFN wrapper."""
    def __init__(self, channels):
        super().__init__()
        self.spatial = HaarWTConv(channels, levels=2)
        self.mlp = nn.Sequential(nn.Conv2d(channels, 2 * channels, 1, bias=False),
            nn.BatchNorm2d(2 * channels), nn.ReLU(),
            nn.Conv2d(2 * channels, channels, 1, bias=False))

    def forward(self, x):
        return x + self.mlp(self.spatial(x))


class DynamicSmallKernel(nn.Module):
    """Per-pixel 3x3 aggregation; channel c uses kernel c modulo kernel_groups.

The implementation is deliberately transparent and autograd-compatible. Its
explicit unfolded tensor has different memory/latency costs from fused Triton.
"""
    def forward(self, x, weights):
        batch, channels, height, width = x.shape
        groups = weights.shape[1]
        if channels % groups or weights.shape != (batch, groups, 9, height, width):
            raise ValueError("weights must be [B, G, 9, H, W] with C divisible by G")
        # Reshape C into [C/G, G], preserving the cyclic (not contiguous) sharing.
        patches = F.unfold(x, kernel_size=3, padding=1).reshape(
            batch, channels // groups, groups, 9, height, width)
        return (patches * weights[:, None]).sum(dim=3).reshape(batch, channels, height, width)


class LSConv(nn.Module):
    """Paper-based large-kernel perception + small-kernel dynamic aggregation.

The source repository's default shares one 3x3 kernel among eight cyclic input
channels (G=C/8). No source code or custom kernel from that repository is vendored.
"""
    def __init__(self, channels: int):
        super().__init__()
        if channels < 8 or channels % 8:
            raise ValueError("LSConv requires channels divisible by eight")
        self.channels, self.kernel_groups = channels, channels // 8
        hidden = channels // 2
        self.perception = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 7, padding=3, groups=hidden, bias=False), nn.BatchNorm2d(hidden),
            nn.Conv2d(hidden, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, 9 * self.kernel_groups, 1),
            nn.GroupNorm(self.kernel_groups, 9 * self.kernel_groups))
        self.aggregate = DynamicSmallKernel()
        self.output_norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        weights = self.perception(x).reshape(x.shape[0], self.kernel_groups, 9, *x.shape[-2:])
        return x + self.output_norm(self.aggregate(x, weights))
