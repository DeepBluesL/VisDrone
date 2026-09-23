# SPDX-License-Identifier: AGPL-3.0-only
"""Detector-ready blocks adapted from the occluded-MNIST project.

The CSP and Ghost-style blocks retain the historical LeakyReLU activation.
The fixed-filter stem stores its bank as a buffer so normal module
initializers and optimizers cannot overwrite it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k: int | Sequence[int], p=None, d: int = 1):
    """Return padding that preserves spatial size for a stride-one convolution."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Convolution, batch normalization, and historical LeakyReLU activation."""

    default_act = nn.LeakyReLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if c1 < 1 or c2 < 1:
            raise ValueError("convolution channel counts must be positive")
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class DWConv(Conv):
    """Grouped convolution using the greatest common divisor of channel counts."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class GhostConv(nn.Module):
    """Ghost convolution with an explicit even-output-channel contract."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        if c2 < 2 or c2 % 2:
            raise ValueError(f"GhostConv requires a positive even c2, got {c2}")
        hidden = c2 // 2
        self.cv1 = Conv(c1, hidden, k, s, None, g, act=act)
        self.cv2 = Conv(hidden, hidden, 5, 1, None, hidden, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), dim=1)


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        hidden = int(c2 * e)
        if hidden < 1:
            raise ValueError("Bottleneck hidden channel count must be positive")
        self.cv1 = Conv(c1, hidden, k[0], 1)
        self.cv2 = Conv(hidden, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        hidden = int(c2 * e)
        if hidden < 1:
            raise ValueError("C3 hidden channel count must be positive")
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.cv3 = Conv(2 * hidden, c2, 1)
        self.m = nn.Sequential(
            *(Bottleneck(hidden, hidden, shortcut, g, ((1, 1), (3, 3)), 1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        if self.c < 1:
            raise ValueError("C2f hidden channel count must be positive")
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, ((3, 3), (3, 3)), 1.0)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(block(y[-1]) for block in self.m)
        return self.cv2(torch.cat(y, dim=1))


class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        hidden = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck(hidden, hidden, shortcut, g, (k, k), 1.0) for _ in range(n))
        )


class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class _GhostBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        if s not in (1, 2):
            raise ValueError("Ghost bottleneck stride must be 1 or 2")
        if s == 1 and c1 != c2:
            raise ValueError("stride-one Ghost bottleneck requires c1 == c2")
        hidden = c2 // 2
        if hidden < 2 or hidden % 2 or c2 % 2:
            raise ValueError("Ghost bottleneck c2 must be divisible by 4")
        self.conv = nn.Sequential(
            GhostConv(c1, hidden, 1, 1),
            DWConv(hidden, hidden, k, s, act=False) if s == 2 else nn.Identity(),
            GhostConv(hidden, c2, 1, 1, act=False),
        )
        self.shortcut = (
            nn.Sequential(
                DWConv(c1, c1, k, s, act=False),
                Conv(c1, c2, 1, 1, act=False),
            )
            if s == 2
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class C3Ghost(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        hidden = int(c2 * e)
        if hidden % 4:
            raise ValueError("C3Ghost hidden channel count must be divisible by 4")
        self.m = nn.Sequential(*(_GhostBottleneck(hidden, hidden) for _ in range(n)))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        if channels < 1 or reduction < 1:
            raise ValueError("channels and reduction must be positive")
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        scale = self.fc(self.avg_pool(x).view(batch, channels)).view(batch, channels, 1, 1)
        return x * scale


class EnhancedSPPF(nn.Module):
    def __init__(self, c1, c2, k=(3, 5, 7)):
        super().__init__()
        if not k or any(size < 1 or size % 2 == 0 for size in k):
            raise ValueError("EnhancedSPPF kernels must be non-empty positive odd integers")
        hidden = c1 // 2
        if hidden < 1:
            raise ValueError("EnhancedSPPF requires c1 >= 2")
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(hidden * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(nn.MaxPool2d(size, 1, size // 2) for size in k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [pool(x) + x for pool in self.m], dim=1))


def _normalize_filters(filters: torch.Tensor) -> torch.Tensor:
    filters = filters - filters.mean(dim=(-2, -1), keepdim=True)
    norms = filters.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-12)
    return filters / norms


def _gabor_bank(kernel_size: int = 5) -> torch.Tensor:
    axis = torch.linspace(-kernel_size // 2 + 1, kernel_size // 2, kernel_size)
    y, x = torch.meshgrid(axis, axis, indexing="xy")
    kernels = []
    for sigma in (0.5, 1.0, 1.5, 2.0):
        for index in range(8):
            theta = index * math.pi / 8
            x_theta = x * math.cos(theta) + y * math.sin(theta)
            y_theta = -x * math.sin(theta) + y * math.cos(theta)
            kernel = torch.exp(
                -(x_theta.square() + 0.5**2 * y_theta.square()) / (2 * sigma**2)
            ) * torch.cos(2 * math.pi * x_theta / 2.0)
            kernels.append(kernel)
    return _normalize_filters(torch.stack(kernels).unsqueeze(1))


def _random_bank(seed: int, kernel_size: int = 5) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    filters = torch.randn(32, 1, kernel_size, kernel_size, generator=generator)
    return _normalize_filters(filters)


class GaborStem(nn.Module):
    """Fixed normalized filter bank followed by a trainable 1x1 projection.

    Both modes use 32 zero-mean, unit-L2 spatial filters. For multi-channel
    input, each filter is replicated and divided by ``c1``, implementing an
    explicit channel average before filtering. The 5x5 stride-two operation
    has the same output spatial size as YOLO's 3x3 stride-two stem.
    """

    def __init__(self, c1, c2, mode="gabor", seed=179):
        super().__init__()
        if c1 < 1 or c2 < 1:
            raise ValueError("GaborStem channel counts must be positive")
        if mode not in {"gabor", "random"}:
            raise ValueError("mode must be 'gabor' or 'random'")
        bank = _gabor_bank() if mode == "gabor" else _random_bank(int(seed))
        bank = bank.repeat(1, c1, 1, 1) / c1
        self.c1 = c1
        self.mode = mode
        self.seed = int(seed)
        self.register_buffer("filter_bank", bank, persistent=True)
        self.proj = Conv(32, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.c1:
            raise ValueError(f"expected BCHW input with {self.c1} channels")
        x = F.conv2d(x, self.filter_bank, stride=2, padding=2)
        return self.proj(x)


__all__ = [
    "Bottleneck",
    "C2f",
    "C3",
    "C3Ghost",
    "C3k",
    "C3k2",
    "Conv",
    "DWConv",
    "EnhancedSPPF",
    "GaborStem",
    "GhostConv",
    "SEBlock",
]
