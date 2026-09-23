# SPDX-License-Identifier: AGPL-3.0-only
"""Detector-ready implementations of FasterNet and StarNet feature blocks.

The FasterNet classes independently implement the paper's partial-convolution
algorithm.  StarBlock is an adaptation of the licensed official implementation.
See ``docs/classic_module_sources.json`` for sources and modification details.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PartialConv3(nn.Module):
    """Apply a 3x3 convolution to one quarter of the input channels."""

    def __init__(self, channels: int, n_div: int = 4):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if n_div < 1:
            raise ValueError("n_div must be positive")
        self.partial_channels = channels // n_div
        if self.partial_channels < 1:
            raise ValueError(f"channels must be at least n_div ({n_div})")
        self.untouched_channels = channels - self.partial_channels
        self.conv = nn.Conv2d(
            self.partial_channels,
            self.partial_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        convolved = self.conv(x[:, : self.partial_channels])
        untouched = x[:, self.partial_channels :]
        return torch.cat((convolved, untouched), dim=1)


class FasterBlock(nn.Module):
    """FasterNet PConv block with a 2x pointwise MLP and residual path."""

    def __init__(self, channels: int):
        super().__init__()
        if channels < 4:
            raise ValueError("FasterBlock requires at least 4 channels")
        hidden = channels * 2
        self.spatial_mixing = PartialConv3(channels, n_div=4)
        self.expand = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.spatial_mixing(x)
        return x + self.project(self.expand(mixed))


class _ConvBN(nn.Sequential):
    """Convolution optionally followed by batch normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        groups: int = 1,
        with_bn: bool = True,
    ):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                groups=groups,
            ),
        )
        if with_bn:
            bn = nn.BatchNorm2d(out_channels)
            nn.init.ones_(bn.weight)
            nn.init.zeros_(bn.bias)
            self.add_module("bn", bn)


class StarBlock(nn.Module):
    """StarNet block using multiplicative pointwise branches."""

    def __init__(self, channels: int):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        hidden = channels * 4
        self.dwconv = _ConvBN(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.f1 = _ConvBN(channels, hidden, with_bn=False)
        self.f2 = _ConvBN(channels, hidden, with_bn=False)
        self.g = _ConvBN(hidden, channels, with_bn=True)
        self.dwconv2 = _ConvBN(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            with_bn=False,
        )
        self.act = nn.ReLU6()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.act(self.f1(x)) * self.f2(x)
        return residual + self.dwconv2(self.g(x))


__all__ = ["FasterBlock", "PartialConv3", "StarBlock"]
