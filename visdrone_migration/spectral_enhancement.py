# SPDX-License-Identifier: AGPL-3.0-only
"""Training utilities adapted from SET (CVPR 2025) for a YOLO feature pyramid.

This is an independent PyTorch implementation. HBS follows the pinned author
code's foreground re-masking and single ReLU; see SET_SOURCE_AUDIT.md for the
differences from the paper's compact equations and the YOLO adaptations.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn


DEFAULT_REDUCTION = 4
DEFAULT_RHO = 1.0
DEFAULT_AUX_WEIGHT = 1.0
YOLO_BRANCHES = ("cls", "box", "dfl")


def kernel_size_for_stride(stride: int) -> int:
    """Return SET Eq. 4's odd smoothing kernel for one pyramid stride."""
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    return math.floor(math.log2(stride) / 2) * 2 + 1


def feature_masks_from_normalized_xywh(
    boxes_xywh: Tensor,
    batch_idx: Tensor,
    *,
    batch_size: int,
    feature_shapes: Sequence[tuple[int, int]],
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    validate: bool = True,
) -> tuple[Tensor, ...]:
    """Rasterize normalized boxes to binary feature-cell overlap masks.

    A cell is foreground when its normalized rectangular extent has positive
    overlap with a ground-truth box. Degenerate/tiny boxes still cover the cell
    containing their clamped center, ensuring at least one cell per valid GT.
    """
    if boxes_xywh.ndim != 2 or boxes_xywh.shape[1] != 4:
        raise ValueError("boxes_xywh must have shape [N, 4]")
    if batch_idx.ndim not in (1, 2) or batch_idx.numel() != boxes_xywh.shape[0]:
        raise ValueError("batch_idx must contain one index per box")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not boxes_xywh.is_floating_point():
        raise ValueError("boxes_xywh must be floating-point")
    target_device = torch.device(device) if device is not None else boxes_xywh.device
    boxes = boxes_xywh.detach().to(device=target_device, dtype=torch.float32)
    raw_indices = batch_idx.detach().reshape(-1).to(device=target_device)
    indices = raw_indices.to(dtype=torch.long)
    if validate:
        if not torch.isfinite(boxes).all():
            raise ValueError("boxes_xywh must be finite")
        if raw_indices.numel() and not torch.equal(raw_indices, indices.to(raw_indices.dtype)):
            raise ValueError("batch_idx values must be integers")
        if indices.numel() and ((indices < 0).any() or (indices >= batch_size).any()):
            raise ValueError("batch_idx contains an out-of-range image index")
        if boxes.numel() and (boxes[:, 2:] < 0).any():
            raise ValueError("normalized box widths and heights must be non-negative")
    masks: list[Tensor] = []
    for height, width in feature_shapes:
        if not isinstance(height, int) or not isinstance(width, int) or height < 1 or width < 1:
            raise ValueError("feature shapes must contain positive integer (height, width) pairs")
        cx, cy, box_width, box_height = boxes.unbind(dim=1)
        left = ((cx - box_width / 2).clamp(0, 1) * width).floor().long().clamp(0, width - 1)
        top = ((cy - box_height / 2).clamp(0, 1) * height).floor().long().clamp(0, height - 1)
        raw_right = ((cx + box_width / 2).clamp(0, 1) * width).ceil().long()
        raw_bottom = ((cy + box_height / 2).clamp(0, 1) * height).ceil().long()
        right = torch.maximum(raw_right, left + 1).clamp(max=width)
        bottom = torch.maximum(raw_bottom, top + 1).clamp(max=height)

        # Rectangle difference rasterization: four integer updates per GT,
        # independent of feature-map area and without device-to-host scalars.
        row_width = width + 1
        plane = (height + 1) * row_width
        base = indices * plane
        corners = torch.cat((
            base + top * row_width + left,
            base + top * row_width + right,
            base + bottom * row_width + left,
            base + bottom * row_width + right,
        ))
        updates = torch.cat((
            torch.ones_like(indices, dtype=torch.int32),
            -torch.ones_like(indices, dtype=torch.int32),
            -torch.ones_like(indices, dtype=torch.int32),
            torch.ones_like(indices, dtype=torch.int32),
        ))
        difference = torch.zeros(batch_size * plane, device=target_device, dtype=torch.int32)
        difference.scatter_add_(0, corners, updates)
        coverage = difference.view(batch_size, height + 1, width + 1)
        coverage = coverage.cumsum(1, dtype=torch.int32).cumsum(2, dtype=torch.int32)
        masks.append((coverage[:, :height, :width] > 0).unsqueeze(1).to(dtype=dtype))
    return tuple(masks)


class HBSLevel(nn.Module):
    """Hierarchical Background Smoothing for one feature level (Eq. 2–4)."""

    def __init__(self, channels: int, stride: int, reduction: int = DEFAULT_REDUCTION):
        super().__init__()
        if channels < 1 or reduction < 1:
            raise ValueError("channels and reduction must be positive")
        hidden = max(1, channels // reduction)
        kernel = kernel_size_for_stride(stride)
        padding = kernel // 2
        self.channels = channels
        self.stride = stride
        self.kernel_size = kernel
        self.reduce = nn.Conv2d(channels, hidden, kernel, padding=padding)
        self.expand = nn.Conv2d(hidden, channels, kernel, padding=padding)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, feature: Tensor, foreground_mask: Tensor) -> Tensor:
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError(f"feature must have shape [B, {self.channels}, H, W]")
        if foreground_mask.ndim != 4 or foreground_mask.shape[1] != 1:
            raise ValueError("foreground_mask must have shape [B, 1, H, W]")
        if foreground_mask.shape[0] != feature.shape[0] or foreground_mask.shape[2:] != feature.shape[2:]:
            raise ValueError("foreground_mask batch/spatial shape must match feature")
        mask = foreground_mask.to(device=feature.device, dtype=feature.dtype)
        background = feature * (1 - mask)
        smoothed = self.expand(self.activation(self.reduce(background))) + background
        # The released implementation re-masks after smoothing to preserve
        # foreground values exactly; its expand convolution has no final ReLU.
        return feature * mask + smoothed * (1 - mask)


class HierarchicalBackgroundSmoothing(nn.Module):
    """Apply independent HBS operations to a sequence of pyramid levels."""

    def __init__(
        self,
        channels: Sequence[int],
        strides: Sequence[int],
        reduction: int = DEFAULT_REDUCTION,
    ):
        super().__init__()
        if len(channels) != len(strides) or not channels:
            raise ValueError("channels and strides must have the same non-zero length")
        self.levels = nn.ModuleList(
            HBSLevel(channel, stride, reduction=reduction)
            for channel, stride in zip(channels, strides)
        )

    def forward(self, features: Sequence[Tensor], masks: Sequence[Tensor]) -> tuple[Tensor, ...]:
        if len(features) != len(self.levels) or len(masks) != len(self.levels):
            raise ValueError("one feature and mask are required for every HBS level")
        return tuple(level(feature, mask) for level, feature, mask in zip(self.levels, features, masks))


def adversarial_feature_perturbations(
    features: Sequence[Tensor],
    branch_losses: Mapping[str, Tensor],
    *,
    rho: float = DEFAULT_RHO,
    branch_weights: Mapping[str, float] | None = None,
    eps: float = 1e-12,
    validate: bool = True,
) -> tuple[Tensor, ...]:
    """Compute detached SET API perturbations for each feature-pyramid level.

    Each YOLO loss branch (normally ``cls``, ``box``, and ``dfl``) is normalized
    independently at each level before the weighted branch perturbations are
    summed. Unused or exactly-zero gradients contribute zero.
    """
    if rho < 0 or eps <= 0:
        raise ValueError("rho must be non-negative and eps must be positive")
    if not features:
        raise ValueError("features must be non-empty")
    if not branch_losses:
        raise ValueError("branch_losses must be non-empty")
    weights = dict(branch_weights or {})
    perturbations = [torch.zeros_like(feature) for feature in features]
    for branch, loss in branch_losses.items():
        if not isinstance(loss, Tensor) or loss.numel() != 1:
            raise ValueError(f"branch loss {branch!r} must be a scalar tensor")
        weight = float(weights.get(branch, 1.0))
        if not math.isfinite(weight):
            raise ValueError(f"branch weight {branch!r} must be finite")
        if not loss.requires_grad or weight == 0 or rho == 0:
            continue
        gradients = torch.autograd.grad(
            loss,
            tuple(features),
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        for index, (feature, gradient) in enumerate(zip(features, gradients)):
            if gradient is None:
                continue
            stable = gradient.detach().to(dtype=torch.float32)
            norm = torch.linalg.vector_norm(stable)
            if validate and not torch.isfinite(norm):
                raise FloatingPointError(f"non-finite gradient norm for branch {branch!r}, level {index}")
            normalized = stable / norm.clamp_min(eps)
            perturbations[index] = perturbations[index] + normalized.to(feature.dtype) * (rho * weight)
    return tuple(value.detach() for value in perturbations)


def inject_adversarial_feature_perturbations(
    features: Sequence[Tensor],
    branch_losses: Mapping[str, Tensor],
    *,
    rho: float = DEFAULT_RHO,
    branch_weights: Mapping[str, float] | None = None,
    eps: float = 1e-12,
    validate: bool = True,
) -> tuple[Tensor, ...]:
    """Add detached API perturbations while retaining feature backpropagation."""
    perturbations = adversarial_feature_perturbations(
        features, branch_losses, rho=rho, branch_weights=branch_weights, eps=eps,
        validate=validate,
    )
    return tuple(feature + perturbation for feature, perturbation in zip(features, perturbations))


__all__ = [
    "DEFAULT_AUX_WEIGHT",
    "DEFAULT_REDUCTION",
    "DEFAULT_RHO",
    "HBSLevel",
    "HierarchicalBackgroundSmoothing",
    "YOLO_BRANCHES",
    "adversarial_feature_perturbations",
    "feature_masks_from_normalized_xywh",
    "inject_adversarial_feature_perturbations",
    "kernel_size_for_stride",
]
