# SPDX-License-Identifier: AGPL-3.0-only
"""Reference implementation of Similarity Distance (SimD).

The equations follow Shi et al., *Similarity Distance-Based Label Assignment
for Tiny Object Detection*, IROS 2024 (arXiv:2407.02394v3, Eqs. 1--5).
This module keeps the train-set normalizers explicit: the paper defines them
from every ground-truth/anchor pair, so they depend on the detector's anchors
as well as the dataset.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class SimDNormalizers:
    """Train-set normalization constants from the paper's Eqs. 4 and 5."""

    m: float
    n: float
    pair_count: int


def _validate_boxes(boxes: Tensor, name: str) -> None:
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"{name} must have shape [N, 4] in xyxy format")
    if not boxes.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if not torch.isfinite(boxes).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    if boxes.numel() and ((boxes[:, 2] <= boxes[:, 0]).any()
                          or (boxes[:, 3] <= boxes[:, 1]).any()):
        raise ValueError(f"{name} must contain boxes with positive width and height")


def _validate_normalizer(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return value


def bbox_simd(
    boxes1: Tensor,
    boxes2: Tensor,
    *,
    m: float,
    n: float,
    aligned: bool = False,
    validate: bool = True,
) -> Tensor:
    """Return the SimD similarity for boxes in ``xyxy`` format.

    With ``aligned=False``, the output has shape ``[len(boxes1), len(boxes2)]``.
    With ``aligned=True``, corresponding rows are compared and the output has
    shape ``[len(boxes1)]``. SimD is in ``[0, 1]``; zero is possible only
    through floating-point underflow for extremely dissimilar boxes. Set
    ``validate=False`` only after the caller has checked box geometry and
    normalizers; it avoids tensor-value checks that synchronize a GPU loop.
    """
    if validate:
        _validate_boxes(boxes1, "boxes1")
        _validate_boxes(boxes2, "boxes2")
        m = _validate_normalizer(m, "m")
        n = _validate_normalizer(n, "n")
    else:
        # The hot-path caller has already checked geometry and constants once.
        # Shape checks do not synchronize accelerator tensor values.
        if boxes1.ndim != 2 or boxes1.shape[-1] != 4:
            raise ValueError("boxes1 must have shape [N, 4] in xyxy format")
        if boxes2.ndim != 2 or boxes2.shape[-1] != 4:
            raise ValueError("boxes2 must have shape [N, 4] in xyxy format")
        m, n = float(m), float(n)
    if boxes1.device != boxes2.device:
        raise ValueError("boxes1 and boxes2 must be on the same device")
    if aligned and len(boxes1) != len(boxes2):
        raise ValueError("aligned boxes must have the same length")

    def components(boxes: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        center_x = (boxes[:, 0] + boxes[:, 2]) / 2
        center_y = (boxes[:, 1] + boxes[:, 3]) / 2
        return center_x, center_y, width, height

    x1, y1, w1, h1 = components(boxes1)
    x2, y2, w2, h2 = components(boxes2)
    if aligned:
        width_scale = (w1 + w2) / m
        height_scale = (h1 + h2) / n
    else:
        x1, y1, w1, h1 = (v[:, None] for v in (x1, y1, w1, h1))
        x2, y2, w2, h2 = (v[None, :] for v in (x2, y2, w2, h2))
        width_scale = (w1 + w2) / m
        height_scale = (h1 + h2) / n

    location = torch.sqrt(((x1 - x2) / width_scale).square()
                          + ((y1 - y2) / height_scale).square())
    shape = torch.sqrt(((w1 - w2) / width_scale).square()
                       + ((h1 - h2) / height_scale).square())
    return torch.exp(-(location + shape))


def estimate_simd_normalizers(
    image_box_pairs: Iterable[tuple[Tensor, Tensor]],
    *,
    anchor_chunk_size: int = 4096,
) -> SimDNormalizers:
    """Estimate ``m`` and ``n`` from per-image ``(ground_truth, anchors)``.

    This streams anchor chunks and accumulates in float64 on CPU. Each input
    tensor uses ``xyxy`` coordinates. Images with no ground truths or anchors
    contribute no pairs, matching the denominator in Eqs. 4 and 5.
    """
    if anchor_chunk_size < 1:
        raise ValueError("anchor_chunk_size must be positive")
    sum_x = 0.0
    sum_y = 0.0
    pair_count = 0
    for ground_truth, anchors in image_box_pairs:
        _validate_boxes(ground_truth, "ground_truth")
        _validate_boxes(anchors, "anchors")
        if not len(ground_truth) or not len(anchors):
            continue
        gt = ground_truth.detach().to(device="cpu", dtype=torch.float64)
        gt_x = ((gt[:, 0] + gt[:, 2]) / 2)[:, None]
        gt_y = ((gt[:, 1] + gt[:, 3]) / 2)[:, None]
        gt_w = (gt[:, 2] - gt[:, 0])[:, None]
        gt_h = (gt[:, 3] - gt[:, 1])[:, None]
        for start in range(0, len(anchors), anchor_chunk_size):
            chunk = anchors[start:start + anchor_chunk_size].detach().to(
                device="cpu", dtype=torch.float64)
            anchor_x = ((chunk[:, 0] + chunk[:, 2]) / 2)[None, :]
            anchor_y = ((chunk[:, 1] + chunk[:, 3]) / 2)[None, :]
            anchor_w = (chunk[:, 2] - chunk[:, 0])[None, :]
            anchor_h = (chunk[:, 3] - chunk[:, 1])[None, :]
            sum_x += (gt_x - anchor_x).abs().div(gt_w + anchor_w).sum().item()
            sum_y += (gt_y - anchor_y).abs().div(gt_h + anchor_h).sum().item()
            pair_count += len(gt) * len(chunk)
    if not pair_count:
        raise ValueError("at least one ground-truth/anchor pair is required")
    m = sum_x / pair_count
    n = sum_y / pair_count
    _validate_normalizer(m, "estimated m")
    _validate_normalizer(n, "estimated n")
    return SimDNormalizers(m, n, pair_count)


def max_simd_assign(
    anchors: Tensor,
    ground_truth: Tensor,
    *,
    m: float,
    n: float,
    positive_threshold: float = 0.7,
    negative_threshold: float = 0.3,
    minimum_positive_threshold: float = 0.3,
) -> tuple[Tensor, Tensor]:
    """Apply the paper's MaxSimDAssigner rule.

    The returned 1-D assignment uses MMDetection conventions: ``-1`` is
    ignored, ``0`` is background, and positive values are one-based ground
    truth indices. The second tensor is each anchor's maximum SimD.
    """
    for value, name in (
        (positive_threshold, "positive_threshold"),
        (negative_threshold, "negative_threshold"),
        (minimum_positive_threshold, "minimum_positive_threshold"),
    ):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if negative_threshold > positive_threshold:
        raise ValueError("negative_threshold cannot exceed positive_threshold")
    _validate_boxes(anchors, "anchors")
    _validate_boxes(ground_truth, "ground_truth")
    if not len(anchors):
        return (torch.empty(0, dtype=torch.long, device=anchors.device),
                anchors.new_empty(0))
    if not len(ground_truth):
        return (torch.zeros(len(anchors), dtype=torch.long, device=anchors.device),
                anchors.new_zeros(len(anchors)))

    similarities = bbox_simd(ground_truth, anchors, m=m, n=n)
    maximum, best_ground_truth = similarities.max(dim=0)
    assigned = torch.full((len(anchors),), -1, dtype=torch.long,
                          device=anchors.device)
    assigned[maximum < negative_threshold] = 0
    positive = maximum >= positive_threshold
    assigned[positive] = best_ground_truth[positive] + 1

    # For every GT, match all tied best anchors when its maximum clears the
    # minimum threshold. This mirrors MMDetection's gt_max_assign_all=True and
    # may overwrite an assignment from the earlier threshold pass.
    best_per_ground_truth = similarities.max(dim=1).values
    for index in range(len(ground_truth)):
        if best_per_ground_truth[index] >= minimum_positive_threshold:
            assigned[similarities[index] == best_per_ground_truth[index]] = index + 1
    return assigned, maximum
