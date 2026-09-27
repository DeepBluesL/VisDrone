# SPDX-License-Identifier: AGPL-3.0-only
"""Memory-bounded exact assignment and tuned reusable-worker data loading."""

from __future__ import annotations

import os

import torch
import torch.distributed as distributed
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.utils import RANK
from ultralytics.utils.tal import TaskAlignedAssigner


class ChunkedTaskAlignedAssigner(TaskAlignedAssigner):
    """Run native task-aligned assignment on independent image chunks.

    Assignment has no cross-image interactions.  Each chunk also removes trailing
    padded GT rows.  A single exceptionally dense image may exceed ``max_pair_elements``;
    it remains one indivisible exact-native assignment rather than changing the
    algorithm by partitioning mutually competing ground truths.
    """

    def __init__(self, *args, max_pair_elements: int = 32_000_000, **kwargs):
        super().__init__(*args, **kwargs)
        if max_pair_elements < 1:
            raise ValueError("max_pair_elements must be positive")
        self.max_pair_elements = int(max_pair_elements)

    def _chunks(self, mask_gt: torch.Tensor, anchors: int):
        valid = mask_gt.squeeze(-1).bool()
        if valid.shape[1]:
            slots = torch.arange(1, valid.shape[1] + 1, device=valid.device)
            # Use last valid index, not count: this preserves original GT indices
            # even if a caller supplies a non-contiguous validity mask.
            extents = (valid * slots).amax(-1)
        else:
            extents = torch.zeros(valid.shape[0], dtype=torch.long, device=valid.device)
        counts = extents.to(device="cpu", dtype=torch.int64).tolist()
        start = 0
        while start < len(counts):
            end = start + 1
            max_gt = counts[start]
            while end < len(counts):
                candidate_max = max(max_gt, counts[end])
                if (end - start + 1) * max(candidate_max, 1) * anchors > self.max_pair_elements:
                    break
                max_gt = candidate_max
                end += 1
            yield start, end, max_gt
            start = end

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        batch = pd_scores.shape[0]
        if batch == 0:
            return super().forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        original_slots = gt_bboxes.shape[1]
        outputs = []
        for start, end, valid_gt in self._chunks(mask_gt, pd_scores.shape[1]):
            # Preserve one zero-padded slot for an empty image when the native
            # batch has GT slots; native target_labels then remains exactly zero.
            slots = min(original_slots, max(valid_gt, 1)) if original_slots else 0
            result = super().forward(
                pd_scores[start:end],
                pd_bboxes[start:end],
                anc_points,
                gt_labels[start:end, :slots].contiguous(),
                gt_bboxes[start:end, :slots].contiguous(),
                mask_gt[start:end, :slots].contiguous(),
            )
            outputs.append(result)
        return tuple(torch.cat(items, dim=0) for items in zip(*outputs))


def install_chunked_assigner(model, *, max_pair_elements: int = 32_000_000):
    """Create the lazy detection criterion if needed and replace only its assigner."""
    if not hasattr(model, "criterion"):
        model.criterion = model.init_criterion()
    native = model.criterion.assigner
    model.criterion.assigner = ChunkedTaskAlignedAssigner(
        topk=native.topk,
        num_classes=native.num_classes,
        alpha=native.alpha,
        beta=native.beta,
        eps=native.eps,
        max_pair_elements=max_pair_elements,
    )
    return model.criterion.assigner


def build_performance_dataloader(
    dataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    rank: int,
    prefetch_factor: int,
):
    """Build an InfiniteDataLoader with explicit staging and reusable workers."""
    batch_size = min(batch_size, len(dataset))
    device_count = torch.cuda.device_count()
    worker_count = min(os.cpu_count() // max(device_count, 1), workers)
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    worker_options = (
        {
            # InfiniteDataLoader already retains its public ``iterator`` across
            # epochs. PyTorch persistent mode additionally retains the same
            # iterator in DataLoader._iterator; Ultralytics reset() replaces
            # only the public reference after close_mosaic, which would leave
            # the old workers alive and prefetching. Disable the second owner so
            # reset() drops the sole old iterator and starts one fresh pool.
            "persistent_workers": False,
            "prefetch_factor": prefetch_factor,
        }
        if worker_count > 0
        else {}
    )
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=worker_count,
        sampler=sampler,
        pin_memory=True,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
        **worker_options,
    )


__all__ = [
    "ChunkedTaskAlignedAssigner",
    "build_performance_dataloader",
    "install_chunked_assigner",
]
