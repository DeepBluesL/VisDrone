import unittest
from unittest.mock import patch

import torch
from ultralytics.utils.tal import TaskAlignedAssigner

from visdrone_migration.realtime_performance import (
    ChunkedTaskAlignedAssigner,
    build_performance_dataloader,
)


def assignment_inputs(counts, anchors=97, classes=5, seed=123):
    generator = torch.Generator().manual_seed(seed)
    batch, slots = len(counts), max(counts, default=0)
    scores = torch.rand(batch, anchors, classes, generator=generator)
    centers = torch.rand(anchors, 2, generator=generator) * 64
    half = torch.rand(batch, anchors, 2, generator=generator) * 8 + 1
    boxes = torch.cat((centers[None] - half, centers[None] + half), dim=-1)
    labels = torch.zeros(batch, slots, 1)
    gt_boxes = torch.zeros(batch, slots, 4)
    mask = torch.zeros(batch, slots, 1, dtype=torch.bool)
    for image, count in enumerate(counts):
        if not count:
            continue
        labels[image, :count, 0] = torch.randint(classes, (count,), generator=generator)
        xy1 = torch.rand(count, 2, generator=generator) * 48
        xy2 = xy1 + torch.rand(count, 2, generator=generator) * 15 + 1
        gt_boxes[image, :count] = torch.cat((xy1, xy2), dim=-1)
        mask[image, :count] = True
    return scores, boxes, centers, labels, gt_boxes, mask


class ChunkedAssignerTests(unittest.TestCase):
    def assert_assignment_equal(self, counts, budget):
        inputs = assignment_inputs(counts)
        kwargs = dict(topk=10, num_classes=5, alpha=0.5, beta=6.0)
        expected = TaskAlignedAssigner(**kwargs)(*inputs)
        actual = ChunkedTaskAlignedAssigner(
            **kwargs, max_pair_elements=budget
        )(*inputs)
        for expected_tensor, actual_tensor in zip(expected, actual):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    def test_exact_for_mixed_dense_empty_and_multiclass_images(self):
        self.assert_assignment_equal([8, 0, 3, 1, 7], budget=97 * 8 * 2)

    def test_exact_when_one_dense_image_exceeds_budget(self):
        self.assert_assignment_equal([902, 2], budget=8_000)

    def test_exact_with_noncontiguous_valid_gt_slots(self):
        inputs = list(assignment_inputs([5, 4, 3]))
        # Create holes without reindexing later valid entries. Native assignment
        # addresses original slots, so trimming by valid count would be wrong.
        inputs[-1][0, 1] = False
        inputs[-1][1, 0] = False
        kwargs = dict(topk=10, num_classes=5, alpha=0.5, beta=6.0)
        expected = TaskAlignedAssigner(**kwargs)(*inputs)
        actual = ChunkedTaskAlignedAssigner(
            **kwargs, max_pair_elements=97 * 3
        )(*inputs)
        for expected_tensor, actual_tensor in zip(expected, actual):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    def test_exact_for_no_ground_truth_slots(self):
        self.assert_assignment_equal([0, 0, 0], budget=100)

    def test_invalid_budget_rejected(self):
        with self.assertRaises(ValueError):
            ChunkedTaskAlignedAssigner(max_pair_elements=0)


class PerformanceLoaderTests(unittest.TestCase):
    def test_infinite_loader_does_not_create_second_persistent_iterator_owner(self):
        class Dataset:
            def __len__(self):
                return 8

        sentinel = object()
        with patch("visdrone_migration.realtime_performance.os.cpu_count", return_value=8), patch(
            "visdrone_migration.realtime_performance.torch.cuda.device_count", return_value=1
        ), patch(
            "visdrone_migration.realtime_performance.InfiniteDataLoader", return_value=sentinel
        ) as loader:
            actual = build_performance_dataloader(
                Dataset(), batch_size=4, workers=4, shuffle=True, rank=-1, prefetch_factor=4
            )
        self.assertIs(actual, sentinel)
        options = loader.call_args.kwargs
        self.assertEqual(options["num_workers"], 4)
        self.assertFalse(options["persistent_workers"])
        self.assertEqual(options["prefetch_factor"], 4)


if __name__ == "__main__":
    unittest.main()
