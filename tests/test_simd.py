import math
import unittest

import torch

from visdrone_migration.simd import (
    bbox_simd,
    estimate_simd_normalizers,
    max_simd_assign,
)


class SimDTests(unittest.TestCase):
    def test_equations_match_manual_location_and_shape_terms(self):
        first = torch.tensor([[0.0, 0.0, 4.0, 2.0]])
        second = torch.tensor([[2.0, 1.0, 8.0, 5.0]])
        actual = bbox_simd(first, second, m=2.0, n=4.0).item()
        location = math.sqrt(((2.0 - 5.0) / ((4.0 + 6.0) / 2.0)) ** 2
                             + ((1.0 - 3.0) / ((2.0 + 4.0) / 4.0)) ** 2)
        shape = math.sqrt(((4.0 - 6.0) / ((4.0 + 6.0) / 2.0)) ** 2
                          + ((2.0 - 4.0) / ((2.0 + 4.0) / 4.0)) ** 2)
        self.assertAlmostEqual(actual, math.exp(-(location + shape)), places=6)

    def test_identical_and_aligned_boxes(self):
        boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [5.0, 4.0, 8.0, 9.0]])
        pairwise = bbox_simd(boxes, boxes, m=6.13, n=4.59)
        aligned = bbox_simd(boxes, boxes, m=6.13, n=4.59, aligned=True)
        torch.testing.assert_close(pairwise.diag(), torch.ones(2))
        torch.testing.assert_close(aligned, torch.ones(2))
        self.assertEqual(pairwise.shape, (2, 2))

    def test_pairwise_broadcasting_is_symmetric(self):
        first = torch.tensor([[0.0, 0.0, 2.0, 3.0], [4.0, 6.0, 9.0, 8.0]])
        second = torch.tensor([[1.0, 1.0, 4.0, 5.0], [2.0, 4.0, 7.0, 9.0],
                               [20.0, 3.0, 22.0, 7.0]])
        forward = bbox_simd(first, second, m=2.3, n=1.7)
        reverse = bbox_simd(second, first, m=2.3, n=1.7)
        self.assertEqual(forward.shape, (2, 3))
        torch.testing.assert_close(forward, reverse.T)
        fast = bbox_simd(first, second, m=2.3, n=1.7, validate=False)
        torch.testing.assert_close(fast, forward)

    def test_extreme_finite_coordinates_remain_bounded(self):
        near = torch.tensor([[1.0e12, 1.0e12, 1.0e12 + 1.0e6,
                              1.0e12 + 2.0e6]], dtype=torch.float64)
        far = torch.tensor([[-1.0e12, -1.0e12, -1.0e12 + 1.0,
                             -1.0e12 + 1.0]], dtype=torch.float64)
        values = bbox_simd(near, torch.cat((near, far)), m=6.13, n=4.59)
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(torch.all((values >= 0) & (values <= 1)))
        self.assertEqual(values[0, 0].item(), 1.0)

    def test_trainset_normalizers_stream_anchor_chunks(self):
        ground_truth = torch.tensor([[0.0, 0.0, 2.0, 2.0], [4.0, 2.0, 8.0, 6.0]])
        anchors = torch.tensor([[1.0, 0.0, 3.0, 2.0], [2.0, 2.0, 6.0, 4.0],
                                [8.0, 6.0, 10.0, 10.0]])
        result = estimate_simd_normalizers([(ground_truth, anchors)], anchor_chunk_size=2)
        gt_center = (ground_truth[:, None, :2] + ground_truth[:, None, 2:]) / 2
        anchor_center = (anchors[None, :, :2] + anchors[None, :, 2:]) / 2
        gt_size = ground_truth[:, None, 2:] - ground_truth[:, None, :2]
        anchor_size = anchors[None, :, 2:] - anchors[None, :, :2]
        expected = ((gt_center - anchor_center).abs() / (gt_size + anchor_size)).mean((0, 1))
        self.assertEqual(result.pair_count, 6)
        self.assertAlmostEqual(result.m, expected[0].item())
        self.assertAlmostEqual(result.n, expected[1].item())

    def test_assignment_thresholds_and_minimum_positive_match(self):
        ground_truth = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        anchors = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.5, 0.0, 2.5, 2.0],
                                [20.0, 20.0, 22.0, 22.0]])
        assigned, maximum = max_simd_assign(
            anchors, ground_truth, m=1.0, n=1.0,
            positive_threshold=0.95, negative_threshold=0.2,
            minimum_positive_threshold=0.7)
        self.assertEqual(assigned.tolist(), [1, -1, 0])
        self.assertTrue(torch.all((maximum > 0) & (maximum <= 1)))

        # No anchor clears the positive threshold, but the best one clears the
        # minimum-positive threshold and is force-matched to the ground truth.
        shifted = torch.tensor([[0.5, 0.0, 2.5, 2.0], [1.0, 0.0, 3.0, 2.0]])
        assigned, _ = max_simd_assign(
            shifted, ground_truth, m=1.0, n=1.0,
            positive_threshold=0.99, negative_threshold=0.2,
            minimum_positive_threshold=0.7)
        self.assertEqual(assigned.tolist(), [1, -1])

    def test_invalid_geometry_and_normalizers_fail_closed(self):
        valid = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        invalid = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        with self.assertRaises(ValueError):
            bbox_simd(invalid, valid, m=1.0, n=1.0)
        with self.assertRaises(ValueError):
            bbox_simd(valid, valid, m=0.0, n=1.0)
        with self.assertRaises(ValueError):
            estimate_simd_normalizers([])
        with self.assertRaises(ValueError):
            estimate_simd_normalizers([(valid, valid)])


if __name__ == "__main__":
    unittest.main()
