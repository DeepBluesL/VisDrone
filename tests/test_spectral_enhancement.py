import unittest

import torch

from visdrone_migration.spectral_enhancement import (
    DEFAULT_AUX_WEIGHT,
    DEFAULT_REDUCTION,
    DEFAULT_RHO,
    HBSLevel,
    HierarchicalBackgroundSmoothing,
    adversarial_feature_perturbations,
    feature_masks_from_normalized_xywh,
    inject_adversarial_feature_perturbations,
    kernel_size_for_stride,
)


class FeatureMaskTests(unittest.TestCase):
    @staticmethod
    def reference(boxes, batch_idx, batch_size, shape):
        height, width = shape
        mask = torch.zeros(batch_size, 1, height, width)
        for box, image in zip(boxes, batch_idx):
            cx, cy, bw, bh = box.tolist()
            left = min(max(int(torch.floor(torch.tensor(max(0.0, cx - bw / 2) * width))), 0), width - 1)
            top = min(max(int(torch.floor(torch.tensor(max(0.0, cy - bh / 2) * height))), 0), height - 1)
            right = min(max(int(torch.ceil(torch.tensor(min(1.0, cx + bw / 2) * width))), left + 1), width)
            bottom = min(max(int(torch.ceil(torch.tensor(min(1.0, cy + bh / 2) * height))), top + 1), height)
            mask[int(image), 0, top:bottom, left:right] = 1
        return mask

    def test_vectorized_difference_rasterizer_matches_loop_reference(self):
        generator = torch.Generator().manual_seed(31415)
        boxes = torch.rand(200, 4, generator=generator)
        boxes[:, 2:] *= 0.35
        boxes[:4] = torch.tensor([
            [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 1.0, 1.0], [0.01, 0.99, 0.001, 0.001],
        ])
        batch_idx = torch.randint(0, 5, (len(boxes),), generator=generator)
        shapes = [(31, 47), (9, 13), (1, 1)]
        actual = feature_masks_from_normalized_xywh(
            boxes, batch_idx, batch_size=5, feature_shapes=shapes
        )
        for result, shape in zip(actual, shapes):
            torch.testing.assert_close(result, self.reference(boxes, batch_idx, 5, shape))

    def test_multiscale_border_tiny_and_empty_images(self):
        boxes = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],       # top-left: still one cell
            [1.0, 1.0, 0.02, 0.02],     # clipped bottom-right
            [0.5, 0.5, 0.25, 0.25],
        ])
        batch_idx = torch.tensor([0, 0, 2])
        masks = feature_masks_from_normalized_xywh(
            boxes, batch_idx, batch_size=4, feature_shapes=[(4, 6), (2, 3)]
        )
        self.assertEqual([tuple(mask.shape) for mask in masks], [(4, 1, 4, 6), (4, 1, 2, 3)])
        for mask in masks:
            self.assertEqual(mask[1].count_nonzero().item(), 0)
            self.assertEqual(mask[3].count_nonzero().item(), 0)
            self.assertGreaterEqual(mask[0].count_nonzero().item(), 2)
            self.assertGreaterEqual(mask[2].count_nonzero().item(), 1)
            self.assertEqual(mask[0, 0, 0, 0].item(), 1)
            self.assertEqual(mask[0, 0, -1, -1].item(), 1)

    def test_empty_box_tensor(self):
        masks = feature_masks_from_normalized_xywh(
            torch.empty(0, 4), torch.empty(0, dtype=torch.long),
            batch_size=2, feature_shapes=[(5, 7)], dtype=torch.bool,
        )
        self.assertEqual(masks[0].count_nonzero().item(), 0)
        self.assertEqual(masks[0].dtype, torch.bool)


class HBSTests(unittest.TestCase):
    def test_kernel_formula_and_defaults(self):
        self.assertEqual([kernel_size_for_stride(s) for s in (4, 8, 16, 32)], [3, 3, 5, 5])
        self.assertEqual((DEFAULT_REDUCTION, DEFAULT_RHO, DEFAULT_AUX_WEIGHT), (4, 1.0, 1.0))

    def test_shape_finiteness_and_gradients(self):
        hbs = HBSLevel(3, stride=8)
        feature = torch.randn(2, 3, 7, 9, requires_grad=True)
        mask = torch.zeros(2, 1, 7, 9)
        mask[:, :, 2:4, 3:5] = 1
        output = hbs(feature, mask)
        self.assertEqual(output.shape, feature.shape)
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(feature.grad)
        self.assertTrue(torch.isfinite(feature.grad).all())
        self.assertTrue(all(parameter.grad is not None for parameter in hbs.parameters()))

    def test_hierarchy_preserves_level_shapes(self):
        module = HierarchicalBackgroundSmoothing([4, 8], [8, 16])
        features = [torch.randn(1, 4, 9, 11), torch.randn(1, 8, 5, 6)]
        masks = [torch.zeros(1, 1, 9, 11), torch.ones(1, 1, 5, 6)]
        outputs = module(features, masks)
        self.assertEqual([x.shape for x in outputs], [x.shape for x in features])

    def test_smoothing_preserves_foreground_and_allows_negative_background(self):
        hbs = HBSLevel(1, stride=8)
        with torch.no_grad():
            hbs.reduce.weight.zero_()
            hbs.reduce.bias.fill_(1)
            hbs.expand.weight.zero_()
            hbs.expand.bias.fill_(-2)
        feature = torch.ones(1, 1, 3, 3)
        mask = torch.zeros(1, 1, 3, 3)
        mask[:, :, 1, 1] = 1
        result = hbs(feature, mask)
        self.assertEqual(result[0, 0, 1, 1].item(), 1)
        self.assertEqual(result[0, 0, 0, 0].item(), -1)


class APITests(unittest.TestCase):
    def test_branches_are_normalized_independently_per_level(self):
        p3 = torch.tensor([1.0, 2.0], requires_grad=True)
        p4 = torch.tensor([3.0, 4.0, 5.0], requires_grad=True)
        cls = 2 * p3.sum() + 4 * p4.sum()
        box = (torch.tensor([3.0, 0.0]) * p3).sum() + (torch.tensor([0.0, 0.0, 5.0]) * p4).sum()
        perturbations = adversarial_feature_perturbations(
            [p3, p4], {"cls": cls, "box": box}, rho=1.0,
            branch_weights={"cls": 1.0, "box": 0.5},
        )
        expected_p3 = torch.tensor([1.0, 1.0]) / torch.sqrt(torch.tensor(2.0)) + torch.tensor([0.5, 0.0])
        expected_p4 = torch.ones(3) / torch.sqrt(torch.tensor(3.0)) + torch.tensor([0.0, 0.0, 0.5])
        torch.testing.assert_close(perturbations[0], expected_p3)
        torch.testing.assert_close(perturbations[1], expected_p4)
        self.assertFalse(perturbations[0].requires_grad)

    def test_zero_and_unused_gradients_are_safe(self):
        used = torch.randn(2, 3, requires_grad=True)
        unused = torch.randn(1, requires_grad=True)
        zero_loss = (used * 0).sum()
        perturbations = adversarial_feature_perturbations(
            [used, unused], {"dfl": zero_loss}, rho=1.0
        )
        self.assertEqual(perturbations[0].count_nonzero().item(), 0)
        self.assertEqual(perturbations[1].count_nonzero().item(), 0)
        self.assertTrue(all(torch.isfinite(value).all() for value in perturbations))

    def test_detached_injection_preserves_auxiliary_backprop(self):
        feature = torch.randn(2, 4, requires_grad=True)
        original_loss = feature.square().sum()
        enhanced, = inject_adversarial_feature_perturbations(
            [feature], {"cls": original_loss}, rho=1.0
        )
        enhanced.sum().backward()
        torch.testing.assert_close(feature.grad, torch.ones_like(feature))

    def test_hot_path_propagates_nonfinite_gradient_without_sync_validation(self):
        feature = torch.ones(2, requires_grad=True)
        loss = (feature * torch.tensor(float("nan"))).sum()
        perturbation, = adversarial_feature_perturbations(
            [feature], {"cls": loss}, validate=False
        )
        self.assertTrue(torch.isnan(perturbation).all())


if __name__ == "__main__":
    unittest.main()
