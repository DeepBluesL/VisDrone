import io
import unittest

import torch
import torch.nn as nn

from visdrone_migration import (
    C2f,
    C3Ghost,
    C3k2,
    EnhancedSPPF,
    GaborStem,
    GhostConv,
    SEBlock,
)


class BlockTests(unittest.TestCase):
    def test_detector_block_shapes_and_gradients(self):
        x = torch.randn(2, 16, 31, 47, requires_grad=True)
        blocks = nn.Sequential(
            C2f(16, 32, n=2),
            C3k2(32, 32, n=2, c3k=True),
            C3Ghost(32, 32, n=1),
            SEBlock(32),
            EnhancedSPPF(32, 48),
        )
        y = blocks(x)
        self.assertEqual(tuple(y.shape), (2, 48, 31, 47))
        y.mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(any(p.grad is not None for p in blocks.parameters()))

    def test_se_supports_channels_below_reduction(self):
        block = SEBlock(4, reduction=16)
        self.assertEqual(tuple(block(torch.rand(2, 4, 5, 5)).shape), (2, 4, 5, 5))

    def test_ghost_rejects_odd_output_channels(self):
        with self.assertRaises(ValueError):
            GhostConv(8, 15)


class GaborStemTests(unittest.TestCase):
    def test_shape_fixed_bank_and_projection_gradients(self):
        stem = GaborStem(3, 16, mode="gabor")
        x = torch.randn(2, 3, 63, 95, requires_grad=True)
        y = stem(x)
        self.assertEqual(tuple(y.shape), (2, 16, 32, 48))
        self.assertIn("filter_bank", dict(stem.named_buffers()))
        self.assertNotIn("filter_bank", dict(stem.named_parameters()))
        y.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(stem.proj.conv.weight.grad)

    def test_bank_survives_initializer_and_optimizer_step(self):
        stem = GaborStem(3, 16, mode="gabor")
        expected = stem.filter_bank.clone()

        def initialize(module):
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.weight, 0.125)

        stem.apply(initialize)
        torch.testing.assert_close(stem.filter_bank, expected)
        optimizer = torch.optim.SGD(stem.parameters(), lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        stem(torch.randn(2, 3, 32, 32)).sum().backward()
        optimizer.step()
        torch.testing.assert_close(stem.filter_bank, expected)

    def test_random_seed_is_local_and_modes_differ(self):
        torch.manual_seed(1234)
        GaborStem(3, 16, mode="gabor")
        expected_next = torch.rand(4)
        torch.manual_seed(1234)
        random_stem = GaborStem(3, 16, mode="random", seed=179)
        actual_next = torch.rand(4)
        torch.testing.assert_close(actual_next, expected_next)

        gabor_stem = GaborStem(3, 16, mode="gabor")
        random_stem.proj.load_state_dict(gabor_stem.proj.state_dict())
        gabor_stem.eval()
        random_stem.eval()
        x = torch.randn(2, 3, 32, 32)
        self.assertFalse(torch.allclose(gabor_stem(x), random_stem(x)))
        self.assertFalse(torch.equal(gabor_stem.filter_bank, random_stem.filter_bank))

    def test_filter_normalization_and_state_dict_roundtrip(self):
        source = GaborStem(3, 16, mode="gabor").eval()
        per_channel = source.filter_bank[:, :1] * 3
        torch.testing.assert_close(
            per_channel.mean(dim=(-2, -1)).squeeze(1),
            torch.zeros(32),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            per_channel.square().sum(dim=(-2, -1)).sqrt().squeeze(1),
            torch.ones(32),
            atol=1e-6,
            rtol=1e-6,
        )
        sample = torch.randn(2, 3, 33, 35)
        expected = source(sample)
        payload = io.BytesIO()
        torch.save(source.state_dict(), payload)
        payload.seek(0)
        target = GaborStem(3, 16, mode="random", seed=1).eval()
        target.load_state_dict(torch.load(payload, weights_only=True))
        torch.testing.assert_close(target.filter_bank, source.filter_bank)
        torch.testing.assert_close(target(sample), expected)


if __name__ == "__main__":
    unittest.main()
