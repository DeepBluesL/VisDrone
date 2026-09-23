import io
import unittest

import torch
import torch.nn as nn

from visdrone_migration.classic_blocks import FasterBlock, PartialConv3, StarBlock


class ClassicBlockTests(unittest.TestCase):
    def test_odd_spatial_shapes_and_finite_gradients(self):
        for block_type in (FasterBlock, StarBlock):
            with self.subTest(block=block_type.__name__):
                block = block_type(8)
                x = torch.randn(2, 8, 15, 23, requires_grad=True)
                y = block(x)
                self.assertEqual(tuple(y.shape), tuple(x.shape))
                y.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())
                gradients = [p.grad for p in block.parameters()]
                self.assertTrue(all(g is not None for g in gradients))
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_state_dict_roundtrip_preserves_forward(self):
        sample = torch.randn(2, 8, 11, 17)
        for block_type in (FasterBlock, StarBlock):
            with self.subTest(block=block_type.__name__):
                source = block_type(8).eval()
                expected = source(sample)
                payload = io.BytesIO()
                torch.save(source.state_dict(), payload)
                payload.seek(0)
                target = block_type(8).eval()
                target.load_state_dict(torch.load(payload, weights_only=True))
                torch.testing.assert_close(target(sample), expected)

    def test_fasternet_partial_convolution_touches_one_quarter(self):
        pconv = PartialConv3(12)
        self.assertEqual(pconv.partial_channels, 3)
        self.assertEqual(pconv.untouched_channels, 9)
        self.assertEqual(pconv.conv.in_channels, 3)

    def test_blocks_use_plain_convolutions_and_expected_expansions(self):
        faster = FasterBlock(8)
        star = StarBlock(8)
        self.assertIsInstance(faster.expand[0], nn.Conv2d)
        self.assertEqual(faster.expand[0].out_channels, 16)
        self.assertEqual(star.f1.conv.out_channels, 32)
        self.assertEqual(star.dwconv.conv.kernel_size, (7, 7))
        self.assertEqual(star.dwconv.conv.groups, 8)


if __name__ == "__main__":
    unittest.main()
