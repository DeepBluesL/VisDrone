import io
import os
from pathlib import Path
import unittest
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).resolve().parents[1] / '.runtime' / 'ultralytics'))
import torch
from torch import nn
from torch.nn import functional as F
from visdrone_migration.spatial_blocks import HaarWTConv, WTBlock, LSConv, DynamicSmallKernel
from visdrone_migration.profiling import convolution_linear_gflops
from visdrone_migration.model import MigratedDetectionModel, NEW_VARIANTS


class SpatialOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_haar_analysis_synthesis_roundtrip_with_odd_padding(self):
        layer = HaarWTConv(8)
        x = torch.randn(2, 8, 13, 15)
        bands = F.conv2d(F.pad(x, (0, 1, 0, 1)), layer.analysis_filter, stride=2, groups=8)
        restored = F.conv_transpose2d(bands, layer.synthesis_filter, stride=2, groups=8)[..., :13, :15]
        torch.testing.assert_close(restored, x, atol=1e-6, rtol=1e-5)

    def test_dynamic_aggregation_matches_independent_shift_reference_and_gradients(self):
        x = torch.randn(2, 16, 5, 7, dtype=torch.double, requires_grad=True)
        weights = torch.randn(2, 2, 9, 5, 7, dtype=torch.double, requires_grad=True)
        actual = DynamicSmallKernel()(x, weights)
        padded = F.pad(x, (1, 1, 1, 1))
        expected = torch.zeros_like(x)
        for row in range(3):
            for column in range(3):
                shared = weights[:, torch.arange(16) % 2, row * 3 + column]
                expected = expected + padded[..., row:row + 5, column:column + 7] * shared
        torch.testing.assert_close(actual, expected)
        expected_grad = torch.autograd.grad(expected.square().sum(), (x, weights), retain_graph=True)
        actual_grad = torch.autograd.grad(actual.square().sum(), (x, weights))
        for one, two in zip(actual_grad, expected_grad):
            torch.testing.assert_close(one, two)

    def test_blocks_train_and_serialize_with_fixed_filters(self):
        for block_type in (WTBlock, LSConv):
            with self.subTest(block=block_type.__name__):
                layer = block_type(16)
                x = torch.randn(2, 16, 15, 19, requires_grad=True)
                before = {name: value.clone() for name, value in layer.named_buffers() if 'filter' in name}
                optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
                y = layer(x)
                self.assertEqual(y.shape, x.shape)
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                optimizer.step()
                for name, value in before.items():
                    torch.testing.assert_close(dict(layer.named_buffers())[name], value)
                target = block_type(16)
                payload = io.BytesIO()
                torch.save(layer.state_dict(), payload)
                payload.seek(0)
                target.load_state_dict(torch.load(payload, weights_only=True))
                torch.testing.assert_close(layer.eval()(x), target.eval()(x))

    def test_wavelet_profiler_includes_fixed_transform_macs(self):
        expected = 2 * 3 * 9 * 9 * 25
        height = 9
        for _ in range(2):
            height = (height + 1) // 2
            expected += 2 * 12 * height * height * 25 + 64 * 3 * height * height
        self.assertAlmostEqual(convolution_linear_gflops(HaarWTConv(3), 9) * 1e9, expected)

    def test_new_arms_preserve_common_weights_rng_and_fusion_outputs(self):
        torch.manual_seed(179)
        baseline = MigratedDetectionModel(variant='baseline').eval()
        expected_rng = torch.rand(4)
        image = torch.randn(1, 3, 128, 128)
        for variant in NEW_VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(179)
                model = MigratedDetectionModel(variant=variant).eval()
                torch.testing.assert_close(torch.rand(4), expected_rng)
                for index in (0, 2, 9, 22):
                    for key, value in baseline.model[index].state_dict().items():
                        torch.testing.assert_close(value, model.model[index].state_dict()[key])
                if variant != 'ghostconv':
                    for index in (4, 6, 8):
                        for name in ('cv1', 'cv2'):
                            expected = getattr(baseline.model[index], name).state_dict()
                            actual = getattr(model.model[index], name).state_dict()
                            for key, value in expected.items():
                                torch.testing.assert_close(value, actual[key])
                with torch.no_grad():
                    before = model(image)[0]
                    model.fuse(verbose=False)
                    after = model(image)[0]
                torch.testing.assert_close(before, after, atol=2e-4, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
