import copy
import io
import types
import unittest
from unittest.mock import patch

import torch
from ultralytics.nn.tasks import DetectionModel

from visdrone_migration.realtime_models import build_realtime_model
from visdrone_migration.tiny_optimization import (
    TinyOptimization, TinyOptimizationTrainer, SimDTaskAlignedAssigner,
    attach_tiny_optimization, build_tiny_model, inference_detector, ablation_options,
)


class TinyOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.base = build_realtime_model("standard", nc=10, seed=179)
        cls.base.args = types.SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    def model(self, options):
        return attach_tiny_optimization(copy.deepcopy(self.base), TinyOptimization(**options), seed=179)

    def batch(self, empty=False):
        return {"img": torch.rand(2, 3, 64, 64),
                "batch_idx": torch.tensor([] if empty else [0., 1.]),
                "cls": torch.empty(0, 1) if empty else torch.tensor([[0.], [2.]]),
                "bboxes": torch.empty(0, 4) if empty else torch.tensor([[.5, .5, .3, .3], [.4, .4, .2, .2]])}

    def test_disabled_path_matches_native_loss_and_gradients(self):
        batch = self.batch()
        native, adapted = copy.deepcopy(self.base).train(), self.model({}).train()
        loss, items = native(batch)
        actual, actual_items = adapted(batch)
        torch.testing.assert_close(actual, loss, rtol=0, atol=0)
        torch.testing.assert_close(actual_items, items, rtol=0, atol=0)
        actual.sum().backward()
        loss.sum().backward()
        torch.testing.assert_close(adapted.model[0].conv.weight.grad, native.model[0].conv.weight.grad, rtol=0, atol=0)

    def test_all_arms_one_step_and_bn_updated_once(self):
        for arm, options in ablation_options().items():
            with self.subTest(arm=arm):
                options = {**options, **({"simd_m": 1.3, "simd_n": 1.1} if options.get("simd") else {})}
                model = self.model(options).train()
                counters = [module for module in model.model[-1].modules() if isinstance(module, torch.nn.BatchNorm2d)]
                before = [int(module.num_batches_tracked) for module in counters]
                optimizer = torch.optim.SGD(model.parameters(), lr=.001)
                loss, items = model(self.batch())
                self.assertTrue(torch.isfinite(loss).all())
                self.assertEqual(tuple(items.shape), (3,))
                loss.sum().backward()
                self.assertTrue(torch.isfinite(model.model[0].conv.weight.grad).all())
                if options.get("hbs"):
                    self.assertTrue(all(parameter.grad is not None for parameter in model.set_hbs.parameters()))
                optimizer.step()
                self.assertEqual([int(module.num_batches_tracked) - old for module, old in zip(counters, before)], [1] * len(counters))
                self.assertFalse(model.model[-1]._forward_pre_hooks)

    def test_empty_targets_api_and_simd_remain_finite(self):
        model = self.model(dict(auxiliary=True, hbs=True, api=True, simd=True, simd_m=1., simd_n=1.)).train()
        loss, _ = model(self.batch(empty=True))
        loss.sum().backward()
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(model.model[0].conv.weight.grad).all())

    def test_inference_does_not_read_labels_and_export_matches_native(self):
        model = self.model(dict(auxiliary=True, hbs=True, api=True)).eval()
        native = copy.deepcopy(self.base).eval()
        batch = self.batch()
        with torch.no_grad():
            actual = model(batch["img"])[0]
            expected = native(batch["img"])[0]
            exported = inference_detector(model)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(exported(batch["img"])[0], expected, rtol=0, atol=0)
        self.assertIs(type(exported), DetectionModel)
        self.assertFalse(hasattr(exported, "set_hbs"))
        self.assertEqual(sum(p.numel() for p in exported.parameters()), sum(p.numel() for p in native.parameters()))

    def test_checkpoint_roundtrip_and_p2(self):
        model, _ = build_tiny_model("p2", config=TinyOptimization(auxiliary=True, hbs=True, api=True), seed=5)
        model.args = self.base.args
        model.train()
        loss, _ = model(self.batch())
        loss.sum().backward()
        stream = io.BytesIO()
        torch.save(model, stream)
        stream.seek(0)
        restored = torch.load(stream, map_location="cpu", weights_only=False)
        self.assertEqual(restored.tiny_optimization, model.tiny_optimization)
        self.assertEqual(len(restored.set_hbs), 4)
        self.assertFalse(restored.model[-1]._forward_pre_hooks)

    def test_trainer_reinstalls_simd_on_raw_and_ema(self):
        trainer = object.__new__(TinyOptimizationTrainer)
        options = dict(simd=True, simd_m=1.1, simd_n=1.2)
        trainer.model = self.model(options)
        trainer.ema = types.SimpleNamespace(ema=self.model(options))
        with patch("visdrone_migration.realtime_models.AP50_95DetectionTrainer._setup_train", return_value=None):
            trainer._setup_train(world_size=1)
        for model in (trainer.model, trainer.ema.ema):
            self.assertIsInstance(model.criterion.assigner, SimDTaskAlignedAssigner)

    def test_simd_needs_explicit_calibration(self):
        with self.assertRaises(ValueError):
            TinyOptimization(simd=True)


if __name__ == "__main__":
    unittest.main()
