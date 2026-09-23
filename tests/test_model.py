"""Graph compatibility and controlled-initialization tests for detector surgery."""
import os
from pathlib import Path
import unittest
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".runtime" / "ultralytics"))
import torch
from visdrone_migration.model import MigratedDetectionModel, MigrationTrainer, VARIANTS


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_variants_preserve_detection_graph(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = MigratedDetectionModel(variant=variant).eval()
                with torch.no_grad():
                    prediction, _ = model(torch.zeros(1, 3, 128, 128))
                self.assertEqual(tuple(prediction.shape), (1, 14, 336))
                self.assertTrue(torch.isfinite(prediction).all())
                self.assertEqual(model.model[-1].f, [15, 18, 21])

    def test_shared_head_weights_and_rng_match_across_arms(self):
        torch.manual_seed(179)
        base = MigratedDetectionModel(variant="baseline")
        base_next = torch.rand(4)
        torch.manual_seed(179)
        full = MigratedDetectionModel(variant="full")
        full_next = torch.rand(4)
        torch.testing.assert_close(base_next, full_next)
        for key, value in base.model[-1].state_dict().items():
            torch.testing.assert_close(value, full.model[-1].state_dict()[key])

    def test_module_initialization_matches_single_and_full(self):
        for variant, index in (("ghost", 4), ("gabor", 0), ("sppf", 9)):
            single = MigratedDetectionModel(variant=variant, module_seed=179)
            full = MigratedDetectionModel(variant="full", module_seed=179)
            for key, value in single.model[index].state_dict().items():
                torch.testing.assert_close(value, full.model[index].state_dict()[key])

    def test_full_model_fusion_preserves_predictions_and_removes_all_bn(self):
        model = MigratedDetectionModel(variant="full").eval()
        image = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            before = model(image)[0]
            model.fuse(verbose=False)
            after = model(image)[0]
        torch.testing.assert_close(before, after, atol=2e-4, rtol=1e-4)
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()))

    def test_checkpoint_selection_uses_only_ap50_95(self):
        trainer = MigrationTrainer.__new__(MigrationTrainer)
        trainer.best_fitness = 0.45
        trainer.selected_epoch = 2
        trainer.epoch = 3
        trainer.validator = lambda _: {"metrics/mAP50(B)": 0.99,
                                      "metrics/mAP50-95(B)": 0.44, "fitness": 0.495}
        metrics, fitness = trainer.validate()
        self.assertEqual(fitness, 0.44)
        self.assertEqual(trainer.selected_epoch, 2)
        self.assertNotIn("fitness", metrics)


if __name__ == "__main__":
    unittest.main()
