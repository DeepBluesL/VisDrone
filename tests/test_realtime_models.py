import json
import unittest

import torch

from visdrone_migration.realtime_models import (
    AP50_95DetectionTrainer,
    build_fair_model_pair,
    build_realtime_model,
)


class RealtimeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = build_realtime_model("standard", nc=80, seed=41)
        with torch.no_grad():
            for index, tensor in enumerate(cls.source.state_dict().values()):
                if tensor.is_floating_point():
                    tensor.fill_((index % 29 + 1) / 31)

    def test_official_standard_and_p2_graphs_preserve_shapes(self):
        standard = build_realtime_model("standard", nc=10, seed=7).eval()
        p2 = build_realtime_model("p2", nc=10, seed=7).eval()
        self.assertEqual(len(standard.model), 23)
        self.assertEqual(len(p2.model), 29)
        self.assertEqual(standard.model[-1].nc, 10)
        self.assertEqual(p2.model[-1].nc, 10)
        with torch.no_grad():
            standard_output = standard(torch.randn(1, 3, 64, 64))[0]
            p2_output = p2(torch.randn(1, 3, 64, 64))[0]
        self.assertEqual(standard_output.shape[1], 14)
        self.assertEqual(p2_output.shape[1], 14)
        self.assertGreater(p2_output.shape[2], standard_output.shape[2])

    def test_fair_transfer_is_semantic_auditable_and_reproducible(self):
        standard, p2, reports = build_fair_model_pair(self.source, nc=10, seed=179)
        standard_state, p2_state = standard.state_dict(), p2.state_dict()
        for key, value in standard_state.items():
            parts = key.split(".", 2)
            if len(parts) == 3 and parts[0] == "model" and int(parts[1]) <= 9:
                torch.testing.assert_close(value, p2_state[key], rtol=0, atol=0)
                torch.testing.assert_close(value, self.source.state_dict()[key], rtol=0, atol=0)

        # P2 layer 21 is a new bottom-up P3 fusion, not standard top-down layer 15.
        new_p3 = next(
            r
            for r in reports["p2"].records
            if r.target_key == "model.21.cv2.conv.weight"
        )
        self.assertEqual(new_p3.status, "skipped")
        self.assertEqual(new_p3.semantic, "P2-only path")

        # P2 layer 24 is the semantic P4 fusion corresponding to standard layer 18.
        target_key = "model.24.cv2.conv.weight"
        source_key = "model.18.cv2.conv.weight"
        torch.testing.assert_close(p2_state[target_key], self.source.state_dict()[source_key])
        record = next(r for r in reports["p2"].records if r.target_key == target_key)
        self.assertEqual(record.source_key, source_key)
        self.assertEqual(record.status, "loaded")
        self.assertTrue(any("classification" in r.semantic for r in reports["p2"].records))
        self.assertLess(reports["p2"].parameter_numel_coverage, 1.0)
        json.dumps(reports["p2"].to_dict())

        # Unmapped P2 tensors are deterministically initialized by the local seed.
        _, p2_again, _ = build_fair_model_pair(self.source, nc=10, seed=179)
        torch.testing.assert_close(
            p2_state["model.18.cv1.conv.weight"],
            p2_again.state_dict()["model.18.cv1.conv.weight"],
            rtol=0,
            atol=0,
        )

    def test_ap50_95_trainer_selection_logic(self):
        trainer = object.__new__(AP50_95DetectionTrainer)
        trainer.best_fitness = 0.25
        trainer.validator = lambda _: {
            "fitness": 999.0,
            "metrics/mAP50-95(B)": 0.3,
            "metrics/mAP50(B)": 0.5,
        }
        metrics, fitness = AP50_95DetectionTrainer.validate(trainer)
        self.assertEqual(fitness, 0.3)
        self.assertEqual(trainer.best_fitness, 0.3)
        self.assertNotIn("fitness", metrics)


if __name__ == "__main__":
    unittest.main()
