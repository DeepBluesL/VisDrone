import random
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from ultralytics.nn.tasks import load_checkpoint

from visdrone_migration.realtime_models import build_realtime_model
from visdrone_migration.realtime_trainer import RealtimeTrainer
from visdrone_migration.realtime_performance import ChunkedTaskAlignedAssigner


class _EMA:
    def __init__(self, model):
        self.ema = model
        self.updates = 12


class _Scaler:
    def state_dict(self):
        return {"scale": 1024.0}


class RealtimeTrainerTests(unittest.TestCase):
    def test_setup_installs_chunked_assignment_on_raw_and_ema_models(self):
        trainer = object.__new__(RealtimeTrainer)
        trainer.model = build_realtime_model("standard", nc=10, seed=7)
        trainer.ema = _EMA(build_realtime_model("standard", nc=10, seed=8))
        loss_args = types.SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        trainer.model.args = loss_args
        trainer.ema.ema.args = loss_args
        with patch(
            "visdrone_migration.realtime_models.AP50_95DetectionTrainer._setup_train",
            return_value=None,
        ):
            trainer._setup_train(world_size=1)
        self.assertIsInstance(trainer.model.criterion.assigner, ChunkedTaskAlignedAssigner)
        self.assertIsInstance(trainer.ema.ema.criterion.assigner, ChunkedTaskAlignedAssigner)

    def test_validation_loader_caps_batch_at_32(self):
        trainer = object.__new__(RealtimeTrainer)
        trainer.args = types.SimpleNamespace(workers=8)
        seen = {}

        def build_dataset(path, mode, batch):
            seen["dataset"] = (path, mode, batch)
            return types.SimpleNamespace(rect=False)

        trainer.build_dataset = build_dataset
        sentinel = object()
        with patch(
            "visdrone_migration.realtime_trainer.build_performance_dataloader",
            return_value=sentinel,
        ) as loader:
            actual = trainer.get_dataloader("validation", batch_size=128, rank=-1, mode="val")
        self.assertIs(actual, sentinel)
        self.assertEqual(seen["dataset"], ("validation", "val", 32))
        self.assertEqual(loader.call_args.kwargs["batch_size"], 32)

    def test_checkpoint_is_atomic_fp32_native_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = object.__new__(RealtimeTrainer)
            trainer.model = build_realtime_model("standard", nc=10, seed=3)
            trainer.ema = _EMA(build_realtime_model("standard", nc=10, seed=4))
            trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
            trainer.scaler = _Scaler()
            trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
                trainer.optimizer, lr_lambda=lambda _: 1.0
            )
            trainer.epoch = 6
            trainer.best_fitness = trainer.fitness = 0.25
            trainer.selected_epoch = 7
            trainer.experiment_identity = {"protocol_sha256": "abc", "arm": "standard_512"}
            trainer.transfer_report = {"loaded_numel": 123}
            trainer.metrics = {"metrics/mAP50-95(B)": 0.25}
            trainer.args = types.SimpleNamespace(plots=False, model="yolov8n.yaml")
            trainer.read_results_csv = lambda: {"epoch": [1, 2]}
            trainer.csv = root / "results.csv"
            trainer.csv.write_text("epoch,metric\n1,0.2\n", encoding="utf-8")
            trainer.last, trainer.best = root / "last.pt", root / "best.pt"
            trainer.wdir, trainer.save_period = root, -1

            trainer.save_model()
            self.assertTrue(trainer.last.is_file())
            self.assertTrue(trainer.best.is_file())
            self.assertFalse(list(root.glob("*.tmp")))
            checkpoint = torch.load(trainer.last, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["selected_epoch"], 7)
            self.assertEqual(checkpoint["experiment_identity"], trainer.experiment_identity)
            self.assertIn("python", checkpoint["rng_state"])
            self.assertIn("numpy", checkpoint["rng_state"])
            self.assertIn("torch_cpu", checkpoint["rng_state"])
            self.assertEqual(checkpoint["scaler"], {"scale": 1024.0})
            self.assertIn("last_epoch", checkpoint["scheduler"])
            self.assertEqual(checkpoint["results_csv_text"], "epoch,metric\n1,0.2\n")
            self.assertTrue(all(p.dtype == torch.float32 for p in checkpoint["model"].parameters()))
            self.assertTrue(all(p.dtype == torch.float32 for p in checkpoint["ema"].parameters()))
            inference_model, native_checkpoint = load_checkpoint(trainer.best)
            self.assertEqual(inference_model.model[-1].nc, 10)
            self.assertEqual(native_checkpoint["experiment_identity"], trainer.experiment_identity)

    def test_semantic_get_model_and_exact_resume_model(self):
        source = build_realtime_model("standard", nc=80, seed=10)
        trainer = object.__new__(RealtimeTrainer)
        trainer.architecture = "p2"
        trainer.initialization_seed = 99
        trainer.pretrained_path = None
        trainer.data = {"nc": 10, "channels": 3}
        trainer.resume = False
        trainer.transfer_report = None
        fresh = trainer.get_model(weights=source, verbose=False)
        self.assertEqual(fresh.model[-1].nc, 10)
        self.assertIsNotNone(trainer.transfer_report)
        self.assertEqual(trainer.transfer_report["architecture"], "p2")

        trainer.resume = True
        resumed = trainer.get_model(weights=fresh, verbose=False)
        for key, value in fresh.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)

    def test_checkpoint_captures_current_rng_states(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        trainer = object.__new__(RealtimeTrainer)
        trainer.model = build_realtime_model("standard", nc=10, seed=5)
        trainer.ema = _EMA(trainer.model)
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        trainer.scaler = _Scaler()
        trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
            trainer.optimizer, lr_lambda=lambda _: 1.0
        )
        trainer.epoch = 0
        trainer.best_fitness = trainer.fitness = 0.1
        trainer.selected_epoch = 1
        trainer.experiment_identity = {"arm": "test"}
        trainer.transfer_report = None
        trainer.metrics = {}
        trainer.args = types.SimpleNamespace()
        trainer.read_results_csv = lambda: {}
        trainer.csv = Path("missing-test-results.csv")
        checkpoint = trainer._checkpoint()
        self.assertEqual(checkpoint["rng_state"]["python"], random.getstate())
        np.testing.assert_equal(checkpoint["rng_state"]["numpy"], np.random.get_state())
        torch.testing.assert_close(checkpoint["rng_state"]["torch_cpu"], torch.get_rng_state())

    def test_resume_reconciles_csv_and_missing_new_best_to_last_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = object.__new__(RealtimeTrainer)
            trainer.csv = root / "results.csv"
            trainer.csv.write_text("header\nold\nahead\n", encoding="utf-8")
            trainer.last = root / "last.pt"
            trainer.last.write_bytes(b"complete-last-checkpoint")
            trainer.best = root / "best.pt"
            trainer.best.write_bytes(b"stale-best")
            trainer.resume_checkpoint_path = trainer.last
            checkpoint = {
                "results_csv_text": "header\nold\n",
                "best_fitness": 0.4,
                "train_metrics": {"fitness": 0.4},
            }
            trainer._reconcile_resume_files(checkpoint)
            self.assertEqual(trainer.csv.read_text(encoding="utf-8"), "header\nold\n")
            self.assertEqual(trainer.best.read_bytes(), trainer.last.read_bytes())

    def test_completed_budget_resume_finalizes_without_training_an_extra_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "last.pt"
            torch.save({"epoch": 4}, checkpoint_path)
            trainer = object.__new__(RealtimeTrainer)
            trainer.resume_checkpoint_path = checkpoint_path
            trainer.epochs = 5
            events = []

            def setup(_world_size):
                trainer.start_epoch = 5

            trainer._setup_train = setup
            trainer.final_eval = lambda: events.append("final_eval")
            trainer.run_callbacks = lambda name: events.append(name)
            trainer._clear_memory = lambda *args: None
            trainer._do_train(world_size=0)
            self.assertEqual(trainer.epoch, 4)
            self.assertEqual(events, ["final_eval", "on_train_end", "teardown"])


if __name__ == "__main__":
    unittest.main()
