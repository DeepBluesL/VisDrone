import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
import importlib.metadata
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_tiny_optimization", ROOT / "scripts" / "train_tiny_optimization.py"
)
assert SPEC and SPEC.loader
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class TinyOptimizationLauncherTests(unittest.TestCase):
    def test_default_is_cuda_free_seven_arm_plan(self):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train_tiny_optimization.py")],
            cwd=ROOT, env=environment, check=True, text=True, capture_output=True,
        )
        value = json.loads(result.stdout)
        self.assertEqual(value["action"], "plan-only")
        self.assertEqual(value["arms"], list(launcher.ARM_NAMES))
        self.assertEqual(len(value["arms"]), 7)
        self.assertFalse(value["cuda_initialized"])

    def test_realtime_stage1_suite_is_rejected(self):
        with self.assertRaises(SystemExit):
            launcher.parse_args(["--suite", "realtime_stage1"])

    def test_execute_requires_one_arm(self):
        args = launcher.parse_args(["--execute"])
        with self.assertRaisesRegex(SystemExit, "requires exactly one"):
            launcher.execute(args)

    def test_calibration_requires_full_matching_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "dataset.yaml"
            pretrained = root / "model.pt"
            calibration = root / "calibration.json"
            data.write_text("data", encoding="utf-8")
            pretrained.write_bytes(b"weights")
            fingerprint = {"images": 6471, "image_and_label_sha256": "train-sha"}
            value = {
                "schema_version": 1, "method": launcher.CALIBRATION_METHOD,
                "status": "completed", "split": "train", "subset": False,
                "max_images": None, "data": str(data.resolve()),
                "train_image_and_label_fingerprint": fingerprint,
                "pretrained": str(pretrained.resolve()),
                "pretrained_sha256": launcher.file_sha256(pretrained),
                "architecture": "standard", "imgsz": 768, "seed": 179, "nc": 10,
                "image_count": 6471, "pair_count": 123, "normalizers": {"m": 2.0, "n": 3.0},
                "code_sha256": {
                    "scripts/calibrate_simd.py": launcher.file_sha256(ROOT / "scripts" / "calibrate_simd.py"),
                    "visdrone_migration/realtime_models.py": launcher.file_sha256(
                        ROOT / "visdrone_migration" / "realtime_models.py"
                    ),
                    "visdrone_migration/simd.py": launcher.file_sha256(
                        ROOT / "visdrone_migration" / "simd.py"
                    ),
                },
                "environment": {"torch": torch.__version__,
                                "ultralytics": importlib.metadata.version("ultralytics")},
            }
            calibration.write_text(json.dumps(value), encoding="utf-8")
            actual = launcher.validate_calibration(
                calibration, data=data, pretrained=pretrained, architecture="standard",
                imgsz=768, seed=179, expected_train_fingerprint=fingerprint,
            )
            self.assertEqual(actual["normalizers"], {"m": 2.0, "n": 3.0})
            value["subset"] = True
            calibration.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "subset"):
                launcher.validate_calibration(
                    calibration, data=data, pretrained=pretrained, architecture="standard",
                    imgsz=768, seed=179, expected_train_fingerprint=fingerprint,
                )

    def test_simd_parameters_are_frozen_into_both_simd_arms(self):
        calibration = {"normalizers": {"m": 1.25, "n": 4.5}}
        options = launcher.optimization_options(calibration)
        self.assertEqual(options["simd"]["simd_m"], 1.25)
        self.assertEqual(options["set_simd"]["simd_n"], 4.5)
        self.assertNotIn("simd_m", options["baseline"])
        explicit = launcher.explicit_optimization_options(calibration)
        self.assertEqual(explicit["baseline"]["branch_weights"], (1 / 3, 1 / 3, 1 / 3))
        self.assertIsNone(explicit["baseline"]["simd_m"])

    def test_export_uses_ema_and_strips_both_native_detector_copies(self):
        from ultralytics.nn.tasks import DetectionModel
        from visdrone_migration.realtime_models import build_realtime_model
        from visdrone_migration.tiny_optimization import TinyOptimization, attach_tiny_optimization

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = attach_tiny_optimization(
                build_realtime_model("standard", nc=10, seed=1),
                TinyOptimization(auxiliary=True, hbs=True), seed=1,
            )
            ema = attach_tiny_optimization(
                build_realtime_model("standard", nc=10, seed=2),
                TinyOptimization(auxiliary=True, hbs=True), seed=2,
            )
            best = root / "best.pt"
            torch.save({"model": raw, "ema": ema, "train_args": {}, "epoch": 3}, best)
            output = root / "inference.pt"
            digest, returned = launcher.export_inference_checkpoint(
                types.SimpleNamespace(best=best, args=types.SimpleNamespace()), output
            )
            checkpoint = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(digest, launcher.file_sha256(output))
            self.assertIsInstance(checkpoint["model"], DetectionModel)
            self.assertIsInstance(checkpoint["ema"], DetectionModel)
            self.assertFalse(hasattr(checkpoint["model"], "tiny_optimization"))
            self.assertFalse(hasattr(checkpoint["ema"], "set_hbs"))
            self.assertIsNone(checkpoint["optimizer"])
            self.assertLess(
                sum(parameter.numel() for parameter in checkpoint["ema"].parameters()),
                sum(parameter.numel() for parameter in ema.parameters()),
            )
            ema_value = next(ema.parameters()).detach()
            torch.testing.assert_close(next(returned.parameters()).detach(), ema_value)

    def test_verify_protocol_rejects_changed_calibration_bytes_with_same_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "dataset.yaml"
            pretrained = root / "pretrained.pt"
            calibration = root / "calibration.json"
            data.write_text("data", encoding="utf-8")
            pretrained.write_bytes(b"weights")
            calibration.write_text('{"normalizers":{"m":1.0,"n":2.0}}', encoding="utf-8")
            code = {"file.py": "code-sha"}
            runtime = {"python": "test"}
            protocol = {
                "data": str(data), "pretrained": str(pretrained),
                "identity": {
                    "training_code": code,
                    "data": {"train": {"images": 6471}, "val": {"images": 548}},
                    "data_yaml_sha256": launcher.file_sha256(data),
                    "pretrained_sha256": launcher.file_sha256(pretrained),
                    "runtime": runtime,
                },
                "calibration": {"path": str(calibration), "sha256": launcher.file_sha256(calibration)},
            }
            with patch.object(launcher, "code_identity", return_value=code), patch(
                "visdrone_migration.evidence.data_fingerprint", return_value=protocol["identity"]["data"]
            ), patch("visdrone_migration.realtime_protocol.runtime_identity", return_value=runtime):
                launcher.verify_protocol(protocol)
                calibration.write_text('{"normalizers": {"m": 1.0, "n": 2.0}}\n', encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "calibration changed"):
                    launcher.verify_protocol(protocol)


if __name__ == "__main__":
    unittest.main()
