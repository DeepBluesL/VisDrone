import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_suite import validate_completed_record
from visdrone_migration.evidence import (
    code_fingerprint,
    data_fingerprint,
    environment_fingerprint,
    file_sha256,
)


class FingerprintTests(unittest.TestCase):
    def test_data_fingerprint_tracks_images_and_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "val"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
                (root / "images" / split / f"{split}.jpg").write_bytes(b"image")
                (root / "labels" / split / f"{split}.txt").write_text(
                    "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                )
                (root / f"{split}.txt").write_text(
                    f"./images/{split}/{split}.jpg\n", encoding="utf-8"
                )
            yaml_path = root / "dataset.yaml"
            yaml_path.write_text(
                f"path: {root.as_posix()}\ntrain: train.txt\nval: val.txt\nnames: {{0: item}}\n",
                encoding="utf-8",
            )
            before = data_fingerprint(yaml_path)
            self.assertEqual(before["train"]["images"], 1)
            (root / "labels" / "train" / "train.txt").write_text(
                "0 0.4 0.5 0.2 0.2\n", encoding="utf-8"
            )
            self.assertNotEqual(before, data_fingerprint(yaml_path))

    def test_code_fingerprint_ignores_runner_but_tracks_training_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "visdrone_migration").mkdir()
            (root / "scripts").mkdir()
            (root / "visdrone_migration" / "model.py").write_text("model=1\n")
            (root / "scripts" / "train.py").write_text("train=1\n")
            (root / "scripts" / "run_suite.py").write_text("runner=1\n")
            before = code_fingerprint(root)
            (root / "scripts" / "run_suite.py").write_text("runner=2\n")
            self.assertEqual(before, code_fingerprint(root))
            (root / "visdrone_migration" / "model.py").write_text("model=2\n")
            self.assertNotEqual(before, code_fingerprint(root))

    def test_environment_identity_has_required_dependencies(self):
        identity = environment_fingerprint()
        self.assertIn("python", identity)
        self.assertEqual(
            set(identity["packages"]),
            {"torch", "torchvision", "ultralytics", "numpy", "Pillow"},
        )


class CompletionValidationTests(unittest.TestCase):
    def test_validates_budget_fingerprints_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"weights")
            identity = {
                "data_fingerprint": {"train": {"images": 2}, "val": {"images": 1}},
                "data_yaml_sha256": "yaml",
                "training_code_sha256": "code",
                "environment_fingerprint": {"python": "3.x", "packages": {}},
            }
            record = {
                "status": "completed",
                "variant": "baseline",
                "seed": 179,
                "epochs": 10,
                "imgsz": 512,
                "batch": 16,
                "training": {"workers": 4},
                **identity,
                "checkpoint_sha256": file_sha256(checkpoint),
            }
            metrics = root / "metrics.json"
            metrics.write_text(json.dumps(record), encoding="utf-8")
            validate_completed_record(
                metrics,
                checkpoint,
                variant="baseline",
                seed=179,
                epochs=10,
                imgsz=512,
                batch=16,
                workers=4,
                identity=identity,
            )
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
                validate_completed_record(
                    metrics,
                    checkpoint,
                    variant="baseline",
                    seed=179,
                    epochs=10,
                    imgsz=512,
                    batch=16,
                    workers=4,
                    identity=identity,
                )


if __name__ == "__main__":
    unittest.main()
