from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("finalize_realtime", ROOT / "scripts" / "finalize_realtime.py")
assert SPEC and SPEC.loader
finalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalizer)


class FinalizeRealtimeTests(unittest.TestCase):
    def test_training_batches_must_match_arm_and_nbs(self):
        training = {"batch": 16, "nbs": 64}
        arm = {"id": "p2_1024_s179", "batch": 32}
        self.assertEqual(
            finalizer.verify_training_batches(
                {"batch": 32, "effective_batch": 64}, arm, training, arm["id"]
            ),
            (32, 64),
        )
        with self.assertRaisesRegex(ValueError, "physical batch"):
            finalizer.verify_training_batches(
                {"batch": 16, "effective_batch": 64}, arm, training, arm["id"]
            )
        with self.assertRaisesRegex(ValueError, "effective batch"):
            finalizer.verify_training_batches(
                {"batch": 32, "effective_batch": 32}, arm, training, arm["id"]
            )

    def test_verify_epochs_requires_selected_best(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epochs.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["epoch", "metrics/mAP50-95(B)"])
                writer.writeheader()
                writer.writerows([
                    {"epoch": 1, "metrics/mAP50-95(B)": 0.1},
                    {"epoch": 2, "metrics/mAP50-95(B)": 0.3},
                    {"epoch": 3, "metrics/mAP50-95(B)": 0.2},
                ])
            self.assertAlmostEqual(finalizer.verify_epochs(path, 2, 3)["best_csv_ap50_95"], 0.3)
            with self.assertRaisesRegex(ValueError, "below CSV best"):
                finalizer.verify_epochs(path, 3, 3)

    def test_readme_update_is_marker_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before, after = "header\n", "\nfooter\n"
            (root / "README.md").write_text(
                before + finalizer.README_START + "\nold\n" + finalizer.README_END + after,
                encoding="utf-8",
            )
            row = {"id": "standard_512_s179", "architecture": "standard", "imgsz": 512,
                   "map50_95": 0.1234, "map50": 0.2345, "latency_p50_ms": 2.5,
                   "latency_p95_ms": 3.5, "latency_fps": 380.0,
                   "physical_batch": 32, "effective_batch": 64}
            finalizer.update_readme(root, [row])
            value = (root / "README.md").read_text(encoding="utf-8")
            self.assertTrue(value.startswith(before) and value.endswith(after))
            self.assertIn("12.34%", value)
            self.assertIn("2.50 / 3.50", value)
            self.assertIn("32 / 64", value)
            (root / "README.md").write_text("no markers", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "marker pair"):
                finalizer.update_readme(root, [row])

    def test_legacy_manifest_detects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(finalizer, "EXPECTED_LEGACY_ARTIFACTS", 1):
            root = Path(directory)
            artifact = root / "results" / "pilot" / "record.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("evidence", encoding="utf-8")
            manifest = {"artifact_count": 1, "artifacts": [{
                "path": "results/pilot/record.json", "bytes": artifact.stat().st_size,
                "sha256": finalizer.sha256(artifact)}]}
            manifest_path = root / "results" / "artifact_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(finalizer.verify_legacy_manifest(root), manifest["artifacts"])
            artifact.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "legacy artifact changed"):
                finalizer.verify_legacy_manifest(root)

    def test_latency_requires_matching_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latency.json"
            summary = {"samples": 100, "mean_ms": 4.0, "p50_ms": 3.8,
                       "p95_ms": 5.0, "fps_from_mean": 250.0}
            value = {"schema_version": 1, "status": "completed", "arm_id": "standard_512_s179",
                     "architecture": "standard", "imgsz": 512, "seed": 179,
                     "checkpoint_sha256": "checkpoint", "precision": "FP16", "fused": True,
                     "batch": 1,
                     "warmup_iterations": 30, "timed_iterations": 100,
                     "settings": {"conf": 0.25, "iou": 0.7, "max_det": 500, "rect": False},
                     "metrics_json_sha256": "metrics", "protocol_json_sha256": "protocol",
                     "pure_model_cuda_event": summary, "synchronized_wall_clock": summary}
            path.write_text(json.dumps(value), encoding="utf-8")
            arm = {"id": "standard_512_s179", "architecture": "standard", "imgsz": 512, "seed": 179}
            result = finalizer.verify_latency(path, arm=arm, checkpoint_hash="checkpoint",
                                               metrics_hash="metrics", protocol_hash="protocol")
            self.assertEqual(result["p95_ms"], 5.0)
            value["checkpoint_sha256"] = "wrong"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                finalizer.verify_latency(path, arm=arm, checkpoint_hash="checkpoint",
                                          metrics_hash="metrics", protocol_hash="protocol")

    def test_latency_rejects_unfused_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latency.json"
            summary = {"samples": 100, "mean_ms": 4.0, "p50_ms": 3.8,
                       "p95_ms": 5.0, "fps_from_mean": 250.0}
            value = {"schema_version": 1, "status": "completed", "arm_id": "standard_512_s179",
                     "architecture": "standard", "imgsz": 512, "seed": 179,
                     "checkpoint_sha256": "checkpoint", "precision": "FP16", "fused": False,
                     "batch": 1, "warmup_iterations": 30, "timed_iterations": 100,
                     "settings": {"conf": 0.25, "iou": 0.7, "max_det": 500, "rect": False},
                     "metrics_json_sha256": "metrics", "protocol_json_sha256": "protocol",
                     "pure_model_cuda_event": summary, "synchronized_wall_clock": summary}
            path.write_text(json.dumps(value), encoding="utf-8")
            arm = {"id": "standard_512_s179", "architecture": "standard", "imgsz": 512, "seed": 179}
            with self.assertRaisesRegex(ValueError, "fused"):
                finalizer.verify_latency(path, arm=arm, checkpoint_hash="checkpoint",
                                          metrics_hash="metrics", protocol_hash="protocol")


if __name__ == "__main__":
    unittest.main()
