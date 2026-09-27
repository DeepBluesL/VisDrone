import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import report_realtime


class RealtimeReportIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.suite = self.root / "results" / "suite"
        self.suite.mkdir(parents=True)
        self.arm = {"id": "standard_512_s179", "architecture": "standard",
                    "imgsz": 512, "seed": 179}
        self.protocol = {"suite": "suite", "training": {"epochs": 100}, "arms": [self.arm]}
        (self.suite / "protocol.json").write_text(json.dumps(self.protocol), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_completed_requires_protocol_identity_and_exported_checkpoint(self):
        run = self.suite / self.arm["id"]
        run.mkdir()
        checkpoint = self.root / "checkpoints" / "suite" / f"{self.arm['id']}.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"weights")
        record = {
            "status": "completed", "arm_id": self.arm["id"], "architecture": "standard",
            "imgsz": 512, "seed": 179, "epochs": 100,
            "experiment_identity": {"protocol_sha256": report_realtime._canonical_sha(self.protocol),
                                    "arm": self.arm},
            "checkpoint_sha256": hashlib.sha256(b"weights").hexdigest(),
            "metrics": {key: 0.2 for key in report_realtime.METRICS.values()},
        }
        (run / "metrics.json").write_text(json.dumps(record), encoding="utf-8")
        normalized_arm = report_realtime._protocol_arms(self.protocol)[0]
        with patch.object(report_realtime, "ROOT", self.root):
            state = report_realtime._run_state(self.suite, normalized_arm)
            self.assertEqual(state["status"], "completed")
            checkpoint.write_bytes(b"wrong")
            state = report_realtime._run_state(self.suite, normalized_arm)
            self.assertEqual(state["status"], "invalid")
            self.assertTrue(any("checkpoint SHA-256" in issue for issue in state["issues"]))

    def test_failed_queue_overrides_stale_running_progress_for_same_arm_only(self):
        run = self.suite / self.arm["id"]
        run.mkdir()
        (run / "progress.json").write_text(json.dumps({"status": "running", "arm_id": self.arm["id"],
            "architecture": "standard", "imgsz": 512, "seed": 179, "epochs": 100, "epoch": 12}),
            encoding="utf-8")
        (self.suite / "queue_state.json").write_text(json.dumps({"status": "failed",
            "arm": self.arm["id"], "error": "child exited 1"}), encoding="utf-8")
        state = report_realtime._run_state(self.suite, report_realtime._protocol_arms(self.protocol)[0])
        self.assertEqual(state["status"], "failed")
        self.assertTrue(any("child exited 1" in issue for issue in state["issues"]))


if __name__ == "__main__":
    unittest.main()
