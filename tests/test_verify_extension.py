import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.verify_results import (
    EFFICIENT_DEPTHS,
    EFFICIENT_UNITS,
    build_parser,
    expected_layers,
    sha256_file,
    verify_snapshot_hashes,
)


class ExtensionVerifierTests(unittest.TestCase):
    def test_expected_run_count_defaults_to_protocol(self):
        self.assertIsNone(build_parser().parse_args([]).expected_runs)

    def test_extension_layer_specs(self):
        ghost, _ = expected_layers("ghostconv")
        for index in (1, 3, 5, 7):
            self.assertEqual(ghost[index], "visdrone_migration.modules.GhostConv")
        for variant in EFFICIENT_UNITS:
            expected, _ = expected_layers(variant)
            for index in EFFICIENT_DEPTHS:
                self.assertEqual(expected[index], "visdrone_migration.model.EfficientC2f")

    def test_snapshot_hashes_are_checked_against_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "results" / "suite" / "data_snapshot"
            snapshot.mkdir(parents=True)
            dataset = snapshot / "dataset.yaml"
            manifest = snapshot / "preparation_manifest.json"
            dataset.write_text("data", encoding="utf-8")
            manifest.write_text("manifest", encoding="utf-8")
            protocol = {
                "data_snapshot": {
                    "dataset_yaml": "results/suite/data_snapshot/dataset.yaml",
                    "preparation_manifest": "results/suite/data_snapshot/preparation_manifest.json",
                },
                "data_yaml_sha256": sha256_file(dataset),
                "preparation_manifest_sha256": sha256_file(manifest),
            }
            with patch("scripts.verify_results.ROOT", root):
                checks, _ = verify_snapshot_hashes(root / "results" / "suite", protocol)
                self.assertTrue(all(checks.values()))
                dataset.write_text("changed", encoding="utf-8")
                checks, _ = verify_snapshot_hashes(root / "results" / "suite", protocol)
                self.assertFalse(checks["snapshot_dataset_yaml_sha256"])


if __name__ == "__main__":
    unittest.main()
