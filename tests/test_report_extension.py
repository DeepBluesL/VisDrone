import unittest
import tempfile
from pathlib import Path

from scripts.report import (
    EXTENSION_ORDER,
    paired_baseline_deltas,
    plot_paired_deltas,
    protocol_run_ids,
)


def run(variant, seed, ap):
    return {"variant": variant, "seed": seed, "_metrics": {"ap50_95": ap}}


class ExtensionReportTests(unittest.TestCase):
    def test_protocol_matrix_drives_expected_runs(self):
        protocol = {
            "content": {"variants": list(EXTENSION_ORDER), "seeds": [179, 2026, 3407]}
        }
        ids = protocol_run_ids(protocol)
        self.assertEqual(len(ids), 18)
        self.assertEqual(ids[0], "baseline_s179")
        self.assertEqual(ids[-1], "lsconv_s3407")

    def test_paired_deltas_match_only_the_same_seed(self):
        runs = [
            run("baseline", 179, 0.10),
            run("baseline", 2026, 0.20),
            run("ghostconv", 179, 0.13),
            run("ghostconv", 2026, 0.18),
            run("ghostconv", 3407, 0.99),  # no same-seed baseline: excluded
        ]
        result = paired_baseline_deltas(runs)
        self.assertEqual(result[0]["paired_seeds"], [179, 2026])
        self.assertAlmostEqual(result[0]["mean_percent_points"], 0.5)
        self.assertAlmostEqual(result[0]["std_percent_points"], 5 / 2**0.5)

    def test_paired_delta_plot_supports_partial_pairs(self):
        paired = [
            {
                "variant": "ghostconv",
                "display_name": "GhostConv",
                "paired_seeds": [179, 2026, 3407],
                "deltas_percent_points": [1.0, -0.5, 0.25],
                "mean_percent_points": 0.25,
                "std_percent_points": 0.75,
            },
            {
                "variant": "pconv",
                "display_name": "PConv",
                "paired_seeds": [179],
                "deltas_percent_points": [0.4],
                "mean_percent_points": 0.4,
                "std_percent_points": None,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "paired.png"
            self.assertTrue(plot_paired_deltas(paired, output))
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
