import argparse
from pathlib import Path
import tempfile
import unittest

from scripts.calibrate_simd import _execute, letterbox_geometry, transform_yolo_ground_truth


class CalibrateSimDTests(unittest.TestCase):
    def test_letterbox_ground_truth_matches_centered_ultralytics_geometry(self):
        ratio, left, top = letterbox_geometry(100, 200, 768)
        self.assertAlmostEqual(ratio, 3.84)
        self.assertEqual((left, top), (0, 192))
        boxes = transform_yolo_ground_truth([[3, .5, .5, .5, .5]],
                                            height=100, width=200, imgsz=768)
        self.assertEqual(boxes, [[192.0, 288.0, 576.0, 480.0]])

    def test_partial_execution_requires_explicit_debug_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                data=Path(directory) / "dataset.yaml", architecture="standard", imgsz=768,
                seed=179, pretrained=Path(directory) / "weights.pt",
                output=Path(directory) / "normalizers.json", device="cpu",
                anchor_chunk_size=4096, max_images=1, debug=False, execute=True,
                resume_paused=True, overwrite=False,
            )
            with self.assertRaisesRegex(SystemExit, "debug-only"):
                _execute(args)


if __name__ == "__main__":
    unittest.main()
