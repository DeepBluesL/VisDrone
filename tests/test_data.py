from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_data import (
    ConversionStats,
    convert_annotation_line,
    prepare_split,
    select_images,
    write_dataset_yaml,
)


class AnnotationConversionTests(unittest.TestCase):
    def test_valid_box_maps_class_and_normalizes(self) -> None:
        stats = ConversionStats()

        result = convert_annotation_line("10,20,30,40,1,4,0,2", 100, 200, stats)

        self.assertEqual(result, "3 0.25000000 0.20000000 0.30000000 0.20000000")
        self.assertEqual(stats.kept_rows, 1)
        self.assertEqual(stats.kept_class_counts[4], 1)
        self.assertEqual(stats.kept_occlusion_counts[2], 1)

    def test_box_is_clipped_to_image_bounds(self) -> None:
        stats = ConversionStats()

        result = convert_annotation_line("-10,80,40,40,1,1,1,1", 100, 100, stats)

        self.assertEqual(result, "0 0.15000000 0.90000000 0.30000000 0.20000000")
        self.assertEqual(stats.clipped_boxes, 1)

    def test_ignored_flags_overlap_but_drop_reason_is_exclusive(self) -> None:
        stats = ConversionStats()

        result = convert_annotation_line("0,0,10,10,0,0,0,0,", 100, 100, stats)

        self.assertIsNone(result)
        self.assertEqual(stats.ignored_flags["score_zero"], 1)
        self.assertEqual(stats.ignored_flags["class_0_ignored_region"], 1)
        self.assertEqual(stats.drop_reasons, {"score_zero": 1})

    def test_class_eleven_and_unsupported_class_are_dropped(self) -> None:
        stats = ConversionStats()

        self.assertIsNone(convert_annotation_line("0,0,1,1,1,11,0,0", 10, 10, stats))
        self.assertIsNone(convert_annotation_line("0,0,1,1,1,12,0,0", 10, 10, stats))

        self.assertEqual(stats.drop_reasons["class_11_others"], 1)
        self.assertEqual(stats.drop_reasons["unsupported_class"], 1)

    def test_invalid_and_outside_boxes_are_accounted(self) -> None:
        stats = ConversionStats()

        self.assertIsNone(convert_annotation_line("bad,row", 10, 10, stats))
        self.assertIsNone(convert_annotation_line("2,2,0,3,1,1,0,0", 10, 10, stats))
        self.assertIsNone(convert_annotation_line("11,2,3,3,1,1,0,0", 10, 10, stats))

        self.assertEqual(stats.total_rows, 3)
        self.assertEqual(sum(stats.drop_reasons.values()), 3)
        self.assertEqual(stats.drop_reasons["malformed_field_count"], 1)
        self.assertEqual(stats.drop_reasons["non_positive_box_size"], 1)
        self.assertEqual(stats.drop_reasons["box_outside_image"], 1)


class DatasetPreparationTests(unittest.TestCase):
    def test_subset_selection_is_reproducible_and_sorted(self) -> None:
        paths = [Path(f"{index:03}.jpg") for index in range(20)]

        first = select_images(paths, limit=5, seed=179, split="train")
        second = select_images(list(reversed(paths)), limit=5, seed=179, split="train")

        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first, key=lambda path: path.name))
        self.assertNotEqual(first, select_images(paths, 5, 180, "train"))

    def test_prepare_split_preserves_source_and_writes_portable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "VisDrone2019-DET-train"
            (source / "images").mkdir(parents=True)
            (source / "annotations").mkdir()
            image_path = source / "images" / "sample.jpg"
            Image.new("RGB", (100, 50), "white").save(image_path)
            annotation_path = source / "annotations" / "sample.txt"
            original = "10,5,20,10,1,2,0,1\n0,0,5,5,0,0,0,0\n"
            annotation_path.write_text(original, encoding="utf-8")
            output = root / "yolo"

            result = prepare_split("train", source, output, 0, 179, "copy")

            self.assertEqual(annotation_path.read_text(encoding="utf-8"), original)
            self.assertTrue((output / "images" / "train" / "sample.jpg").is_file())
            label = (output / "labels" / "train" / "sample.txt").read_text(
                encoding="utf-8"
            )
            self.assertEqual(label, "1 0.20000000 0.20000000 0.20000000 0.20000000\n")
            self.assertEqual(
                (output / "train.txt").read_text(encoding="utf-8"),
                "./images/train/sample.jpg\n",
            )
            self.assertEqual(result["source_split"], "VisDrone2019-DET-train")
            self.assertEqual(result["selected_filenames"], ["sample.jpg"])
            self.assertEqual(result["stats"]["annotation_rows"]["dropped"], 1)
            self.assertEqual(result["stats"]["ignored_flags_nonexclusive"]["score_zero"], 1)

    def test_dataset_yaml_uses_absolute_posix_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            yaml_path = root / "dataset.yaml"

            write_dataset_yaml(yaml_path, root / "yolo")

            text = yaml_path.read_text(encoding="utf-8")
            expected = (root / "yolo").resolve().as_posix()
            self.assertIn(json.dumps(expected), text)
            self.assertIn("train: train.txt", text)
            self.assertIn("val: val.txt", text)
            self.assertIn("9: motor", text)


if __name__ == "__main__":
    unittest.main()
