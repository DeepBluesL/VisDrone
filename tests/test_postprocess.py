import json
import tempfile
import unittest
from pathlib import Path

from scripts.diagnostic_provenance import raw_inputs_sha256
from scripts.postprocess_suite import reusable_occlusion, valid_diagnostic_metrics


def record(matched=1, total=1):
    return {"match_count": matched, "total": total, "recall": matched / total if total else None}


def metrics():
    return {
        "overall": record(),
        "occlusion": {"0": record(), "1": record(0, 0), "2": record(0, 0)},
        "truncation": {"0": record(), "1": record(0, 0), "2": record(0, 0)},
        "size": {"small": record(), "medium": record(0, 0), "large": record(0, 0)},
        "class": {"1:pedestrian": record()},
    }


class DiagnosticProvenanceTests(unittest.TestCase):
    def test_raw_digest_tracks_image_and_annotation_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            (root / "annotations").mkdir()
            image = root / "images" / "a.jpg"
            annotation = root / "annotations" / "a.txt"
            image.write_bytes(b"image")
            annotation.write_bytes(b"annotation")
            before = raw_inputs_sha256(root, [image])
            annotation.write_bytes(b"changed")
            self.assertNotEqual(before, raw_inputs_sha256(root, [image]))

    def test_metric_validator_rejects_inconsistent_counts(self):
        value = metrics()
        self.assertTrue(valid_diagnostic_metrics(value))
        value["size"]["small"]["recall"] = 0.5
        self.assertFalse(valid_diagnostic_metrics(value))
        value = metrics()
        value["size"]["small"] = record(0, 1)
        self.assertFalse(valid_diagnostic_metrics(value))

    def test_reuse_requires_version_semantics_and_raw_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "raw"
            data.mkdir()
            weights = root / "model.pt"
            weights.write_bytes(b"weights")
            output = root / "occlusion.json"
            payload = {
                "schema_version": 2,
                "diagnostic_version": "visdrone_raw_recall_v2",
                "metrics": metrics(),
                "settings": {
                    "source_image_count": 1, "evaluated_image_count": 1,
                    "is_subset_smoke_run": False, "max_images": None,
                    "imgsz": 512, "batch": 16, "max_det": 500, "device": "cpu",
                    "data_root": str(data.resolve()), "weights": str(weights.resolve()),
                    "confidence_threshold": 0.05, "nms_iou_threshold": 0.5,
                    "matching_iou_threshold": 0.5,
                    "eligible_ground_truth": "score > 0, source class 1..10, valid after clipping",
                    "small_object_definition": "clipped raw-image area < 32^2 pixels (COCO convention)",
                    "size_bins": "small < 32^2; medium 32^2 to < 96^2; large >= 96^2 raw pixels",
                },
                "provenance": {"weights_sha256": "weights", "raw_image_annotation_sha256": "raw"},
            }
            output.write_text(json.dumps(payload), encoding="utf-8")
            kwargs = dict(checkpoint_sha256="weights", image_count=1, data_root=data,
                          weights=weights, imgsz=512, device="cpu", raw_inputs_hash="raw")
            self.assertTrue(reusable_occlusion(output, **kwargs))
            payload["diagnostic_version"] = "old"
            output.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(reusable_occlusion(output, **kwargs))


if __name__ == "__main__":
    unittest.main()
