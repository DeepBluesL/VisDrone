#!/usr/bin/env python3
"""Measure natural-occlusion GT recall for a trained VisDrone detector.

This is a class-aware IoU=0.5 recall diagnostic over eligible VisDrone ground
truth, not official VisDrone AP. Ignored regions and class 11 are excluded and
the official ignored-region matching rules are not reproduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))

CLASS_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class GroundTruth:
    box: tuple[float, float, float, float]
    class_id: int  # zero-based detector class
    truncation: int
    occlusion: int

    @property
    def area(self) -> float:
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])

    @property
    def size(self) -> str:
        # COCO area convention, measured in original-image pixels.
        if self.area < 32**2:
            return "small"
        if self.area < 96**2:
            return "medium"
        return "large"


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    class_id: int
    confidence: float


def _integer(value: str) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"expected integer-valued field, got {value!r}")
    return int(number)


def parse_annotations(
    lines: Iterable[str], image_width: int, image_height: int
) -> tuple[list[GroundTruth], Counter[str]]:
    """Parse, filter, and clip original VisDrone annotations."""
    ground_truths: list[GroundTruth] = []
    excluded: Counter[str] = Counter()
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip().rstrip(",")
        if not line:
            continue
        fields = [value.strip() for value in line.split(",")]
        if len(fields) < 8:
            raise ValueError(f"annotation line {line_number} has fewer than 8 fields")
        try:
            x, y, width, height, score = (float(value) for value in fields[:5])
            source_class = _integer(fields[5])
            truncation = _integer(fields[6])
            occlusion = _integer(fields[7])
        except ValueError as error:
            raise ValueError(f"invalid annotation line {line_number}: {error}") from error
        if not all(math.isfinite(value) for value in (x, y, width, height, score)):
            raise ValueError(f"annotation line {line_number} contains a non-finite number")
        if score <= 0:
            excluded["score_nonpositive"] += 1
            continue
        if source_class == 0:
            excluded["class_0_ignored_region"] += 1
            continue
        if source_class == 11:
            excluded["class_11_others"] += 1
            continue
        if not 1 <= source_class <= 10:
            excluded["unsupported_class"] += 1
            continue
        x1 = min(max(x, 0.0), float(image_width))
        y1 = min(max(y, 0.0), float(image_height))
        x2 = min(max(x + width, 0.0), float(image_width))
        y2 = min(max(y + height, 0.0), float(image_height))
        if x2 <= x1 or y2 <= y1:
            excluded["invalid_after_clipping"] += 1
            continue
        ground_truths.append(
            GroundTruth((x1, y1, x2, y2), source_class - 1, truncation, occlusion)
        )
    return ground_truths, excluded


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_ground_truths(
    ground_truths: Sequence[GroundTruth],
    detections: Sequence[Detection],
    iou_threshold: float = 0.5,
) -> set[int]:
    """Greedily match confidence-ordered detections to all eligible GT once."""
    unmatched = set(range(len(ground_truths)))
    matched: set[int] = set()
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        candidates = [
            (box_iou(detection.box, ground_truths[index].box), index)
            for index in unmatched
            if detection.class_id == ground_truths[index].class_id
        ]
        if not candidates:
            continue
        overlap, index = max(candidates)
        if overlap >= iou_threshold:
            unmatched.remove(index)
            matched.add(index)
    return matched


def _record(match_count: int, total: int) -> dict[str, int | float | None]:
    return {
        "match_count": match_count,
        "total": total,
        "recall": match_count / total if total else None,
    }


def aggregate_recall(
    ground_truths: Sequence[GroundTruth], matched: set[int]
) -> dict[str, object]:
    """Stratify after global matching; no stratum can rematch a detection."""
    def grouped(
        values: Iterable[tuple[str, int]], required_keys: Sequence[str]
    ) -> dict[str, dict[str, int | float | None]]:
        totals: Counter[str] = Counter()
        matches: Counter[str] = Counter()
        for label, index in values:
            totals[label] += 1
            matches[label] += index in matched
        keys = list(required_keys) + sorted(set(totals) - set(required_keys))
        return {key: _record(matches[key], totals[key]) for key in keys}

    return {
        "overall": _record(len(matched), len(ground_truths)),
        "occlusion": grouped(
            ((str(gt.occlusion), index) for index, gt in enumerate(ground_truths)),
            ("0", "1", "2"),
        ),
        "truncation": grouped(
            ((str(gt.truncation), index) for index, gt in enumerate(ground_truths)),
            ("0", "1", "2"),
        ),
        "size": grouped(
            ((gt.size, index) for index, gt in enumerate(ground_truths)),
            ("small", "medium", "large"),
        ),
        "class": grouped(
            (
                (f"{gt.class_id + 1}:{CLASS_NAMES[gt.class_id]}", index)
                for index, gt in enumerate(ground_truths)
            ),
            tuple(f"{index + 1}:{name}" for index, name in enumerate(CLASS_NAMES)),
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filename_digest(paths: Sequence[Path]) -> str:
    content = "".join(f"{path.name}\n" for path in paths).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _image_paths(data_root: Path) -> list[Path]:
    images_dir = data_root / "images"
    annotations_dir = data_root / "annotations"
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise FileNotFoundError(f"expected images/ and annotations/ under {data_root}")
    paths = sorted(
        (path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name,
    )
    if not paths:
        raise FileNotFoundError(f"no supported images found under {images_dir}")
    missing = [path.name for path in paths if not (annotations_dir / f"{path.stem}.txt").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images lack annotations; first is {missing[0]}")
    return paths


def _detections_from_result(result) -> list[Detection]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(np.int64)
    confidences = boxes.conf.detach().cpu().numpy()
    return [
        Detection(tuple(float(value) for value in box), int(class_id), float(confidence))
        for box, class_id, confidence in zip(xyxy, classes, confidences)
    ]


def predict_batched(detector, paths, **kwargs):
    """Bound list inputs explicitly; upstream treats a list as one batch."""
    batch = int(kwargs["batch"])
    for start in range(0, len(paths), batch):
        yield from detector.predict(source=[str(path) for path in paths[start:start + batch]], **kwargs)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    weights = args.weights.resolve()
    data_root = args.data_root.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint not found: {weights}")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max-images must be positive")
    paths = _image_paths(data_root)
    source_image_count = len(paths)
    if args.max_images is not None:
        paths = paths[: args.max_images]

    all_ground_truths: list[GroundTruth] = []
    image_ground_truths: dict[str, list[GroundTruth]] = {}
    offsets: dict[str, int] = {}
    excluded: Counter[str] = Counter()
    for image_path in paths:
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        annotation_path = data_root / "annotations" / f"{image_path.stem}.txt"
        parsed, dropped = parse_annotations(
            annotation_path.read_text(encoding="utf-8-sig").splitlines(),
            image_width,
            image_height,
        )
        offsets[image_path.name] = len(all_ground_truths)
        image_ground_truths[image_path.name] = parsed
        all_ground_truths.extend(parsed)
        excluded.update(dropped)

    # Import custom model definitions before checkpoint deserialization.
    (ROOT / ".runtime" / "ultralytics").mkdir(parents=True, exist_ok=True)
    from visdrone_migration import model as _custom_model  # noqa: F401
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    SETTINGS.update(
        {
            key: False
            for key in (
                "sync",
                "clearml",
                "comet",
                "dvc",
                "hub",
                "mlflow",
                "neptune",
                "raytune",
                "tensorboard",
                "wandb",
            )
        }
    )
    detector = YOLO(str(weights))
    results = predict_batched(
        detector, paths,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
        max_det=args.max_det,
        stream=True,
        verbose=False,
    )
    matched_global: set[int] = set()
    seen: set[str] = set()
    for result in results:
        name = Path(result.path).name
        if name not in image_ground_truths:
            raise RuntimeError(f"prediction returned unexpected image: {result.path}")
        if name in seen:
            raise RuntimeError(f"prediction returned image more than once: {name}")
        seen.add(name)
        local_matches = match_ground_truths(
            image_ground_truths[name], _detections_from_result(result), args.match_iou
        )
        matched_global.update(offsets[name] + index for index in local_matches)
    missing_results = sorted(set(image_ground_truths) - seen)
    if missing_results:
        raise RuntimeError(f"no prediction returned for {len(missing_results)} images")

    payload: dict[str, object] = {
        "diagnostic": "class-aware ground-truth recall under natural VisDrone occlusion",
        "protocol": (
            "Predictions are greedily matched by descending confidence against all eligible "
            "ground truth in each image. Matching is class-aware, one-to-one, and performed "
            "before stratification. This is not AP and does not implement official VisDrone "
            "ignored-region handling."
        ),
        "metrics": aggregate_recall(all_ground_truths, matched_global),
        "settings": {
            "data_root": str(data_root),
            "weights": str(weights),
            "imgsz": args.imgsz,
            "confidence_threshold": args.conf,
            "nms_iou_threshold": args.iou,
            "matching_iou_threshold": args.match_iou,
            "device": str(args.device),
            "batch": args.batch,
            "max_det": args.max_det,
            "max_images": args.max_images,
            "source_image_count": source_image_count,
            "evaluated_image_count": len(paths),
            "is_subset_smoke_run": args.max_images is not None and len(paths) < source_image_count,
            "eligible_ground_truth": "score > 0, source class 1..10, valid after clipping",
            "small_object_definition": "clipped raw-image area < 32^2 pixels (COCO convention)",
            "size_bins": "small < 32^2; medium 32^2 to < 96^2; large >= 96^2 raw pixels",
        },
        "excluded_ground_truth_counts": dict(sorted(excluded.items())),
        "provenance": {
            "weights_sha256": _sha256_file(weights),
            "filenames_sha256": _filename_digest(paths),
            "filenames_digest_definition": "SHA-256 of sorted evaluated basenames, each followed by newline",
        },
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "data" / "raw" / "VisDrone2019-DET-val"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "occlusion" / "occlusion.json")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5, help="prediction NMS IoU threshold")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--max-images", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.imgsz < 1 or args.batch < 1 or args.max_det < 1:
        raise ValueError("--imgsz, --batch, and --max-det must be positive")
    if not 0 <= args.conf <= 1 or not 0 < args.iou <= 1 or not 0 < args.match_iou <= 1:
        raise ValueError("thresholds must lie in their probability/IoU ranges")
    payload = evaluate(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "images": payload["settings"]["evaluated_image_count"],
                "overall": payload["metrics"]["overall"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
