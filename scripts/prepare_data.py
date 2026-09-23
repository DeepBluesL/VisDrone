#!/usr/bin/env python3
"""Convert VisDrone2019-DET train/val annotations to Ultralytics YOLO format.

The source tree is read-only from this script's point of view. Images are hard
linked into ``data/yolo`` when possible (with a copy fallback), and annotations
are converted into new label files. The official train and validation splits
are never mixed or re-split.

VisDrone evaluation has special handling for ignored regions. Metrics computed
from these converted YOLO labels are therefore *YOLO-converted evaluation*, not
the official VisDrone DET protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image


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
SOURCE_SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
EVALUATION_NOTE = (
    "Evaluation using these labels is YOLO-converted evaluation. The official "
    "VisDrone DET evaluator applies ignored-region matching rules that are not "
    "represented by standard YOLO labels."
)


@dataclass
class ConversionStats:
    """Counters for one source split."""

    total_rows: int = 0
    kept_rows: int = 0
    clipped_boxes: int = 0
    empty_label_files: int = 0
    missing_annotation_files: int = 0
    link_methods: Counter[str] = field(default_factory=Counter)
    drop_reasons: Counter[str] = field(default_factory=Counter)
    ignored_flags: Counter[str] = field(default_factory=Counter)
    raw_class_counts: Counter[int] = field(default_factory=Counter)
    kept_class_counts: Counter[int] = field(default_factory=Counter)
    raw_occlusion_counts: Counter[int] = field(default_factory=Counter)
    kept_occlusion_counts: Counter[int] = field(default_factory=Counter)
    raw_truncation_counts: Counter[int] = field(default_factory=Counter)
    kept_truncation_counts: Counter[int] = field(default_factory=Counter)

    def drop(self, reason: str) -> None:
        self.drop_reasons[reason] += 1

    def as_dict(self) -> dict[str, object]:
        dropped = sum(self.drop_reasons.values())
        if self.total_rows != self.kept_rows + dropped:
            raise AssertionError("annotation accounting invariant failed")

        def ordered(counter: Counter[object]) -> dict[str, int]:
            return {
                str(key): counter[key]
                for key in sorted(counter, key=lambda value: str(value))
            }

        return {
            "annotation_rows": {
                "total": self.total_rows,
                "kept": self.kept_rows,
                "dropped": dropped,
            },
            "drop_reasons": ordered(self.drop_reasons),
            # These flags intentionally overlap. For example, an ignored-region
            # row commonly has both score=0 and class=0.
            "ignored_flags_nonexclusive": ordered(self.ignored_flags),
            "raw_class_counts": ordered(self.raw_class_counts),
            "kept_class_counts": ordered(self.kept_class_counts),
            "raw_occlusion_counts": ordered(self.raw_occlusion_counts),
            "kept_occlusion_counts": ordered(self.kept_occlusion_counts),
            "raw_truncation_counts": ordered(self.raw_truncation_counts),
            "kept_truncation_counts": ordered(self.kept_truncation_counts),
            "clipped_boxes": self.clipped_boxes,
            "empty_label_files": self.empty_label_files,
            "missing_annotation_files": self.missing_annotation_files,
            "image_materialization": ordered(self.link_methods),
        }


def _parse_integer(value: str) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(value)
    return int(number)


def convert_annotation_line(
    line: str,
    image_width: int,
    image_height: int,
    stats: ConversionStats,
) -> str | None:
    """Convert one VisDrone row, updating ``stats`` and returning a YOLO row.

    Drop reasons are exclusive, while ignored flags are nonexclusive. Coordinates
    are clipped to image bounds before normalization. Source class IDs 1..10 are
    mapped to YOLO IDs 0..9.
    """
    stats.total_rows += 1
    if image_width <= 0 or image_height <= 0:
        stats.drop("invalid_image_size")
        return None

    fields = [field.strip() for field in line.strip().split(",")]
    while fields and fields[-1] == "":
        fields.pop()
    if len(fields) != 8:
        stats.drop("malformed_field_count")
        return None

    try:
        x, y, width, height, score = (float(value) for value in fields[:5])
        class_id = _parse_integer(fields[5])
        truncation = _parse_integer(fields[6])
        occlusion = _parse_integer(fields[7])
    except (ValueError, OverflowError):
        stats.drop("non_numeric_or_non_integer_metadata")
        return None

    numeric_values = (x, y, width, height, score)
    if not all(math.isfinite(value) for value in numeric_values):
        stats.drop("non_finite_value")
        return None

    stats.raw_class_counts[class_id] += 1
    stats.raw_occlusion_counts[occlusion] += 1
    stats.raw_truncation_counts[truncation] += 1

    if score == 0:
        stats.ignored_flags["score_zero"] += 1
    if class_id == 0:
        stats.ignored_flags["class_0_ignored_region"] += 1
    if class_id == 11:
        stats.ignored_flags["class_11_others"] += 1

    # A fixed priority gives every dropped row exactly one primary reason.
    if score == 0:
        stats.drop("score_zero")
        return None
    if score < 0:
        stats.drop("negative_score")
        return None
    if class_id == 0:
        stats.drop("class_0_ignored_region")
        return None
    if class_id == 11:
        stats.drop("class_11_others")
        return None
    if not 1 <= class_id <= 10:
        stats.drop("unsupported_class")
        return None
    if width <= 0 or height <= 0:
        stats.drop("non_positive_box_size")
        return None

    x1 = max(0.0, min(float(image_width), x))
    y1 = max(0.0, min(float(image_height), y))
    x2 = max(0.0, min(float(image_width), x + width))
    y2 = max(0.0, min(float(image_height), y + height))
    if x2 <= x1 or y2 <= y1:
        stats.drop("box_outside_image")
        return None

    if (x1, y1, x2, y2) != (x, y, x + width, y + height):
        stats.clipped_boxes += 1

    clipped_width = x2 - x1
    clipped_height = y2 - y1
    center_x = (x1 + x2) / 2.0 / image_width
    center_y = (y1 + y2) / 2.0 / image_height
    normalized_width = clipped_width / image_width
    normalized_height = clipped_height / image_height

    stats.kept_rows += 1
    stats.kept_class_counts[class_id] += 1
    stats.kept_occlusion_counts[occlusion] += 1
    stats.kept_truncation_counts[truncation] += 1
    return (
        f"{class_id - 1} {center_x:.8f} {center_y:.8f} "
        f"{normalized_width:.8f} {normalized_height:.8f}"
    )


def select_images(
    image_paths: Sequence[Path], limit: int, seed: int, split: str
) -> list[Path]:
    """Select a stable subset without changing split membership."""
    ordered = sorted(image_paths, key=lambda path: path.name)
    if limit < 0:
        raise ValueError(f"{split} limit must be non-negative")
    if limit == 0:
        return ordered
    if limit > len(ordered):
        raise ValueError(
            f"{split} limit {limit} exceeds the {len(ordered)} available images"
        )
    # Different deterministic streams prevent one split's membership from
    # depending on the other split's size or selection.
    split_offset = 0 if split == "train" else 1_000_003
    rng = random.Random(seed + split_offset)
    return sorted(rng.sample(ordered, limit), key=lambda path: path.name)


def _sha256_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _clear_files(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            raise RuntimeError(f"unexpected directory in generated output: {child}")


def _materialize_image(source: Path, destination: Path, mode: str) -> str:
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def _image_paths(images_dir: Path) -> list[Path]:
    return [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def prepare_split(
    split: str,
    source_dir: Path,
    output_root: Path,
    limit: int,
    seed: int,
    link_mode: str,
) -> dict[str, object]:
    images_dir = source_dir / "images"
    annotations_dir = source_dir / "annotations"
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise FileNotFoundError(
            f"expected images/ and annotations/ under source split: {source_dir}"
        )

    all_images = _image_paths(images_dir)
    if not all_images:
        raise FileNotFoundError(f"no supported images found in {images_dir}")
    selected = select_images(all_images, limit, seed, split)

    # Validate every pair before replacing generated output.
    missing = [path.stem for path in selected if not (annotations_dir / f"{path.stem}.txt").is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} selected {split} images lack annotations (first: {preview})"
        )

    output_images = output_root / "images" / split
    output_labels = output_root / "labels" / split
    _clear_files(output_images)
    _clear_files(output_labels)

    stats = ConversionStats()
    for image_path in selected:
        destination = output_images / image_path.name
        method = _materialize_image(image_path, destination, link_mode)
        stats.link_methods[method] += 1

        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
        except Exception as exc:
            raise RuntimeError(f"failed to read image dimensions: {image_path}") from exc

        annotation_path = annotations_dir / f"{image_path.stem}.txt"
        converted: list[str] = []
        for line in annotation_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            converted_line = convert_annotation_line(
                line, image_width, image_height, stats
            )
            if converted_line is not None:
                converted.append(converted_line)
        if not converted:
            stats.empty_label_files += 1
        label_text = "\n".join(converted)
        if label_text:
            label_text += "\n"
        _atomic_text(output_labels / f"{image_path.stem}.txt", label_text)

    # YOLO resolves './' entries relative to the list file. Lists live in the
    # output root so these remain portable across machines.
    list_lines = [f"./images/{split}/{path.name}" for path in selected]
    list_path = output_root / f"{split}.txt"
    _atomic_text(list_path, "".join(f"{line}\n" for line in list_lines))

    selected_filenames = [path.name for path in selected]
    return {
        "source_split": source_dir.name,
        "source_image_count": len(all_images),
        "selected_image_count": len(selected),
        "selection_limit": limit,
        "selected_filenames": selected_filenames,
        "selected_filenames_sha256": _sha256_lines(selected_filenames),
        "yolo_list_file": list_path.name,
        "yolo_list_sha256": _sha256_lines(list_lines),
        "stats": stats.as_dict(),
    }


def write_dataset_yaml(path: Path, output_root: Path) -> None:
    absolute_root = output_root.resolve().as_posix()
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    text = (
        f"path: {json.dumps(absolute_root)}\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "names:\n"
        f"{names}\n"
    )
    _atomic_text(path, text)


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=project_root / "data" / "raw")
    parser.add_argument(
        "--output-root", type=Path, default=project_root / "data" / "yolo"
    )
    parser.add_argument(
        "--dataset-yaml", type=Path, default=project_root / "data" / "dataset.yaml"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "data" / "preparation_manifest.json",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="fixed-size deterministic train subset; 0 uses the full official split",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=0,
        help="fixed-size deterministic val subset; 0 uses all 548 official val images",
    )
    parser.add_argument("--seed", type=int, default=179)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="hardlink uses a copy fallback when the filesystem rejects linking",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()

    split_results: dict[str, object] = {}
    for split, source_name in SOURCE_SPLITS.items():
        limit = args.train_limit if split == "train" else args.val_limit
        print(f"Preparing {split} from {raw_root / source_name} ...", flush=True)
        split_results[split] = prepare_split(
            split=split,
            source_dir=raw_root / source_name,
            output_root=output_root,
            limit=limit,
            seed=args.seed,
            link_mode=args.link_mode,
        )

    write_dataset_yaml(args.dataset_yaml.resolve(), output_root)
    manifest = {
        "schema_version": 1,
        "dataset": "VisDrone2019-DET",
        "class_names": list(CLASS_NAMES),
        "seed": args.seed,
        "raw_root": raw_root.as_posix(),
        "output_root": output_root.as_posix(),
        "source_splits_preserved": True,
        "evaluation_note": EVALUATION_NOTE,
        "splits": split_results,
    }
    _atomic_text(
        args.manifest.resolve(),
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )

    for split, result in split_results.items():
        stats = result["stats"]
        rows = stats["annotation_rows"]
        print(
            f"{split}: {result['selected_image_count']} images, "
            f"{rows['kept']} boxes kept, {rows['dropped']} rows dropped"
        )
    print(f"Dataset YAML: {args.dataset_yaml.resolve()}")
    print(f"Preparation manifest: {args.manifest.resolve()}")
    print(f"NOTICE: {EVALUATION_NOTE}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
