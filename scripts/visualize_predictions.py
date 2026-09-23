#!/usr/bin/env python3
"""Render deterministic ground-truth and multi-checkpoint VisDrone predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from scripts.evaluate_occlusion import CLASS_NAMES, GroundTruth, parse_annotations


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_COLORS = tuple(plt.get_cmap("tab10")(index) for index in range(10))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_images(
    data_root: Path,
    filenames: Sequence[str] | None,
    seed: int,
    count: int = 3,
) -> list[Path]:
    """Choose source images independently of model outputs."""
    images_dir = data_root / "images"
    annotations_dir = data_root / "annotations"
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise FileNotFoundError(f"expected images/ and annotations/ under {data_root}")
    available = sorted(
        (
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name,
    )
    by_name = {path.name: path for path in available}
    if filenames is not None:
        if len(filenames) != count:
            raise ValueError(f"--filenames must provide exactly {count} image basenames")
        if len(set(filenames)) != len(filenames):
            raise ValueError("--filenames must not contain duplicates")
        invalid = [name for name in filenames if Path(name).name != name or name not in by_name]
        if invalid:
            raise FileNotFoundError(f"unknown or invalid image filename: {invalid[0]}")
        selected = [by_name[name] for name in filenames]
    else:
        if len(available) < count:
            raise ValueError(f"need at least {count} validation images, found {len(available)}")
        selected = random.Random(seed).sample(available, count)
    missing = [
        path.name
        for path in selected
        if not (annotations_dir / f"{path.stem}.txt").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"selected image lacks annotation file: {missing[0]}")
    return selected


def load_ground_truth(data_root: Path, image_path: Path) -> list[GroundTruth]:
    with Image.open(image_path) as image:
        width, height = image.size
    annotation = data_root / "annotations" / f"{image_path.stem}.txt"
    ground_truths, _ = parse_annotations(
        annotation.read_text(encoding="utf-8-sig").splitlines(), width, height
    )
    return ground_truths


def _prediction_records(result) -> list[dict[str, object]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    coordinates = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(np.int64)
    confidences = boxes.conf.detach().cpu().numpy()
    records = [
        {
            "box": tuple(float(value) for value in box),
            "class_id": int(class_id),
            "confidence": float(confidence),
        }
        for box, class_id, confidence in zip(coordinates, classes, confidences)
    ]
    return sorted(records, key=lambda item: item["confidence"], reverse=True)


def predict_checkpoints(args, selected: Sequence[Path]) -> list[dict[str, list[dict[str, object]]]]:
    """Return one filename-keyed prediction mapping for each checkpoint."""
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
    sources = [str(path) for path in selected]
    all_predictions: list[dict[str, list[dict[str, object]]]] = []
    for weights in args.weights:
        detector = YOLO(str(weights))
        results = detector.predict(
            source=sources,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            batch=len(sources),
            max_det=args.max_det,
            stream=False,
            verbose=False,
        )
        predictions = {
            Path(result.path).name: _prediction_records(result) for result in results
        }
        missing = [path.name for path in selected if path.name not in predictions]
        if missing:
            raise RuntimeError(f"checkpoint returned no result for {missing[0]}")
        all_predictions.append(predictions)
        del detector
    return all_predictions


def _draw_boxes(ax, records: Sequence[dict[str, object]], linewidth: float) -> None:
    for record in records:
        x1, y1, x2, y2 = record["box"]
        class_id = int(record["class_id"])
        color = CLASS_COLORS[class_id] if 0 <= class_id < len(CLASS_COLORS) else "white"
        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                alpha=0.9,
            )
        )


def render_figure(
    args,
    selected: Sequence[Path],
    ground_truth: dict[str, list[GroundTruth]],
    predictions: Sequence[dict[str, list[dict[str, object]]]],
) -> list[dict[str, object]]:
    columns = len(args.weights) + 1
    rows = len(selected)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.3 * columns, 3.8 * rows + 1.3),
        squeeze=False,
        constrained_layout=False,
    )
    figure_records: list[dict[str, object]] = []
    for row, image_path in enumerate(selected):
        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert("RGB"))
        image_ground_truth = ground_truth[image_path.name]
        gt_records = [
            {"box": item.box, "class_id": item.class_id} for item in image_ground_truth
        ]
        ax = axes[row][0]
        ax.imshow(pixels)
        _draw_boxes(ax, gt_records, linewidth=1.25)
        ax.set_title(
            f"{image_path.name}\nGround truth: {len(gt_records)} eligible boxes",
            fontsize=10,
        )
        ax.axis("off")

        per_model = []
        for column, (label, model_predictions) in enumerate(
            zip(args.labels, predictions), start=1
        ):
            records = model_predictions[image_path.name]
            shown = records[: args.max_draw]
            ax = axes[row][column]
            ax.imshow(pixels)
            _draw_boxes(ax, shown, linewidth=1.1)
            cap_note = (
                f"; top {len(shown)} shown" if len(shown) < len(records) else ""
            )
            ax.set_title(
                f"{label}\nPredictions: {len(records)}{cap_note}",
                fontsize=10,
            )
            ax.axis("off")
            per_model.append(
                {
                    "label": label,
                    "prediction_count": len(records),
                    "rendered_count": len(shown),
                    "render_limit_applied": len(shown) < len(records),
                }
            )
        figure_records.append(
            {
                "filename": image_path.name,
                "eligible_ground_truth_count": len(gt_records),
                "models": per_model,
            }
        )

    legend = [
        Line2D([0], [0], color=CLASS_COLORS[index], lw=3, label=f"{index + 1}: {name}")
        for index, name in enumerate(CLASS_NAMES)
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.035),
    )
    fig.text(
        0.5,
        0.006,
        "Data: VisDrone2019-DET, VisDrone team at Tianjin University "
        "(github.com/VisDrone/VisDrone-Dataset). Detector scaffold: Ultralytics YOLO. "
        "Qualitative diagnostic, not official VisDrone evaluation.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.suptitle(
        f"VisDrone qualitative detections · confidence ≥ {args.conf:g} · NMS IoU {args.iou:g}",
        fontsize=14,
        y=0.995,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.11, wspace=0.025, hspace=0.12)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp" + output.suffix)
    fig.savefig(
        temporary,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        format=output.suffix.lstrip(".") or "png",
    )
    plt.close(fig)
    temporary.replace(output)
    return figure_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "assets" / "pilot" / "predictions.png"
    )
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "data" / "raw" / "VisDrone2019-DET-val"
    )
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--max-draw", type=int, default=80)
    parser.add_argument("--seed", type=int, default=179)
    parser.add_argument("--filenames", nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.weights) != len(args.labels):
        raise ValueError("--weights and --labels must contain the same number of values")
    if not args.weights:
        raise ValueError("at least one checkpoint is required")
    args.weights = [path.resolve() for path in args.weights]
    missing_weights = [path for path in args.weights if not path.is_file()]
    if missing_weights:
        raise FileNotFoundError(f"checkpoint not found: {missing_weights[0]}")
    if args.imgsz < 1 or args.max_det < 1 or args.max_draw < 1:
        raise ValueError("--imgsz, --max-det, and --max-draw must be positive")
    if not 0 <= args.conf <= 1 or not 0 < args.iou <= 1:
        raise ValueError("--conf must be in [0,1] and --iou in (0,1]")
    args.data_root = args.data_root.resolve()
    selected = select_images(args.data_root, args.filenames, args.seed)
    ground_truth = {
        path.name: load_ground_truth(args.data_root, path) for path in selected
    }
    predictions = predict_checkpoints(args, selected)
    image_records = render_figure(args, selected, ground_truth, predictions)

    sidecar = {
        "figure": str(args.output.resolve()),
        "data_root": str(args.data_root),
        "selection": {
            "method": "explicit filenames" if args.filenames is not None else "local random sample from sorted filenames",
            "seed": None if args.filenames is not None else args.seed,
            "filenames": [path.name for path in selected],
        },
        "inference": {
            "imgsz": args.imgsz,
            "confidence_threshold": args.conf,
            "nms_iou_threshold": args.iou,
            "max_det": args.max_det,
            "max_draw": args.max_draw,
            "device": str(args.device),
        },
        "checkpoints": [
            {
                "label": label,
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for label, path in zip(args.labels, args.weights)
        ],
        "images": image_records,
        "ground_truth_protocol": "score > 0, source class 1..10, valid clipped boxes",
        "credits": (
            "VisDrone2019-DET validation images and annotations, VisDrone team at "
            "Tianjin University (https://github.com/VisDrone/VisDrone-Dataset); "
            "detector scaffold from Ultralytics YOLO."
        ),
    }
    sidecar_path = args.output.resolve().with_suffix(".json")
    temporary = sidecar_path.with_name(sidecar_path.name + ".tmp")
    temporary.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    temporary.replace(sidecar_path)
    print(
        json.dumps(
            {
                "figure": str(args.output.resolve()),
                "sidecar": str(sidecar_path),
                "filenames": [path.name for path in selected],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
