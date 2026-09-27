#!/usr/bin/env python3
"""Plan or explicitly execute train-only SimD normalizer calibration."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))


def letterbox_geometry(height: int, width: int, imgsz: int) -> tuple[float, int, int]:
    """Return Ultralytics LetterBox(auto=False, center=True) scale and left/top pad."""
    if min(height, width, imgsz) < 1:
        raise ValueError("image dimensions and imgsz must be positive")
    ratio = min(imgsz / height, imgsz / width)
    resized_width, resized_height = round(width * ratio), round(height * ratio)
    half_width = (imgsz - resized_width) / 2
    half_height = (imgsz - resized_height) / 2
    return ratio, int(round(half_width - 0.1)), int(round(half_height - 0.1))


def transform_yolo_ground_truth(rows: list[list[float]], *, height: int, width: int,
                                imgsz: int) -> list[list[float]]:
    """Convert normalized YOLO xywh rows to letterboxed pixel xyxy boxes."""
    ratio, left, top = letterbox_geometry(height, width, imgsz)
    boxes = []
    for row in rows:
        if len(row) != 5 or any(not math.isfinite(value) for value in row):
            raise ValueError("each label row must contain five finite values")
        _, x, y, w, h = row
        if w <= 0 or h <= 0:
            raise ValueError("ground-truth width and height must be positive")
        x1, y1 = (x - w / 2) * width, (y - h / 2) * height
        x2, y2 = (x + w / 2) * width, (y + h / 2) * height
        boxes.append([x1 * ratio + left, y1 * ratio + top,
                      x2 * ratio + left, y2 * ratio + top])
    return boxes


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _load_rows(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append([float(value) for value in line.split()])
            except ValueError as exc:
                raise ValueError(f"invalid label at {path}:{line_number}") from exc
    return rows


def _train_paths(data: Path) -> tuple[list[Path], Path]:
    from ultralytics.utils import YAML

    config = YAML.load(data)
    if "train" not in config or "val" not in config:
        raise ValueError("dataset YAML must explicitly define train and val splits")
    dataset_root = Path(config.get("path", data.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data.parent / dataset_root).resolve()
    train_spec = Path(config["train"])
    train_list = train_spec if train_spec.is_absolute() else dataset_root / train_spec
    if train_list.suffix.lower() != ".txt" or not train_list.is_file():
        raise ValueError("train split must be an existing image-list text file")
    images = []
    for raw in train_list.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        path = Path(raw.strip())
        path = path if path.is_absolute() else (train_list.parent / path).resolve()
        normalized = path.as_posix().lower()
        if "/images/train/" not in normalized or "/val/" in normalized or "/test" in normalized:
            raise ValueError(f"non-training image in calibration input: {path}")
        images.append(path)
    if len(images) != len(set(images)):
        raise ValueError("train image list contains duplicates")
    return images, dataset_root


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    matches = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not matches:
        raise ValueError(f"image path has no images directory: {image}")
    parts[matches[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def _train_fingerprint(images: list[Path]) -> dict[str, Any]:
    """Hash only the selected training image/label pairs; never inspect val/test."""
    from visdrone_migration.evidence import file_sha256

    digest = hashlib.sha256()
    for image in images:
        label = _label_path(image)
        digest.update(image.name.encode())
        digest.update(file_sha256(image).encode())
        digest.update(label.read_bytes() if label.exists() else b"<background>")
    return {"images": len(images), "image_and_label_sha256": digest.hexdigest()}


def _execute(args: argparse.Namespace) -> None:
    pause_path = ROOT / ".runtime" / "training_paused_by_user.json"
    if pause_path.exists():
        pause = json.loads(pause_path.read_text(encoding="utf-8"))
        if pause.get("automatic_restart_allowed") is False and not args.resume_paused:
            raise SystemExit("Execution is paused by user request; --resume-paused requires an explicit user request.")
    if args.max_images is not None and not args.debug:
        raise SystemExit("--max-images is debug-only and requires --debug")
    if args.debug and args.max_images is None:
        raise SystemExit("--debug requires --max-images so subset calibration is explicit")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {args.output}; use --overwrite to replace")

    import cv2
    import numpy as np
    import torch
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.ops import xywh2xyxy
    from visdrone_migration.evidence import file_sha256
    from visdrone_migration.realtime_models import build_realtime_model, transfer_coco_weights
    from visdrone_migration.simd import estimate_simd_normalizers

    images, _ = _train_paths(args.data.resolve())
    if args.max_images is None and len(images) != 6471:
        raise RuntimeError(f"formal calibration requires all 6471 train images, found {len(images)}")
    selected = images[:args.max_images] if args.max_images is not None else images
    if not selected:
        raise ValueError("calibration image set is empty")
    if not args.pretrained.is_file():
        raise FileNotFoundError(f"local pretrained checkpoint missing: {args.pretrained}")
    input_identity = {
        "train_image_and_label_fingerprint": _train_fingerprint(selected),
        "pretrained_sha256": file_sha256(args.pretrained),
        "code_sha256": {
            "scripts/calibrate_simd.py": file_sha256(Path(__file__).resolve()),
            "visdrone_migration/realtime_models.py": file_sha256(ROOT / "visdrone_migration" / "realtime_models.py"),
            "visdrone_migration/simd.py": file_sha256(ROOT / "visdrone_migration" / "simd.py"),
        },
    }
    device = torch.device(args.device)
    source = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    model = build_realtime_model(args.architecture, nc=10, seed=args.seed, verbose=False)
    transfer = transfer_coco_weights(model, source, architecture=args.architecture)
    model.to(device).eval()
    transform = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, scale_fill=False,
                          scaleup=True, center=True, stride=32)
    counts = {"images": 0, "ground_truth": 0, "anchors": 0, "pairs": 0}

    def pairs() -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        with torch.inference_mode():
            for image_path in selected:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"cannot decode training image: {image_path}")
                height, width = image.shape[:2]
                gt = transform_yolo_ground_truth(_load_rows(_label_path(image_path)),
                                                  height=height, width=width, imgsz=args.imgsz)
                resized = transform(image=image)
                tensor = torch.from_numpy(np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1)))
                tensor = tensor.unsqueeze(0).to(device=device, dtype=torch.float32).div_(255)
                prediction = model(tensor)
                decoded = prediction[0] if isinstance(prediction, tuple) else prediction
                anchors = xywh2xyxy(decoded[0, :4].transpose(0, 1)).detach().to("cpu", torch.float32)
                ground_truth = torch.tensor(gt, dtype=torch.float32).reshape(-1, 4)
                counts["images"] += 1
                counts["ground_truth"] += len(ground_truth)
                counts["anchors"] += len(anchors)
                counts["pairs"] += len(ground_truth) * len(anchors)
                yield ground_truth, anchors

    normalizers = estimate_simd_normalizers(pairs(), anchor_chunk_size=args.anchor_chunk_size)
    final_identity = {
        "train_image_and_label_fingerprint": _train_fingerprint(selected),
        "pretrained_sha256": file_sha256(args.pretrained),
        "code_sha256": {
            "scripts/calibrate_simd.py": file_sha256(Path(__file__).resolve()),
            "visdrone_migration/realtime_models.py": file_sha256(ROOT / "visdrone_migration" / "realtime_models.py"),
            "visdrone_migration/simd.py": file_sha256(ROOT / "visdrone_migration" / "simd.py"),
        },
    }
    if final_identity != input_identity:
        changed = [key for key in input_identity if input_identity[key] != final_identity.get(key)]
        raise RuntimeError(f"calibration inputs or code changed during execution: {changed}")
    record = {
        "schema_version": 1,
        "method": "frozen-initial-predictions: YOLO adaptation, not original fixed-anchor estimator",
        "status": "completed",
        "split": "train",
        "subset": args.max_images is not None,
        "max_images": args.max_images,
        "data": str(args.data.resolve()),
        "train_image_and_label_fingerprint": input_identity["train_image_and_label_fingerprint"],
        "pretrained": str(args.pretrained.resolve()),
        "pretrained_sha256": input_identity["pretrained_sha256"],
        "code_sha256": input_identity["code_sha256"],
        "environment": {"torch": torch.__version__,
                        "ultralytics": importlib.metadata.version("ultralytics")},
        "architecture": args.architecture,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "nc": 10,
        "device": str(device),
        "image_count": counts["images"],
        "ground_truth_count": counts["ground_truth"],
        "anchor_count_per_image": counts["anchors"] // counts["images"],
        "pair_count": normalizers.pair_count,
        "normalizers": {"m": normalizers.m, "n": normalizers.n},
        "prediction_boxes": "all decoded pre-NMS detector anchors, xywh converted to xyxy",
        "semantic_transfer": {"loaded_parameter_numel": transfer.loaded_parameter_numel,
                              "target_parameter_numel": transfer.target_parameter_numel,
                              "parameter_numel_coverage": transfer.parameter_numel_coverage},
        "note": "Only the non-augmented training split was used; validation and test images were prohibited.",
    }
    if (normalizers.pair_count <= 0 or normalizers.pair_count != counts["pairs"]
            or counts["images"] != len(selected)):
        raise RuntimeError("incomplete calibration stream")
    _atomic_json(args.output.resolve(), record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "full" / "dataset.yaml")
    parser.add_argument("--architecture", choices=("standard", "p2"), default="standard")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--seed", type=int, default=179)
    parser.add_argument("--pretrained", type=Path, default=ROOT / "pretrained" / "yolov8n.pt")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime" / "simd_normalizers.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--anchor-chunk-size", type=int, default=4096)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-paused", action="store_true",
                        help="allow execution only after the user explicitly requests work to resume")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.imgsz < 32 or args.imgsz % 32 or args.anchor_chunk_size < 1:
        raise SystemExit("imgsz must be a positive multiple of 32 and anchor chunk size must be positive")
    if args.max_images is not None and args.max_images < 1:
        raise SystemExit("--max-images must be positive")
    plan = {"action": "execute" if args.execute else "plan-only", "data": str(args.data.resolve()),
            "split": "train only", "architecture": args.architecture, "imgsz": args.imgsz,
            "seed": args.seed, "nc": 10, "pretrained": str(args.pretrained.resolve()),
            "output": str(args.output.resolve()), "device": args.device,
            "max_images": args.max_images, "subset": args.max_images is not None}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    _execute(args)
    print(json.dumps({**plan, "status": "completed"}))


if __name__ == "__main__":
    main()
