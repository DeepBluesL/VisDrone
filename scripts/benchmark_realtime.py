#!/usr/bin/env python3
"""Benchmark completed real-time suite checkpoints on decoded validation images."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import tempfile
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from ultralytics.utils import SETTINGS


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def latency_statistics(milliseconds: list[float]) -> dict[str, float]:
    """Summarize positive, finite per-image latency samples."""
    if not milliseconds or any(not math.isfinite(x) or x <= 0 for x in milliseconds):
        raise ValueError("latency samples must be a non-empty list of positive finite values")
    ordered = np.asarray(milliseconds, dtype=np.float64)
    mean = float(np.mean(ordered))
    return {"samples": len(milliseconds), "mean_ms": mean,
            "p50_ms": float(np.percentile(ordered, 50)),
            "p95_ms": float(np.percentile(ordered, 95)),
            "fps_from_mean": 1000.0 / mean}


def _valid_latency_summary(value: Any, samples: int) -> bool:
    if not isinstance(value, dict) or value.get("samples") != samples:
        return False
    numbers = [value.get(key) for key in ("mean_ms", "p50_ms", "p95_ms", "fps_from_mean")]
    if any(not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0
           for number in numbers):
        return False
    return value["p95_ms"] >= value["p50_ms"]


def _protocol_arms(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    arms = protocol.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("protocol must define a non-empty arms list")
    output = []
    for arm in arms:
        if not isinstance(arm, dict):
            raise ValueError("each protocol arm must be an object")
        arm_id = arm.get("id") or arm.get("run_id") or arm.get("name")
        if not isinstance(arm_id, str) or Path(arm_id).name != arm_id:
            raise ValueError(f"invalid arm id {arm_id!r}")
        if not isinstance(arm.get("imgsz"), int) or not isinstance(arm.get("seed"), int):
            raise ValueError(f"arm {arm_id} requires integer imgsz and seed")
        output.append({**arm, "id": arm_id})
    return output


def _find_images(root: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(path for path in root.rglob("*") if path.suffix.lower() in extensions)
    if not images:
        raise ValueError(f"no validation images found under {root}")
    return images


def _time_cuda(forward: Callable[[], Any], warmup: int, samples: int) -> list[float]:
    for _ in range(warmup):
        forward()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        forward()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def _time_wall(forward: Callable[[int], Any], warmup: int, samples: int) -> list[float]:
    for index in range(warmup):
        forward(index)
    torch.cuda.synchronize()
    values = []
    for index in range(samples):
        started = time.perf_counter_ns()
        forward(index)
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return values


def benchmark_arm(
    arm: dict[str, Any], checkpoint: Path, decoded_images: list[np.ndarray], *,
    warmup: int, samples: int, device: str, conf: float, iou: float, max_det: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("this benchmark requires CUDA and an explicit CUDA device")
    detector = YOLO(str(checkpoint))  # local path only; never requests a model by name
    # Fuse Conv+BN in FP32 once, then use this same fused FP16 graph for both
    # direct model timing and predictor timing.
    detector.model.fuse(verbose=False)
    detector.model.to(device).half().eval()
    imgsz = int(arm["imgsz"])
    tensor = torch.rand((1, 3, imgsz, imgsz), device=device, dtype=torch.float16)
    with torch.inference_mode():
        pure = _time_cuda(lambda: detector.model(tensor), warmup, samples)

        def predict(index: int) -> Any:
            # Inputs were decoded before timing. predict includes resize/pad, normalization,
            # host-to-device transfer, FP16 model execution, decode and NMS.
            return detector.predict(source=decoded_images[index % len(decoded_images)], imgsz=imgsz,
                                    batch=1, device=device, half=True, rect=False, conf=conf,
                                    iou=iou, max_det=max_det, verbose=False, save=False)

        end_to_end = _time_wall(predict, warmup, samples)
    return {
        "schema_version": 1,
        "status": "completed",
        "arm_id": arm["id"],
        "architecture": arm.get("architecture"),
        "imgsz": imgsz,
        "seed": arm["seed"],
        "checkpoint_sha256": _sha256(checkpoint),
        "precision": "FP16",
        "fused": True,
        "batch": 1,
        "warmup_iterations": warmup,
        "timed_iterations": samples,
        "settings": {"conf": conf, "iou": iou, "max_det": max_det, "rect": False},
        "pure_model_cuda_event": latency_statistics(pure),
        "synchronized_wall_clock": latency_statistics(end_to_end),
        "timing_boundaries": {
            "pure_model_cuda_event": "preallocated 1x3ximgszximgsz CUDA FP16 tensor through native eval-model forward, including Detect box decode; excludes preprocessing, H2D and NMS",
            "synchronized_wall_clock": "decoded BGR numpy image through square resize/pad, normalization, H2D, FP16 model, box decode and NMS; excludes JPEG decode and disk I/O",
        },
        "hardware": {"gpu": torch.cuda.get_device_name(torch.device(device)),
                     "torch": torch.__version__, "cuda": torch.version.cuda,
                     "python": platform.python_version(), "platform": platform.platform()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="realtime_stage1")
    parser.add_argument("--raw-val-root", type=Path,
                        default=ROOT / "data" / "raw" / "VisDrone2019-DET-val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arm", help="benchmark only this protocol arm")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--iou", type=float, default=.7)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    SETTINGS.update({key: False for key in ("sync", "clearml", "comet", "dvc", "hub", "mlflow",
                                           "neptune", "raytune", "tensorboard", "wandb")})
    if Path(args.suite).name != args.suite or args.warmup < 1 or args.samples < 1:
        raise SystemExit("invalid suite name or iteration count")
    suite_dir = ROOT / "results" / args.suite
    protocol = _read_json(suite_dir / "protocol.json")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise SystemExit("benchmark requires CUDA and an explicit CUDA device")
    images = _find_images(args.raw_val_root.resolve())
    rng = random.Random(179)
    selected = rng.sample(images, min(args.samples, len(images)))
    # Ultralytics' numpy-source contract is HWC BGR. Decode outside timing and
    # convert PIL RGB to a contiguous BGR array so prediction content is realistic.
    decoded = [np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy() for path in selected]
    image_selection = {
        "population": len(images), "decoded_images": len(selected), "selection_seed": 179,
        "relative_paths": [path.relative_to(args.raw_val_root.resolve()).as_posix() for path in selected],
        "note": "Images were decoded and converted to contiguous BGR before either timing boundary.",
    }
    arms = _protocol_arms(protocol)
    if args.arm:
        arms = [arm for arm in arms if arm["id"] == args.arm]
        if not arms:
            raise SystemExit(f"unknown protocol arm: {args.arm}")
    protocol_hash = _sha256(suite_dir / "protocol.json")
    current_hardware = {"gpu": torch.cuda.get_device_name(torch.device(args.device)),
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "python": platform.python_version(), "platform": platform.platform()}
    for arm in arms:
        run_dir = suite_dir / arm["id"]
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            raise SystemExit(f"arm is not complete: {arm['id']}")
        metrics = _read_json(metrics_path)
        if metrics.get("status") != "completed":
            raise SystemExit(f"arm is not complete: {arm['id']}")
        checkpoint = ROOT / "checkpoints" / args.suite / f"{arm['id']}.pt"
        if not checkpoint.is_file():
            raise SystemExit(f"checkpoint missing: {checkpoint}")
        checkpoint_hash = _sha256(checkpoint)
        if metrics.get("checkpoint_sha256") != checkpoint_hash:
            raise SystemExit(f"checkpoint hash does not match metrics.json: {arm['id']}")
        output = run_dir / "latency.json"
        expected = {"schema_version": 1, "status": "completed", "arm_id": arm["id"],
                    "imgsz": arm["imgsz"], "seed": arm["seed"],
                    "checkpoint_sha256": checkpoint_hash, "precision": "FP16", "fused": True, "batch": 1,
                    "warmup_iterations": args.warmup, "timed_iterations": args.samples,
                    "settings": {"conf": args.conf, "iou": args.iou,
                                 "max_det": args.max_det, "rect": False},
                    "metrics_json_sha256": _sha256(metrics_path),
                    "protocol_json_sha256": protocol_hash,
                    "image_selection": image_selection, "hardware": current_hardware}
        if output.exists() and not args.overwrite:
            try:
                existing = _read_json(output)
            except (OSError, ValueError, json.JSONDecodeError):
                existing = {}
            if (all(existing.get(key) == value for key, value in expected.items())
                    and _valid_latency_summary(existing.get("pure_model_cuda_event"), args.samples)
                    and _valid_latency_summary(existing.get("synchronized_wall_clock"), args.samples)):
                print(json.dumps({"arm": arm["id"], "status": "reused",
                                  "latency": str(output)}), flush=True)
                continue
            raise SystemExit(f"latency result exists but its provenance/settings do not match: {output}; pass --overwrite to replace")
        result = benchmark_arm(arm, checkpoint, decoded, warmup=args.warmup,
                               samples=args.samples, device=args.device, conf=args.conf,
                               iou=args.iou, max_det=args.max_det)
        result["image_selection"] = image_selection
        result["metrics_json_sha256"] = _sha256(metrics_path)
        result["protocol_json_sha256"] = protocol_hash
        _atomic_json(output, result)
        print(json.dumps({"arm": arm["id"], "pure_model": result["pure_model_cuda_event"],
                          "end_to_end": result["synchronized_wall_clock"]}), flush=True)


if __name__ == "__main__":
    main()
