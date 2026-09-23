#!/usr/bin/env python3
"""Run one controlled from-scratch VisDrone detection experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from ultralytics.utils import SETTINGS
from visdrone_migration.model import MigrationTrainer, VARIANTS
from visdrone_migration.profiling import convolution_linear_gflops
from visdrone_migration.evidence import code_fingerprint, data_fingerprint, environment_fingerprint


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=list(VARIANTS), default="baseline")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "dataset.yaml")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=179)
    p.add_argument("--device", default="0")
    p.add_argument("--suite", default="pilot")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    name = f"{args.variant}_s{args.seed}"
    attempt_name = f"{name}_attempt_{time.time_ns()}"
    destination = ROOT / "results" / args.suite / name
    if (destination / "metrics.json").exists() and not args.overwrite:
        raise SystemExit(f"Completed run exists: {destination}. Choose a new suite or --overwrite.")
    if args.overwrite:
        (destination / "metrics.json").unlink(missing_ok=True)
    SETTINGS.update({key: False for key in ("sync", "clearml", "comet", "dvc", "hub", "mlflow",
                                           "neptune", "raytune", "tensorboard", "wandb")})
    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    overrides = dict(model="yolov8n.yaml", data=str(args.data.resolve()), epochs=args.epochs,
        imgsz=args.imgsz, batch=args.batch, workers=args.workers, seed=args.seed, device=args.device,
        project=str(ROOT / "runs" / args.suite), name=attempt_name, exist_ok=False,
        pretrained=False, optimizer="AdamW", lr0=args.lr, lrf=0.1, weight_decay=0.0005, nbs=args.batch,
        momentum=0.9, warmup_epochs=min(1.0, args.epochs / 10), cos_lr=True, patience=0,
        deterministic=True, amp=False, cache=False, plots=True, save=True, val=True,
        close_mosaic=0, mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0,
        translate=0.1, scale=0.3, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, max_det=500, verbose=False)
    destination.mkdir(parents=True, exist_ok=True)
    started = time.time()
    trainer = MigrationTrainer(variant=args.variant, overrides=overrides)
    data_identity = data_fingerprint(args.data)
    code_identity = code_fingerprint(ROOT)
    environment_identity = environment_fingerprint()
    # Record actual epoch progress without scraping terminal output.
    def progress(t):
        record = {"variant": args.variant, "seed": args.seed, "epoch": t.epoch + 1,
                  "epochs": args.epochs, "elapsed_seconds": round(time.time() - started, 1),
                  "metrics": {k: float(v) for k, v in (t.metrics or {}).items()}}
        pending = destination / "progress.json.tmp"
        pending.write_text(json.dumps(record, indent=2), encoding="utf-8")
        pending.replace(destination / "progress.json")
    trainer.add_callback("on_fit_epoch_end", progress)
    trainer.train()
    elapsed = time.time() - started
    validator = trainer.validator
    metric = validator.metrics
    params = sum(p.numel() for p in trainer.model.parameters())
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    record = {
        "status": "completed", "variant": args.variant, "factors": list(VARIANTS[args.variant]),
        "seed": args.seed, "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "initialization": "from scratch; no pretrained checkpoint", "evaluation_split": "official validation",
        "evaluation_protocol": "Ultralytics YOLO-converted 10-class AP; not official VisDrone ignore-region evaluation",
        "checkpoint_selection": "best validation mAP50-95 within fixed epoch budget",
        "selected_epoch": trainer.selected_epoch,
        "metrics": {k: float(v) for k, v in metric.results_dict.items() if k != "fitness"},
        "per_class": [{"id": int(c), "name": trainer.data["names"][int(c)],
                       "ap50": float(metric.box.all_ap[i, 0]),
                       "ap50_95": float(np.mean(metric.box.all_ap[i]))}
                      for i, c in enumerate(metric.box.ap_class_index)],
        "speed_ms": {k: float(v) for k, v in metric.speed.items()},
        "parameters": params, "trainable_parameters": trainable,
        "gflops_at_imgsz": convolution_linear_gflops(trainer.model, args.imgsz),
        "gflops_method": "Convolution and linear MACs x 2, batch 1, includes fixed Gabor/Haar analysis and synthesis plus LS dynamic spatial MACs; excludes normalization, pooling, activation, elementwise gates/scales, decode and NMS; not total hardware FLOPs",
        "training_and_final_validation_seconds": elapsed,
        "peak_cuda_reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else None,
        "data_yaml_sha256": sha256(args.data), "checkpoint_sha256": sha256(trainer.best),
        "data_fingerprint": data_identity, "training_code_sha256": code_identity,
        "environment_fingerprint": environment_identity,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                        **{pkg: importlib.metadata.version(pkg) for pkg in
                           ("torchvision", "ultralytics", "numpy", "Pillow", "matplotlib")}},
        "training": {k: v for k, v in overrides.items() if k not in ("project", "data", "name")},
        "run_directory": trainer.save_dir.relative_to(ROOT).as_posix(),
    }
    if (data_fingerprint(args.data) != data_identity or code_fingerprint(ROOT) != code_identity
            or environment_fingerprint() != environment_identity):
        raise RuntimeError("Data, code or environment changed during the run; result not marked complete.")
    shutil.copy2(trainer.csv, destination / "epochs.csv")
    weights = ROOT / "checkpoints" / args.suite
    weights.mkdir(parents=True, exist_ok=True)
    temporary_weights = weights / f"{name}.pt.tmp"
    shutil.copy2(trainer.best, temporary_weights)
    if sha256(temporary_weights) != record["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint export checksum mismatch")
    temporary_weights.replace(weights / f"{name}.pt")
    for filename in ("results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png"):
        source = trainer.save_dir / filename
        if source.exists():
            shutil.copy2(source, destination / filename)
    pending = destination / "metrics.json.tmp"
    pending.write_text(json.dumps(record, indent=2), encoding="utf-8")
    pending.replace(destination / "metrics.json")
    print("COMPLETED " + json.dumps({"variant": args.variant, "seed": args.seed,
                                      "metrics": record["metrics"], "seconds": elapsed}), flush=True)


if __name__ == "__main__":
    main()
