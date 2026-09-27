#!/usr/bin/env python3
"""Train one immutable full-data, COCO-initialized resolution/P2 experiment."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
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
from visdrone_migration.evidence import file_sha256
from visdrone_migration.profiling import convolution_linear_gflops
from visdrone_migration.realtime_protocol import atomic_json, canonical_sha, verify_inputs
from visdrone_migration.realtime_trainer import RealtimeTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    identity = verify_inputs(ROOT, protocol)
    arm = next(a for a in protocol["arms"] if a["id"] == args.arm)
    recipe = {**protocol["training"], "batch": arm.get("batch", protocol["training"]["batch"])}
    suite = protocol["suite"]
    destination = protocol_path.parent / arm["id"]
    destination.mkdir(parents=True, exist_ok=True)
    experiment_identity = {"protocol_sha256": canonical_sha(protocol), "arm": arm}
    metrics_path = destination / "metrics.json"
    if metrics_path.exists():
        raise RuntimeError("Completed result exists; use the suite runner to verify/reuse it")
    run_dir = ROOT / "runs" / suite / arm["id"]
    last = run_dir / "weights" / "last.pt"
    if run_dir.exists() and not args.resume:
        raise RuntimeError("Run directory already exists; explicit --resume is required")
    if args.resume and not last.exists():
        raise RuntimeError("Resume requested but no complete last checkpoint exists")
    SETTINGS.update({key: False for key in ("sync", "clearml", "comet", "dvc", "hub", "mlflow",
                                           "neptune", "raytune", "tensorboard", "wandb")})
    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats()
    overrides = {**recipe, "model": "yolov8n-p2.yaml" if arm["architecture"] == "p2" else "yolov8n.yaml",
                 "data": protocol["data"], "imgsz": arm["imgsz"], "seed": arm["seed"],
                 "project": str(ROOT / "runs" / suite), "name": arm["id"], "exist_ok": True,
                 "pretrained": False}
    if args.resume:
        overrides["resume"] = str(last)
    started = time.time()
    previous = json.loads((destination / "progress.json").read_text()) if (destination / "progress.json").exists() else {}
    preceding_seconds = previous.get("elapsed_seconds", 0) if args.resume else 0
    resume_count = int(previous.get("resume_count", 0)) + int(args.resume)
    failure_path = destination / "failure.json"
    if failure_path.exists():
        failure_path.replace(destination / f"failure_attempt_{time.time_ns()}.json")
    trainer = RealtimeTrainer(architecture=arm["architecture"], initialization_seed=arm["seed"],
                              pretrained_path=Path(protocol["pretrained"]),
                              experiment_identity=experiment_identity, overrides=overrides)
    common = dict(arm_id=arm["id"], architecture=arm["architecture"], seed=arm["seed"],
                  imgsz=arm["imgsz"], epochs=recipe["epochs"], experiment_identity=experiment_identity,
                  resume_count=resume_count)

    def ready(t):
        if bool(t.amp) != recipe["amp"]:
            raise RuntimeError("Actual AMP state differs from the frozen recipe")
        if t.batch_size != recipe["batch"]:
            raise RuntimeError("Actual physical batch differs from frozen recipe")
        transfer = t.transfer_report
        atomic_json(destination / "initialization.json", transfer.to_dict() if hasattr(transfer, "to_dict") else transfer)
        atomic_json(destination / "progress.json", {**common, "status": "running", "epoch": t.start_epoch,
                    "epochs": recipe["epochs"], "elapsed_seconds": preceding_seconds, "amp": bool(t.amp)})

    def progress(t):
        epoch_metrics = {k: float(v) for k, v in (t.metrics or {}).items()}
        if not all(np.isfinite(v) for v in epoch_metrics.values()):
            raise RuntimeError("Non-finite validation metrics")
        if t.csv.exists():
            pending_csv = destination / "epochs.csv.tmp"
            shutil.copyfile(t.csv, pending_csv)
            pending_csv.replace(destination / "epochs.csv")
        atomic_json(destination / "progress.json", {**common, "status": "running", "epoch": t.epoch + 1,
                    "elapsed_seconds": round(preceding_seconds + time.time() - started, 1),
                    "metrics": epoch_metrics, "selected_epoch": t.selected_epoch,
                    "peak_cuda_reserved_gb": torch.cuda.max_memory_reserved() / 1e9})

    trainer.add_callback("on_pretrain_routine_end", ready)
    trainer.add_callback("on_fit_epoch_end", progress)
    try:
        trainer.train()
        with trainer.csv.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        epochs = [int(row["epoch"]) for row in rows]
        if epochs != list(range(1, recipe["epochs"] + 1)):
            raise RuntimeError("Epoch history is not a complete continuous fixed-budget run")
        verify_inputs(ROOT, protocol)
        metric = trainer.validator.metrics
        initialization = json.loads((destination / "initialization.json").read_text())
        elapsed = preceding_seconds + time.time() - started
        record = {**common, "status": "completed", "batch": recipe["batch"], "effective_batch": recipe["nbs"],
            "validation_batch": trainer.test_loader.batch_size,
            "loader": {"train_workers": trainer.train_loader.num_workers,
                       "validation_workers": trainer.test_loader.num_workers,
                       "train_prefetch": trainer.train_loader.prefetch_factor,
                       "pin_memory": trainer.train_loader.pin_memory},
            "initialization": "COCO-pretrained YOLOv8n with explicit semantic transfer; new layers seeded",
            "pretrained_parameter_coverage": initialization["parameter_numel_coverage"],
            "evaluation_protocol": "Ultralytics YOLO-converted 10-class AP; not official VisDrone ignore-region evaluation",
            "checkpoint_selection": "highest validation AP50-95 in fixed epoch budget; ties select latest",
            "selected_epoch": trainer.selected_epoch,
            "selection_metrics": {
                "csv_ap50_95": float(rows[trainer.selected_epoch - 1]["metrics/mAP50-95(B)"]),
                "final_minus_selection_ap50_95": float(metric.results_dict["metrics/mAP50-95(B)"])
                    - float(rows[trainer.selected_epoch - 1]["metrics/mAP50-95(B)"]),
            },
            "metrics": {k: float(v) for k, v in metric.results_dict.items() if k != "fitness"},
            "per_class": [{"id": int(c), "name": trainer.data["names"][int(c)],
                           "ap50": float(metric.box.all_ap[i, 0]), "ap50_95": float(np.mean(metric.box.all_ap[i]))}
                          for i, c in enumerate(metric.box.ap_class_index)],
            "speed_ms": {k: float(v) for k, v in metric.speed.items()},
            "speed_note": "Validation batch timing; not batch-1 realtime latency. See latency.json.",
            "parameters": sum(p.numel() for p in trainer.model.parameters()),
            "gflops_at_imgsz": convolution_linear_gflops(trainer.model, arm["imgsz"]),
            "gflops_method": "Conv/linear MACs x 2, batch 1; excludes decode, NMS, normalization and elementwise ops",
            "training_and_final_validation_seconds": elapsed,
            "peak_cuda_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "amp": bool(trainer.amp), "training": recipe,
            "checkpoint_sha256": file_sha256(trainer.best), "identity": identity,
            "resume_count": resume_count,
            "resume_note": "Epoch-boundary state restored; worker augmentation/prefetch state prevents a bitwise uninterrupted-run guarantee",
            "run_directory": trainer.save_dir.relative_to(ROOT).as_posix()}
        weights = ROOT / "checkpoints" / suite / f"{arm['id']}.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        temporary = weights.with_suffix(".pt.tmp")
        shutil.copy2(trainer.best, temporary)
        if file_sha256(temporary) != record["checkpoint_sha256"]:
            raise RuntimeError("Exported checkpoint hash differs")
        temporary.replace(weights)
        for name in ("results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "args.yaml"):
            if (trainer.save_dir / name).exists():
                shutil.copy2(trainer.save_dir / name, destination / name)
        atomic_json(metrics_path, record)
        atomic_json(destination / "progress.json", {**common, "status": "completed", "epoch": recipe["epochs"],
                    "elapsed_seconds": elapsed, "metrics": record["metrics"]})
        print("COMPLETED " + json.dumps({"arm": arm["id"], "metrics": record["metrics"]}), flush=True)
    except BaseException as exc:
        atomic_json(destination / "failure.json", {**common, "error": repr(exc), "time": time.time(),
                    "checkpoint_exists": last.exists()})
        raise


if __name__ == "__main__":
    main()
