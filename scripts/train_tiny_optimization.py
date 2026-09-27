#!/usr/bin/env python3
"""Plan seven tiny-object ablations, or explicitly train one frozen arm."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ARM_NAMES = ("baseline", "aux_control", "hbs", "api", "set", "simd", "set_simd")
SIMD_ARMS = {"simd", "set_simd"}
EXTRA_CODE_FILES = (
    "scripts/train_tiny_optimization.py",
    "visdrone_migration/tiny_optimization.py",
    "visdrone_migration/spectral_enhancement.py",
    "visdrone_migration/simd.py",
)
CALIBRATION_METHOD = "frozen-initial-predictions: YOLO adaptation, not original fixed-anchor estimator"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        pending = Path(stream.name)
    pending.replace(path)


def optimization_options(calibration: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    base = {
        "baseline": {},
        "aux_control": {"auxiliary": True},
        "hbs": {"auxiliary": True, "hbs": True},
        "api": {"auxiliary": True, "api": True},
        "set": {"auxiliary": True, "hbs": True, "api": True},
        "simd": {"simd": True},
        "set_simd": {"auxiliary": True, "hbs": True, "api": True, "simd": True},
    }
    if calibration is not None:
        normalizers = calibration["normalizers"]
        for name in SIMD_ARMS:
            base[name].update(simd_m=normalizers["m"], simd_n=normalizers["n"])
    return base


def explicit_optimization_options(calibration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Expand every dataclass default into auditable frozen protocol fields."""
    from dataclasses import asdict
    from visdrone_migration.tiny_optimization import TinyOptimization

    return {
        name: asdict(TinyOptimization(**options))
        for name, options in optimization_options(calibration).items()
    }


def plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "action": "execute-one-arm" if args.execute else "plan-only",
        "suite": args.suite,
        "selected_arm": args.arm,
        "arms": list(ARM_NAMES),
        "architecture": args.architecture,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "seed": args.seed,
        "data": str(args.data.resolve()),
        "pretrained": str(args.pretrained.resolve()),
        "calibration": str(args.calibration.resolve()),
        "calibration_required_for_protocol_freeze": True,
        "cuda_initialized": False,
    }


def validate_calibration(
    path: Path, *, data: Path, pretrained: Path, architecture: str, imgsz: int,
    seed: int, expected_train_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("calibration must be a JSON object")
    expected = {
        "schema_version": 1,
        "method": CALIBRATION_METHOD,
        "status": "completed",
        "split": "train",
        "subset": False,
        "max_images": None,
        "data": str(data.resolve()),
        "pretrained": str(pretrained.resolve()),
        "pretrained_sha256": file_sha256(pretrained),
        "architecture": architecture,
        "imgsz": imgsz,
        "seed": seed,
        "nc": 10,
        "image_count": 6471,
        "train_image_and_label_fingerprint": expected_train_fingerprint,
    }
    mismatches = [key for key, wanted in expected.items() if value.get(key) != wanted]
    if mismatches:
        raise ValueError(f"calibration provenance mismatch: {mismatches}")
    normalizers = value.get("normalizers")
    if not isinstance(normalizers, dict):
        raise ValueError("calibration has no normalizers")
    for key in ("m", "n"):
        number = normalizers.get(key)
        if not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
            raise ValueError(f"calibration normalizer {key} must be finite and positive")
    if not isinstance(value.get("pair_count"), int) or value["pair_count"] <= 0:
        raise ValueError("calibration pair_count must be positive")
    expected_code = {
        "scripts/calibrate_simd.py": file_sha256(ROOT / "scripts" / "calibrate_simd.py"),
        "visdrone_migration/realtime_models.py": file_sha256(ROOT / "visdrone_migration" / "realtime_models.py"),
        "visdrone_migration/simd.py": file_sha256(ROOT / "visdrone_migration" / "simd.py"),
    }
    if value.get("code_sha256") != expected_code:
        raise ValueError("calibration code provenance does not match current frozen code")
    import torch
    expected_environment = {
        "torch": torch.__version__, "ultralytics": importlib.metadata.version("ultralytics")
    }
    if value.get("environment") != expected_environment:
        raise ValueError("calibration runtime provenance does not match current runtime")
    return value


def code_identity() -> dict[str, str]:
    from visdrone_migration.realtime_protocol import TRAINING_FILES

    names = sorted(set(TRAINING_FILES) | set(EXTRA_CODE_FILES))
    return {name: file_sha256(ROOT / name) for name in names}


def build_protocol(args: argparse.Namespace, calibration: dict[str, Any]) -> dict[str, Any]:
    from visdrone_migration.evidence import data_fingerprint
    from visdrone_migration.realtime_protocol import runtime_identity

    stage1_path = ROOT / "results" / "realtime_stage1" / "protocol.json"
    stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))
    if stage1.get("suite") != "realtime_stage1" or not isinstance(stage1.get("training"), dict):
        raise ValueError("invalid frozen realtime_stage1 protocol")
    recipe = dict(stage1["training"])
    recipe.update(epochs=args.epochs, batch=args.batch, workers=args.workers)
    if recipe.get("nbs") != 64:
        raise ValueError("stage1 effective batch must remain nbs=64")
    runtime = runtime_identity()
    configurations = explicit_optimization_options(calibration)
    arms = [
        {
            "id": name, "architecture": args.architecture, "imgsz": args.imgsz,
            "seed": args.seed, "batch": args.batch, "optimization": configurations[name],
        }
        for name in ARM_NAMES
    ]
    return {
        "schema_version": 1,
        "suite": args.suite,
        "title": "Full-data tiny-object optimization ablation",
        "purpose": "Controlled SET/SimD training-only ablation on one frozen detector configuration",
        "data": str(args.data.resolve()),
        "pretrained": str(args.pretrained.resolve()),
        "training": recipe,
        "arms": arms,
        "optimization_configs": configurations,
        "calibration": {
            "path": str(args.calibration.resolve()),
            "sha256": file_sha256(args.calibration),
            "method": calibration["method"],
            "train_image_and_label_fingerprint": calibration["train_image_and_label_fingerprint"],
            "normalizers": calibration["normalizers"],
            "code_sha256": calibration["code_sha256"],
            "environment": calibration["environment"],
            "manifest": calibration,
        },
        "selection": "Best validation AP50-95 over the complete fixed budget; ties select latest",
        "evaluation": "Converted ten-class YOLO validation AP; not official VisDrone ignore matching",
        "validation_batch": 32,
        "identity": {
            "training_code": code_identity(),
            "data": data_fingerprint(args.data),
            "data_yaml_sha256": file_sha256(args.data),
            "pretrained_sha256": file_sha256(args.pretrained),
            "runtime": runtime,
        },
    }


def verify_protocol(protocol: dict[str, Any]) -> None:
    from visdrone_migration.evidence import data_fingerprint
    from visdrone_migration.realtime_protocol import runtime_identity

    identity = protocol.get("identity", {})
    actual = {
        "training_code": code_identity(),
        "data": data_fingerprint(Path(protocol["data"])),
        "data_yaml_sha256": file_sha256(Path(protocol["data"])),
        "pretrained_sha256": file_sha256(Path(protocol["pretrained"])),
    }
    for key, value in actual.items():
        if identity.get(key) != value:
            raise RuntimeError(f"frozen tiny-optimization input changed: {key}")
    if identity.get("runtime") != runtime_identity():
        raise RuntimeError("frozen tiny-optimization runtime changed")
    calibration = protocol.get("calibration", {})
    calibration_path = Path(calibration.get("path", ""))
    if not calibration_path.is_file() or calibration.get("sha256") != file_sha256(calibration_path):
        raise RuntimeError("frozen SimD calibration changed")


def freeze_protocol(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    from visdrone_migration.evidence import data_fingerprint
    from scripts.calibrate_simd import _train_fingerprint, _train_paths

    if not args.data.is_file() or not args.pretrained.is_file() or not args.calibration.is_file():
        raise FileNotFoundError("data, pretrained, and completed calibration files are required")
    data_identity = data_fingerprint(args.data)
    if data_identity.get("train", {}).get("images") != 6471:
        raise ValueError("formal tiny optimization requires all 6471 training images")
    if data_identity.get("val", {}).get("images") != 548:
        raise ValueError("formal tiny optimization requires the fixed 548-image validation split")
    calibration_images, _ = _train_paths(args.data.resolve())
    if len(calibration_images) != 6471:
        raise ValueError("formal SimD calibration identity requires all 6471 training images")
    calibration_train_fingerprint = _train_fingerprint(calibration_images)
    calibration = validate_calibration(
        args.calibration, data=args.data, pretrained=args.pretrained,
        architecture=args.architecture, imgsz=args.imgsz, seed=args.seed,
        expected_train_fingerprint=calibration_train_fingerprint,
    )
    suite_dir = ROOT / "results" / args.suite
    protocol_path = suite_dir / "protocol.json"
    # Normalize tuples (for example branch weights) exactly as persisted JSON so
    # a second invocation compares equal to the immutable on-disk protocol.
    candidate = json.loads(json.dumps(build_protocol(args, calibration), allow_nan=False))
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol != candidate:
            raise RuntimeError("existing suite protocol differs; frozen protocol cannot be modified")
    else:
        atomic_json(protocol_path, candidate)
        protocol = candidate
    verify_protocol(protocol)
    return protocol_path, protocol


def export_inference_checkpoint(trainer: Any, destination: Path) -> tuple[str, Any]:
    import torch
    from ultralytics import __version__ as ultralytics_version
    from visdrone_migration.tiny_optimization import inference_detector

    selected = torch.load(trainer.best, map_location="cpu", weights_only=False)
    model = selected.get("ema")
    if model is None:
        raise RuntimeError("selected checkpoint has no EMA model; refusing raw-model deployment export")
    detector = inference_detector(model.float().cpu())
    record = {
        "model": deepcopy(detector),
        "ema": detector,
        "optimizer": None,
        "train_args": selected.get("train_args", vars(trainer.args)),
        "epoch": selected.get("epoch", -1),
        "best_fitness": selected.get("best_fitness"),
        "date": selected.get("date"),
        "version": ultralytics_version,
        "license": "AGPL-3.0 (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(record, pending)
    pending.replace(destination)
    return file_sha256(destination), detector


def train_arm(args: argparse.Namespace, protocol_path: Path, protocol: dict[str, Any]) -> None:
    import numpy as np
    import torch
    from ultralytics.utils import SETTINGS
    from visdrone_migration.profiling import convolution_linear_gflops
    from visdrone_migration.tiny_optimization import TinyOptimization, TinyOptimizationTrainer

    arm = next(item for item in protocol["arms"] if item["id"] == args.arm)
    recipe = {**protocol["training"], "batch": arm["batch"]}
    destination = protocol_path.parent / arm["id"]
    destination.mkdir(parents=True, exist_ok=True)
    experiment_identity = {"protocol_sha256": canonical_sha(protocol), "arm": arm}
    metrics_path = destination / "metrics.json"
    if metrics_path.exists():
        raise RuntimeError("completed metrics already exist")
    run_dir = ROOT / "runs" / args.suite / arm["id"]
    last = run_dir / "weights" / "last.pt"
    if run_dir.exists() and not args.resume:
        raise RuntimeError("run directory exists; explicit --resume is required")
    if args.resume and not last.is_file():
        raise RuntimeError("resume requested but last.pt is missing")
    SETTINGS.update({key: False for key in (
        "sync", "clearml", "comet", "dvc", "hub", "mlflow", "neptune", "raytune", "tensorboard", "wandb"
    )})
    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats()
    overrides = {
        **recipe,
        "model": "yolov8n-p2.yaml" if args.architecture == "p2" else "yolov8n.yaml",
        "data": protocol["data"], "imgsz": args.imgsz, "seed": args.seed,
        "project": str(ROOT / "runs" / args.suite), "name": arm["id"],
        "exist_ok": True, "pretrained": False,
    }
    if args.resume:
        overrides["resume"] = str(last)
    previous_path = destination / "progress.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    preceding_seconds = previous.get("elapsed_seconds", 0) if args.resume else 0
    resume_count = int(previous.get("resume_count", 0)) + int(args.resume)
    failure_path = destination / "failure.json"
    if failure_path.exists():
        failure_path.replace(destination / f"failure_attempt_{time.time_ns()}.json")
    optimization_values = dict(arm["optimization"])
    optimization_values["branch_weights"] = tuple(optimization_values["branch_weights"])
    optimization = TinyOptimization(**optimization_values)
    trainer = TinyOptimizationTrainer(
        optimization=optimization, architecture=args.architecture,
        initialization_seed=args.seed, pretrained_path=Path(protocol["pretrained"]),
        experiment_identity=experiment_identity, overrides=overrides,
    )
    started = time.time()
    common = {
        "arm_id": arm["id"], "architecture": args.architecture, "seed": args.seed,
        "imgsz": args.imgsz, "epochs": recipe["epochs"],
        "experiment_identity": experiment_identity, "resume_count": resume_count,
        "optimization": arm["optimization"],
    }

    def ready(current):
        if bool(current.amp) != recipe["amp"] or current.batch_size != recipe["batch"]:
            raise RuntimeError("actual AMP/batch differs from frozen recipe")
        transfer = current.transfer_report
        atomic_json(destination / "initialization.json", transfer.to_dict() if hasattr(transfer, "to_dict") else transfer)
        atomic_json(destination / "progress.json", {
            **common, "status": "running", "epoch": current.start_epoch,
            "elapsed_seconds": preceding_seconds, "amp": bool(current.amp),
        })

    def progress(current):
        values = {key: float(value) for key, value in (current.metrics or {}).items()}
        if not all(np.isfinite(value) for value in values.values()):
            raise RuntimeError("non-finite validation metrics")
        for label, tensor in (
            ("training loss", getattr(current, "loss", None)),
            ("training loss items", getattr(current, "loss_items", None)),
        ):
            if tensor is not None and not torch.isfinite(tensor.detach()).all():
                raise RuntimeError(f"non-finite {label}; refusing to continue or publish completed metrics")
        if current.csv.exists():
            pending = destination / "epochs.csv.tmp"
            shutil.copyfile(current.csv, pending)
            pending.replace(destination / "epochs.csv")
        atomic_json(destination / "progress.json", {
            **common, "status": "running", "epoch": current.epoch + 1,
            "elapsed_seconds": round(preceding_seconds + time.time() - started, 1),
            "metrics": values, "selected_epoch": current.selected_epoch,
            "peak_cuda_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        })

    trainer.add_callback("on_pretrain_routine_end", ready)
    trainer.add_callback("on_fit_epoch_end", progress)
    try:
        # Ultralytics' AMP compatibility check resolves yolo11n.pt from CWD.
        # Use the pre-provisioned local asset and always restore the caller's
        # directory. All experiment data/project/checkpoint paths are absolute.
        previous_cwd = Path.cwd()
        amp_check_dir = ROOT / ".runtime" / "amp-check"
        if not (amp_check_dir / "yolo11n.pt").is_file():
            raise FileNotFoundError("local AMP-check asset is missing: .runtime/amp-check/yolo11n.pt")
        try:
            os.chdir(amp_check_dir)
            trainer.train()
        finally:
            os.chdir(previous_cwd)
        with trainer.csv.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if [int(row["epoch"]) for row in rows] != list(range(1, recipe["epochs"] + 1)):
            raise RuntimeError("epoch CSV is not a complete continuous fixed-budget run")
        verify_protocol(protocol)
        metric = trainer.validator.metrics
        selected_epoch = trainer.selected_epoch
        if not isinstance(selected_epoch, int) or not 1 <= selected_epoch <= recipe["epochs"]:
            raise RuntimeError("invalid selected epoch")
        initialization = json.loads((destination / "initialization.json").read_text())
        inference_path = ROOT / "checkpoints" / args.suite / f"{arm['id']}.pt"
        inference_hash, deployment_model = export_inference_checkpoint(trainer, inference_path)
        elapsed = preceding_seconds + time.time() - started
        selected_csv = float(rows[selected_epoch - 1]["metrics/mAP50-95(B)"])
        final_ap = float(metric.results_dict["metrics/mAP50-95(B)"])
        final_metrics = {key: float(value) for key, value in metric.results_dict.items() if key != "fitness"}
        if not all(np.isfinite(value) for value in final_metrics.values()):
            raise RuntimeError("non-finite final validation metrics")
        record = {
            **common, "status": "completed", "batch": recipe["batch"],
            "effective_batch": recipe["nbs"], "validation_batch": trainer.test_loader.batch_size,
            "calibration_sha256": protocol["calibration"]["sha256"],
            "calibration_provenance": protocol["calibration"],
            "initialization": "shared COCO-pretrained semantic transfer; auxiliary state seeded",
            "pretrained_parameter_coverage": initialization["parameter_numel_coverage"],
            "evaluation_protocol": "Ultralytics converted ten-class AP; not official VisDrone ignore matching",
            "checkpoint_selection": "highest validation AP50-95 in fixed budget; ties select latest",
            "selected_epoch": selected_epoch,
            "selection_metrics": {"csv_ap50_95": selected_csv,
                                  "final_minus_selection_ap50_95": final_ap - selected_csv},
            "metrics": final_metrics,
            "per_class": [{"id": int(category), "name": trainer.data["names"][int(category)],
                           "ap50": float(metric.box.all_ap[index, 0]),
                           "ap50_95": float(np.mean(metric.box.all_ap[index]))}
                          for index, category in enumerate(metric.box.ap_class_index)],
            "speed_ms": {key: float(value) for key, value in metric.speed.items()},
            "parameters_training": sum(parameter.numel() for parameter in trainer.model.parameters()),
            "parameters_inference": sum(parameter.numel() for parameter in deployment_model.parameters()),
            "parameters": sum(parameter.numel() for parameter in deployment_model.parameters()),
            "gflops_at_imgsz": convolution_linear_gflops(deployment_model, args.imgsz),
            "training_and_final_validation_seconds": elapsed,
            "peak_cuda_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "amp": bool(trainer.amp), "training": recipe,
            "resume_checkpoint": last.relative_to(ROOT).as_posix(),
            "resume_checkpoint_sha256": file_sha256(last),
            "inference_checkpoint": inference_path.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": inference_hash,
            "identity": protocol["identity"],
            "resume_note": "Epoch-boundary process RNG restored; worker prefetch state is not bitwise resumable",
            "run_directory": trainer.save_dir.relative_to(ROOT).as_posix(),
        }
        for name in ("results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "args.yaml"):
            source = trainer.save_dir / name
            if source.exists():
                shutil.copy2(source, destination / name)
        atomic_json(metrics_path, record)
        atomic_json(destination / "progress.json", {
            **common, "status": "completed", "epoch": recipe["epochs"],
            "elapsed_seconds": elapsed, "metrics": record["metrics"],
        })
        print("COMPLETED " + json.dumps({"arm": arm["id"], "metrics": record["metrics"]}), flush=True)
    except BaseException as exc:
        atomic_json(destination / "failure.json", {
            **common, "error": repr(exc), "time": time.time(), "checkpoint_exists": last.exists(),
        })
        raise


def execute(args: argparse.Namespace) -> None:
    if args.arm is None:
        raise SystemExit("--execute requires exactly one --arm")
    pause_path = ROOT / ".runtime" / "training_paused_by_user.json"
    if pause_path.exists():
        pause = json.loads(pause_path.read_text(encoding="utf-8"))
        if pause.get("automatic_restart_allowed") is False and not args.resume_paused:
            raise SystemExit("Execution is paused by user request; explicit --resume-paused authorization is required.")
    import msvcrt

    lock_path = ROOT / ".runtime" / "realtime_training.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("another realtime/tiny training process holds realtime_training.lock") from exc
        try:
            protocol_path, protocol = freeze_protocol(args)
            train_arm(args, protocol_path, protocol)
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="tiny_optimization")
    parser.add_argument("--arm", choices=ARM_NAMES)
    parser.add_argument("--architecture", choices=("standard", "p2"), default="standard")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=179)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "full" / "dataset.yaml")
    parser.add_argument("--pretrained", type=Path, default=ROOT / "pretrained" / "yolov8n.pt")
    parser.add_argument("--calibration", type=Path, default=ROOT / ".runtime" / "simd_normalizers.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-paused", action="store_true")
    args = parser.parse_args(argv)
    reserved = {"pilot", "module_extension", "realtime_stage1"}
    if (not args.suite or Path(args.suite).name != args.suite or args.suite in {".", ".."}
            or args.suite in reserved or args.suite.startswith("realtime")):
        parser.error("suite must be a safe new result directory name")
    if (args.imgsz < 32 or args.imgsz % 32 or min(args.epochs, args.batch, args.seed) < 1
            or args.workers < 0 or 64 % args.batch):
        parser.error("invalid image size, epoch, batch, worker, or seed value")
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    return args


def main() -> None:
    args = parse_args()
    if not args.execute:
        print(json.dumps(plan(args), indent=2))
        return
    execute(args)


if __name__ == "__main__":
    main()
