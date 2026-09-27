#!/usr/bin/env python3
"""Profile fresh real-time training configurations without producing experiment results."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import ultralytics.engine.trainer as trainer_module
from ultralytics.utils.torch_utils import autocast
from visdrone_migration.evidence import file_sha256
from visdrone_migration.realtime_protocol import canonical_sha, training_code_identity
from visdrone_migration.realtime_trainer import RealtimeTrainer


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"{path} must contain an object")
    return result


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("samples must be finite, nonnegative, and non-empty")
    array = np.asarray(values, dtype=np.float64)
    return {"samples": len(values), "mean": float(array.mean()),
            "p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95)),
            "max": float(array.max())}


class NvidiaSampler:
    """Collect low-rate telemetry with one persistent nvidia-smi process."""

    def __init__(self, interval_ms: int = 200):
        self.interval_ms = interval_ms
        self.rows: list[dict[str, float]] = []
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        command = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm",
                   "--format=csv,noheader,nounits", f"--loop-ms={self.interval_ms}"]
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                            text=True, creationflags=flags)
        except OSError:
            return

        def read() -> None:
            assert self.process and self.process.stdout
            for line in self.process.stdout:
                try:
                    values = [float(part.strip()) for part in line.split(",")]
                    if len(values) == 4:
                        self.rows.append(dict(gpu_util_percent=values[0], memory_used_mib=values[1],
                                              power_w=values[2], sm_clock_mhz=values[3]))
                except ValueError:
                    continue

        self.thread = threading.Thread(target=read, daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread:
            self.thread.join(timeout=3)
        keys = ("gpu_util_percent", "memory_used_mib", "power_w", "sm_clock_mhz")
        return {key: summarize([row[key] for row in self.rows]) for key in keys} if self.rows else {
            "unavailable": "nvidia-smi telemetry could not be collected"}


def _next_batch(iterator: Any, loader: Any) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _child(protocol_path: Path, arm_id: str, batch_size: int, workers: int,
           warmup: int, steps: int, output: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(8)
    protocol = _read_json(protocol_path)
    arm = next((item for item in protocol["arms"] if item["id"] == arm_id), None)
    if arm is None:
        raise ValueError(f"unknown arm {arm_id!r}")
    recipe = dict(protocol["training"])
    overrides = {**recipe, "model": "yolov8n-p2.yaml" if arm["architecture"] == "p2" else "yolov8n.yaml",
                 "data": protocol["data"], "imgsz": arm["imgsz"], "seed": arm["seed"],
                 "batch": batch_size, "workers": workers, "nbs": recipe.get("nbs", batch_size),
                 "epochs": 1, "val": False, "save": False, "plots": False,
                 "project": str(ROOT / ".runtime" / "performance" / "runs"),
                 "name": f"{arm_id}_b{batch_size}_w{workers}_{time.time_ns()}",
                 "exist_ok": False, "pretrained": False, "verbose": False}
    trainer = RealtimeTrainer(architecture=arm["architecture"], initialization_seed=arm["seed"],
                              pretrained_path=Path(protocol["pretrained"]),
                              experiment_identity={"profile": True, "protocol_sha256": canonical_sha(protocol),
                                                   "arm": arm}, overrides=overrides)
    # Avoid Ultralytics' unrelated YOLO11n AMP self-test/download. The formal
    # trainer performs that check; this profiler records and exercises AMP directly.
    original_check_amp = trainer_module.check_amp
    trainer_module.check_amp = lambda model: True
    try:
        trainer._setup_train(world_size=1)
    finally:
        trainer_module.check_amp = original_check_amp
    trainer._model_train()
    trainer.optimizer.zero_grad()
    iterator = iter(trainer.train_loader)
    accumulate = max(round(recipe.get("nbs", batch_size) / batch_size), 1)
    telemetry = NvidiaSampler()
    data_wait_ms: list[float] = []
    wall_step_ms: list[float] = []
    cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    images_seen = 0
    measured_started = 0
    try:
        for index in range(warmup + steps):
            wait_started = time.perf_counter_ns()
            batch, iterator = _next_batch(iterator, trainer.train_loader)
            wait_ms = (time.perf_counter_ns() - wait_started) / 1e6
            if index == warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                telemetry.start()
                measured_started = time.perf_counter_ns()
            step_started = time.perf_counter_ns()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            with autocast(trainer.amp):
                batch = trainer.preprocess_batch(batch)
                loss, _ = trainer.model(batch)
                loss = loss.sum()
            trainer.scaler.scale(loss).backward()
            if (index + 1) % accumulate == 0:
                trainer.optimizer_step()
            end.record()
            if index >= warmup:
                data_wait_ms.append(wait_ms)
                wall_step_ms.append((time.perf_counter_ns() - step_started) / 1e6)
                cuda_events.append((start, end))
                images_seen += int(batch["img"].shape[0])
        torch.cuda.synchronize()
        total_wall_seconds = (time.perf_counter_ns() - measured_started) / 1e9
        gpu_step_ms = [float(start.elapsed_time(end)) for start, end in cuda_events]
        result = {
            "schema_version": 1, "status": "completed", "purpose": "throughput profile; not a training result",
            "protocol_sha256": canonical_sha(protocol), "arm": arm, "batch": batch_size,
            "training_code_sha256": training_code_identity(ROOT),
            "profile_script_sha256": file_sha256(Path(__file__).resolve()),
            "workers": workers, "effective_batch": batch_size * accumulate, "accumulate": accumulate,
            "amp": bool(trainer.amp), "warmup_steps": warmup, "measured_steps": steps,
            "amp_check": "bypassed profiler-only YOLO11n comparison/download; FP16 path exercised directly",
            "images_seen": images_seen, "total_wall_seconds": total_wall_seconds,
            "throughput_images_per_second": images_seen / total_wall_seconds,
            "gpu_step_ms": summarize(gpu_step_ms),
            "host_next_batch_ms": summarize(data_wait_ms),
            "host_dispatch_ms": summarize(wall_step_ms),
            "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_cuda_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "telemetry": telemetry.stop(),
            "timing_note": "CUDA events cover H2D preprocessing, forward/loss, backward, and optimizer/EMA on accumulation boundaries. Total wall preserves asynchronous overlap and synchronizes only at measurement boundaries. host_next_batch_ms is host blocking time and may overlap outstanding GPU work.",
            "environment": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                            "cuda": torch.version.cuda, "python": platform.python_version(),
                            "platform": platform.platform()},
        }
        _atomic_json(output, result)
    finally:
        if telemetry.process and telemetry.process.poll() is None:
            telemetry.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "results" / "realtime_stage1" / "protocol.json")
    parser.add_argument("--arm", action="append", help="protocol arm id; repeatable (default: all)")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, nargs="+", default=[8])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.batch < 1 or any(worker < 0 for worker in args.workers) or args.warmup < 1 or args.steps < 1:
        raise SystemExit("batch, warmup and steps must be positive; workers must be nonnegative")
    protocol_path = args.protocol.resolve()
    if args._child:
        if len(args.arm or []) != 1 or len(args.workers) != 1 or args.output is None:
            raise SystemExit("internal child invocation requires one arm, one workers value, and --output")
        _child(protocol_path, args.arm[0], args.batch, args.workers[0], args.warmup, args.steps,
               args.output.resolve())
        return
    protocol = _read_json(protocol_path)
    known = [arm["id"] for arm in protocol["arms"]]
    arms = args.arm or known
    unknown = sorted(set(arms) - set(known))
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")
    output_dir = ROOT / ".runtime" / "performance"
    for arm in arms:
        for workers in args.workers:
            output = output_dir / f"{arm}_b{args.batch}_w{workers}.json"
            if output.exists() and not args.overwrite:
                print(f"SKIP existing {output}", flush=True)
                continue
            command = [sys.executable, str(Path(__file__).resolve()), "--_child",
                       "--protocol", str(protocol_path), "--arm", arm, "--batch", str(args.batch),
                       "--workers", str(workers), "--warmup", str(args.warmup), "--steps", str(args.steps),
                       "--output", str(output)]
            print(f"PROFILE {arm} batch={args.batch} workers={workers}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            result = _read_json(output)
            print(json.dumps({"arm": arm, "batch": args.batch, "workers": workers,
                              "images_per_second": result["throughput_images_per_second"],
                              "gpu_util_percent": result["telemetry"].get("gpu_util_percent"),
                              "peak_reserved_gb": result["peak_cuda_reserved_gb"]}), flush=True)


if __name__ == "__main__":
    main()
