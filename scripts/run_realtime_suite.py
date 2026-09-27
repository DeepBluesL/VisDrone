#!/usr/bin/env python3
"""Prepare or execute the five-arm full-data pretrained resolution/P2 study."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")
os.environ.setdefault("GCM_INTERACTIVE", "Never")

from visdrone_migration.evidence import file_sha256
from visdrone_migration.realtime_protocol import atomic_json, canonical_sha, input_identity, verify_inputs


def make_protocol(suite, data, *, epochs=100, batch=64, workers=8, smoke=False, profiled_batches=False):
    pretrained = ROOT / "pretrained" / "yolov8n.pt"
    recipe = dict(epochs=epochs, batch=batch, workers=workers, device="0",
                  optimizer="AdamW", lr0=.001, lrf=.01, momentum=.9, weight_decay=.0005,
                  nbs=64, warmup_epochs=3., warmup_bias_lr=0., cos_lr=True, patience=0,
                  deterministic=True, amp=True, cache="disk", plots=True, save=True, val=True,
                  close_mosaic=10 if epochs > 10 else 0, mosaic=.3, mixup=0., copy_paste=0.,
                  degrees=0., translate=.1, scale=.3, shear=0., perspective=0.,
                  flipud=0., fliplr=.5, hsv_h=.015, hsv_s=.7, hsv_v=.4,
                  max_det=500, verbose=False)
    configurations = [("standard", 512), ("standard", 768), ("standard", 1024), ("p2", 768), ("p2", 1024)]
    if smoke:
        configurations = [("p2", 1024), ("standard", 512)]
    return dict(schema_version=1, suite=suite,
                title="Full-data COCO-pretrained VisDrone: resolution and P2 screening",
                purpose="Optimization steps 1 and 2; single-seed screening before repeated finalist runs",
                data=str(data.resolve()), pretrained=str(pretrained), training=recipe,
                arms=[dict(id=f"{architecture}_{size}_s179", architecture=architecture, imgsz=size, seed=179,
                           batch=(32 if architecture == "p2" and size == 1024 else 64) if profiled_batches else batch)
                      for architecture, size in configurations],
                initialization="Semantic COCO transfer with seeded new/classification layers; per-tensor audit retained",
                selection="Best validation AP50-95 over the full fixed budget; no early stopping",
                evaluation="Converted ten-class validation AP, not official ignore-region matching",
                validation_batch=32,
                precision_speed_goal="RTX 5090 D batch-1 FP16 latency after each completed run; no deployment FPS target assumed",
                sampling="All training images once per epoch, ordinary shuffled loader; no class resampling",
                effective_batch_note="nbs=64 after warmup; native warmup interpolates accumulation from 1",
                physical_batch_note="Physical batch may differ by arm for GPU utilization. BatchNorm statistics and per-microbatch loss normalization are not identical despite equal effective batch; this is practical configuration screening.",
                performance="Disk-decoded image cache, pinned persistent loader workers and bounded per-image assignment; identical loss/assignment objective",
                resume_note="Atomic FP32 training-state checkpoints; worker prefetch/augmentation state is not bitwise reproducible",
                smoke=smoke, identity=input_identity(ROOT, data, pretrained))


def report(suite):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "report_realtime.py"), "--suite", suite],
                   cwd=ROOT, check=True)


def monitor_child(child, suite_dir, arm, state):
    """Keep observable throughput/GPU telemetry while the sequential queue runs."""
    run_dir = suite_dir / arm["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = run_dir / "gpu_telemetry.csv"
    has_header = telemetry_path.exists() and telemetry_path.stat().st_size > 0
    last_reported_epoch = 0
    with telemetry_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if not has_header:
            writer.writerow(["unix_time", "gpu_util_percent", "memory_util_percent", "memory_used_mib", "power_watts", "temperature_c"])
        while child.poll() is None:
            try:
                row = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
                     "--format=csv,noheader,nounits"], text=True, timeout=5).strip().splitlines()[0]
                fields = [value.strip() for value in row.split(",")]
                writer.writerow([round(time.time(), 3), *fields])
                stream.flush()
                state.update(updated=time.time(), gpu_latest=fields)
                atomic_json(suite_dir / "queue_state.json", state)
                progress_path = run_dir / "progress.json"
                if progress_path.exists():
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    epoch = int(progress.get("epoch", 0))
                    if epoch >= last_reported_epoch + 10:
                        report(suite_dir.name)
                        last_reported_epoch = epoch
            except (OSError, subprocess.SubprocessError):
                pass  # Monitoring is advisory and never interrupts training.
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
    return child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="realtime_stage1")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "full" / "dataset.yaml")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--profiled-batches", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--finalize", action="store_true", help="verify/archive/commit after all five arms complete")
    parser.add_argument("--push", action="store_true", help="push verified final results to the existing authorized repository")
    args = parser.parse_args()
    if Path(args.suite).name != args.suite:
        raise SystemExit("Suite must be a single directory name")
    if args.smoke and not args.suite.startswith("realtime_smoke"):
        raise SystemExit("Smoke runs require a realtime_smoke-prefixed suite")
    if args.push and not args.finalize:
        raise SystemExit("--push requires --finalize")
    if args.finalize and (args.suite != "realtime_stage1" or args.skip_benchmark):
        raise SystemExit("Finalization requires realtime_stage1 and completed speed benchmarks")
    suite_dir = ROOT / "results" / args.suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = suite_dir / "protocol.json"
    if not protocol_path.exists():
        protocol = make_protocol(args.suite, args.data, epochs=args.epochs, batch=args.batch,
                                 workers=args.workers, smoke=args.smoke, profiled_batches=args.profiled_batches)
        atomic_json(protocol_path, protocol)
        shutil.copy2(args.data, suite_dir / "dataset.yaml")
        manifest = args.data.parent / "preparation_manifest.json"
        if manifest.exists():
            shutil.copy2(manifest, suite_dir / "preparation_manifest.json")
        shutil.copy2(ROOT / "pretrained" / "source_provenance.json", suite_dir / "pretrained_provenance.json")
        raw_source = ROOT / "data" / "source_provenance.json"
        if raw_source.exists():
            shutil.copy2(raw_source, suite_dir / "data_source_provenance.json")
        cache_source = ROOT / ".runtime" / "disk_cache_provenance.json"
        if cache_source.exists():
            shutil.copy2(cache_source, suite_dir / "disk_cache_provenance.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    verify_inputs(ROOT, protocol)
    report(args.suite)
    if args.prepare_only:
        print("PREPARED " + str(protocol_path), flush=True)
        return
    lock_path = ROOT / ".runtime" / "realtime_training.lock"
    # OS-backed lock is released automatically on process death; stale files are harmless.
    import msvcrt
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        state = {"pid": os.getpid(), "suite": args.suite, "status": "running", "started": time.time()}
        atomic_json(suite_dir / "queue_state.json", state)
        try:
            for arm in protocol["arms"]:
                state.update(arm=arm["id"], updated=time.time())
                atomic_json(suite_dir / "queue_state.json", state)
                result_path = suite_dir / arm["id"] / "metrics.json"
                expected_identity = {"protocol_sha256": canonical_sha(protocol), "arm": arm}
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    checkpoint = ROOT / "checkpoints" / args.suite / f"{arm['id']}.pt"
                    if (result.get("status") != "completed" or result.get("experiment_identity") != expected_identity
                            or result.get("checkpoint_sha256") != file_sha256(checkpoint)):
                        raise RuntimeError(f"Cannot reuse incompatible result: {arm['id']}")
                else:
                    command = [sys.executable, str(ROOT / "scripts" / "train_realtime.py"),
                               "--protocol", str(protocol_path), "--arm", arm["id"]]
                    last = ROOT / "runs" / args.suite / arm["id"] / "weights" / "last.pt"
                    if last.exists():
                        command.append("--resume")
                    log_dir = ROOT / ".runtime" / "logs" / args.suite
                    log_dir.mkdir(parents=True, exist_ok=True)
                    with (log_dir / f"{arm['id']}.log").open("a", encoding="utf-8") as log:
                        log.write(f"\nATTEMPT {time.time()} {command}\n")
                        log.flush()
                        print("TRAINING " + arm["id"], flush=True)
                        child = subprocess.Popen(command, cwd=ROOT / ".runtime" / "amp-check",
                                                 stdout=log, stderr=subprocess.STDOUT)
                        state.update(child_pid=child.pid)
                        atomic_json(suite_dir / "queue_state.json", state)
                        exit_code = monitor_child(child, suite_dir, arm, state)
                    if exit_code:
                        raise RuntimeError(f"Training failed ({exit_code}): {arm['id']}; see {log_dir}")
                if not args.skip_benchmark:
                    subprocess.run([sys.executable, str(ROOT / "scripts" / "benchmark_realtime.py"),
                                    "--suite", args.suite, "--arm", arm["id"]], cwd=ROOT, check=True)
                report(args.suite)
            state.update(status="completed", updated=time.time())
            atomic_json(suite_dir / "queue_state.json", state)
        except BaseException as exc:
            state.update(status="failed", error=repr(exc), updated=time.time())
            atomic_json(suite_dir / "queue_state.json", state)
            report(args.suite)
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    if args.finalize:
        command = [sys.executable, str(ROOT / "scripts" / "finalize_realtime.py"), "--commit"]
        if args.push:
            command.append("--push")
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
