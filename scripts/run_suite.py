#!/usr/bin/env python3
"""Sequential GPU suite; completed runs are reused only for the same protocol."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
from visdrone_migration.model import VARIANTS
from visdrone_migration.evidence import (
    code_fingerprint,
    data_fingerprint,
    environment_fingerprint,
    file_sha256,
)


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _atomic_json(path: Path, content: dict) -> None:
    _atomic_bytes(path, (json.dumps(content, indent=2) + "\n").encode("utf-8"))


def _snapshot(source: Path, destination: Path) -> None:
    content = source.read_bytes()
    if destination.exists() and destination.read_bytes() != content:
        raise RuntimeError(
            f"suite snapshot already exists with different content: {destination}"
        )
    if not destination.exists():
        _atomic_bytes(destination, content)


def _identity(root: Path, data: Path, preparation_manifest: Path) -> dict:
    return {
        "data_fingerprint": data_fingerprint(data),
        "data_yaml_sha256": file_sha256(data),
        "preparation_manifest_sha256": file_sha256(preparation_manifest),
        "training_code_sha256": code_fingerprint(root),
        "environment_fingerprint": environment_fingerprint(),
    }


def _assert_live_identity(expected: dict, root: Path, data: Path, manifest: Path) -> None:
    current = _identity(root, data, manifest)
    mismatches = [key for key, value in current.items() if value != expected[key]]
    if mismatches:
        raise RuntimeError(
            "experiment inputs changed while the suite was active: "
            + ", ".join(mismatches)
        )


def validate_completed_record(
    metrics_path: Path,
    checkpoint_path: Path,
    *,
    variant: str,
    seed: int,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    identity: dict,
) -> dict:
    """Validate a completion record and its exported checkpoint or raise."""
    try:
        record = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable completion record {metrics_path}: {exc}") from exc

    expected_fields = {
        "status": "completed",
        "variant": variant,
        "seed": seed,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
    }
    mismatches = [
        f"{key}={record.get(key)!r} (expected {value!r})"
        for key, value in expected_fields.items()
        if record.get(key) != value
    ]
    training = record.get("training")
    if not isinstance(training, dict) or training.get("workers") != workers:
        mismatches.append(
            f"training.workers={training.get('workers') if isinstance(training, dict) else None!r} "
            f"(expected {workers!r})"
        )
    for key in (
        "data_fingerprint",
        "data_yaml_sha256",
        "training_code_sha256",
        "environment_fingerprint",
    ):
        if record.get(key) != identity[key]:
            mismatches.append(f"{key} does not match suite protocol")

    if not checkpoint_path.is_file():
        mismatches.append(f"exported checkpoint is missing: {checkpoint_path}")
    else:
        actual_checkpoint_hash = file_sha256(checkpoint_path)
        if record.get("checkpoint_sha256") != actual_checkpoint_hash:
            mismatches.append("checkpoint SHA-256 does not match completion record")
    if mismatches:
        raise ValueError("; ".join(mismatches))
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[179])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--suite", default="pilot")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "dataset.yaml")
    p.add_argument("--expected-train", type=int)
    p.add_argument("--expected-val", type=int)
    a = p.parse_args()
    data = a.data.resolve()
    preparation_manifest = data.parent / "preparation_manifest.json"
    if not data.is_file():
        raise SystemExit(f"Dataset YAML not found: {data}")
    if not preparation_manifest.is_file():
        raise SystemExit(f"Preparation manifest not found: {preparation_manifest}")

    output = ROOT / "results" / a.suite
    output.mkdir(parents=True, exist_ok=True)
    identity = _identity(ROOT, data, preparation_manifest)
    counts = {
        split: int(identity["data_fingerprint"][split]["images"])
        for split in ("train", "val")
    }
    print(f"DATA train={counts['train']} val={counts['val']}", flush=True)
    for split, expected in (("train", a.expected_train), ("val", a.expected_val)):
        if expected is not None and counts[split] != expected:
            raise SystemExit(
                f"Expected {expected} {split} images, found {counts[split]} in {data}"
            )

    snapshot_dir = output / "data_snapshot"
    dataset_snapshot = snapshot_dir / "dataset.yaml"
    manifest_snapshot = snapshot_dir / "preparation_manifest.json"
    _snapshot(data, dataset_snapshot)
    _snapshot(preparation_manifest, manifest_snapshot)
    protocol = {
        "variants": a.variants,
        "seeds": a.seeds,
        "epochs": a.epochs,
        "imgsz": a.imgsz,
        "batch": a.batch,
        "workers": a.workers,
        "suite": a.suite,
        "data": str(data),
        "expected_train": a.expected_train,
        "expected_val": a.expected_val,
        "actual_image_counts": counts,
        **identity,
        "data_snapshot": {
            "dataset_yaml": dataset_snapshot.relative_to(ROOT).as_posix(),
            "preparation_manifest": manifest_snapshot.relative_to(ROOT).as_posix(),
        },
    }
    protocol_file = output / "protocol.json"
    if protocol_file.exists():
        try:
            existing_protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Existing suite protocol is unreadable: {exc}") from exc
        if existing_protocol != protocol:
            raise SystemExit("Suite protocol changed; choose a different --suite name.")
    else:
        _atomic_json(protocol_file, protocol)

    for seed in a.seeds:
        for variant in a.variants:
            _assert_live_identity(identity, ROOT, data, preparation_manifest)
            name = f"{variant}_s{seed}"
            metrics_path = output / name / "metrics.json"
            checkpoint_path = ROOT / "checkpoints" / a.suite / f"{name}.pt"
            if metrics_path.exists():
                try:
                    validate_completed_record(
                        metrics_path,
                        checkpoint_path,
                        variant=variant,
                        seed=seed,
                        epochs=a.epochs,
                        imgsz=a.imgsz,
                        batch=a.batch,
                        workers=a.workers,
                        identity=identity,
                    )
                except ValueError as exc:
                    raise SystemExit(f"Cannot reuse invalid completed run {name}: {exc}") from exc
                print(f"SKIP completed {name}", flush=True)
                continue
            logfile = output / f"{name}.log"
            command = [sys.executable, "-u", str(ROOT / "scripts" / "train.py"),
                "--variant", variant, "--seed", str(seed), "--epochs", str(a.epochs),
                "--imgsz", str(a.imgsz), "--batch", str(a.batch), "--workers", str(a.workers),
                "--suite", a.suite, "--data", str(data), "--overwrite"]
            print(f"START {name}; log: {logfile}", flush=True)
            start = time.time()
            with logfile.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            if completed.returncode:
                print(logfile.read_text(encoding="utf-8", errors="replace")[-10000:], flush=True)
                raise SystemExit(f"FAILED {name}, code={completed.returncode}")
            _assert_live_identity(identity, ROOT, data, preparation_manifest)
            try:
                validate_completed_record(
                    metrics_path,
                    checkpoint_path,
                    variant=variant,
                    seed=seed,
                    epochs=a.epochs,
                    imgsz=a.imgsz,
                    batch=a.batch,
                    workers=a.workers,
                    identity=identity,
                )
            except ValueError as exc:
                raise SystemExit(f"Run {name} exited successfully but evidence is invalid: {exc}") from exc
            print(f"FINISH {name} in {time.time() - start:.1f}s", flush=True)
    print("SUITE COMPLETED", flush=True)


if __name__ == "__main__":
    main()
