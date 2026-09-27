"""Immutable experiment identities and atomic records for the full-data study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from .evidence import data_fingerprint, environment_fingerprint, file_sha256


TRAINING_FILES = (
    "scripts/train_realtime.py",
    "visdrone_migration/realtime_models.py",
    "visdrone_migration/realtime_trainer.py",
    "visdrone_migration/realtime_performance.py",
    "visdrone_migration/realtime_protocol.py",
    "visdrone_migration/evidence.py",
    "visdrone_migration/profiling.py",
)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def training_code_identity(root):
    return {name: file_sha256(Path(root) / name) for name in TRAINING_FILES}


def runtime_identity():
    import torch
    result = environment_fingerprint()
    result.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
    result["driver"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).strip()
    return result


def input_identity(root, data, pretrained):
    return dict(training_code=training_code_identity(root), data=data_fingerprint(data),
                data_yaml_sha256=file_sha256(data), pretrained_sha256=file_sha256(pretrained),
                runtime=runtime_identity())


def validate_protocol(protocol):
    arms = protocol["arms"]
    ids = [arm["id"] for arm in arms]
    if not arms or len(ids) != len(set(ids)):
        raise ValueError("Protocol needs unique non-empty arms")
    for arm in arms:
        if Path(arm["id"]).name != arm["id"] or arm["architecture"] not in ("standard", "p2"):
            raise ValueError("Invalid arm id or architecture")
        if arm["imgsz"] % 32 or arm["imgsz"] < 32:
            raise ValueError("Input resolution must be a positive multiple of 32")
    recipe = protocol["training"]
    if recipe["batch"] < 1 or recipe["nbs"] % recipe["batch"]:
        raise ValueError("Effective batch must be divisible by the physical batch")
    if any(a.get("batch", recipe["batch"]) < 1 or recipe["nbs"] % a.get("batch", recipe["batch"]) for a in arms):
        raise ValueError("Every arm's physical batch must divide the common effective batch")
    if recipe["epochs"] < 1:
        raise ValueError("Epoch budget must be positive")


def verify_inputs(root, protocol):
    validate_protocol(protocol)
    actual = input_identity(root, protocol["data"], protocol["pretrained"])
    if actual != protocol["identity"]:
        different = [key for key in actual if actual[key] != protocol["identity"].get(key)]
        raise RuntimeError(f"Frozen experiment inputs changed: {different}")
    return actual
