# SPDX-License-Identifier: AGPL-3.0-only
"""YOLOv8n standard/P2 builders and auditable COCO weight transfer.

This module does not download weights.  Callers must provide an already loaded
Ultralytics model, checkpoint mapping, or state dictionary.  P2 transfer uses
explicit feature-level semantics rather than coincident numeric graph indices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel


MODEL_CONFIGS = {
    "standard": "yolov8n.yaml",
    "p2": "yolov8n-p2.yaml",
}

# P2 target graph index -> standard YOLOv8 source index.  Layers 0--15 are
# structurally and semantically shared.  P2-only fusions at 18 and 21 have no source.
_P2_NECK_MAP = {
    22: 16,  # P3 -> P4 downsample
    24: 18,  # bottom-up P4 fusion
    25: 19,  # P4 -> P5 downsample
    27: 21,  # bottom-up P5 fusion
}


@dataclass(frozen=True)
class TensorTransfer:
    target_key: str
    source_key: str | None
    semantic: str
    status: str
    numel: int
    reason: str | None = None


@dataclass(frozen=True)
class TransferReport:
    architecture: str
    source_tensor_count: int
    target_tensor_count: int
    loaded_tensor_count: int
    loaded_numel: int
    target_numel: int
    loaded_parameter_numel: int
    target_parameter_numel: int
    records: tuple[TensorTransfer, ...]

    @property
    def tensor_coverage(self) -> float:
        return self.loaded_tensor_count / self.target_tensor_count

    @property
    def numel_coverage(self) -> float:
        return self.loaded_numel / self.target_numel

    @property
    def parameter_numel_coverage(self) -> float:
        return self.loaded_parameter_numel / self.target_parameter_numel

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            tensor_coverage=self.tensor_coverage,
            numel_coverage=self.numel_coverage,
            parameter_numel_coverage=self.parameter_numel_coverage,
        )
        return result


def build_realtime_model(
    architecture: str,
    *,
    nc: int = 10,
    channels: int = 3,
    seed: int = 0,
    verbose: bool = False,
) -> DetectionModel:
    """Build an official YOLOv8n standard or P2 graph with local RNG isolation."""
    if architecture not in MODEL_CONFIGS:
        raise ValueError(f"unknown architecture {architecture!r}; expected {tuple(MODEL_CONFIGS)}")
    if nc < 1 or channels < 1:
        raise ValueError("nc and channels must be positive")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return DetectionModel(
            MODEL_CONFIGS[architecture], nc=nc, ch=channels, verbose=verbose
        )


def _state_dict(source: nn.Module | Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    if isinstance(source, nn.Module):
        return source.state_dict()
    if not isinstance(source, Mapping):
        raise TypeError("source must be an nn.Module, checkpoint mapping, or state dict")
    for key in ("ema", "model"):
        value = source.get(key)
        if isinstance(value, nn.Module):
            return value.state_dict()
        if isinstance(value, Mapping) and value and all(
            isinstance(item, torch.Tensor) for item in value.values()
        ):
            return value
    if all(isinstance(value, torch.Tensor) for value in source.values()):
        return source  # type: ignore[return-value]
    raise TypeError("checkpoint does not contain a model/ema module or tensor state dict")


def _split_model_key(key: str) -> tuple[int, str] | None:
    parts = key.split(".", 2)
    if len(parts) != 3 or parts[0] != "model" or not parts[1].isdigit():
        return None
    return int(parts[1]), parts[2]


def _standard_source_key(target_key: str) -> tuple[str | None, str, str | None]:
    parsed = _split_model_key(target_key)
    if parsed is None:
        return target_key, "model metadata", None
    index, suffix = parsed
    if index == 22 and suffix.startswith("cv3."):
        return None, "Detect classification branch", "class-dependent output skipped"
    return target_key, "same graph role", None


def _p2_source_key(target_key: str) -> tuple[str | None, str, str | None]:
    parsed = _split_model_key(target_key)
    if parsed is None:
        return target_key, "model metadata", None
    index, suffix = parsed
    if index <= 15:
        return target_key, "shared backbone/top-down neck", None
    if index in _P2_NECK_MAP:
        source_index = _P2_NECK_MAP[index]
        return f"model.{source_index}.{suffix}", f"semantic neck {source_index}->{index}", None
    if index == 28:
        if suffix.startswith("cv3."):
            return None, "Detect classification branch", "class-dependent output skipped"
        if suffix.startswith("cv2."):
            parts = suffix.split(".", 2)
            branch = int(parts[1])
            if branch == 0:
                return None, "Detect P2 regression branch", "new P2 scale has no source"
            return (
                f"model.22.cv2.{branch - 1}.{parts[2]}",
                f"Detect regression P{branch + 2}",
                None,
            )
        return f"model.22.{suffix}", "shared Detect state", None
    return None, "P2-only path", "new P2 topology has no semantic source"


def transfer_coco_weights(
    target: DetectionModel,
    source: nn.Module | Mapping[str, Any],
    *,
    architecture: str,
) -> TransferReport:
    """Transfer locally supplied COCO weights and return a complete audit report."""
    if architecture not in MODEL_CONFIGS:
        raise ValueError(f"unknown architecture {architecture!r}")
    source_state = _state_dict(source)
    target_state = target.state_dict()
    parameter_keys = {name for name, _ in target.named_parameters()}
    mapper = _standard_source_key if architecture == "standard" else _p2_source_key
    updates: dict[str, torch.Tensor] = {}
    records: list[TensorTransfer] = []
    loaded_parameter_numel = 0

    for target_key, target_value in target_state.items():
        source_key, semantic, reason = mapper(target_key)
        if source_key is None:
            records.append(
                TensorTransfer(target_key, None, semantic, "skipped", target_value.numel(), reason)
            )
            continue
        source_value = source_state.get(source_key)
        if source_value is None:
            records.append(
                TensorTransfer(
                    target_key, source_key, semantic, "skipped", target_value.numel(),
                    "source tensor missing",
                )
            )
            continue
        if source_value.shape != target_value.shape:
            records.append(
                TensorTransfer(
                    target_key, source_key, semantic, "skipped", target_value.numel(),
                    f"shape mismatch {tuple(source_value.shape)} != {tuple(target_value.shape)}",
                )
            )
            continue
        updates[target_key] = source_value.detach().to(
            device=target_value.device, dtype=target_value.dtype
        )
        if target_key in parameter_keys:
            loaded_parameter_numel += target_value.numel()
        records.append(
            TensorTransfer(target_key, source_key, semantic, "loaded", target_value.numel())
        )

    target.load_state_dict(updates, strict=False)
    loaded = [record for record in records if record.status == "loaded"]
    return TransferReport(
        architecture=architecture,
        source_tensor_count=len(source_state),
        target_tensor_count=len(target_state),
        loaded_tensor_count=len(loaded),
        loaded_numel=sum(record.numel for record in loaded),
        target_numel=sum(value.numel() for value in target_state.values()),
        loaded_parameter_numel=loaded_parameter_numel,
        target_parameter_numel=sum(parameter.numel() for parameter in target.parameters()),
        records=tuple(records),
    )


def build_fair_model_pair(
    source: nn.Module | Mapping[str, Any],
    *,
    nc: int = 10,
    channels: int = 3,
    seed: int = 0,
) -> tuple[DetectionModel, DetectionModel, dict[str, TransferReport]]:
    """Build standard/P2 arms with identical transferred backbone tensors."""
    standard = build_realtime_model("standard", nc=nc, channels=channels, seed=seed)
    p2 = build_realtime_model("p2", nc=nc, channels=channels, seed=seed)
    reports = {
        "standard": transfer_coco_weights(standard, source, architecture="standard"),
        "p2": transfer_coco_weights(p2, source, architecture="p2"),
    }
    standard_state, p2_state = standard.state_dict(), p2.state_dict()
    for key, value in standard_state.items():
        parsed = _split_model_key(key)
        if parsed is not None and parsed[0] <= 9:
            if not torch.equal(value, p2_state[key]):
                raise RuntimeError(f"shared backbone tensor differs between arms: {key}")
    return standard, p2, reports


class AP50_95DetectionTrainer(DetectionTrainer):
    """Select best checkpoints by AP50-95 while retaining native resume behavior."""

    def validate(self):
        metrics = self.validator(self)
        metrics.pop("fitness", None)
        fitness = float(metrics["metrics/mAP50-95(B)"])
        if self.best_fitness is None or fitness >= self.best_fitness:
            self.best_fitness = fitness
        return metrics, fitness


__all__ = [
    "AP50_95DetectionTrainer",
    "MODEL_CONFIGS",
    "TensorTransfer",
    "TransferReport",
    "build_fair_model_pair",
    "build_realtime_model",
    "transfer_coco_weights",
]
