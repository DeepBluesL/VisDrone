# SPDX-License-Identifier: AGPL-3.0-only
"""Resumable trainer for the standard/P2 real-time VisDrone study.

Checkpoint resume restores process RNG state, but cannot promise bitwise batch
continuation: PyTorch DataLoader worker/prefetch state is not serializable through
the Ultralytics lifecycle.  Epoch-boundary continuation is the supported model.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import io
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from ultralytics import __version__ as ultralytics_version
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first

from .realtime_models import (
    AP50_95DetectionTrainer,
    TransferReport,
    build_realtime_model,
    transfer_coco_weights,
)
from .realtime_performance import build_performance_dataloader, install_chunked_assigner


def _atomic_write(path: Path, content: bytes) -> None:
    """Durably replace one checkpoint without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fp32_optimizer_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu() if value.is_floating_point() else value.detach().cpu()
    if isinstance(value, dict):
        return {key: _fp32_optimizer_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_fp32_optimizer_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_fp32_optimizer_state(item) for item in value)
    return deepcopy(value)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state.get("torch_cuda", [])
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("CUDA device count differs from saved RNG state")
        torch.cuda.set_rng_state_all(cuda_states)


class RealtimeTrainer(AP50_95DetectionTrainer):
    """Semantic COCO initialization plus durable epoch-boundary continuation."""

    def __init__(
        self,
        *,
        architecture: str,
        initialization_seed: int,
        pretrained_path: str | Path | None,
        experiment_identity: Mapping[str, Any],
        overrides: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        resume_value = (overrides or {}).get("resume")
        self.resume_checkpoint_path = (
            Path(resume_value).resolve() if isinstance(resume_value, (str, Path)) else None
        )
        self.architecture = architecture
        self.initialization_seed = int(initialization_seed)
        self.pretrained_path = Path(pretrained_path).resolve() if pretrained_path else None
        self.experiment_identity = deepcopy(dict(experiment_identity))
        self.transfer_report: dict[str, Any] | None = None
        self.selected_epoch: int | None = None
        super().__init__(overrides=dict(overrides or {}), **kwargs)

    def _pretrained_source(self):
        if self.pretrained_path is None:
            return None
        if not self.pretrained_path.is_file():
            raise FileNotFoundError(f"pretrained checkpoint not found: {self.pretrained_path}")
        return torch.load(self.pretrained_path, map_location="cpu", weights_only=False)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build the requested graph; use exact raw weights only while resuming."""
        model = build_realtime_model(
            self.architecture,
            nc=self.data["nc"],
            channels=self.data.get("channels", 3),
            seed=self.initialization_seed,
            verbose=verbose and RANK == -1,
        )
        if self.resume:
            if weights is None:
                raise RuntimeError("resume requested but checkpoint raw model is missing")
            raw = weights.float() if isinstance(weights, nn.Module) else weights
            state = raw.state_dict() if isinstance(raw, nn.Module) else raw
            model.load_state_dict(state, strict=True)
            return model

        source = self._pretrained_source()
        if source is None:
            source = weights
        if source is None:
            raise RuntimeError("a local COCO pretrained checkpoint is required for a fresh run")
        report: TransferReport = transfer_coco_weights(
            model, source, architecture=self.architecture
        )
        self.transfer_report = report.to_dict()
        return model

    def setup_model(self):
        """Use saved raw model for resume; Ultralytics normally prefers checkpoint EMA."""
        if isinstance(self.model, nn.Module):
            return None
        cfg, weights, checkpoint = self.model, None, None
        if str(self.model).endswith(".pt"):
            loaded, checkpoint = load_checkpoint(self.model)
            if self.resume:
                raw = checkpoint.get("model")
                if raw is None:
                    raise RuntimeError("resume checkpoint has no raw model state")
                weights = raw.float()
                cfg = raw.yaml
            else:
                weights, cfg = loaded, loaded.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = load_checkpoint(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)
        return checkpoint

    def validate(self):
        previous = self.best_fitness
        metrics, fitness = super().validate()
        if previous is None or fitness >= previous:
            self.selected_epoch = self.epoch + 1
        return metrics, fitness

    def _checkpoint(self) -> dict[str, Any]:
        raw_model = deepcopy(de_parallel(self.model)).float().cpu()
        ema_model = deepcopy(self.ema.ema).float().cpu() if self.ema else None
        scaler_state = deepcopy(self.scaler.state_dict()) if getattr(self, "scaler", None) else None
        return {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "selected_epoch": self.selected_epoch,
            "model": raw_model,
            "ema": ema_model,
            "updates": self.ema.updates if self.ema else 0,
            "optimizer": _fp32_optimizer_state(self.optimizer.state_dict()),
            "scheduler": deepcopy(self.scheduler.state_dict()),
            "scaler": scaler_state,
            "rng_state": _rng_state(),
            "experiment_identity": deepcopy(self.experiment_identity),
            "transfer_report": deepcopy(self.transfer_report),
            "resume_scope": "epoch-boundary process RNG; DataLoader worker/prefetch state not captured",
            "train_args": vars(self.args),
            "train_metrics": {**self.metrics, "fitness": self.fitness},
            "train_results": self.read_results_csv(),
            "results_csv_text": Path(self.csv).read_text(encoding="utf-8") if Path(self.csv).exists() else "",
            "date": datetime.now().isoformat(),
            "version": ultralytics_version,
            "license": "AGPL-3.0 (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }

    def save_model(self):
        """Atomically save resumable FP32 raw/EMA checkpoints in native .pt form."""
        buffer = io.BytesIO()
        torch.save(self._checkpoint(), buffer)
        content = buffer.getvalue()
        _atomic_write(Path(self.last), content)
        if self.best_fitness == self.fitness:
            _atomic_write(Path(self.best), content)
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            _atomic_write(Path(self.wdir) / f"epoch{self.epoch}.pt", content)

    def resume_training(self, checkpoint):
        """Restore optimizer/EMA/scaler/RNG after native trainer construction."""
        if checkpoint is None or not self.resume:
            return
        saved_identity = checkpoint.get("experiment_identity")
        if saved_identity != self.experiment_identity:
            raise RuntimeError("resume checkpoint experiment identity does not match this run")
        if checkpoint.get("model") is None:
            raise RuntimeError("resume checkpoint lacks the raw model required for continuation")
        if checkpoint.get("results_csv_text") is None:
            raise RuntimeError("resume checkpoint lacks its exact CSV boundary")
        self._reconcile_resume_files(checkpoint)
        super().resume_training(checkpoint)
        if checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        if checkpoint.get("rng_state") is None:
            raise RuntimeError("resume checkpoint lacks process RNG state")
        _restore_rng_state(checkpoint["rng_state"])
        self._resume_scheduler_state = deepcopy(checkpoint.get("scheduler"))
        self.selected_epoch = checkpoint.get("selected_epoch")
        self.transfer_report = deepcopy(checkpoint.get("transfer_report"))

    def _reconcile_resume_files(self, checkpoint: Mapping[str, Any]) -> None:
        """Roll CSV/best artifacts back or forward to the atomic last boundary."""
        _atomic_write(Path(self.csv), checkpoint["results_csv_text"].encode("utf-8"))
        # save_model writes last first. If a crash occurred before best, last can
        # reconstruct best exactly whenever the saved epoch established the best.
        train_fitness = checkpoint.get("train_metrics", {}).get("fitness")
        if (
            train_fitness == checkpoint.get("best_fitness")
            and self.resume_checkpoint_path is not None
            and self.resume_checkpoint_path.is_file()
        ):
            _atomic_write(Path(self.best), self.resume_checkpoint_path.read_bytes())

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        install_chunked_assigner(de_parallel(self.model), max_pair_elements=32_000_000)
        # Ultralytics validates the EMA copy during training. Its criterion is
        # lazy and independent from the raw model criterion, so installing only
        # on self.model silently restores the native, full-batch assigner for
        # validation and can trigger CUDA OOM/CPU fallback on dense images.
        if self.ema is not None:
            install_chunked_assigner(de_parallel(self.ema.ema), max_pair_elements=32_000_000)
        state = getattr(self, "_resume_scheduler_state", None)
        if state is not None:
            self.scheduler.load_state_dict(state)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Build fixed-staging persistent loaders without native val-worker doubling."""
        if mode not in {"train", "val"}:
            raise ValueError(f"mode must be 'train' or 'val', got {mode!r}")
        # Use one validation geometry for every arm. Besides comparability, this
        # bounds dense-target assignment memory for both in-training EMA
        # validation and the native final evaluation loader.
        if mode == "val":
            batch_size = min(32, batch_size)
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with shuffle, setting shuffle=False")
            shuffle = False
        workers = int(self.args.workers) if mode == "train" else min(int(self.args.workers), 4)
        prefetch = 4 if mode == "train" else 2
        return build_performance_dataloader(
            dataset,
            batch_size=batch_size,
            workers=workers,
            shuffle=shuffle,
            rank=rank,
            prefetch_factor=prefetch,
        )

    def _do_train(self, world_size=1):
        """Finalize an already-complete budget without repeating its last epoch."""
        completed_checkpoint = None
        if self.resume_checkpoint_path is not None and self.resume_checkpoint_path.is_file():
            completed_checkpoint = torch.load(
                self.resume_checkpoint_path, map_location="cpu", weights_only=False
            )
            completed_epochs = int(completed_checkpoint.get("epoch", -1)) + 1
            if completed_epochs > self.epochs:
                raise RuntimeError(
                    f"resume checkpoint has {completed_epochs} epochs, exceeding requested {self.epochs}"
                )
            if completed_epochs < self.epochs:
                completed_checkpoint = None
        if completed_checkpoint is None:
            return super()._do_train(world_size)

        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)
        self.epoch = self.start_epoch - 1
        LOGGER.info(
            "Resume checkpoint already completed the requested epoch budget; "
            "running final validation without another training epoch."
        )
        if RANK in {-1, 0}:
            self.final_eval()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        self.run_callbacks("teardown")

    def final_eval(self):
        """Validate best EMA checkpoint without stripping resumable training state."""
        if Path(self.best).exists():
            LOGGER.info(f"\nValidating {self.best}...")
            self.validator.args.plots = self.args.plots
            self.metrics = self.validator(model=self.best)
            self.metrics.pop("fitness", None)
            self.run_callbacks("on_fit_epoch_end")


__all__ = ["RealtimeTrainer"]
