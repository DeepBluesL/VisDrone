# SPDX-License-Identifier: AGPL-3.0-only
"""Training-only SET and SimD-TAL adaptations for the existing YOLOv8 study.

The original papers use different detectors. These are explicitly named YOLO
adaptations; see docs/SET_SIMD_EXPERIMENTS.md for the controlled comparisons.
The inference graph is the unchanged standard/P2 detector.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import de_parallel

from .realtime_models import build_realtime_model, transfer_coco_weights
from .realtime_performance import ChunkedTaskAlignedAssigner
from .realtime_trainer import RealtimeTrainer
from .simd import bbox_simd


@dataclass(frozen=True)
class TinyOptimization:
    auxiliary: bool = False
    hbs: bool = False
    api: bool = False
    simd: bool = False
    reduction: int = 4
    rho: float = 1.0
    auxiliary_weight: float = 1.0
    # YOLO's three weighted loss branches: box, classification, DFL.
    # The released FCOS implementation averages its three normalized gradients.
    branch_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    simd_m: float | None = None
    simd_n: float | None = None

    def __post_init__(self):
        if (self.hbs or self.api) and not self.auxiliary:
            raise ValueError("HBS/API require the auxiliary training branch")
        if self.reduction < 1 or not math.isfinite(self.rho) or self.rho < 0:
            raise ValueError("invalid reduction or perturbation radius")
        if not math.isfinite(self.auxiliary_weight) or self.auxiliary_weight <= 0:
            raise ValueError("auxiliary_weight must be finite and positive")
        if len(self.branch_weights) != 3 or any(not math.isfinite(x) or x < 0 for x in self.branch_weights):
            raise ValueError("three finite nonnegative API branch weights are required")
        if self.simd and any(x is None or not math.isfinite(x) or x <= 0 for x in (self.simd_m, self.simd_n)):
            raise ValueError("SimD requires explicit positive train-set-calibrated m and n")


class SimDTaskAlignedAssigner(ChunkedTaskAlignedAssigner):
    """Replace TAL's box-similarity term; keep CIoU regression and IoU NMS."""

    def __init__(self, *, m: float, n: float, **kwargs):
        super().__init__(**kwargs)
        if any(not math.isfinite(x) or x <= 0 for x in (m, n)):
            raise ValueError("SimD m/n must be finite and positive")
        self.m, self.n = float(m), float(n)

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        # Assignment is detached. FP32 avoids AMP underflow in the exponential;
        # no synchronous tensor validation occurs in this hot path.
        return bbox_simd(gt_bboxes.float(), pd_bboxes.float(), m=self.m, n=self.n,
                         aligned=True, validate=False).to(pd_bboxes.dtype)


@contextmanager
def auxiliary_batch_norm(head):
    """Use batch statistics twice but update shared BN running buffers only once."""
    modules = [module for module in head.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    states = [module.track_running_stats for module in modules]
    try:
        for module in modules:
            module.track_running_stats = False
        yield
    finally:
        for module, state in zip(modules, states):
            module.track_running_stats = state


class TinyDetectionModel(DetectionModel):
    """Ordinary inference; label-dependent enhancement is confined to loss()."""

    def init_criterion(self):
        criterion = v8DetectionLoss(self)
        config = self.tiny_optimization
        options = dict(topk=10, num_classes=self.model[-1].nc, alpha=0.5, beta=6.0,
                       max_pair_elements=32_000_000)
        criterion.assigner = (SimDTaskAlignedAssigner(m=config.simd_m, n=config.simd_n, **options)
                              if config.simd else ChunkedTaskAlignedAssigner(**options))
        return criterion

    def loss(self, batch, preds=None):
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        config = self.tiny_optimization
        if not self.training or not config.auxiliary:
            if preds is None:
                preds = self.predict(batch["img"])
            return self.criterion(preds, batch)
        if preds is not None:
            raise ValueError("SET training needs loss(batch) to capture the original neck features")

        captured = []

        def capture(_module, arguments):
            # Detect mutates its input list, so retain the original tensor refs.
            captured.extend(list(arguments[0]))

        handle = self.model[-1].register_forward_pre_hook(capture)
        try:
            predictions = self.predict(batch["img"])
        finally:
            handle.remove()
        main_loss, main_items = self.criterion(predictions, batch)
        if len(captured) != len(self.model[-1].stride):
            raise RuntimeError("Unexpected detection feature pyramid")
        # Loss components returned by v8DetectionLoss retain gradients. The
        # second return value is detached and must NOT be used for API.
        from .spectral_enhancement import adversarial_feature_perturbations, feature_masks_from_normalized_xywh
        names = ("box", "cls", "dfl")
        perturbations = (adversarial_feature_perturbations(
                            captured, dict(zip(names, main_loss.unbind())), rho=config.rho,
                            branch_weights=dict(zip(names, config.branch_weights)), validate=False)
                         if config.api else [torch.zeros_like(feature) for feature in captured])
        if config.hbs:
            masks = feature_masks_from_normalized_xywh(
                batch["bboxes"], batch["batch_idx"], batch_size=captured[0].shape[0],
                feature_shapes=[tuple(feature.shape[-2:]) for feature in captured],
                device=captured[0].device, dtype=captured[0].dtype, validate=False)
            enhanced = [block(feature, mask) for block, feature, mask in zip(self.set_hbs, captured, masks)]
        else:
            enhanced = captured
        with auxiliary_batch_norm(self.model[-1]):
            auxiliary_predictions = self.model[-1]([feature + delta for feature, delta in zip(enhanced, perturbations)])
        auxiliary_loss, auxiliary_items = self.criterion(auxiliary_predictions, batch)
        # Three native logging slots contain the actual combined optimized loss;
        # validation still reports the ordinary main-head detection loss.
        return (main_loss + config.auxiliary_weight * auxiliary_loss,
                main_items + config.auxiliary_weight * auxiliary_items)


def attach_tiny_optimization(model, config: TinyOptimization, *, seed: int):
    """Attach auxiliary modules without perturbing shared initialization/RNG."""
    from .spectral_enhancement import HBSLevel
    model.__class__ = TinyDetectionModel
    model.tiny_optimization = config
    head = model.model[-1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model.set_hbs = nn.ModuleList([
            HBSLevel(branch[0].conv.in_channels, int(stride), reduction=config.reduction)
            for branch, stride in zip(head.cv2, head.stride)
        ] if config.hbs else [])
    # A serialized or already-used detector may contain a stale criterion.
    if hasattr(model, "criterion"):
        del model.criterion
    return model


def build_tiny_model(architecture, *, config: TinyOptimization, nc=10, seed=179, source=None):
    model = build_realtime_model(architecture, nc=nc, seed=seed)
    report = transfer_coco_weights(model, source, architecture=architecture) if source is not None else None
    return attach_tiny_optimization(model, config, seed=seed), report


def inference_detector(model):
    """Strip all auxiliary training state from a copied deployment detector."""
    result = deepcopy(model).eval()
    result.__class__ = DetectionModel
    for name in ("set_hbs", "tiny_optimization", "criterion"):
        if hasattr(result, name):
            delattr(result, name)
    return result


class TinyOptimizationTrainer(RealtimeTrainer):
    def __init__(self, *, optimization: TinyOptimization, **kwargs):
        self.optimization = optimization
        super().__init__(**kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model, report = build_tiny_model(self.architecture, config=self.optimization,
                                        nc=self.data["nc"], seed=self.initialization_seed,
                                        source=None if self.resume else self._pretrained_source())
        if self.resume:
            if weights is None:
                raise RuntimeError("Resume requires the raw training model")
            if getattr(weights, "tiny_optimization", None) != self.optimization:
                raise RuntimeError("Resume optimization configuration changed")
            state = weights.float().state_dict() if isinstance(weights, nn.Module) else weights
            model.load_state_dict(state, strict=True)
        else:
            if report is None:
                raise RuntimeError("A local pretrained checkpoint is required")
            self.transfer_report = report.to_dict()
        return model

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        # The inherited performance setup installs the native chunked assigner.
        # Rebuild criteria afterward for BOTH raw and EMA validation copies.
        for model in [de_parallel(self.model)] + ([de_parallel(self.ema.ema)] if self.ema else []):
            model.criterion = model.init_criterion()


def ablation_options():
    """Calibration m/n are deliberately absent until train-only calibration."""
    return {
        "baseline": {},
        "aux_control": {"auxiliary": True},
        "hbs": {"auxiliary": True, "hbs": True},
        "api": {"auxiliary": True, "api": True},
        "set": {"auxiliary": True, "hbs": True, "api": True},
        "simd": {"simd": True},
        "set_simd": {"auxiliary": True, "hbs": True, "api": True, "simd": True},
    }
