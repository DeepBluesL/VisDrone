# SPDX-License-Identifier: AGPL-3.0-only
"""Detection scaffold with explicit, serializable MNIST module interventions."""
from __future__ import annotations

from copy import deepcopy
import torch
from torch import nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import fuse_conv_and_bn
from ultralytics.nn.modules.block import C2f as DetectorC2f
from .modules import C3k2, C3Ghost, Conv, EnhancedSPPF, GaborStem, SEBlock, GhostConv
from .classic_blocks import FasterBlock, StarBlock
from .spatial_blocks import WTBlock, LSConv

FACTORS = ("c3k2", "ghost", "se", "sppf", "gabor")
VARIANTS = {"baseline": (), **{name: (name,) for name in FACTORS}, "full": FACTORS,
            **{f"without_{name}": tuple(x for x in FACTORS if x != name) for name in FACTORS},
            "random_stem": ("random",)}
NEW_VARIANTS = ("ghostconv", "pconv", "star", "wtconv", "lsconv")
VARIANTS.update({name: (name,) for name in NEW_VARIANTS})


class EfficientC2f(DetectorC2f):
    """Preserve the YOLO C2f projections/depth and replace its internal units."""
    def __init__(self, original, kind):
        nn.Module.__init__(self)
        self.kind = kind
        self.c = original.c
        self.cv1, self.cv2 = original.cv1, original.cv2
        block = {"pconv": FasterBlock, "star": StarBlock,
                 "wtconv": WTBlock, "lsconv": LSConv}[kind]
        self.m = nn.ModuleList(block(self.c) for _ in original.m)


class AttentionStage(nn.Module):
    """Keep the original graph index and original block state_dict names in base."""
    def __init__(self, base, channels):
        super().__init__()
        self.base = base
        self.attention = SEBlock(channels)

    def forward(self, x):
        return self.attention(self.base(x))


class MigratedDetectionModel(DetectionModel):
    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=10, verbose=False,
                 variant="baseline", module_seed=179):
        if isinstance(cfg, dict):
            cfg = deepcopy(cfg)
            variant = cfg.get("migration_variant", variant)
            module_seed = cfg.get("migration_module_seed", module_seed)
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}")
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.variant = variant
        self.yaml["migration_variant"] = variant
        self.yaml["migration_module_seed"] = module_seed
        factors = VARIANTS[variant]

        def replace(index, factory):
            old = self.model[index]
            # Per-location seeds make each intervention identical in single/full arms.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(module_seed + 1009 * index)
                new = factory(old)
            for key in ("i", "f"):
                setattr(new, key, getattr(old, key))
            new.type = f"{type(new).__module__}.{type(new).__name__}"
            new.np = sum(p.numel() for p in new.parameters())
            self.model[index] = new

        if "gabor" in factors or "random" in factors:
            replace(0, lambda old: GaborStem(ch, old.conv.out_channels,
                    mode="random" if "random" in factors else "gabor", seed=module_seed))
        if "c3k2" in factors:
            replace(2, lambda old: C3k2(old.cv1.conv.in_channels,
                    old.cv2.conv.out_channels, n=len(old.m)))
        if "ghost" in factors:
            for index in (4, 6, 8):
                replace(index, lambda old: C3Ghost(old.cv1.conv.in_channels,
                        old.cv2.conv.out_channels, n=len(old.m)))
        if "sppf" in factors:
            replace(9, lambda old: EnhancedSPPF(old.cv1.conv.in_channels,
                    old.cv2.conv.out_channels))
        if "se" in factors:
            replace(2, lambda old: AttentionStage(old, old.cv2.conv.out_channels))

        if "ghostconv" in factors:
            def downsample_ghost(old):
                block = GhostConv(old.conv.in_channels, old.conv.out_channels,
                                  k=old.conv.kernel_size[0], s=old.conv.stride[0], act=nn.SiLU())
                # Match the unchanged YOLO activation and BatchNorm configuration.
                for submodule in block.modules():
                    if isinstance(submodule, nn.BatchNorm2d):
                        submodule.eps, submodule.momentum = old.bn.eps, old.bn.momentum
                return block
            for index in (1, 3, 5, 7):
                replace(index, downsample_ghost)
        if variant in ("pconv", "star", "wtconv", "lsconv"):
            for index in (4, 6, 8):
                replace(index, lambda old: EfficientC2f(old, variant))

    def fuse(self, verbose=True):
        # Upstream recognizes only its own Conv class. Apply the same inference
        # optimization to migrated Conv blocks for comparable validation timing.
        super().fuse(verbose=False)
        for module in self.model.modules():
            if isinstance(module, Conv) and hasattr(module, "bn"):
                module.conv = fuse_conv_and_bn(module.conv, module.bn)
                delattr(module, "bn")
                module.forward = module.forward_fuse
        # New reference blocks express Conv-BN pairs as ordinary Sequential
        # modules. Give them the same inference fusion as the original arms.
        for sequence in tuple(self.model.modules()):
            if isinstance(sequence, nn.Sequential):
                children = list(sequence._modules.items())
                for (conv_key, conv), (bn_key, bn) in zip(children, children[1:]):
                    if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                        sequence._modules[conv_key] = fuse_conv_and_bn(conv, bn)
                        sequence._modules[bn_key] = nn.Identity()
        if verbose:
            self.info(verbose=True)
        return self


class MigrationTrainer(DetectionTrainer):
    def __init__(self, *args, variant="baseline", **kwargs):
        self.migration_variant = variant
        super().__init__(*args, **kwargs)

    def validate(self):
        metrics = self.validator(self)
        metrics.pop("fitness", None)
        # Upstream combines AP50 and AP50-95; this experiment preregisters AP50-95.
        fitness = float(metrics["metrics/mAP50-95(B)"])
        if self.best_fitness is None or fitness >= self.best_fitness:
            self.best_fitness = fitness
            self.selected_epoch = self.epoch + 1
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = MigratedDetectionModel(cfg or "yolov8n.yaml", ch=self.data.get("channels", 3),
                    nc=self.data["nc"], variant=self.migration_variant,
                    module_seed=self.args.seed, verbose=False)
        if weights is not None:
            model.load(weights)
        return model
