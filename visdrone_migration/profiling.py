"""Explicit convolution/linear FLOP estimate, including functional Gabor filters."""
from copy import deepcopy
import torch
from torch import nn
from .modules import GaborStem
from .spatial_blocks import HaarWTConv, DynamicSmallKernel


def convolution_linear_gflops(model, imgsz):
    """Count 2 FLOPs per multiply-accumulate at batch 1; omit other operations."""
    candidate = deepcopy(model).cpu().float().eval()
    operations = []
    hooks = []

    def count(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            operations.append(2 * output.numel() * (module.in_channels // module.groups)
                              * module.kernel_size[0] * module.kernel_size[1])
        elif isinstance(module, nn.Linear):
            operations.append(2 * output.numel() * module.in_features)
        elif isinstance(module, GaborStem):
            # Projection Conv/BN are separate children; only count fixed F.conv2d here.
            n, c, h, w = inputs[0].shape
            operations.append(2 * n * 32 * ((h + 1) // 2) * ((w + 1) // 2) * c * 5 * 5)
        elif isinstance(module, HaarWTConv):
            n, c, h, w = inputs[0].shape
            for _ in range(module.levels):
                h, w = (h + 1) // 2, (w + 1) // 2
                # Both fixed 2x2 analysis and transposed-convolution synthesis:
                # 4 bands/channel, 4 coefficients/band, 2 FLOPs/MAC.
                operations.append(2 * 2 * n * c * 4 * h * w * 4)
        elif isinstance(module, DynamicSmallKernel):
            operations.append(2 * output.numel() * 9)

    for module in candidate.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, GaborStem, HaarWTConv, DynamicSmallKernel)):
            hooks.append(module.register_forward_hook(count))
    try:
        with torch.no_grad():
            candidate(torch.zeros(1, 3, imgsz, imgsz))
    finally:
        for hook in hooks:
            hook.remove()
    return sum(operations) / 1e9
