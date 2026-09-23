# New lightweight-module comparison: research record and preregistration

## Scope

This extension compares five lightweight feature operators against a freshly
trained YOLOv8n baseline on VisDrone. The candidates are GhostConv, FasterNet
partial convolution, StarNet's StarBlock, WTConv, and LSConv. They were chosen
to cover complementary mechanisms: cheap feature synthesis, partial-channel
spatial mixing, multiplicative feature interaction, multiresolution wavelet
mixing, and spatially varying small-kernel aggregation.

This is a bounded candidate set, not a claim that these are the five newest
modules published by September 2026. The comparison includes recent candidates
where they were sufficiently reproducible, but it also retains older,
well-defined lightweight operators when they provide a cleaner controlled
intervention.

This document preregisters the extension before its accuracy results are
reported. It does not state or imply that any arm has completed training or
outperforms another arm. Actual results, after verification, belong in the
project README and generated result artifacts.

## Fixed experiment

The experiment has six arms:

1. `baseline`: a fresh YOLOv8n initialized from scratch.
2. `ghostconv`: GhostConv in the four backbone downsampling locations.
3. `pconv`: FasterBlock inside the three backbone C2f stages.
4. `star`: StarBlock inside the same C2f stages.
5. `wtconv`: WTBlock inside the same C2f stages.
6. `lsconv`: LSConv inside the same C2f stages.

Every arm is run with seeds **179, 2026, and 3407**, for 18 planned runs in
total. Each run uses the same fixed 2,048-image training subset and all 548
official validation images, 10 epochs, 512 x 512 inputs, batch size 16, AdamW,
and training from scratch. Dataset preparation, augmentation, optimizer
settings, validation procedure, checkpoint selection, and reporting code are
held constant across arms. Comparisons are paired by seed; the primary summary
is the mean paired change from the fresh baseline across the three seeds.

The extension is intentionally a short-budget screening experiment. Its
results estimate behavior under this exact training budget and must not be
presented as converged VisDrone performance, an official VisDrone leaderboard
score, or evidence of deployment speed on a different runtime.

## Interventions

### GhostConv

`ghostconv` uses the GhostConv class already migrated into this project. At
backbone graph indices 1, 3, 5, and 7, it replaces the original stride-2
convolution with a ratio-2 GhostConv: a primary 3 x 3 convolution followed by a
depthwise 5 x 5 cheap-feature branch. The replacement uses explicit SiLU and
copies the baseline BatchNorm epsilon and momentum. Input/output channels,
stride, graph connectivity, and activation family therefore remain aligned
with the corresponding baseline locations.

This intervention is distinct from the earlier `ghost`/C3Ghost experiment.
The earlier arm replaced feature-processing stages at indices 4, 6, and 8;
this arm tests GhostConv specifically in backbone downsampling convolutions.

Primary paper: [GhostNet: More Features from Cheap Operations (CVPR
2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Han_GhostNet_More_Features_From_Cheap_Operations_CVPR_2020_paper.html).
The migrated source attribution is recorded in
[`ATTRIBUTION.md`](ATTRIBUTION.md).

### FasterNet partial convolution

`pconv` preserves each original C2f outer projection (`cv1` and `cv2`) and the
original internal depths 2, 2, and 1 at graph indices 4, 6, and 8. Only each
internal unit is replaced with a FasterBlock. The local block applies a 3 x 3
convolution to the first quarter of the channels, concatenates the untouched
three quarters, then applies a 2x pointwise FFN with BatchNorm, ReLU, projection,
and a residual connection. DropPath and layer scale are disabled.

Primary paper: [Run, Don't Walk: Chasing Higher FLOPS for Faster Neural
Networks (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.html).
The algorithm was cross-checked against the official FasterNet repository at
commit [`e8fba4465ae912359c9f661a72b14e39347e4954`](https://github.com/JierunChen/FasterNet/tree/e8fba4465ae912359c9f661a72b14e39347e4954).
No license file or license declaration was present there at the audited commit.
The local block is an independent paper-based implementation; no FasterNet
source is vendored and this project does not describe that upstream code as
MIT-licensed.

### StarBlock

`star` uses the same C2f-preserving wrapper and depths as `pconv`. Each internal
unit follows the original StarNet block ordering: a 7 x 7 depthwise convolution,
two pointwise branches expanded by 4x, ReLU6 on one branch, elementwise product,
pointwise projection, a second 7 x 7 depthwise convolution, and a residual
connection. DropPath and external model-registration dependencies are removed.

Primary paper: [Rewrite the Stars (CVPR
2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Rewrite_the_Stars_CVPR_2024_paper.html).
The adaptation is pinned to the official implementation at commit
[`c999eb50840a44f9f1d92e8f7d2c22cd645a6d5e`](https://github.com/ma-xu/Rewrite-the-Stars/tree/c999eb50840a44f9f1d92e8f7d2c22cd645a6d5e).
The upstream license is Apache-2.0 and its retained notice is
[`licenses/Rewrite-the-Stars-Apache-2.0.txt`](../licenses/Rewrite-the-Stars-Apache-2.0.txt).

### WTConv

`wtconv` again preserves the C2f outer projections and depths. Its internal
WTBlock uses a two-level db1/Haar WTConv with 5 x 5 depthwise convolutions on
the base feature and wavelet bands. The Haar analysis and synthesis filters are
fixed buffers. Odd feature dimensions are padded before analysis and cropped
after reconstruction. The spatial output enters a Faster-style 2x pointwise
FFN and an outer residual connection.

Primary paper: [Wavelet Convolutions for Large Receptive Fields (ECCV
2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07137.pdf).
The implementation is adapted from the official WTConv repository at commit
[`4a84da2218924b3973de2235d641f2ecf969c346`](https://github.com/BGU-CS-VIL/WTConv/tree/4a84da2218924b3973de2235d641f2ecf969c346).
The upstream license is MIT and is retained in
[`licenses/WTConv-LICENSE`](../licenses/WTConv-LICENSE).

### LSConv

`lsconv` preserves the same C2f shell and depths but replaces each internal
unit with the paper-based LS operator and adds no separate FFN. The large-kernel
perception path reduces channels by half, applies a depthwise 7 x 7
convolution, and predicts spatially varying 3 x 3 aggregation weights. There
are `C/8` kernel channels; input channel `c` uses kernel `c mod (C/8)`, so each
kernel is shared cyclically by eight channels. GroupNorm normalizes the
predicted weights, ordinary PyTorch `unfold` performs the dynamic aggregation,
and BatchNorm plus a residual connection produces the output.

Primary paper: [LSNet: See Large, Focus Small (CVPR
2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_LSNet_See_Large_Focus_Small_CVPR_2025_paper.html).
The equations and channel-sharing convention were cross-checked against the
official repository at commit
[`cbe737c92b7c43ecf02d08545a07f03f1010177c`](https://github.com/THU-MIG/lsnet/tree/cbe737c92b7c43ecf02d08545a07f03f1010177c).
That repository had no license file or license declaration at the audited
commit. No LSNet source or Triton kernel is vendored, and the project does not
claim the official source is MIT-licensed. The local implementation expresses
the paper's operation independently with standard autograd-compatible PyTorch.

## What the comparison can establish

The five arms do not isolate pure operator effects. `ghostconv` changes four
downsampling layers, whereas the other four arms change internal units in three
C2f stages. FasterBlock, StarBlock, and WTBlock include different pointwise
FFN or gating structures; LSConv intentionally has no extra FFN. Consequently,
the arms differ in placement, trainable capacity, normalization, nonlinearities,
residual topology, memory traffic, and arithmetic count as well as in their
named spatial operator.

The valid question is therefore: **which complete, explicitly specified
detector intervention performs best under the fixed short-budget protocol?**
Results cannot support a causal claim that PConv, star multiplication, wavelet
decomposition, or dynamic aggregation alone caused a difference. Parameter
counts, counted Conv/Linear FLOPs, and measured latency should accompany
accuracy so that capacity and systems tradeoffs remain visible. In particular,
the unfolded LSConv implementation is mathematically aligned with the fused
operator but has different memory and latency behavior, and theoretical FLOPs
do not determine Windows/PyTorch throughput.

## Verification before accepting results

Before a run enters the comparison, verification must confirm the requested
variant and seed in the serialized model, the exact replacement locations and
depths, channel and spatial-shape preservation, finite forward/backward values,
and inference-time Conv-BatchNorm fusion where applicable. The implementation
has already passed basic spatial-block parity/gradient tests and fusion tests;
these are implementation checks rather than training results. Completed runs
must additionally satisfy the repository's artifact, dataset-count, epoch,
metric, and checkpoint verification.

No post-hoc arm removal or hyperparameter tuning is planned from intermediate
validation accuracy. If a run fails for a technical reason, the failure and
any rerun must be recorded. Any protocol change must be documented before the
affected results are interpreted.

## Candidates examined but excluded

- **[ConvNeur, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Efficiency_Follows_Global-Local_Decoupling_CVPR_2026_paper.html)**
  was examined from its [official Apache-2.0
  repository](https://github.com/ZhenyuYang01/ConvNeur).
  It combines a local path with chunked neural memory and gated global-to-local
  modulation. It is a stage-level backbone design with stateful memory updates,
  fixed chunking choices, and extra `einx`/`tensordict` dependencies, rather
  than a small convolutional unit that can be exchanged under the shared C2f
  wrapper. Detection code was also absent from the initial two-commit release.
  It was retained as a research lead, not forced into this comparison.
- **The [official LSNet SKA execution
  kernel](https://github.com/THU-MIG/lsnet/blob/cbe737c92b7c43ecf02d08545a07f03f1010177c/model/ska.py)**
  uses custom Triton forward and
  backward kernels. Native Windows support and export portability would make
  it an uncontrolled systems dependency here. The experiment instead uses a
  transparent mathematical equivalent based on `torch.nn.functional.unfold`,
  while explicitly accepting different memory and latency characteristics.
- **[ShiftwiseConv, CVPR
  2025](https://openaccess.thecvf.com/content/CVPR2025/html/Li_ShiftwiseConv_Small_Convolutional_Kernel_with_Large_Kernel_Effect_CVPR_2025_paper.html)**
  depends on a custom ShiftAdd CUDA extension in its [official
  repository](https://github.com/lidc54/shift-wiseConv), and
  its official environment targets old PyTorch/CUDA versions. That makes a
  clean native-Windows RTX 5090 comparison substantially riskier than the
  standard-PyTorch arms.
- **[SET, CVPR
  2025](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html)**
  targets tiny-object detection but its [official
  repository](https://github.com/HuixinSun/SET) provides a training framework
  built around an older MMDetection/MMCV stack rather than a small matched
  feature block. Its official repository also lacked a clear source license at
  review time.
- **[UniConvNet, ICCV
  2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_UniConvNet_Expanding_Effective_Receptive_Field_while_Maintaining_Asymptotically_Gaussian_Distribution_ICCV_2025_paper.html)**
  is a complete backbone family rather than a compact plug-in, and its
  [official repository](https://github.com/ai-paperwithcode/UniConvNet) listed
  downstream detection transfer as forthcoming when reviewed.

These exclusions reflect comparability, implementation maturity, licensing,
and the chosen Windows/PyTorch execution envelope. They are not claims that the
excluded methods are inferior.

## Machine-readable provenance

Exact source files, commits, and hashes used for the source audit are recorded
in [`classic_module_sources.json`](classic_module_sources.json),
[`wtconv_sources.json`](wtconv_sources.json), and
[`lsconv_sources.json`](lsconv_sources.json). These
records establish what was consulted; they do not broaden any upstream
license.
