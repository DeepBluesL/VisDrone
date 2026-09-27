# Sources and licensing

This project adapts the active feature-extraction modules from
[DeepBluesL/occluded-mnist-comparison](https://github.com/DeepBluesL/occluded-mnist-comparison),
audited at commit `eec8e3c8f1aa042ea9171581c1dab7e99926c03c`.
The historical model was co-developed in the Group 7 course project; individual
module authorship is not asserted. The original maintainer is Hongyi Liu.

The CSP/C3/C2f/C3k2, convolution, and Ghost blocks derive from
[Ultralytics](https://github.com/ultralytics/ultralytics/tree/v8.3.195/ultralytics/nn/modules).
The detection scaffold, head, loss, training engine, NMS and AP computation use
Ultralytics 8.3.195. Local changes retain the historical LeakyReLU activations.
SE is standard squeeze-and-excitation. EnhancedSPPF retains the project's parallel
3/5/7 pooling branches and per-branch residual additions.

The Ghost feature-generation reference is
[GhostNet](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnet_pytorch).
Its Apache-2.0 license and notice are retained in `licenses/`.
Code in this repository is distributed under GNU AGPL-3.0 (`LICENSE`), consistent
with the imported source. No originality claim is made for these building blocks.

The original Gabor-named stem was overwritten by Xavier initialization and then
frozen. This migration intentionally corrects it using a persistent analytic
filter buffer and supplies a normalized random-bank control. Therefore these
experiments do not reproduce the historical MNIST checkpoint.

VisDrone2019-DET comes from the
[VisDrone project](https://github.com/VisDrone/VisDrone-Dataset).
Downloads use archives referenced by the official repository and the
[Ultralytics dataset configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VisDrone.yaml).
Dataset rights and terms remain with their original owners and are separate from
the repository's code license. Original images and annotations are not bundled.
Download checksums and transformation manifests document the locally used files.

The additional lightweight-module study is specified in
[NEW_MODULES.md](NEW_MODULES.md). StarBlock is adapted from the Apache-2.0
StarNet implementation and HaarWTConv from the MIT WTConv implementation;
their license texts are retained in `licenses/`. FasterNet partial convolution
and LSConv are independent implementations of the published algorithms. The
audited FasterNet and LSNet repositories did not declare source-code licenses;
their source files are not redistributed. Pinned source URLs and audit hashes
are recorded in `classic_module_sources.json`, `wtconv_sources.json`, and
`lsconv_sources.json`. These experiments adapt operators into a YOLO detector
and do not reproduce the authors' complete networks or optimized runtimes.

The prospective training-only tiny-object study adapts two additional papers.
HBS and API follow [SET: Spectral Enhancement for Tiny Object Detection (CVPR
2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html).
The authors' [official repository](https://github.com/HuixinSun/SET) was audited
at commit
[`9208fbc4cfe571be4c15dccad8db1665cfdcb9d6`](https://github.com/HuixinSun/SET/tree/9208fbc4cfe571be4c15dccad8db1665cfdcb9d6).
It had no top-level license file at that revision, so none of its source code is
copied here. The local implementation independently expresses the mathematical
method for YOLO under this project's AGPL-3.0 license. Its source mapping is in
[`SET_SOURCE_AUDIT.md`](SET_SOURCE_AUDIT.md).

The SimD reference follows [Similarity Distance-Based Label Assignment for Tiny
Object Detection (IROS 2024)](https://arxiv.org/html/2407.02394v3). Its
[official repository](https://github.com/cszzshi/SimD) was audited at commit
[`2b16d4ea2f823e0cb8a0c87a89f7f0296be53a83`](https://github.com/cszzshi/SimD/tree/2b16d4ea2f823e0cb8a0c87a89f7f0296be53a83)
and carries an Apache-2.0 OpenMMLab license notice. No upstream implementation
is vendored. The local formula, streaming estimator, and YOLO TAL adaptation
were independently implemented under AGPL-3.0; known upstream issues and the
anchor-based-to-anchor-free boundary are in
[`SIMD_SOURCE_AUDIT.md`](SIMD_SOURCE_AUDIT.md).
