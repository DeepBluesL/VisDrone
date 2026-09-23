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
