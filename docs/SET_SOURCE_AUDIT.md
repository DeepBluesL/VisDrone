# SET source and adaptation audit

Audit date: 2026-09-27.

## Primary sources

- Huixin Sun, Runqi Wang, Yanjing Li, Linlin Yang, Shaohui Lin, Xianbin Cao,
  and Baochang Zhang, “[SET: Spectral Enhancement for Tiny Object
  Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html),”
  CVPR 2025, pp. 4713–4723.
- The local paper is `.runtime/references/SET_CVPR2025.pdf`, SHA-256
  `25af910ba1a4276894f65dc06bcb0fd5603aad6b1942174b3dfbef2065fadf5f`.
  Equations were also checked visually on rendered pages 3–5.
- The authors subsequently released the
  [official SET repository](https://github.com/HuixinSun/SET). This audit pins
  commit [`9208fbc4cfe571be4c15dccad8db1665cfdcb9d6`](https://github.com/HuixinSun/SET/tree/9208fbc4cfe571be4c15dccad8db1665cfdcb9d6)
  from 2026-06-16. The core file is
  [`fcos_set.py`](https://github.com/HuixinSun/SET/blob/9208fbc4cfe571be4c15dccad8db1665cfdcb9d6/mmdet/models/detectors/fcos_set.py)
  (blob `08a35240814b0dda3a1d20505b692d84133eeec2`); its AI-TOD
  [configuration](https://github.com/HuixinSun/SET/blob/9208fbc4cfe571be4c15dccad8db1665cfdcb9d6/configs/aitod/fcos_r50_set.py)
  is blob `e53993044039f1e36aa109d9b736b9bee0443947`.
- At the pinned revision GitHub reports `license: null`, and a top-level
  `LICENSE` request returns 404. No official source is copied here; the local
  code independently implements the mathematical method under AGPL-3.0.

## Pinned author-code behavior

The official FCOS implementation extracts backbone/FPN features once. It calls
the same `self.bbox_head.forward_train` first on ordinary features and then on
SET features, so no separately parameterized auxiliary head exists. Main losses
remain in the result and three `loss_set_*` auxiliary entries are added.

The AI-TOD configuration uses five strides `[8, 16, 32, 64, 128]`, HBS
reduction 4, auxiliary `scale=1`, and perturbation factor 1 at every level. Its
classification, box, and centerness loss weights are all 1. The FCOS head sets
`norm_cfg=None`, so it has no head BatchNorm running buffers.

For one background feature tensor, the author HBS block computes
`x + Conv2(ReLU(Conv1(x)))`. Both convolutions use `bias=True`; ReLU is not
in-place and appears only between them. Kernel size is
`(floor(log2(stride)/2) * 2) + 1` with symmetric padding. The caller then
re-masks the entire smoothed result as background and adds the untouched
foreground, preventing convolution support from changing foreground cells.

For API, each feature level receives separate gradients from FCOS
classification, box, and centerness losses. `torch.norm(gradient)` produces one
scalar over the complete `B,C,H,W` tensor for that branch and level. After adding
`1e-12`, each gradient is normalized independently. The configured perturbation
is

```text
(g_box + g_cls + g_centerness) / 3 * 1.
```

Thus the three official branch coefficients are each `1/3`, not 1. The code
does not call `detach`, but `torch.autograd.grad` uses the default
`create_graph=False`, so the returned gradients and perturbation do not require
grad. Adding the perturbation to the feature retains the ordinary identity
gradient path for the auxiliary loss.

HBS and API appear only in `forward_train`. The inherited inference path never
calls the denoisers or auxiliary head forward. This supports a structural
training-only statement, not a measured latency claim.

## Author code and local YOLO mapping

| Element | Pinned FCOS code | Local YOLO adaptation |
| --- | --- | --- |
| Foreground mask | Writes integer-coordinate GT rectangles into an image-sized mask, then nearest-neighbor resizes each FPN mask | Rasterizes augmented normalized boxes directly by feature-cell overlap; tiny valid boxes receive at least one cell |
| HBS block | Background residual plus two biased convolutions with one intermediate ReLU; re-masks output before restoring foreground | Follows the same block and re-mask behavior after the source audit |
| API branches | FCOS `cls`, `bbox`, `centerness` | YOLO `box`, `cls`, `dfl` |
| API normalization | One L2 norm over all `B,C,H,W`, separately per branch and level | Same normalization domain, evaluated in FP32 with zero-gradient handling |
| API combination | Equal `1/3` coefficients and level factor 1 | Equal `1/3` branch weights and `rho=1` |
| Auxiliary loss | Same FCOS head twice; main plus separately named auxiliary losses, scale 1 | Same YOLO Detect head twice; main plus `lambda_aux=1` auxiliary loss |
| Inference | SET absent from test forward | `inference_detector` strips HBS/config/criterion from a copied detector |

The local cell-overlap mask is intentionally different. It protects tiny
ground truths that could disappear under image-mask nearest-neighbor resizing,
and avoids an image-sized mask, but it must be described as a YOLO adaptation.
Boxes must be supplied after geometric augmentation so masks align with current
features.

The local shared YOLO Detect head contains BatchNorm, unlike the official FCOS
head. The main forward updates running buffers once. The auxiliary forward uses
batch statistics without a second running-buffer update. This policy and the
`aux_control` arm are required to separate HBS/API effects from duplicated
supervision and shared-head reuse.

The official auxiliary-loss code multiplies returned components by the loss
object's configured weight and by `scale`. MMDetection normally applies the
loss weight inside the loss object as well, but every published weight is 1, so
this distinction does not affect the released configuration.

Official code passes `allow_unused=True` but immediately norms the returned
value; an actually unused feature would fail. Local code treats unused or zero
gradients as zero perturbations. FP32 norm evaluation, empty-gradient handling,
YOLO DFL mapping, feature-cell masks, and shared-head BatchNorm policy are local
choices absent from the paper and author FCOS code.

The source identity and author-code behavior are high-confidence. This project
does not claim exact paper reproduction because the dataset, detector, loss
branches, mask rasterization, and training engine differ.
