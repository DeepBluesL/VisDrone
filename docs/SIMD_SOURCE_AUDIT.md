# SimD primary-source audit

Audit date: 2026-09-27. “SimD” here means **Similarity Distance**, not CPU
single-instruction/multiple-data execution and not the similarly named dataset.

## Identity and primary sources

- Shuohao Shi, Qiang Fang, Xin Xu, and Tong Zhao, “Similarity Distance-Based
  Label Assignment for Tiny Object Detection,” *2024 IEEE/RSJ International
  Conference on Intelligent Robots and Systems (IROS)*, pp. 13711–13718.
  The [IEEE record](https://ieeexplore.ieee.org/document/10801448) is the
  publication record. The author-hosted [arXiv v3 HTML](https://arxiv.org/html/2407.02394v3)
  contains selectable equations and reports that v3 was posted 2024-07-26;
  the work was accepted at IROS 2024. The arXiv manuscript is CC BY 4.0.
- The authors identify [cszzshi/SimD](https://github.com/cszzshi/SimD) as the
  official implementation. This audit pins code to commit
  [`2b16d4ea2f823e0cb8a0c87a89f7f0296be53a83`](https://github.com/cszzshi/SimD/tree/2b16d4ea2f823e0cb8a0c87a89f7f0296be53a83)
  (2026-03-16) rather than relying on a moving branch. Its top-level license is
  Apache-2.0 and retains the OpenMMLab copyright notice.

Confidence is high for the paper identity, equations, thresholds, and published
VisDrone configuration. Confidence is low that the public repository can
reproduce the reported numbers without repair and missing preprocessing, for
the code issues documented below.

## Algorithm

For ground-truth box `g` and anchor box `a`, both represented by center
coordinates `(x, y)`, width `w`, and height `h`, paper Eqs. 1–3 define

```text
location = sqrt(((xg-xa) / ((wg+wa)/m))^2
              + ((yg-ya) / ((hg+ha)/n))^2)

shape    = sqrt(((wg-wa) / ((wg+wa)/m))^2
              + ((hg-ha) / ((hg+ha)/n))^2)

SimD(g,a) = exp(-(location + shape)).
```

The result is one for identical boxes and approaches zero as location or shape
diverge. Unlike IoU, it remains informative for disjoint boxes.

Paper Eqs. 4–5 do not make `m` and `n` learned neural-network parameters. They
are deterministic train-set statistics over every ground-truth/anchor pair in
each image:

```text
m = mean(|x_gt - x_anchor| / (w_gt + w_anchor))
n = mean(|y_gt - y_anchor| / (h_gt + h_anchor)).
```

They therefore depend on the training images **and on the detector's anchor
generator**. Values from another dataset, resize pipeline, or anchor layout are
not portable without an ablation.

The paper's MaxSimDAssigner replaces the similarity calculation in
MaxIoUAssigner and keeps its hard assignment logic. Its published thresholds
are positive `0.7`, negative `0.3`, and minimum-positive `0.3`. Following
MMDetection's default `gt_max_assign_all=True` low-quality matching, every
ground truth force-matches all of its tied best anchors when that maximum clears
the minimum-positive threshold. This step runs for every ground truth and may
overwrite an anchor assignment made by the earlier threshold pass.

## Published VisDrone protocol

The paper and [official VisDrone config](https://github.com/cszzshi/SimD/blob/2b16d4ea2f823e0cb8a0c87a89f7f0296be53a83/configs/simd/visdrone/faster-rcnn_r50_fpn_1x_sim_visdrone.py)
use Faster R-CNN with a ResNet-50 FPN backbone pretrained on ImageNet. The
configuration trains for 12 epochs with batch size 2, SGD (`lr=0.005`,
`momentum=0.9`, `weight_decay=0.0001`), and learning-rate drops at epochs 8 and
11. It keeps 3,000 RPN proposals.

Only the **RPN** assigner changes to `MaxSiMAssigner`; the second-stage RCNN
assigner remains `MaxIoUAssigner`. Evaluation keeps ordinary IoU NMS with
threshold 0.7 for RPN and 0.5 for RCNN. Although the paper describes SimD NMS
as possible, its experimental settings explicitly use IoU NMS, so SimD NMS is
not part of the reported VisDrone comparison.

## Official-code reproducibility findings

1. [`bbox_overlaps.py`](https://github.com/cszzshi/SimD/blob/2b16d4ea2f823e0cb8a0c87a89f7f0296be53a83/mmdet/structures/bbox/bbox_overlaps.py)
   asserts that `mode` is one of `iou`, `iof`, or `giou` before reaching its
   `if mode == 'sim'` branch. [`BboxSiM2D`](https://github.com/cszzshi/SimD/blob/2b16d4ea2f823e0cb8a0c87a89f7f0296be53a83/mmdet/models/task_modules/assigners/sim2d_calculator.py)
   calls this function with `mode='sim'` by default. At the pinned commit, the
   advertised path therefore raises an assertion unless that assertion is
   repaired.
2. The same file hardcodes `x=6.13` and `y=4.59` where the equations require
   train-set/anchor statistics `m` and `n`. Repository search found no script
   that derives those values and no dataset-specific parameter injection. The
   VisDrone configuration uses the same global implementation. Their exact
   provenance cannot be verified from the released code.
3. [`sim_ota_assigner.py`](https://github.com/cszzshi/SimD/blob/2b16d4ea2f823e0cb8a0c87a89f7f0296be53a83/mmdet/models/task_modules/assigners/sim_ota_assigner.py)
   is the ordinary SimOTA implementation and defaults to `BboxOverlaps2D`.
   It does not connect SimD by itself.

The local [`simd.py`](../visdrone_migration/simd.py) is an independent
implementation of the published equations. It requires explicit `m` and `n`,
offers a streaming estimator, rejects non-finite and degenerate boxes, and
implements the documented MaxSimD hard-assignment rule. It does not copy the
official implementation or its unexplained constants.

## Boundary for a YOLOv8 adaptation

Ultralytics YOLOv8 is anchor-free and uses Task-Aligned Assignment (TAL). TAL
ranks candidates with a joint classification/alignment score and top-k logic;
it does not run the RPN MaxIoUAssigner evaluated in the SimD paper. Replacing
TAL's IoU overlap term with SimD would be a new adaptation, not a faithful
reproduction of the published method.

There is a second mismatch: the paper estimates `m,n` from fixed, sized
anchors. YOLOv8 anchor points have locations and strides but no box widths and
heights. Computing SimD from detached decoded predictions, or constructing
stride-sized pseudo-boxes around anchor points, changes the method. The latter
also introduces a box-scale choice. Either design must be declared and ablated.

Recommended experiment order:

1. Keep CIoU/DFL regression and IoU NMS unchanged. SimD is an assignment
   similarity in the validated experiment, not a regression loss.
2. Establish the unmodified TAL baseline.
3. Add a separate experimental flag that replaces only TAL's overlap factor,
   while retaining the classification term, top-k selection, and all losses.
4. Fit `m,n` using exactly the box representation selected for that experiment
   and the training split only. Save the values and the data/anchor fingerprint.
5. Report positive counts and AP by size along with total AP. Do not attribute
   a result to the IROS method without stating the anchor-free adaptation.

The planned local integration is therefore named **SimD-TAL adaptation**. A
`ChunkedTaskAlignedAssigner` subclass replaces only its aligned IoU calculation
with `bbox_simd(..., aligned=True, validate=False)`. Native TAL behavior remains:
top-k 10, `alpha=0.5`, `beta=6`, inside-ground-truth candidate filtering,
conflict resolution, and quality-target construction. CIoU regression, DFL,
and IoU NMS stay unchanged. The normalizers are checked once when the assigner
is constructed, so the hot-path `validate=False` avoids tensor-value checks
that would synchronize the GPU.

Because YOLO has no fixed sized anchors, this adaptation will calibrate `m,n`
from decoded predictions of a frozen pretrained detector on the **training
split only**. No validation boxes and no guessed or author-repository constants
may enter calibration. This makes the box representation concrete and avoids
statistics changing during training, but it remains a deliberate departure
from the paper's fixed-anchor definition. The calibration manifest must record
the frozen checkpoint hash, train-data fingerprint, preprocessing, image size,
and resulting pair count/values. No SimD arm should start unless explicit
calibrated `m,n` are present.

For the current reference API, `m` and `n` deliberately have no defaults. The
assignment thresholds default to the paper's `0.7/0.3/0.3`. The constants
`6.13/4.59` may be supplied only to diagnose the pinned official code; they are
not recommended defaults for this VisDrone/YOLOv8 pipeline.
