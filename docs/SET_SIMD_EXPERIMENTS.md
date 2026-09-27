# Planned SET and SimD-TAL experiment

## Status and question

This is a prospective experiment design. The implementation has received
focused CPU verification, but **none of the seven arms has been trained** and no
GPU calibration, memory profile, accuracy measurement, or latency benchmark has
run for this study. All AP and latency results remain pending. The existing GPU
queue is paused and this document does not authorize a restart.

The study asks whether two training-time changes improve tiny-object detection
without adding operators to the exported detector: a YOLO adaptation of SET
spectral enhancement and a YOLO Task-Aligned Assigner adaptation of SimD. These
are adaptations across detector families, not reproductions of the published
FCOS/Faster R-CNN experiments. Published paper AP must not be mixed with future
measurements from this repository. Primary-source audits are in
[`SET_SOURCE_AUDIT.md`](SET_SOURCE_AUDIT.md) and
[`SIMD_SOURCE_AUDIT.md`](SIMD_SOURCE_AUDIT.md).

## Seven-arm matrix

| Arm | Training intervention | Inference graph | Current results |
| --- | --- | --- | --- |
| `baseline` | Ordinary YOLOv8n loss and native chunked TAL | Standard detector | Not trained; AP and latency pending |
| `aux_control` | Second loss-bearing shared Detect-head forward on unchanged neck features | Standard detector after export | Not trained; AP and latency pending |
| `hbs` | Auxiliary forward after HBS background smoothing | Standard detector after export | Not trained; AP and latency pending |
| `api` | Auxiliary forward after API perturbation | Standard detector after export | Not trained; AP and latency pending |
| `set` | HBS followed by API in the auxiliary forward | Standard detector after export | Not trained; AP and latency pending |
| `simd` | SimD replaces only TAL's aligned overlap term | Standard detector | Blocked on train-only `m,n` calibration; AP and latency pending |
| `set_simd` | Combined SET auxiliary path and SimD-TAL assignment | Standard detector after export | Blocked on calibration; AP and latency pending |

The `aux_control` arm is required. HBS, API, and SET all add a second loss,
reuse the shared Detect head, and backpropagate through another head forward.
Without this control, a difference could arise from duplicated supervision,
head reuse, or the altered gradient path. In the auxiliary forward, the shared
head uses batch statistics but does not update its BatchNorm running buffers a
second time. The ordinary forward updates those buffers once per batch.

## SET adaptation

**HBS** builds foreground masks only from augmented training ground truths. At
each detection level, the author-code-aligned block applies a biased reduction
convolution, ReLU, biased expansion convolution, and background residual. It
then re-masks the result so foreground features remain exact. Kernel size grows
with stride and channel reduction is `r=4`. The local feature-cell-overlap mask
differs from the author's image-mask/nearest-resize path to keep tiny valid boxes
from disappearing; this is an explicit YOLO adaptation.

**API** obtains differentiable main-loss components for `box`, `cls`, and `dfl`.
For every branch and feature level it computes one L2 norm over the complete
`B,C,H,W` gradient tensor. The three independently normalized directions use
equal coefficients `1/3`, are summed, multiplied by `rho=1`, and detached before
the auxiliary forward. The auxiliary loss weight is `lambda_aux=1`. This matches
the pinned author code's averaging and first-order perturbation while mapping
FCOS centerness to YOLO DFL.

Because the norm covers the whole batch, the effective perturbation per image
depends on physical batch composition and size. All seven arms must therefore
use the same physical batch, currently proposed as 32. A smaller final batch is
unavoidable but is shared because every arm uses the same ordered dataset.
SET's peak memory is still unknown, so a future GPU memory profile must either
confirm batch 32 for every arm or re-freeze one lower physical batch for all
seven arms before training. Nominal batch size remains `nbs=64`.

The `set` arm applies HBS and API to captured neck features and sends them
through the shared Detect head. The optimized loss is the main loss plus
`lambda_aux` times the auxiliary loss. Validation uses the ordinary detector
path. `inference_detector` removes HBS modules, optimization configuration, and
training criterion from a deployment copy. The exported graph therefore has no
HBS/API operation or auxiliary forward. Real latency has not been measured, so
this structural property is not evidence of measured “zero overhead.”

Focused CPU structure checks measured the following parameter counts. These are
implementation facts, not training, AP, throughput, memory, or latency results.

| Structure | Parameters |
| --- | ---: |
| Standard YOLOv8n, 10 classes | 3,012,798 |
| SET training model with HBS | 4,055,790 |
| HBS-only addition during training | 1,042,992 |
| Detector returned by `inference_detector` | 3,012,798 |

API and SimD add no learned parameters; their criterion/assignment state is not
a trainable module. The stripped detector matched the native evaluation forward
exactly in the focused CPU check. GPU memory, AMP behavior, training throughput,
and real latency remain unmeasured.

## SimD-TAL adaptation

The IROS paper evaluates SimD in an anchor-based MaxIoU/MaxSimD procedure,
principally at a Faster R-CNN RPN. This project calls its intervention
**SimD-TAL adaptation**. It replaces only the aligned box-similarity calculation
inside the chunked YOLO Task-Aligned Assigner. Native TAL behavior remains:

- top-k 10, `alpha=0.5`, and `beta=6`;
- candidate centers must lie inside a ground-truth box;
- conflict resolution and quality-target construction are unchanged;
- CIoU/DFL regression and IoU NMS remain unchanged.

Preserving the inside-ground-truth mask means SimD-TAL cannot fix a tiny object
with no eligible candidate center. It changes the similarity and ranking of
already eligible candidates. It is neither a regression loss nor an NMS change.

The paper's `m,n` depend on fixed anchors, while YOLOv8 is anchor-free. This
adaptation defines them from all decoded pre-NMS prediction boxes produced by a
frozen COCO-initialized detector on every one of the 6,471 training images at
the frozen preprocessing and image size. Each image's predictions are paired
with its training ground truths. Validation and test images are prohibited.
The record must include checkpoint hash, data fingerprint, architecture, image
size, seed, box/pair counts, and resulting values.

The launcher freezes all seven configurations together, including the calibration
manifest and its exact SHA. It therefore requires a completed calibration before
the first arm can execute, even if that arm is the baseline.

Calibration has not run. No guessed values, author-repository constants, or
validation-derived values may be substituted. Both SimD arms must refuse to
launch until explicit positive calibrated `m,n` exist. Frozen initial
predictions make the estimator reproducible, but this remains a stated YOLO
adaptation rather than the paper's fixed-anchor estimator.

The SimD criterion is also installed on the validation model, so validation
loss for a SimD arm is computed with SimD-TAL while baseline/SET-only validation
loss uses native TAL. AP still comes from the same native inference, decoding,
and IoU NMS path. Validation loss is therefore a within-arm diagnostic and must
not be compared directly across native-TAL and SimD-TAL arms; AP remains the
cross-arm accuracy measure.

## Prospective screening protocol

The planned screen uses standard YOLOv8n at 768-pixel input, official COCO
initialization, all 6,471 training images, all 548 validation images, 100 epochs,
and seed 179. Standard/768 is a prospective controlled setting; it is not an
observed winner from the paused `realtime_stage1` study, whose first
standard/512 run stopped at epoch 31.

Physical batch 32 and `nbs=64` remain tentative until the SET memory profile.
Any necessary physical-batch change applies to all seven arms. No GPU work may
start while the user pause remains in force.

Seed 179 is for screening. Formal comparative conclusions require the same
frozen protocol at three prespecified seeds. One seed may motivate follow-up but
cannot establish a stable benefit. The full validation set serves model
selection and reporting, so results will be internal validation evidence rather
than official VisDrone test or challenge scores.

## Interpretation boundary

The completed pilot and module-extension studies found that named feature
modules, lower parameter counts, and lower accounted FLOPs did not reliably
improve AP or runtime. This motivates prioritizing training-only interventions
that can leave the deployment graph unchanged. It does not make SET or SimD
more likely to work: SET adds a label-dependent auxiliary path, and SimD changes
a prediction-dependent TAL ranking outside its published detector setting.

The matrix addresses major alternatives: `aux_control` measures the second loss
and shared-head reuse; `hbs` and `api` isolate SET components; `set` tests their
combination; `simd` isolates assignment; and `set_simd` tests interaction. A
positive screen remains a hypothesis until it repeats over the prespecified
three seeds and a controlled latency benchmark. Until authorized training is
complete, every arm is “not trained; AP and latency pending.”

## Entry points and verification

These commands only print plans; they do not initialize CUDA or start training:

```powershell
.venv\Scripts\python.exe scripts/calibrate_simd.py
.venv\Scripts\python.exe scripts/train_tiny_optimization.py
```

Actual calibration and single-arm training require `--execute`. The current user
pause additionally requires `--resume-paused`, which is only for a future explicit
request to resume. Both training launchers share the same process lock. Completed
arms can use `scripts/report_realtime.py --suite tiny_optimization` and the common
batch-1 benchmarking script after execution is authorized. The five-arm stage-1
finalizer does not publish this seven-arm study.

The [implementation verification record](../results/tiny_optimization/implementation_verification.json)
contains CPU checks and structural parameter counts. It is not a trained-model
result or an accuracy/latency benchmark. Resumable raw/EMA/optimizer checkpoints
stay under `runs/`; deployment exports are separately stripped from the selected
best EMA, with distinct hashes and parameter counts.
