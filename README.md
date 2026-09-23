# VisDrone module migration experiments

This repository studies whether feature-extraction modules from an occluded-MNIST
project transfer usefully to ten-class object detection on VisDrone2019-DET. The
detector uses the YOLOv8n training, detection-head, localization, classification,
and distribution-focal-loss machinery. The MNIST comparison head, global pooling,
left/right label swapping, and feature-norm ranking loss were intentionally left
behind because they do not define a corresponding object-detection intervention.

The repository contains two experiment generations:

- The completed 13-arm `pilot` migration study evaluates the modules that were
  active in the original course model, plus ablations and a matched random-filter
  control.
- The six-arm `module_extension` study compares five additional efficient-module interventions with
  a freshly trained baseline across three model seeds. All 18 runs are complete.

Implementation details for the extension modules are documented in
[NEW_MODULES.md](docs/NEW_MODULES.md). Source attribution and licenses are recorded
in [ATTRIBUTION.md](docs/ATTRIBUTION.md),
[classic_module_sources.json](docs/classic_module_sources.json), and
[WTConv](docs/wtconv_sources.json) and [LSConv](docs/lsconv_sources.json) source
records.

## Completed pilot: protocol and results

All 13 pilot runs completed under one fixed protocol: 2,048 training images, all
548 validation images, 10 epochs, 512-pixel input, batch size 16, seed 179, and
training from scratch. The downloaded source data contain all 8,629 images across
train, validation, and test-dev; test-dev was not used for training, checkpoint
selection, or reported metrics. Dataset provenance is stored in
[source_provenance.json](results/data/source_provenance.json).

AP values below are percentages. The prespecified primary metric is validation
AP50-95.

| Pilot arm | AP50 | AP50-95 | Parameters |
| --- | ---: | ---: | ---: |
| Baseline | 9.643 | 4.471 | 3,012,798 |
| C3k2 | 9.145 | 4.249 | 3,010,478 |
| Ghost | 8.484 | 3.859 | 2,499,310 |
| SE | 9.447 | 4.386 | 3,012,960 |
| Enhanced SPPF | 9.159 | 4.200 | 3,012,798 |
| Gabor stem | 10.296 | 4.866 | 3,012,878 |
| Full five-module model | 8.764 | 4.044 | 2,497,232 |
| Full minus C3k2 | 9.083 | 4.194 | 2,499,552 |
| Full minus Ghost | 8.506 | 4.119 | 3,010,720 |
| Full minus SE | 8.288 | 3.726 | 2,497,070 |
| Full minus Enhanced SPPF | 8.953 | 4.069 | 2,497,232 |
| Full minus Gabor | 8.421 | 3.853 | 2,497,152 |
| Matched random-filter stem | **10.491** | **4.991** | 3,012,878 |

The random-filter control produced the highest AP50-95 in this run. Gabor exceeded
the baseline by 0.395 percentage points, but the matched random filters exceeded
Gabor by 0.125 points. The evidence therefore does not isolate a benefit from the
Gabor prior itself. It suggests only that the fixed-filter stem intervention may
have helped under this particular short training budget.

The full model used about 17.1% fewer parameters and 10.7% fewer counted
convolution/linear FLOPs than the baseline, while its AP50-95 was 0.428 points
lower. In the leave-one-out runs, removing SE or Gabor reduced the full model's
score, while removing C3k2, Ghost, or Enhanced SPPF increased it slightly. These
are context-dependent interactions within the combined model, not independent
module effects.

The complete evidence is available in the [generated report](results/pilot/REPORT.md),
[summary CSV](results/pilot/summary.csv), [research findings](docs/FINDINGS.md),
[verification record](results/pilot/verification.json), and
[artifact manifest](results/artifact_manifest.json). The released checkpoints are
under `checkpoints/pilot/`.

![Baseline and completed single-module comparisons](assets/pilot/comparison.png)

![Full-model ablations and fixed-filter control](assets/pilot/ablation.png)

![Recall grouped by natural occlusion level](assets/pilot/occlusion_recall.png)

![Predictions from the baseline, full model, and best pilot arm](assets/pilot/predictions.png)

## What was migrated

The pilot deliberately migrated feature extraction rather than the entire MNIST
task. Its interventions occupy different architectural locations:

| Factor | Detector intervention |
| --- | --- |
| C3k2 | Replace the early backbone C2f at layer 2 |
| Ghost | Replace backbone C2f stages at layers 4, 6, and 8 |
| SE | Apply channel attention after the layer-2 backbone block |
| Enhanced SPPF | Replace the backbone SPPF at layer 9 |
| Gabor | Replace the input stem with a fixed analytic filter bank plus learned projection |
| Random control | Use the same stem shape and normalization with fixed seeded random filters |

This placement difference is a real experimental confound: the pilot compares
complete interventions, not one interchangeable operator inserted at one common
site. Some variants also replace an entire C2f wrapper while others wrap or replace
only a stem or pooling stage. Parameter count, receptive field, normalization,
depth, and optimization path can therefore change together.

The extension study narrows this issue. PConv, Star, WTConv, and large-small
convolution blocks replace the internal units of otherwise retained C2f wrappers
at backbone layers 4, 6, and 8. The GhostConv arm instead replaces four stride-2
downsampling convolutions, so comparisons involving GhostConv still include a
placement and wrapper difference. See [NEW_MODULES.md](docs/NEW_MODULES.md) for the
exact definitions and source provenance.

## The original Gabor initialization bug

In the historical MNIST code, the Gabor convolution weights were created and
frozen, but a later generic model initializer applied Xavier initialization to
convolution modules. Setting `requires_grad=False` prevents optimizer updates; it
does not protect a tensor from an explicit initialization function. Consequently,
the layer described as Gabor no longer contained the intended analytic filters by
the time training began.

This migration corrects that problem by storing the analytic bank as a persistent
buffer and applying it with a functional convolution before a learned projection.
Generic module initialization and optimizers cannot overwrite that buffer. A
normalized fixed random bank with identical shape provides the necessary control.
Because this is a correction, the detector experiment does not reproduce the
historical MNIST checkpoint's actual computation.

## Completed module extension study

The prespecified matrix contains six arms: `baseline`, `ghostconv`, `pconv`,
`star`, `wtconv`, and `lsconv`. All 18 runs completed: each arm was trained from
scratch for 10 epochs at image size 512 and batch size 16, using model/training
seeds 179, 2026, and 3407. Every run used the same 2,048-image training subset,
selected once during data preparation with dataset-selection seed 179, and all
548 validation images. Changing a model seed did not resample the dataset.

AP values are percentages; `±` is the sample standard deviation across the three
model seeds. The paired delta subtracts the freshly trained baseline with the same
seed and is measured in percentage points.

The added operators come from GhostNet (CVPR 2020), FasterNet (CVPR 2023),
StarNet (CVPR 2024), WTConv (ECCV 2024), and LSNet (CVPR 2025). These are
explicit detector adaptations, not reproductions of their full published networks.
The [research record](docs/NEW_MODULES.md) links their papers and explains why
other candidates, including a CVPR 2026 architecture, were excluded.

| Extension arm | AP50 mean ± SD | AP50-95 mean ± SD | Paired ΔAP50-95 mean ± SD | Parameters | Reduction vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 9.540 ± 0.288 | 4.431 ± 0.149 | — | 3.013M | — |
| GhostConv downsampling | 8.703 ± 0.260 | 3.975 ± 0.175 | -0.457 ± 0.124 | 2.823M | 6.3% |
| PConv / FasterBlock | 8.772 ± 0.096 | 3.974 ± 0.087 | -0.457 ± 0.084 | 2.655M | 11.9% |
| StarBlock | 9.219 ± 0.128 | 4.200 ± 0.056 | -0.232 ± 0.195 | 2.888M | 4.1% |
| WTConv / WTBlock | 9.212 ± 0.146 | 4.192 ± 0.057 | -0.240 ± 0.120 | 2.715M | 9.9% |
| LSConv (large-small convolution) | 9.154 ± 0.058 | 4.208 ± 0.005 | -0.224 ± 0.147 | 2.578M | 14.4% |

All five paired deltas were negative at every seed. The per-seed AP50-95 deltas
were -0.60/-0.38/-0.39 for GhostConv, -0.52/-0.48/-0.36 for PConv,
-0.24/-0.42/-0.03 for StarBlock, -0.32/-0.30/-0.10 for WTConv, and
-0.27/-0.34/-0.06 for LSConv at seeds 179/2026/3407. Thus none of the additional
interventions exceeded the matched baseline under this short training budget.
LSConv had the highest mean AP50-95 among the five interventions and the largest
parameter reduction, while StarBlock had the highest mean AP50 among them. These
descriptions do not establish a converged ranking. LSConv's nominal AP50-95 lead
over StarBlock and WTConv was less than 0.016 percentage points, far too small to
support a meaningful ordering or stability claim from three seeds.

| Extension arm | Accounted GFLOPs | Validator inference mean ± SD (ms/image) |
| --- | ---: | ---: |
| Baseline | 5.179 | 0.274 ± 0.021 |
| GhostConv downsampling | 4.902 | 0.317 ± 0.063 |
| PConv / FasterBlock | 4.616 | 0.287 ± 0.043 |
| StarBlock | 5.011 | 0.269 ± 0.029 |
| WTConv / WTBlock | 4.648 | 0.359 ± 0.061 |
| LSConv (large-small convolution) | 4.508 | 0.538 ± 0.245 |

The accounted GFLOPs use each run's shared method: convolution and linear MACs
multiplied by two, plus fixed Haar analysis/synthesis and LSConv dynamic spatial
MACs. They exclude normalization, pooling, activation, elementwise gates and
scales, decoding, and NMS, so they are not total hardware FLOPs. Inference time is
the per-image inference-stage time reported by the batched Ultralytics validator
in the recorded RTX 5090 D and software environment. It is not isolated
single-image end-to-end latency or a deployment benchmark. The result illustrates
that lower accounted arithmetic did not guarantee lower measured time: LSConv had
the fewest accounted GFLOPs but the highest mean validator inference time, consistent
with the memory-heavy standard-PyTorch `unfold` implementation.
Timing SD describes variation between the three validator runs, not variation
between individual images or repeated controlled deployment measurements.

Fourteen of 18 runs selected epoch 10; baseline seed 3407, StarBlock seed 179,
WTConv seed 2026, and LSConv seed 2026 selected epoch 9. This concentration at the
budget boundary reinforces that the study measures early learning rather than
convergence. The comparatively small across-seed AP50-95 SD for LSConv,
StarBlock, and WTConv is descriptive across only three observations and is not a
95% confidence interval. Because all runs use one fixed subset, these SDs reflect
only training variation from initialization and randomized training operations;
they do not capture dataset-resampling or population-generalization uncertainty.

The baseline was rerun under the extension code. No archived checkpoint
was used for training. As a deterministic regression check, the fresh seed-179
baseline reproduced the archived pilot seed-179 metrics exactly, matched the data
fingerprint and parameter count, and matched all 355 checkpoint state tensors
exactly despite the changed source-code fingerprint. See the
[baseline regression record](results/module_extension/baseline_regression.json).

The four C2f-based arms are aligned in placement: PConv, StarBlock, WTBlock, and
LSConv replace internal units while preserving the C2f outer projections and
depths at backbone layers 4, 6, and 8. Their internal FFNs, gates, normalization,
capacity, and residual paths still differ. GhostConv changes stride-2 downsampling
convolutions at layers 1, 3, 5, and 7, so its placement and wrapper are not matched
to the other four. The results compare these complete detector interventions and
cannot attribute differences solely to partial convolution, star multiplication,
wavelets, or dynamic aggregation.

Full numerical evidence is available in the extension
[report](results/module_extension/REPORT.md),
[summary CSV](results/module_extension/summary.csv),
[verification record](results/module_extension/verification.json), and
[comparison](assets/module_extension/comparison.png),
[efficiency](assets/module_extension/efficiency.png), and
[paired-delta](assets/module_extension/paired_deltas.png) figures.

![Three-seed module comparison](assets/module_extension/comparison.png)

![Same-seed AP50-95 changes relative to the baseline](assets/module_extension/paired_deltas.png)

All 18 checkpoints were also evaluated on the same 548 raw validation images
for occlusion and size diagnostics. Of 38,759 eligible ground-truth boxes,
26,575 (68.6%) have clipped raw-image area below 32² pixels. Baseline recall
was 13.54% for these small objects, 52.79% for medium objects, and 72.57% for
large objects. By natural occlusion level, baseline recall was 36.19% without
occlusion, 20.44% with partial occlusion, and 9.84% with heavy occlusion.
These are three-seed means at confidence 0.05, NMS IoU 0.5, and class-aware
one-to-one matching IoU 0.5. They are ground-truth recall diagnostics, not
AP by size or official challenge metrics; size is measured before input resizing.

Some diagnostic slices improved slightly while overall AP declined. StarBlock's
small-object recall was 13.79%, versus the baseline's 13.54%; GhostConv's
heavy-occlusion recall was 10.19%, versus 9.84%. Neither observation establishes
an overall improvement or a specific causal benefit for occlusion handling.
Recall at one confidence threshold does not summarize precision, confidence
ranking, or localization across AP thresholds, and size and occlusion are not
independently controlled here.

![Recall grouped by clipped raw-image object size](assets/module_extension/size_recall.png)

The [occlusion plot](assets/module_extension/occlusion_recall.png),
[learning curves](assets/module_extension/learning_curves.png), and
[per-class AP plot](assets/module_extension/per_class_ap.png) provide the remaining
diagnostics. [Qualitative predictions](assets/module_extension/predictions.png)
compare all six arms at the fixed model seed 179 on the same three images,
selected independently of predictions. The
[selection record](results/module_extension/selection.json) and
[figure metadata](assets/module_extension/predictions.json) record this choice.

## Reproducing the experiments

Use the repository virtual environment explicitly. On Windows PowerShell:

```powershell
py -3.10 -m venv .venv
# Install a CUDA-enabled PyTorch build suitable for your GPU in this environment first.
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\prepare_data.py --train-limit 2048 --seed 179
```

The archived pilot was produced from source commit `af2026e`. Current core code
contains the extension modules, so it has a different training fingerprint and
must not be used to append runs to the archived `pilot` suite. Create an isolated
worktree at the recorded source commit and use a fresh suite name:

```powershell
git worktree add ..\VisDrone-pilot-reproduction af2026e
Push-Location ..\VisDrone-pilot-reproduction
py -3.10 -m venv .venv
# Install a CUDA-enabled PyTorch build suitable for your GPU first.
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\prepare_data.py --raw-root ..\VisDrone\data\raw --train-limit 2048 --seed 179
.\.venv\Scripts\python.exe scripts\run_suite.py --suite pilot_reproduction --epochs 10 --imgsz 512 --batch 16 --seeds 179 --expected-train 2048 --expected-val 548
.\.venv\Scripts\python.exe scripts\report.py --suite pilot_reproduction
Pop-Location
```

The extension suite uses the current source and a new archive name:

```powershell
.\.venv\Scripts\python.exe scripts\run_suite.py --suite module_extension_reproduction --variants baseline ghostconv pconv star wtconv lsconv --epochs 10 --imgsz 512 --batch 16 --seeds 179 2026 3407 --expected-train 2048 --expected-val 548
.\.venv\Scripts\python.exe scripts\postprocess_suite.py --suite module_extension_reproduction --check-current-code
```

The published extension run uses `module_extension`; reproduction uses
`module_extension_reproduction` so immutable evidence is never silently reused or
overwritten. `run_suite.py` verifies dataset counts and records the dataset YAML,
preparation manifest, data-content fingerprint, training-code fingerprint, and
environment versions. If any identity field differs, it refuses to reuse the
suite. `runs/` stores complete trainer output, `results/` stores metrics and epoch
CSVs, `checkpoints/` stores the selected validation checkpoints, and `assets/`
contains plots generated from recorded results.

To verify the archived pilot evidence and regenerate diagnostics:

```powershell
.\.venv\Scripts\python.exe scripts\verify_results.py --suite pilot --expected-runs 13
.\.venv\Scripts\python.exe scripts\visualize_predictions.py --weights checkpoints\pilot\baseline_s179.pt checkpoints\pilot\full_s179.pt checkpoints\pilot\random_stem_s179.pt --labels baseline full random_stem --output assets\pilot\predictions.png
.\.venv\Scripts\python.exe scripts\evaluate_occlusion.py --weights checkpoints\pilot\baseline_s179.pt --output results\pilot\baseline_s179\occlusion.json
```

The occlusion diagnostic uses confidence 0.05 and matching IoU 0.5. Image
selection does not depend on predictions. Run it separately for each checkpoint
and regenerate the report after all records are present.

To use all 6,471 training images for a later study, prepare the data with
`--train-limit 0` and choose another suite name. The 1,610 downloaded test-dev
images remain excluded unless a separate evaluation protocol explicitly uses
them.

## Interpretation and research reflection

The pilot is an engineering and hypothesis-generation study under a fixed small
budget. Ten epochs from scratch are not a convergence study. Twelve of the 13
pilot arms selected their checkpoint at epoch 10, which is direct evidence that
the observed ranking may reflect early optimization speed rather than stable
final performance. The validation set is used both for checkpoint selection and
reporting, and there is no independent test score.

The pilot also has only one seed. Differences of a few tenths of a percentage
point therefore have no uncertainty estimate and must not be called statistically
significant. The three-seed extension improves visibility into seed sensitivity,
but its sample SD remains descriptive and is not a 95% confidence interval.

The primary metric is AP50-95 on ten-class labels converted to YOLO format. The
conversion excludes ignored regions, score-zero annotations, and non-target
categories, while the validator does not implement the official VisDrone ignored-
region matching behavior. These numbers must not be presented as official
VisDrone challenge scores.

The strongest lesson from the pilot is methodological. A named prior should be
tested against a matched structural control: without the random-filter arm, the
Gabor result could easily have been overinterpreted. Likewise, module comparisons
need aligned placement and wrappers before architectural names can explain an
observed difference. Longer training, more seeds, a held-out evaluation protocol,
and controlled insertion points are required before making performance claims.

The extension adds a practical lesson: fewer parameters and fewer counted
operations are useful resource measurements, but neither guarantees improved
detection or lower runtime. Small objects and heavily occluded objects remain
weak points across all six arms. A follow-up should first test longer training
on all 6,471 training images, then separately control input resolution or feature
stride, insertion placement, and capacity. Deployment timing should be measured
in its own warmed-up, repeated benchmark rather than inferred from FLOP counts.

The code is distributed under AGPL-3.0. VisDrone images and annotations retain
their original dataset terms and are not bundled in this repository.
