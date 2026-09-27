# Full-data pretrained resolution and P2 study

Current status: **paused**, with 0/5 arms completed and the first arm at epoch
31/100. The resumable checkpoint is recorded in
[pause_snapshot.json](../results/realtime_stage1/pause_snapshot.json).
The queue refuses ordinary execution while the user pause marker is present;
`--resume-paused` is reserved for an explicit user request to resume.

This study implements optimization steps 1 and 2 after the short module experiments.
The goal is to find a useful accuracy/latency tradeoff on the current RTX 5090 D.
The final deployment device and FPS requirement have not been fixed.

## Design

| Arm | Architecture | Input size | Training budget | Seed |
| --- | --- | ---: | ---: | ---: |
| standard_512_s179 | Official YOLOv8n P3/P4/P5 | 512 | 100 epochs | 179 |
| standard_768_s179 | Official YOLOv8n P3/P4/P5 | 768 | 100 epochs | 179 |
| standard_1024_s179 | Official YOLOv8n P3/P4/P5 | 1024 | 100 epochs | 179 |
| p2_768_s179 | Official YOLOv8n P2/P3/P4/P5 | 768 | 100 epochs | 179 |
| p2_1024_s179 | Official YOLOv8n P2/P3/P4/P5 | 1024 | 100 epochs | 179 |

All arms use 6,471 training and 548 validation images. Test-dev is excluded.
Each arm starts independently from the same official COCO YOLOv8n checkpoint;
one resolution is never fine-tuned from another arm. The optimizer is AdamW,
with initial LR 0.001, cosine decay, three warmup epochs, and weight decay 0.0005.
Mosaic probability is 0.3 and is disabled for the last ten epochs. Other spatial
augmentations are mild. There is no early stopping or class resampling.
The frozen `results/realtime_stage1/protocol.json` is authoritative for exact settings.
The common effective batch is 64 after warmup. Physical batches are chosen using GPU
memory and throughput profiles, with gradient accumulation for smaller microbatches.
Different physical batches change BatchNorm statistics and per-microbatch loss
normalization; the study compares practical configurations and does not claim to
remove that confound. Bias LR warms up from zero.
Validation uses a common batch of 32 in every arm. The memory-bounded assigner is
installed on both the training model and its EMA validation copy, avoiding the
native dense-matching CPU fallback during validation.

## Pretrained initialization

The P2 graph shifts layer indices. Weight transfer uses explicit semantic mappings
rather than copying arbitrary equal-numbered layers. Both architectures share exactly
the same pretrained backbone. New P2 fusion paths retain seeded initialization.
Classification branches are reinitialized for ten VisDrone classes in both arms.
Each run exports an `initialization.json` with every loaded/skipped tensor and
parameter coverage. Coverage differs between architectures, so this is a comparison
of complete pretrained configurations, not a perfectly isolated architecture effect.

## GPU throughput and memory

The training pipeline uses FP16 automatic mixed precision, pinned memory, persistent
data-loader workers, and prefetching. An official decoded-image disk cache avoids
repeated JPEG decoding. The full cache occupies approximately 32.1 GB. It is kept
outside Git; the cache provenance is retained with the experiment.

Dense P2 training can expand task-aligned matching tensors to hundreds of millions
of candidate/GT pairs. The new assigner partitions only the independent image batch
dimension and removes trailing padded GT slots before invoking the original matcher.
It does not split competing GTs within an image. Exact tensor-equivalence tests cover
empty, dense, multiclass, padded, and non-contiguous valid-GT cases. This preserves
the assignment objective while reducing peak temporary memory.

`scripts/profile_realtime.py` measures warmed training throughput, CPU batch wait,
CUDA step time, utilization, and peak memory in separate processes. It does not
produce accuracy results. The GPU utilization counter alone is not a speed metric;
images/second and wall-clock epoch duration determine the selected configuration.
`gpu_telemetry.csv` records utilization, memory, power, and temperature during each run.
Measured settings and throughput are summarized in [GPU_THROUGHPUT.md](GPU_THROUGHPUT.md).

## Execution and recovery

Use the project virtual environment on the experiment host:

```powershell
.\.venv\Scripts\python.exe scripts/run_realtime_suite.py --suite realtime_stage1 --profiled-batches
```

The queue runs one GPU experiment at a time and benchmarks each completed checkpoint.
Optional `--finalize --push` verifies all five complete runs, preserves the previous
290 archived artifacts, updates the English README/report/manifest, and commits and
pushes only the experiment publication paths to the existing repository. No partial
training run is presented as a completed result. Finalization errors remain recorded
in `.runtime/finalization_state.json` and can be retried separately.
An OS-backed lock prevents overlapping queues. Input images/labels, code, environment,
pretrained weights, protocol, and checkpoints have explicit identities. A changed
identity stops reuse/resume rather than silently mixing incompatible runs.

Atomic checkpoints preserve FP32 raw model, EMA, optimizer, scheduler, AMP scaler,
process RNG state, selected epoch, and the exact CSV boundary. Resume restores the
checkpoint epoch and reconciles CSV/best-checkpoint state. Worker augmentation and
prefetch state are not serializable, so resumed training is not claimed to be bitwise
identical to an uninterrupted run. Completed-budget recovery performs final evaluation
without adding another training epoch.

## Evaluation and interpretation

The selected checkpoint maximizes validation AP50-95 within the fixed epoch budget.
AP50, per-class AP, training curves, parameter count, counted convolution/linear
GFLOPs, memory, and training duration accompany it. This uses converted ten-class
Ultralytics AP, not the official VisDrone ignore-region evaluation.

Batch-one FP16 speed measurements include 30 warmup and 100 timed iterations:

- CUDA events measure the model forward, including box decode, on a preallocated tensor.
- Synchronized wall time measures an already decoded BGR image through preprocessing,
  H2D, model/decode, and NMS. JPEG decoding, camera acquisition, and disk I/O are excluded.

The report includes mean FPS and p50/p95 latency at each arm's own input size.
No batched validation timing is presented as deployment FPS. These are PyTorch results,
not TensorRT deployment measurements.

Each configuration has one seed at this screening stage. No significance or robustness
claim is made. Finalists and the matched baseline should subsequently receive repeated
seeds. Improvements relative to the old 10-epoch, 2,048-image scratch baseline combine
more data, pretraining, and a longer budget; they cannot be assigned to one factor alone.
