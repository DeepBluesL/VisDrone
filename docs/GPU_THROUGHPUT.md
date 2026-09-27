# RTX 5090 D training throughput tuning

These short profiles choose execution settings before the five 100-epoch runs. They are not accuracy results. CUDA work includes forward/loss, backward, and optimizer/EMA steps; wall timing preserves asynchronous overlap. The complete per-profile settings, step counts, hardware and source fingerprints accompany the JSON evidence.

| Architecture | Input | Physical batch | Images/s | Mean GPU utilization | p95 utilization | Peak reserved GB | Mean next-batch wait ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 512 | 64 | 492.7 | 60.0% | 67.0% | 6.58 | 0.232 |
| standard | 768 | 64 | 268.8 | 73.4% | 86.4% | 12.81 | 0.237 |
| standard | 1024 | 64 | 183.9 | 81.0% | 99.0% | 22.54 | 0.233 |
| p2 | 768 | 64 | 148.8 | 75.3% | 98.0% | 20.82 | 0.232 |
| p2 | 1024 | 32 | 90.2 | 74.8% | 90.4% | 20.31 | 0.217 |

At standard/512 with eight workers, increasing physical batch from 16 to 32 to 64 yielded approximately 236, 377 and 493 images/s. Four/eight/twelve workers at batch 16 yielded 231/236/239 images/s. Data wait was already below 0.3 ms; adding workers alone did not fill the GPU. Eight workers provide a common setting for the larger images.

The density stress set combines the 32 most densely annotated training images with 32 fixed random images. Standard/1024 and P2/768 at batch 64 peaked at about 22.3 and 21.4 GB reserved. An earlier 32-image P2/1024 stress test at batch 32 peaked at 25.1 GB; its original log is retained. A single image with 902 target boxes remains one native assignment, preserving within-image competition.

The full decoded cache is 32.1 GB. Host RAM is 64 GB. The selected physical batches avoid exhausting 32 GB of VRAM. GPU utilization depends on image density, input size, Python/kernel dispatch, validation and checkpointing; sustained 100% is not claimed.

Formal runs use effective batch 64 via accumulation and a zero-to-base bias-LR warmup. Profiling uses a manual fixed-LR training loop, bypasses the unrelated YOLO11n AMP comparison, and is not an exact epoch-time prediction. Some exploration profiles used nbs=16, as recorded; therefore optimizer cadence/decay differs from the final protocol. The final trainer also fixes the extra persistent-loader reference at mosaic shutdown; ordinary-epoch loading remains the same.

The physical-batch difference can affect BatchNorm and microbatch loss normalization. These are practical configuration comparisons; equal effective batch does not eliminate every training confound.
