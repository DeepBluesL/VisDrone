# Full-data COCO-pretrained VisDrone: resolution and P2 screening

Status: **0/5 completed**, 0 running, 5 pending, 0 failed, 0 invalid.

Each row is one protocol arm and one seed. Incomplete arms are excluded from final accuracy and efficiency comparisons. No across-arm mean, standard deviation, confidence interval, or significance claim is reported.

| Arm | Architecture | Image size | Seed | Status | Epoch | Precision | Recall | AP50 | AP50–95 | Params | GFLOPs¹ | Validator ms² | E2E p50/p95 ms³ | FPS³ |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| standard_512_s179 | standard | 512 | 179 | pending | 0/100 | — | — | — | — | — | — | — | —/— | — |
| standard_768_s179 | standard | 768 | 179 | pending | 0/100 | — | — | — | — | — | — | — | —/— | — |
| standard_1024_s179 | standard | 1024 | 179 | pending | 0/100 | — | — | — | — | — | — | — | —/— | — |
| p2_768_s179 | p2 | 768 | 179 | pending | 0/100 | — | — | — | — | — | — | — | —/— | — |
| p2_1024_s179 | p2 | 1024 | 179 | pending | 0/100 | — | — | — | — | — | — | — | —/— | — |

¹ GFLOPs are shown at each arm's actual input size and are comparable only when the recorded accounting method agrees. ² Validator inference is batched stage timing, not end-to-end latency. ³ E2E latency is the matched-provenance synchronized batch-1 benchmark; FPS is derived from mean latency, while the table shows p50/p95.

## Interpretation limits

The arms intentionally change input resolution and architecture. Compute, memory, and speed therefore belong to each arm's actual configuration. A single seed cannot quantify training variance. COCO-pretrained coverage may also differ for newly introduced P2 layers; consult each run's initialization and coverage metadata before attributing a difference solely to architecture.

This validation uses the protocol's converted VisDrone validation set and is not the official VisDrone ignore-region evaluation unless the protocol explicitly says otherwise.
