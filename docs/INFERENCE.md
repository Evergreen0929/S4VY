# Choosing an inference configuration

`infer_segmenter.py` has these settings:

| Flag | Values | Default | In the paper |
|---|---|---|---|
| `--mask-id-enhance` | `on` (owned decoding, `masks_to_idmap_owned`), `off` (the paper's decoder, `masks_to_idmap_paper`) | `on` | **no** (the paper used `off`) |
| `--speedup` | `on` (exact attention-mask downsample, BF16 weight cache, `torch.compile`; LoRA not merged), `off` (plain BF16-autocast forward) | `on` | **no** (the paper used `off`) |
| `--cuda-graph` | with `--speedup on`, also replay a CUDA graph per repeated clip shape (exact; extra GPU memory, see [Memory](#memory)) | off | no |
| `--input-long-side` | `native` (the frames as given; scaled down only past 1024 px long side or 1024×768 px), or `16`–`1024` | `native` | `native` |
| `--low-memory`, `--offload-backbone` | opt-in memory reductions, outputs unchanged (see [Long clips](#long-clips-and-saved-mask-comparisons)) | off | off |

`demo/forward_video.py` takes `--mask-id-enhance`, `--speedup` and `--long_side` with the same
values. `--long_side` defaults to `512` there, and the demo skips CUDA graphs since it runs a
single forward per video.

**`--mask-id-enhance` and `--speedup` are not part of the paper.** Every number in the paper was
produced with both off. With both off and native input, `infer_segmenter.py` reproduces our
evaluation of the released checkpoint for the paper bitwise (id maps and scores) on all 1,875
benchmark clips; `segmenter/s4vy.pt` is that checkpoint, tensor for tensor.

## Quick pick

| Situation | Flags | Expect |
|---|---|---|
| Reproduce our paper evaluation of this checkpoint | `--speedup off --mask-id-enhance off` | bitwise identical id maps and scores |
| Benchmark sweeps, any GPU with ≥ 48 GB | *(defaults)* | 67.7 mean score (both off: 62.2); warm forward 0.2–7.3 s/clip on an A100 |
| Highest accuracy, ≥ 80 GB | `--input-long-side 1024` | +0.8 mean score over native, ~3× slower |
| Smaller GPU (≥ 32 GB), or fastest | `--input-long-side 512` | −3.9 mean score vs native, about half the time |
| One short video | `demo/forward_video.py --speedup off` | no compilation wait |

## Accuracy

Evaluation setup:

- **Clips:** every prompt-free cohort of the paper's segmentation tables (the paper's exported
  frames and ground truth): IGGT-Bench ScanNet (Table 1) and its IGGT-protocol arms (Table 1b),
  ScanNet200 (Table 1c), DAVIS/LVOS dense, sparse, shuffled and hard (Table 2), windowed full-frame
  LVOS (Table 2b), VIPSeg (Table 3), strict 4D (Table 4), HOI4D, EPIC-KITCHENS VISOR and KITTI-STEP.
- **Scoring:** the paper's protocol and scorers (greedy whole-track matching, J&F, T-mIoU/T-SR,
  strict 4D IoU, VIPSeg clip-mean STQ and pooled thing mAP).
- One NVIDIA A100-SXM4-80GB, PyTorch 2.5.0, released weights, native input.

### By switch

"Both off" is our evaluation of the released checkpoint for the paper, re-scored; with both switches
off the code reproduces it bitwise. Values are ×100.

| Benchmark | Metric | Both off | `--mask-id-enhance on` | `--speedup on` | **Default (both on)** |
|---|---|---:|---:|---:|---:|
| ScanNet, IGGT-Bench (Tab. 1) | J&F | 78.1 | **79.7** | 78.1 | **79.7** |
| ScanNet, IGGT-Bench (Tab. 1) | T-mIoU | 80.3 | **82.1** | 80.2 | **82.1** |
| ScanNet, IGGT-Bench (Tab. 1) | T-SR | 79.2 | **80.8** | 77.9 | **80.8** |
| ScanNet, IGGT protocol (Tab. 1b) | T-mIoU | 79.9 | **82.8** | 79.7 | 82.7 |
| ScanNet, IGGT protocol (Tab. 1b) | T-SR | **97.0** | 95.5 | **97.0** | 95.5 |
| ScanNet++, IGGT protocol (Tab. 1b) | T-mIoU | 78.0 | **80.3** | 77.8 | 80.2 |
| ScanNet++, IGGT protocol (Tab. 1b) | T-SR | 100.0 | 100.0 | 100.0 | 100.0 |
| ScanNet200 val (Tab. 1c) | J&F | 43.6 | **46.2** | 43.4 | **46.2** |
| ScanNet200 val (Tab. 1c) | T-mIoU | 49.2 | **52.8** | 49.0 | **52.8** |
| DAVIS 2017 (Tab. 2) | J&F | 80.3 | **84.0** | 80.3 | **84.0** |
| DAVIS sparse | J&F | 80.7 | 83.7 | 79.8 | **83.8** |
| DAVIS shuffled | J&F | 80.0 | **83.9** | 79.9 | **83.9** |
| LVOS (Tab. 2) | J&F | 60.2 | **67.9** | 58.7 | 67.8 |
| LVOS sparse | J&F | 64.5 | **73.6** | 65.0 | 73.3 |
| LVOS shuffled | J&F | 65.6 | 73.1 | 65.8 | **73.2** |
| LVOS hard | J&F | 58.8 | 67.5 | 59.0 | **67.6** |
| LVOS full-frame, 64-frame windows (Tab. 2b) | J&F | 54.7 | **61.5** | 54.9 | **61.5** |
| LVOS full-frame, 128-frame windows (Tab. 2b) | J&F | 53.1 | **59.9** | 53.2 | **59.9** |
| VIPSeg (Tab. 3) | STQ | 79.5 | **82.4** | 79.5 | **82.4** |
| VIPSeg (Tab. 3) | AQ | 65.4 | **70.4** | 65.4 | **70.4** |
| VIPSeg (Tab. 3) | SQ | **98.0** | 97.4 | **98.0** | 97.4 |
| VIPSeg (Tab. 3) | thing mAP | 35.2 | 44.5 | 35.2 | **44.6** |
| VIPSeg sparse | STQ | 79.6 | **82.4** | 79.6 | **82.4** |
| VIPSeg sparse | thing mAP | 38.5 | 48.6 | 38.6 | **48.7** |
| DAVIS 2017, strict 4D (Tab. 4) | IoU4D | 71.5 | **79.1** | 71.5 | **79.1** |
| DAVIS 2017, strict 4D (Tab. 4) | SR4D | 91.4 | **95.3** | 91.4 | **95.3** |
| LVOS, strict 4D (Tab. 4) | IoU4D | 53.3 | **63.3** | 52.1 | 63.2 |
| LVOS, strict 4D (Tab. 4) | SR4D | 57.3 | **75.3** | 54.3 | **75.3** |
| HOI4D | J&F | 62.4 | **66.5** | 62.4 | 66.4 |
| EPIC-KITCHENS VISOR | J&F | 31.3 | 35.6 | 31.3 | **35.8** |
| KITTI-STEP | J&F | 50.1 | **52.2** | 49.6 | 51.9 |

`--mask-id-enhance` raises every row except VIPSeg SQ (−0.6) and the IGGT-protocol ScanNet T-SR
(−1.5); the IGGT-protocol ScanNet++ T-SR is 100 either way. Most of the gain is on long videos and
crowded scenes (LVOS, VISOR, VIPSeg mAP), where large masks used to paint over smaller objects.
Metrics that charge false positives rise too (VIPSeg thing mAP, STQ's association term AQ), so the
gain is not bought with extra predictions. At the decoder's default P(object) > 0.1 it predicts
15–45% more instances per clip than the paper's decoder; at P(object) > 0.3 it predicts only 2–10%
more, and no score moves by more than 1.0 (KITTI-STEP, −1.0), so the gain comes from the ownership rule,
not the threshold.

`--speedup` changes scores only through `torch.compile`'s floating-point order. With the owned
decoder, the median clip keeps 99.82% of its pixels' instance (5th percentile 93.2%, over 1,875
clips), and the 31 scores above move by 0.06 on average (largest 0.3, KITTI-STEP J&F). With the
paper's decoder they move by 0.35 on average; the largest change is −3.0 on LVOS strict-4D SR4D
(−1.5 on LVOS J&F).


### By input resolution

This sweep was measured by the S4VY streaming-service team with the owned decoder and an earlier
speedup build that also merged the LoRA adapters (NVIDIA GB300, PyTorch 2.9). "Paper" is the value
printed in the paper. Its native column agrees with the default column of the switch table above to
within 0.3 on every shared row.

| Cohort | Metric | Paper | 512 | native | 1024 |
|---|---|---:|---:|---:|---:|
| ScanNet (IGGT) | J&F | 78.10 | **80.87** | 79.78 | 80.03 |
| DAVIS 2017 | J&F | 80.30 | 80.76 | 84.00 | **85.04** |
| DAVIS 2017 | IoU4D | 71.50 | 74.64 | 79.14 | **80.94** |
| LVOS | J&F | 60.20 | 62.74 | 67.88 | **68.35** |
| LVOS | IoU4D | 53.30 | 58.14 | 63.15 | **64.38** |
| VIPSeg | STQ | 79.50 | 80.38 | **82.41** | 81.78 |
| VIPSeg | mAP | 35.20 | 39.26 | **44.74** | 44.66 |
| ScanNet200 | J&F | 45.40 | **46.37** | 46.15 | 44.91 |
| HOI4D | J&F | 62.40 | 64.63 | 66.64 | **67.41** |
| EPIC-KITCHENS VISOR | J&F | 31.30 | 33.92 | 35.78 | **40.24** |
| KITTI-STEP | J&F | 50.10 | 37.90 | 52.03 | **52.28** |
| DAVIS sparse | J&F | 80.70 | 80.93 | 83.70 | **85.05** |
| LVOS sparse | J&F | 64.50 | 68.94 | 73.28 | **75.73** |
| DAVIS shuffled | J&F | 80.00 | 80.82 | 83.93 | **85.32** |
| LVOS shuffled | J&F | 65.60 | 68.59 | 72.99 | **74.98** |
| LVOS hard / shuffled | J&F | 58.80 | 62.00 | **67.60** | 64.85 |
| **Mean** | | 62.31 | 63.80 | 67.70 | **68.50** |

Model input sizes for the exported frames:

| Export | 512 | native | 1024 |
|---|---|---|---|
| 688×384 (DAVIS, LVOS, VIPSeg, HOI4D, VISOR) | 512×272 | 688×384 | 1024×560 |
| 592×448 (ScanNet, ScanNet200) | 512×384 | 592×448 | 1008×768 |
| 928×288 (KITTI-STEP) | 512×144 | 928×288 | 1024×304 |

Against 512, native is significantly better on 14 of the 16 rows (paired bootstrap 95% CI). It
loses 0.2 on ScanNet200 and is within noise on ScanNet (IGGT). 1024 adds more on DAVIS, LVOS and
VISOR but loses 1.5 on ScanNet200 against 512.

## Speed

Per clip on one NVIDIA A100-SXM4-80GB (PyTorch 2.5.0), native input, all 1,875 benchmark clips,
without CUDA graphs. "Forward" is the model forward alone; "warm" excludes the first clip of each new
shape, which pays compilation under `--speedup on`.

| Clips | Frames (mean) | Forward, `--speedup off` | Forward, `--speedup on` (warm) | Speedup | Decode, paper | Decode, owned |
|---|---:|---:|---:|---:|---:|---:|
| ScanNet, IGGT-Bench | 8 | 0.43 s | 0.24 s | 1.82× | 0.12 s | 0.025 s |
| ScanNet200 | 8 | 0.40 s | 0.25 s | 1.61× | 0.08 s | 0.007 s |
| DAVIS | 66 | 4.94 s | 4.03 s | 1.23× | 0.31 s | 0.033 s |
| LVOS | 100 | 8.64 s | 7.16 s | 1.21× | 0.56 s | 0.052 s |
| HOI4D | 100 | 8.73 s | 7.17 s | 1.22× | 0.29 s | 0.037 s |
| VISOR | 85 | 7.11 s | 6.97 s | 1.02× | 1.12 s | 0.082 s |
| KITTI-STEP | 100 | 8.86 s | 7.27 s | 1.22× | 1.17 s | 0.096 s |
| VIPSeg | 24 | 1.33 s | 0.94 s | 1.42× | 0.09 s | 0.011 s |
| 16-frame sparse / shuffled sets | 16 | 0.80 s | 0.53 s | 1.52× | 0.06 s | 0.008 s |

**First call per clip shape.** `--speedup on` compiles every new (frames, height, width) shape, so that
first call is slower (4–18 s for 10–90% of first calls here, forward included); later clips of the same shape reuse the
compiled kernels, so sweeps over clips of one size get the warm numbers above. A set of clips with
all-different shapes pays the first-call cost each time and is better served by `--speedup off`.

**CUDA graphs** (`--cuda-graph`, off by default) replay a captured graph per clip shape, keeping the
last two shapes. They are exact (on 887 clips every instance count and per-clip pixel agreement matched the run
without graphs) and
make the warm forward 1–5% faster on the cohorts we measured (ScanNet200 +5%, DAVIS sparse +2%, LVOS sparse +1%, VIPSeg sparse +1%, HOI4D +2%, LVOS +2%). Each capture costs a few extra forwards the
first time a shape is seen, and each cached graph keeps its own GPU memory (see [Memory](#memory)).

## Memory

Peak allocated GiB for one 100-frame 688×384 clip:

| Input long side | Model input | `--speedup off` |
|---|---|---:|
| 512 | 512×272 | 22.2 |
| native | 688×384 | 37.0 |
| 1024 | 1024×560 | 73.7 |

Memory grows with frames × pixels, so shorter clips need proportionally less. `--speedup on` adds
the BF16 copies of the aggregator weights (1.8 GB).

**CUDA graphs** (`--cuda-graph`) add memory for their own buffers, which is why they are off by
default. Under an 80 GB cap, a 100-frame native clip still captures a graph (39 GiB peak), but each
of the two cached graphs keeps its memory between clips: on 40–90-frame DAVIS clips they held about
30 GB of an 80 GB A100 together. If a capture does not fit (for example 100 frames at 1024),
`GraphRunner` runs that shape compiled without a graph; the output is identical and only the
graph's gain is lost. On GPUs shared with other processes, leave headroom: 1024 at 100 frames
needs nearly all of an 80 GB card.

## Long clips and saved-mask comparisons

Both inference entry points accept `--low-memory` to discard intermediate decoder
predictions after their attention masks are built and release consumed pyramid features.
All refinement layers still run; final logits and the decoder settings are unchanged.
This is opt-in, and training/grad-enabled forwards retain auxiliary supervision.

`--offload-backbone` also enables these reductions and stores cached backbone features
in CPU RAM while the heads run, transferring each requested layer back to its original
device. It reduces head-stage GPU residency, not the peak within the backbone, and adds
transfer time and host-memory use (about 59.5 GB for four FP32 caches at 782 × 1024×576).
It does not load or reuse features from disk. `GraphRunner` bypasses CUDA capture for
this mode because the CPU transfers cannot be captured. The other `--speedup` optimizations
remain available. Configure memory options before creating a graph runner.

```bash
python demo/forward_video.py --video clip.mov --out outputs/clip \
  --nframes 782 --long_side native --speedup off --offload-backbone
```

This selects up to 782 frames uniformly; choose the actual frame count to retain every
frame. Memory availability still limits clip size. No universal maximum follows from the
225/226-frame concatenation boundary discussed below.

Verify bitwise parity and record peak allocation on a short sample before a large run:

```bash
python tools/check_inference_memory.py --checkpoint /path/to/segmenter/s4vy.pt \
  --frames /path/to/source/pngs --nframes 10 --long-side 1024 \
  --out outputs/memory-parity.json
```

With the released checkpoint on ten sampled 1024×576 frames, both modes produced bitwise
identical final logits, mask embeddings, and decoded instance IDs. Head tests also cover
prompt outputs and training gradients. This check does not establish segmentation accuracy.

Export a synchronized 2×2 comparison from existing predictions (FFmpeg required):

```bash
python tools/export_comparison.py --video clip.mov \
  --offline outputs/offline/masks --stream outputs/stream/masks --sam outputs/sam/masks \
  --offline-color 169=55,200,245 --offline-color 147=255,145,35 \
  --sam-label 'SAM3: robot arm' --out outputs/comparison.mp4
```

Every mask directory must contain one PNG per source frame in the same temporal order.
The exporter accepts uint16 IDs and legacy packed RGB IDs (`R + 256*G`), resizes IDs with
nearest neighbors, copies available source audio through AAC encoding, and saves a JSON
manifest beside the video. Existing output videos are not overwritten. Optional `--duration`
sets the total constant-frame-rate video duration; otherwise source FPS is used. Variable
source timestamps are not preserved. Color overrides affect only the offline display;
saved masks and IDs are unchanged. IDs 169/147 are example queries from the grasping clip,
not universal arm/object classes. Override panel labels to describe the actual saved runs.

To inspect candidate merges, using an earlier run as a diagnostic reference:

```bash
python tools/audit_instance_ids.py --reference outputs/earlier/masks \
  --candidate outputs/offline/masks --object-id 227 --arm-id 169 \
  --out outputs/instance-id-audit.json
```

This compares majority IDs inside eroded reference interiors. Filenames must identify
the same source frames; missing candidate frames are reported explicitly. The reference
is not ground truth. Shared IDs are candidates for visual review, not an accuracy metric.
Distinct IDs can receive nearly identical palette colors. Conversely, the corrected native
grasping clip still has brief genuine object/arm merges at contact or occlusion; recoloring
does not fix those predictions. Same-weight 512-resolution controls separated the object
in some of these frames. Global mask agreement alone does not establish small-object quality.

The concatenation microbenchmark is also available:

```bash
python tools/benchmark_cuda_cat.py --large --out outputs/cuda-cat-benchmark.json
```

`--large` includes the 226/256-frame cached-feature and 782-frame register-token layouts
and needs about 24 GiB free GPU memory. Replay is checked against changed inputs and eager
output. On the tested GB300, graph replay improved the 4.3–7.4 GB cases by about 0.2–0.4%,
versus about 2–2.5× for a 19 MB case. These are resident concatenation timings, excluding model
attention, CPU transfers, warmup and capture; they are not end-to-end model speedups.

## Notes

- **Large CUDA concatenations:** backbone forwards automatically use explicit slice copies
  for homogeneous CUDA `torch.cat` outputs of at least 4 GiB. NVIDIA PyTorch 25.10
  (`2.9.0a0+145a3a7`) silently corrupted some such outputs in our reproducer. The guard is
  scoped to the backbone call, covers the external VGGT-Ω checkout without editing it, and
  retains native concatenation for smaller tensors. It preserves weights, attention context,
  and decoding settings, and supports compiled blocks and CUDA graph capture.
  At 1024×576, cached FP32 features have shape `[1, frames, 2321, 2048]`: 225 frames is
  below 4 GiB, while 226 crosses it. A separate register-token concatenation also failed at
  782 frames. These are implementation failure boundaries, not learned context limits.
  Explicit copies restored coherent predictions on the tested 256- and 782-frame clips.
  The 782-frame diagnostic used lossless CPU feature storage and discarded unused training
  auxiliaries to fit approximately 200 GB of GPU allocation; this is not a memory guarantee
  for the standard inference script. Frame limits still depend on resolution and memory.
  Regenerate features cached by affected runs: correcting the operator cannot repair an
  existing corrupted feature cache.
- Run the concatenation checks with `python -m unittest discover -s tests -v`.
  Set `S4VY_RUN_COMPILE_TESTS=1` for the Inductor test and
  `S4VY_RUN_LARGE_CUDA_TESTS=1` for the actual 4-GiB boundary and 782-frame register-layout
  regressions (24 GiB of free GPU memory recommended).
- `--speedup` never merges the LoRA adapters into the backbone weights and does not cast them, so
  `set_lora_enabled(False)` still yields the original VGGT-Ω backbone (the geometry path). The BF16
  weight cache covers only the aggregator; the FP32 geometry heads are untouched. CUDA graphs are keyed
  on the LoRA state, so toggling LoRA never replays a graph captured in the other state.
- `torch.compile` runs with Inductor's pattern-rewrite passes off: on PyTorch 2.5 its int8
  weight-only quantization pattern misreads the unmerged LoRA's scalar multiply and fails to compile.
- The switch, speed and accuracy tables use the pinned PyTorch 2.5.0; the input-resolution sweep and
  the memory table use PyTorch 2.9.
- The GB300 used for the input-resolution sweep and the memory table was shared with other resident
  processes; expect some run-to-run variation.
