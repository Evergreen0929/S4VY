# Choosing an inference configuration

`infer_segmenter.py` has three settings that trade accuracy, speed and GPU memory:

| Flag | Values | Default |
|---|---|---|
| `--decoder` | `owned` (per-pixel score ownership), `legacy` (area-ordered paint + `edge_snap`) | `owned` |
| `--input-long-side` | `16`–`1024`, or `native` (the frames' own size, long side capped at 1024, input at 1024×768 px) | `native` |
| `--fast` | `off` (eager), `exact` (bitwise-identical speedups + CUDA graphs), `full` (+ merged LoRA, BF16 weights, `torch.compile`) | `full` |

`demo/forward_video.py` takes `--long_side` and `--fast` with the same values but defaults to
`512` and `off`, since it runs a single forward per video.

All numbers below are measured with the released `segmenter/s4vy.pt`.

## Quick pick

| Situation | Flags | Expect |
|---|---|---|
| Benchmark sweeps, any GPU with ≥ 48 GB | *(defaults)* | 67.7 mean score, 0.2–4.2 s/clip |
| Highest accuracy, ≥ 80 GB | `--input-long-side 1024` | 68.5 mean score, ~3× slower than native |
| Smaller GPU (≥ 32 GB), or fastest | `--input-long-side 512` | 63.8 mean score, 0.2–2.0 s/clip |
| Outputs bitwise identical to the eager model | `--fast exact` | same accuracy as `off`, 1.1–1.4× faster than `off` |
| The released behaviour before these options existed | `--decoder legacy --input-long-side 512 --fast off` | 60.5 mean score, 1.6–12.8 s/clip |
| One video, interactive | `demo/forward_video.py --fast exact` | no compilation wait |

Mean score = mean of the 16 cohort metrics in the accuracy table below. Times are warm
end-to-end seconds per clip on an NVIDIA GB300 for 8–100-frame clips; memory is for
100-frame clips (see [Memory](#memory)).

## Accuracy

Evaluation setup:

- **Clips:** every prompt-free cohort of paper Tables 2–5 (1,374 clips; the paper's exported
  frames and ground truth).
- **Scoring:** the paper's protocol (greedy whole-track matching, J&F, strict 4D IoU on DAVIS and
  LVOS, VIPSeg clip-mean STQ and pooled mAP).

### By input resolution (`--decoder owned --fast full`)

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

### By decoder (mean of the 16 rows, `--fast full`)

| Decoder | 512 | native | 1024 |
|---|---:|---:|---:|
| `legacy` (area paint + `edge_snap`) | 60.35 | 63.50 | 63.63 |
| `legacy` without `edge_snap` | 59.77 | 63.48 | 64.90 |
| **`owned`** | **63.80** | **67.70** | **68.50** |
| `owned` + `edge_snap` | 63.86 | 66.83 | 66.61 |

`edge_snap` adds nothing on top of `owned` and costs ~95 ms per frame on the CPU.

### By fast mode

`--fast exact` is bitwise identical to `--fast off`. `--fast full` changes floating-point order,
so 99.4–99.99% of pixels keep their instance. Over the full 1,374-clip evaluation at 512 with
the `owned` decoder, the paired `full` − `off` change per cohort metric ranges from −0.81 (LVOS
hard / shuffled) to +0.32 (LVOS sparse), and every paired 95% CI includes zero. Individual clips
do move slightly in both directions.

## Speed

### End-to-end seconds per clip (paper Table 11 format)

PNG decode + resize + forward + decode to a full-resolution id map, on an even sample of the
Table 11 cohorts (10 ScanNet, 30 VIPSeg, 30 DAVIS, 20 LVOS, 20 HOI4D clips). Warm = the second run of the same clip. One NVIDIA GB300, PyTorch 2.9.

| Configuration | ScanNet (8 f) | VIPSeg (21 f) | DAVIS (66 f) | LVOS (100 f) | HOI4D (100 f) |
|---|---:|---:|---:|---:|---:|
| Paper, A100 | 0.5 | 3.0 | 6.6 | 10.1 | 10.2 |
| `legacy`, 512, `off` (the earlier release) | 1.63 | 3.13 | 8.89 | 12.77 | 11.79 |
| `owned`, 512, `off` | 0.30 | 0.53 | 1.70 | 2.64 | 2.71 |
| `owned`, 512, `exact` | 0.21 | 0.45 | 1.50 | 2.34 | 2.40 |
| `owned`, 512, `full` | 0.17 | 0.37 | 1.24 | 1.95 | 2.02 |
| **`owned`, native, `full` (default)** | **0.19** | **0.57** | **2.48** | **4.10** | **4.17** |
| `owned`, 1024, `full` | 0.57 | 1.51 | 7.33 | 13.31 | 13.39 |

Frame counts are this sample's means; the paper's VIPSeg clips average 41 frames and its DAVIS clips 71.

**First call per clip shape.** `full` compiles every new (frames, height, width) shape, and `exact` and `full` both
capture a CUDA graph for it. That first call takes seconds to tens of seconds (~35 s for a short
clip on a cold compile cache, ~50 s for 100 frames × batch 4). Later clips of the same shape
replay the graph. `infer_segmenter.py` keeps graphs for the last two shapes, so sweeps over
clips of one size get the warm numbers above; a set of clips with all-different shapes pays the
first-call cost each time and is better served by `--fast off` or `exact`.

### Forward throughput at 688×384 (paper Table 12 format)

Frames per second for the model forward alone, at the paper's GB200 batch sizes (in parentheses).

| Frames per clip | Paper A100 | Paper GB200 | GB300 `off` | GB300 `exact` | GB300 `full` |
|---:|---:|---:|---:|---:|---:|
| 1 | 25.85 (32) | 63.15 (32) | 38.09 (32) | 44.57 (32) | 65.88 (32) |
| 4 | 25.44 (32) | 65.19 (32) | 39.02 (32) | 45.70 (32) | 67.30 (32) |
| 8 | 24.37 (16) | 64.40 (32) | 38.15 (32) | 44.50 (32) | 64.56 (32) |
| 16 | 22.28 (8) | 61.61 (16) | 35.86 (16) | 41.33 (16) | 58.08 (16) |
| 32 | 18.98 (4) | 56.69 (8) | 32.04 (8) | 36.21 (8) | 48.48 (8) |
| 64 | 14.72 (2) | 48.64 (4) | 26.32 (4) | 28.51 (4) | 35.59 (4) |
| 100 | 11.64 (1) | 42.11 (4) | 22.02 (4) | 25.50 (4) | 30.96 (4) |

## Memory

Peak allocated GiB for one 100-frame 688×384 clip:

| Input long side | Model input | `off` | `full`, no CUDA graph |
|---|---|---:|---:|
| 512 | 512×272 | 22.2 | 22.1 |
| native | 688×384 | 37.0 | 36.9 |
| 1024 | 1024×560 | 73.7 | 73.6 |

Memory grows with frames × pixels, so shorter clips need proportionally less.

**CUDA graphs** add memory for their own buffers. Under an 80 GB cap, a 100-frame native clip
still captures a graph (39 GiB peak). If a capture does not fit (for example 100 frames at
1024), `GraphRunner` runs that shape compiled without a graph; the output is identical and
only the graph's few-percent speedup is lost. On GPUs shared with other processes, leave
headroom: 1024 at 100 frames needs nearly all of an 80 GB card.

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
this mode because the CPU transfers cannot be captured. The other `--fast` optimizations
remain available. Configure memory options before creating a graph runner.

```bash
python demo/forward_video.py --video clip.mov --out outputs/clip \
  --nframes 782 --long_side native --fast off --offload-backbone
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
- `--fast full` merges the LoRA adapters into the backbone weights in memory at load time; the
  checkpoint on disk is unchanged.
- Results above use PyTorch 2.9. `fast_inference.py` also supports the pinned PyTorch 2.5.0,
  but timings there have not been measured.
- The GB300 was shared with other resident processes during the measurements; expect some
  run-to-run variation.
