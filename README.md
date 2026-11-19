# S4VY: Segment Anything in Feed-Forward 4D Visual Geometry

[![Paper](https://img.shields.io/badge/arXiv-2609.36875-b31b1b.svg)](https://arxiv.org/abs/2609.36875)
[![License](https://img.shields.io/badge/License-NVIDIA%20Source%20Code%20License--NC-76b900.svg)](LICENSE)
[![Weights](https://img.shields.io/badge/Weights-FAIR%20NC%20%2B%20Apache--2.0-lightgrey.svg)](https://huggingface.co/jdzhang0929/S4VY)

Official implementation of **S4VY**, a promptable instance segmentation model built on
feed-forward 4D visual geometry. S4VY turns RGB observations into class-agnostic 4D
instance masks, where each persistent object query tracks one entity across every view,
and grounds them in natural language through an agentic system.

**This is a research project from [NVIDIA Research](https://www.nvidia.com/en-us/research/).**

Jingdong Zhang, Xin Li, Jan Kautz, Wenping Wang, Chris Choy

## What is here

Two stages that can be used independently or chained:

**Segmenter** — a query-based decoder on a LoRA-tuned [VGGT-Ω](https://github.com/facebookresearch/vggt-omega) backbone. 400 persistent
object queries produce class-agnostic masks for every object in the scene, from ordered
video or from sparse, non-consecutive views. A prompt encoder matches point/box prompts
to the corresponding object query.

**Grounding** — given a language expression, Active Tree Search (ATS) adaptively explores
the observation set instead of sweeping every frame; a dual-stream grounder predicts both
an object query and a box prompt, yielding two mask candidates; a critic renders both and
picks the one that matches the expression.

```
S4VY  (segmenter)
  ├─ prompt encoder ────────► Grounder, box stream
  └─ frozen features ───────► Grounder, query-matching stream
                                └─ Critic picks the final mask
```

Configs and launchers use internal run ids (`s4vy` is S4VY, `s4vy_joint` is S4VY-J in the
paper's reference rows). Training S4VY also needs a frozen teacher checkpoint, produced by
the `teacher` recipe included under `segomega/configs/`.

## Install

```bash
# Segmenter
pip install -r requirements.txt
cp env.example.sh env.sh && $EDITOR env.sh

# Grounder / Critic
pip install -r grounding/requirements.txt
cp grounding/env.example.sh grounding/env.sh && $EDITOR grounding/env.sh
```

`grounding/requirements-vllm.txt` is only needed for the vLLM serving utilities.

[VGGT-Ω](https://github.com/facebookresearch/vggt-omega) (Wang et al.,
[arXiv:2605.15195](https://arxiv.org/abs/2605.15195)) is a separate upstream dependency and is not
redistributed here. Obtain it and accept its licence before training or inference, and point
`VGGT_OMEGA_ROOT` at the checkout.

## Weights

Released at [huggingface.co/jdzhang0929/S4VY](https://huggingface.co/jdzhang0929/S4VY).

| Path | What it is |
|---|---|
| `segmenter/s4vy.pt` | **S4VY** — the segmenter behind every headline number |
| `segmenter/s4vy_prompt.pt` | prompt encoder for point/box prompts |
| `segmenter/s4vy_joint.pt` | S4VY-J, the from-scratch reference variant |
| `grounder/full_model.pt` | dual-stream grounder, on [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| `critic/full_model.pt` | critic, on [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |

Grounding results use the 4B grounder with the 2B critic.

## Training

```bash
bash scripts/submit_segmenter.sh teacher      # frozen teacher, consumed by s4vy
bash scripts/submit_segmenter.sh s4vy         # S4VY
bash scripts/submit_segmenter.sh s4vy_prompt  # prompt encoder, on the frozen S4VY
bash scripts/submit_segmenter.sh s4vy_joint   # S4VY-J

bash grounding/scripts/train_grounder_4b.sh
bash grounding/scripts/train_critic_2b.sh
```

Every launcher takes `--dry-run`, which verifies that all configured inputs resolve
without allocating a node. Run it first.

## Inference

```bash
# Class-agnostic 4D segmentation over a clip directory
S4VY_CKPT=<segmenter.pt> python infer_segmenter.py --root <clips>

# Language grounding, end to end (segmenter -> ATS -> grounder -> critic)
python grounding/run_pipeline_scene.py --grounder <g> --critic <c> ...   # static scenes
python grounding/run_pipeline_video.py --grounder <g> --critic <c> ...   # video
```

`demo/forward_video.py` and `demo/grounding_video.py` are self-contained: they pull the weights
from Hugging Face and need only a video file.

### Two inference options that are not in the paper

`infer_segmenter.py` and `demo/forward_video.py` have two switches that were added after the paper.
**Neither is described in the paper, and no number in the paper uses either of them.** Both are on by
default, and they are independent of each other:

| Switch | `on` (default) | `off` (what the paper used) |
|---|---|---|
| `--mask-id-enhance` | Owned id-map decoding: each pixel goes to the most confident query | The paper's decoder: masks painted in order of area |
| `--speedup` | Faster forward: exact attention-mask downsample, cached BF16 weights, `torch.compile` | Plain BF16-autocast forward |

To reproduce the paper's segmentation outputs, turn both off:

```bash
S4VY_CKPT=<segmenter.pt> python infer_segmenter.py --root <clips> --speedup off --mask-id-enhance off
```

With both off and the default `--input-long-side native`, the id maps and scores are bitwise
identical to our evaluation of this checkpoint for the paper; we checked this on all 1,875
benchmark clips. The released `segmenter/s4vy.pt` is that checkpoint, tensor for tensor.

**`--mask-id-enhance`** (`masks_to_idmap_owned` vs `masks_to_idmap_paper` in
`segomega/engine/decode.py`) changes only how the 400 query masks become one id map. The paper's
decoder keeps queries with P(object) > 0.3, binarizes each mask at 0.5 at the model's 1/4 mask
resolution, upsamples it nearest-neighbour and paints the masks smallest first, so where two masks
overlap the larger one wins. The owned decoder upsamples each frame's mask logits bilinearly, and every
pixel goes to the query with the highest P(object) × P(mask) among queries with P(object) > 0.1 and
P(mask) > 0.65; pixels with no such query are background. Overlaps are therefore settled by
confidence instead of by area, so a small object in front of a larger mask is no longer painted
over. Ids are query index + 1, so an id names the same query in every frame of one model call. The
gain comes from the ownership rule, not the lower P(object) threshold: at P(object) > 0.3 it
predicts only 2–10% more instances than the paper's decoder (15–45% more at 0.1) and no score in
the table below moves by more than 1.0.

**`--speedup`** (`segomega/engine/fast_inference.py`) leaves the model's weights and the LoRA
adapters as they are. LoRA is not merged into the backbone, so `set_lora_enabled(False)` still
gives the original VGGT-Ω backbone for geometry. The attention-mask downsample and the BF16 weight
cache are exact. `torch.compile` changes floating-point order, so a small share of pixels can change
instance (scores below). Every new clip shape is compiled on its first call, which makes that call
slower (4–18 s on our benchmarks, forward included), so for a single short video `--speedup off`
can finish sooner. CUDA graphs are a separate opt-in, `infer_segmenter.py --cuda-graph`: they are
exact and make the warm forward 1–5% faster, but each cached graph keeps its own GPU memory (on 40–90-frame DAVIS clips the two cached graphs held about 30 GB of an
80 GB A100).

Effect on the paper's benchmarks (same weights, native input; full tables in
[docs/INFERENCE.md](docs/INFERENCE.md)):

| Benchmark | Metric | Both off (paper setting) | `--mask-id-enhance on` | Default (both on) |
|---|---|---:|---:|---:|
| ScanNet, IGGT-Bench (Tab. 1) | J&F | 78.1 | **79.7** | **79.7** |
| ScanNet, IGGT protocol (Tab. 1b) | T-SR | **97.0** | 95.5 | 95.5 |
| ScanNet200 val (Tab. 1c) | J&F | 43.6 | **46.2** | **46.2** |
| DAVIS 2017 (Tab. 2) | J&F | 80.3 | **84.0** | **84.0** |
| LVOS (Tab. 2) | J&F | 60.2 | **67.9** | 67.8 |
| LVOS hard | J&F | 58.8 | 67.5 | **67.6** |
| LVOS full-frame, 64-frame windows (Tab. 2b) | J&F | 54.7 | **61.5** | **61.5** |
| VIPSeg (Tab. 3) | STQ | 79.5 | **82.4** | **82.4** |
| VIPSeg (Tab. 3) | SQ | **98.0** | 97.4 | 97.4 |
| VIPSeg (Tab. 3) | thing mAP | 35.2 | 44.5 | **44.6** |
| DAVIS 2017, strict 4D (Tab. 4) | IoU4D | 71.5 | **79.1** | **79.1** |
| LVOS, strict 4D (Tab. 4) | IoU4D | 53.3 | **63.3** | 63.2 |
| HOI4D | J&F | 62.4 | **66.5** | 66.4 |
| EPIC-KITCHENS VISOR | J&F | 31.3 | 35.6 | **35.8** |
| KITTI-STEP | J&F | 50.1 | **52.2** | 51.9 |

### Input resolution

`infer_segmenter.py` feeds the frames to the model as they are (`--input-long-side native`, the
default, as in the paper); frames longer than 1024 px or larger than 1024×768 are scaled down.
Pass `--input-long-side 512` for a smaller input: up to half the time and 60% of the memory, but
lower accuracy on every benchmark except the two ScanNet ones. `demo/forward_video.py --long_side`
takes the same values and defaults to 512, since videos are often larger than the exported
benchmark frames. [docs/INFERENCE.md](docs/INFERENCE.md) has accuracy, speed and memory for each
setting.

## What is not included

The benchmark evaluation harness is not part of this release — the AP / IoU scoring, the
benchmark protocols and the baseline adapters are omitted. Training and inference do not
depend on it; the periodic validation hook in the training loop is a soft import and is
skipped when it is absent. See `docs/CODE_STRUCTURE.md`.

## Data

Training data is **not** redistributed. Several sources prohibit redistribution or require
a signed agreement. `docs/DATA.md` lists every source, its license, and how to obtain it.

## License

The code in this repository is released under the
[NVIDIA Source Code License-NC](LICENSE) — noncommercial research use only.

It is our own implementation and does not redistribute any upstream source. The VGGT-Ω backbone
and the Qwen3-VL base models are external dependencies that you obtain and accept separately.

**The released weights are licensed differently and more restrictively.** The segmenter
checkpoints contain VGGT-Ω weights and are therefore distributed under the FAIR
Noncommercial Research License, which permits redistribution only under its own terms; the
grounder and critic checkpoints contain Qwen3-VL weights under Apache-2.0. See the
[weights repository](https://huggingface.co/jdzhang0929/S4VY) for the per-file breakdown.

Training data terms apply on top of both. Several sources are noncommercial-only and
KITTI-STEP carries CC BY-NC-SA; see `docs/DATA.md`.

## Citation

```bibtex
@article{zhang2026s4vy,
  title   = {S4VY: Segment Anything in Feed-Forward 4D Visual Geometry},
  author  = {Zhang, Jingdong and Li, Xin and Kautz, Jan and Wang, Wenping and Choy, Chris},
  journal = {arXiv preprint arXiv:2609.36875},
  year    = {2026}
}
```
