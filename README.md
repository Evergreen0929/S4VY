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

### Extra parameters

`infer_segmenter.py` and `demo/forward_video.py` take these extra parameters, which are not part of
the paper. Both switches are on by default; pass `--speedup off --mask-id-enhance off` to reproduce
the paper's outputs exactly.

- `--mask-id-enhance on|off`: how the 400 query masks are merged into one instance-id map. `on`
  gives each pixel to the query with the highest P(object) × P(mask), so a small object in front of
  a larger mask keeps its pixels. `off` is the paper's decoder, which paints the masks in order of
  area, so the larger mask wins where two overlap.
- `--speedup on|off`: faster inference with the same weights, using `torch.compile` on the backbone
  blocks, cached BF16 weights and an exact attention-mask downsample. The LoRA adapters are not
  merged, so `set_lora_enabled(False)` still gives the original VGGT-Ω geometry. Compilation changes
  floating-point order, so a few pixels can change instance, and each new clip shape is compiled on
  its first call.
- `--cuda-graph` (`infer_segmenter.py` only, off by default): with `--speedup on`, also replays a CUDA
  graph for repeated clip shapes. It is about 1–5% faster, but the cached graphs hold extra GPU memory.

Accuracy, speed and memory for each setting are in [docs/INFERENCE.md](docs/INFERENCE.md).

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
