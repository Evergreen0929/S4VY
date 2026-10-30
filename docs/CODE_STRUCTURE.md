# Code structure

Two model stages that can be used independently or chained.

## How the pieces connect

```text
S4VY segmenter  (segomega/)
  ├─ prompt encoder ──────────────► grounder, box stream
  └─ frozen features ─────────────► grounder, query-matching stream
                                      └─ critic picks the final mask
```

Training S4VY also needs a frozen teacher, produced by `segomega/configs/teacher.yaml`. The
teacher is a training-time artifact only; it is not a released model.

`s4vy_joint` is the from-scratch joint variant (S4VY-J), which appears in the paper's reference
rows. It is trained by the same entry point with a different config.

## Directory map

| Path | What it holds |
|---|---|
| `segomega/models/` | backbone wrapper, query decoder, mask head, prompt encoder, LoRA |
| `segomega/losses/` | Hungarian matcher and the set-prediction criterion |
| `segomega/data/` | scene / video loaders, prompt sampling |
| `segomega/engine/` | training loops, EMA pseudo-labelling, optimizer, mask decoding |
| `segomega/configs/` | one YAML per recipe: `teacher`, `s4vy`, `s4vy_prompt`, `s4vy_joint` |
| `train_segmenter.py` | training entry point for `teacher`, `s4vy`, `s4vy_joint` |
| `train_prompt.py` | training entry point for `s4vy_prompt` |
| `infer_segmenter.py` | class-agnostic 4D inference over a clip directory |
| `grounding/agent_model.py` | dual-stream grounder on top of [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) |
| `grounding/grounder_infer.py` | grounder inference, **including Active Tree Search (`mf_tree_windows`)** |
| `grounding/train_agent.py` | grounder trainer |
| `grounding/train_critic.py` | critic trainer |
| `grounding/prompt_infer.py` | prompt-encoder inference used by the box stream |
| `grounding/cache_scene*.py`, `cache_video_ref.py` | frozen-segmenter feature caches for grounder training |
| `grounding/build_cot_data.py`, `gen_spp_cot.py` | geometry-distilled chain-of-thought supervision |
| `grounding/cards_*.py`, `perframe_cards.py`, `render_pairframe.py` | critic card rendering |
| `grounding/align3d.py` | segmenter adapter over SegVGGT's scene loader (2D masks to point cloud) |
| `grounding/selfcorrect_data.py` | two-pass self-correction dataset for the grounder |
| `grounding/run_pipeline_scene.py` | end-to-end grounding on static scenes |
| `grounding/run_pipeline_video.py` | end-to-end grounding on video |
| `grounding/_env.py` | required-environment lookup shared by the grounding scripts |
| `demo/` | self-contained inference examples |

## Launchers

```
scripts/submit_segmenter.sh <teacher|s4vy|s4vy_prompt|s4vy_joint>
grounding/scripts/train_grounder_4b.sh
grounding/scripts/train_critic_2b.sh
```

Each accepts `--dry-run`, which resolves every configured input without allocating a node. The
Slurm scripts underneath are `scripts/train_*.sbatch` and `grounding/scripts/train_*.sbatch`.

## What is not here

The benchmark evaluation harness is not part of this release. `segomega/engine/decode.py`
decodes masks for inference, but the AP / IoU scoring, the benchmark protocols and the
baseline adapters are omitted. The periodic validation hook in `train_segmenter.py` is a soft
import and is skipped when the harness is absent, so training runs unchanged without it.
