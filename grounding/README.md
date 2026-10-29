# Grounding: dual-stream grounder + critic

Given a language expression and a set of observations, this stage selects the 4D mask of
the referred object out of the segmenter's candidates.

## How it works

**Active Tree Search (ATS)** — `grounder_infer.mf_tree_windows`. Instead of sweeping every
frame with a fixed window, ATS starts from a global window over the whole observation set.
At each level the grounder predicts target presence for every sampled observation, and the
neighbourhood of each positive prediction defines the subrange examined next. Recursion
stops when a window holds only adjacent frames. Predictions from all visited windows are
aggregated by voting.

**Dual-stream grounder** — `agent_model.AgentGrounder`. Per-frame camera and register
tokens from the frozen segmenter backbone are injected into Qwen3-VL as compact geometric
context, avoiding the redundancy of feeding every patch token. The model emits two things
per window: an object-query prediction (matched against the segmenter's persistent
queries) and a box prompt (encoded by the prompt encoder). Each yields a 4D mask
candidate. Training is supervised with a geometry-distilled chain of thought covering
object size, height and inter-object distance.

**Critic** — `train_critic.py`, cards from `perframe_cards.py` / `render_pairframe.py`.
Both candidates are rendered as overlays on a shared frame set and presented together with
the expression; the critic predicts a pairwise preference and the winner becomes the final
mask.

## Training

```bash
cp env.example.sh env.sh && $EDITOR env.sh
pip install -r requirements.txt

bash scripts/train_grounder_4b.sh --dry-run   # verify inputs resolve
bash scripts/train_grounder_4b.sh             # Qwen3-VL-4B, 4 epochs
bash scripts/train_critic_2b.sh            # Qwen3-VL-2B, 3 epochs
```

Both self-chain across allocations and resume from their output directory. A `DONE` file
marks completion and blocks accidental resubmission.

## Inference

```bash
python run_pipeline_scene.py ...   # static scenes (ScanNet++ style)
python run_pipeline_video.py ...   # video (Ref-DAVIS / LVOS / MeViS style)
```

One process runs segmenter -> ATS -> grounder -> critic with no intermediate cache, so the
segmenter forward happens once.

## Data preparation

`cache_scene*.py` and `cache_video_ref.py` build the frozen-segmenter feature caches.
`build_cot_data.py` and `gen_spp_cot.py` distil the chain-of-thought supervision from scene
geometry. See `../docs/DATA.md` for how to obtain the underlying datasets — none of them
are redistributed here.

## Optional: vLLM

`vllm_box_backend.py` and `vllm_regplugin.py` serve the box stream through vLLM, with
s4vy registers added as a custom modality on top of stock Qwen3-VL. Install
`requirements-vllm.txt` only if you need this path; it is not required for training or for
the pipelines above.
