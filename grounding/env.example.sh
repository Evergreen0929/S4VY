# =============================================================================
# Grounder / critic / grounding-eval environment.
#
# Copy to `grounding/env.sh`, edit, then source it before using any canonical
# launcher under grounding/scripts/.  The local env.sh is ignored by git.
# =============================================================================

# --- repository and Slurm ----------------------------------------------------
export GROUNDING_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SEG4D_REPO="$(cd "${GROUNDING_REPO}/.." && pwd)"
export GROUNDING_ACCOUNT="YOUR_SLURM_ACCOUNT"
export GROUNDING_TRAIN_PARTITIONS="YOUR_PARTITION"        # comma-separated Slurm partitions
export GROUNDING_EVAL_PARTITIONS="YOUR_PARTITION"         # partitions for inference jobs

# --- runtime ----------------------------------------------------------------
export GROUNDING_CONDA_ROOT="/path/to/miniforge3"
export GROUNDING_CONDA_ENV="groundsft"
export GROUNDING_CONTAINER_IMAGE="/path/to/pytorch_24.10_py3.sqsh"
export GROUNDING_CONTAINER_MOUNTS="/data:/data"
export GROUNDING_LOG_ROOT="${GROUNDING_REPO}/logs"
export GROUNDING_HF_HOME="/path/to/.cache/huggingface"

# --- data/checkpoints --------------------------------------------------------
# Extract the archived jdzhang0929/S4VY-Grounder-Dataset snapshot so these
# paths reproduce its top-level data and data_log layout.
export GROUNDING_DATA_ROOT="/path/to/data"
export GROUNDING_DATA_LOG_ROOT="/path/to/data_log"
export AGENT_CKPT_ROOT="/path/to/checkpoints/agent"

export GROUNDER_BIG_MANIFEST="${GROUNDING_DATA_LOG_ROOT}/joint_train_big.jsonl"
export GROUNDER_LEGACY_MANIFEST="${GROUNDING_DATA_LOG_ROOT}/joint_train_srpp.jsonl"
export CRITIC_CARD_ROOTS="${GROUNDING_DATA_ROOT}/critic_cards_scanrefer:${GROUNDING_DATA_ROOT}/critic_cards_scene:${GROUNDING_DATA_ROOT}/critic_cards_video"

# --- base VLMs ---------------------------------------------------------------
export QWEN3_VL_2B="/path/to/hf_models/Qwen3-VL-2B-Instruct"
export QWEN3_VL_4B="/path/to/hf_models/Qwen3-VL-4B-Instruct"

# --- frozen segmenter pair used by both grounding arms ----------------------
export VGGT_OMEGA_ROOT="/path/to/vggt-omega"
export VGGT_OMEGA_CKPT="/path/to/checkpoints/VGGT-Omega/vggt_omega_1b_512.pt"
export S4VY_CKPT="/path/to/checkpoints/seg4d/s4vy/segomega_final.pt"
# Report/e2e protocol uses this selected checkpoint, not prompt_final.pt.
export S4VY_PROMPT_CKPT="/path/to/checkpoints/seg4d/s4vy_prompt/prompt_13500.pt"

# --- external geometry/data bridges used by static grounding eval -----------
# align_eval imports SegVGGT's released point-cloud evaluator for the 2D->3D
# lift. The segmenter model itself is imported from SEG4D_REPO above.
export SEGVGGT_ROOT="/path/to/codebase/SegVGGT"
export SCANNET200_2D_ROOT="${SEGVGGT_ROOT}/data_processing/scannet200/scenes_2d"
export SCANNET200_POINTS_ROOT="${SEGVGGT_ROOT}/data_processing/scannet200/points"
export SCANNET_RAW_VAL_ROOT="/path/to/data/scannet_raw_val/scans"
export SCANNET_RAW_TRAIN_ROOT="/path/to/data/scannet_raw_train/scans"
export SCANNET_VAL_LIST="/path/to/datasets/scannet/scannetv2_val.txt"
export INSSCENE_SPP_ROOT="/path/to/data/InsScene-15K/processed_scannetpp_v2/processed_scannetpp_v2"
export SPACEFORMER_SPP_ROOT="/path/to/datasets/spaceformer/v4/scannetpp_v4gt_qwen"
export SPP_EVAL_MANIFEST="${GROUNDING_DATA_LOG_ROOT}/scannetpp_eval_filt.jsonl"
export SCANREFER_QUERY_FILTER="${GROUNDING_DATA_ROOT}/critic_openloop/query_filter_700.json"
