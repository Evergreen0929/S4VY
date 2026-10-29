# ============================================================================
# S4VY segmenter — cluster / path configuration.
# COPY this file to `env.sh` and edit for your environment, then:
#     source env.sh
#     sbatch --nodes=8 scripts/train_teacher.sbatch      # etc.
# The training launchers `source` this file for all paths, and SLURM reads the
# SBATCH_* vars below as submission defaults (so account/partition need not be
# hard-coded in the #SBATCH headers, and self-chained re-submits inherit them).
# ============================================================================

# --- SLURM submission defaults (SLURM honors these env vars) -----------------
export SBATCH_ACCOUNT="YOUR_SLURM_ACCOUNT"                 # your Slurm account
export SBATCH_PARTITION="YOUR_PARTITION"                   # your Slurm partition(s)

# --- Repo + runtime ----------------------------------------------------------
export SEG4D_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this repo (auto)
export CONDA_ROOT="/path/to/miniforge3"                    # conda install root
export CONDA_ENV="seg4d"                                   # conda env name (see requirements.txt)
export CONTAINER_IMAGE="/path/to/pytorch_24.10_py3.sqsh"   # enroot/pyxis image (or "" to run bare)
export CONTAINER_MOUNTS="/data:/data"                  # srun --container-mounts (host:container)

# --- VGGT-Omega backbone (EXTERNAL upstream dependency) ----------------------
# Clone the released VGGT-Omega repo separately and point these at it.
export VGGT_OMEGA_ROOT="/path/to/vggt-omega"                       # the VGGT-Omega code repo
export VGGT_OMEGA_CKPT="/path/to/VGGT-Omega/vggt_omega_1b_512.pt"  # 1B/512px backbone checkpoint

# --- Data + checkpoint output roots ------------------------------------------
export DATA_ROOT="/path/to/InsScene-15K"                  # training data root (see README §Data)
export CKPT_ROOT="/path/to/checkpoints/seg4d"             # where teacher/s4vy/s4vy_joint write ckpts

# --- Teacher checkpoint (produced by teacher, consumed by s4vy/s4vy_joint) -------
# After teacher finishes, its final ckpt becomes the frozen teacher. Set this to
# $CKPT_ROOT/teacher/segomega_50000.pt (matches the yaml `ema_init_ckpt`).
export TEACHER_CKPT="${CKPT_ROOT}/teacher/segomega_50000.pt"

# --- s4vy checkpoint (frozen base for the optional prompt-encoder run, STEP 2b) ------
# After s4vy finishes, set this to its final ckpt (init_from for s4vy_prompt).
export S4VY_CKPT="${CKPT_ROOT}/s4vy/segomega_final.pt"

# --- s4vy_joint offline pseudo prompts -------------------------------------------
# Required by s4vy_joint's prompt branch. The directory must contain the active
# `dl3dv` (v2 chunk store, 53,968 units) and `matterport` entries. These may be
# directories or symlinks, but must be prepared before submission.
export PROMPT_PSEUDO_ROOT="/path/to/data/prompt_pseudo_active"

# --- exported-frame segmenter evaluation ------------------------------------
# Root produced by export/build_* tools; contains <bench>/clip_*/{frames,gt.npz}.
export SEG_EVAL_ROOT="/path/to/data/baseline_eval"
