#!/bin/bash
#SBATCH --job-name=s4vy-critic
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --open-mode=append
#SBATCH --output=slurm-%x.%j.out
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_load_env.sh"
CONDA=${GROUNDING_CONDA_ROOT:?}
IMG=${GROUNDING_CONTAINER_IMAGE:?}
D=${GROUNDING_DATA_ROOT:?}
CK=${AGENT_CKPT_ROOT:?}/critic_2b
[ -f "$CK/DONE" ] && { echo "[critic] DONE -> stop"; exit 0; }
# --mem=0 gives the step the whole node's memory. Without it srun grants only
# DefMemPerCPU x cpus-per-task, which is far below what this step needs: three card directories
# plus one host-resident copy of the 2B model per rank (~41 GB peak).
srun --mem=0 --container-image="$IMG" --container-mounts="${GROUNDING_CONTAINER_MOUNTS}" --container-workdir="$GROUNDING_REPO" --no-container-remap-root \
  bash -lc "
    source $CONDA/etc/profile.d/conda.sh; conda activate $GROUNDING_CONDA_ENV
    export HF_HOME=$GROUNDING_HF_HOME HF_HUB_OFFLINE=1
    export MODEL_PATH=$QWEN3_VL_2B
    export GROUNDING_DATA_ROOT=$GROUNDING_DATA_ROOT
    export CARDS=${CARDS:-$CRITIC_CARD_ROOTS} OUT_DIR=$CK
    export EPOCHS=${EPOCHS:-3} LR=${LR:-5e-6} BS=1 GRAD_ACC=8
    cd $GROUNDING_REPO
    accelerate launch --num_processes 4 --mixed_precision bf16 --use_deepspeed --zero_stage 2 train_critic.py
  "
