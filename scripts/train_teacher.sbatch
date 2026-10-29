#!/bin/bash
#SBATCH --job-name=seg4d21
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --time=3:55:00
#SBATCH --open-mode=append
#SBATCH --output=slurm-%x.%j.out
# Base training. 8-node afterany self-chain.
# RUN: source env.sh && sbatch --nodes=8 scripts/train_teacher.sbatch
# (account/partition from SBATCH_ACCOUNT/PARTITION in env.sh; paths inherited via --export=ALL.)
set -u
: "${SEG4D_REPO:?source env.sh first}" "${CKPT_ROOT:?}" "${DATA_ROOT:?}" "${VGGT_OMEGA_ROOT:?}" "${VGGT_OMEGA_CKPT:?}"
: "${CONDA_ROOT:?}" "${CONDA_ENV:?}" "${CONTAINER_IMAGE:?}" "${CONTAINER_MOUNTS:?}"
REPO="$SEG4D_REPO"
CKDIR="$CKPT_ROOT/teacher"
mkdir -p "$CKDIR"
[ -f "$CKDIR/DONE" ] && { echo "[chain] DONE -> stop"; exit 0; }   # trainer writes DONE at total_steps
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
PORT=$((20000 + SLURM_JOB_ID % 20000))
STEPS=${STEPS:-50000}
# self-replenish: always queue the next 4h link (afterany survives preempt/timeout); successor's DONE check stops the chain
(cd "$REPO" && sbatch --dependency=afterany:$SLURM_JOB_ID --nodes=$SLURM_NNODES scripts/train_teacher.sbatch) && echo "[chain] queued successor"
srun --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" --container-workdir="$REPO" --no-container-remap-root \
  bash -lc "
    source $CONDA_ROOT/etc/profile.d/conda.sh; conda activate $CONDA_ENV
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 NCCL_ASYNC_ERROR_HANDLING=1
    export VGGT_OMEGA_ROOT=$VGGT_OMEGA_ROOT VGGT_OMEGA_CKPT=$VGGT_OMEGA_CKPT
    export WANDB_RUN_ID=teacher WANDB_MODE=offline WANDB_DIR=$CKDIR/wandb
    cd $REPO
    torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 --node_rank=\$SLURM_NODEID \
      --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER:$PORT \
      train_segmenter.py --config segomega/configs/teacher.yaml \
      --opts train.num_workers=8 train.save_every=500 train.ckpt_dir=$CKDIR train.total_steps=$STEPS data.root=$DATA_ROOT
  "
