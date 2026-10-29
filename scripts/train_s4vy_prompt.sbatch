#!/bin/bash
#SBATCH --job-name=seg4d23gp
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --time=3:55:00
#SBATCH --open-mode=append
#SBATCH --output=slurm-%x.%j.out
# Train only the prompt encoder on a frozen base (entry: train_prompt.py). Single node, afterany chain.
# PREREQ: S4VY_CKPT set in env.sh (the frozen base checkpoint).
# RUN: source env.sh && sbatch --nodes=1 scripts/train_s4vy_prompt.sbatch
set -u
: "${SEG4D_REPO:?source env.sh first}" "${CKPT_ROOT:?}" "${DATA_ROOT:?}" "${VGGT_OMEGA_ROOT:?}" "${VGGT_OMEGA_CKPT:?}"
: "${CONDA_ROOT:?}" "${CONDA_ENV:?}" "${CONTAINER_IMAGE:?}" "${CONTAINER_MOUNTS:?}" "${S4VY_CKPT:?}"
REPO="$SEG4D_REPO"
CKDIR="$CKPT_ROOT/s4vy_prompt"
mkdir -p "$CKDIR"
[ -f "$CKDIR/DONE" ] && {   # training complete -> drain the dep chain and stop
  for j in $(squeue -h -u $USER -n "$SLURM_JOB_NAME" -t PD -o %i); do scancel "$j"; done
  echo "[chain] DONE -> stop"; exit 0; }
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
PORT=$((20000 + SLURM_JOB_ID % 20000))
STEPS=${STEPS:-15000}
# dependency chain: keep exactly one successor queued (survives preempt/timeout)
NPD=$(squeue -h -u $USER -n "$SLURM_JOB_NAME" -t PD -o %i | wc -l)
[ "$NPD" -eq 0 ] && (cd "$REPO" && sbatch --dependency=afterany:$SLURM_JOB_ID --nodes=$SLURM_NNODES \
   scripts/train_s4vy_prompt.sbatch) && echo "[chain] dep successor submitted"
srun --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" --container-workdir="$REPO" --no-container-remap-root \
  bash -lc "
    source $CONDA_ROOT/etc/profile.d/conda.sh; conda activate $CONDA_ENV
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 NCCL_ASYNC_ERROR_HANDLING=1
    export VGGT_OMEGA_ROOT=$VGGT_OMEGA_ROOT VGGT_OMEGA_CKPT=$VGGT_OMEGA_CKPT
    export SEG_NO_FRAMEPOS=1     # match s4vy student (no decoder frame_pos)
    export WANDB_RUN_ID=s4vy_prompt WANDB_MODE=offline WANDB_DIR=$CKDIR/wandb
    cd $REPO
    torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 --node_rank=\$SLURM_NODEID \
      --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER:$PORT \
      train_prompt.py --config segomega/configs/s4vy_prompt.yaml \
      --opts train.num_workers=16 train.save_every=500 train.ckpt_dir=$CKDIR train.total_steps=$STEPS \
             data.root=$DATA_ROOT train.init_from=$S4VY_CKPT
  "
