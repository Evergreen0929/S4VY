#!/bin/bash
#SBATCH --job-name=seg4d23g
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --time=3:55:00
#SBATCH --open-mode=append
#SBATCH --output=slurm-%x.%j.out
# Training with a frozen-teacher pseudo-label branch. 8-node self-chain.
# PREREQ: TEACHER_CKPT set in env.sh (a finished base checkpoint).
# RUN: source env.sh && sbatch --nodes=8 scripts/train_s4vy.sbatch
set -u
: "${SEG4D_REPO:?source env.sh first}" "${CKPT_ROOT:?}" "${DATA_ROOT:?}" "${VGGT_OMEGA_ROOT:?}" "${VGGT_OMEGA_CKPT:?}"
: "${CONDA_ROOT:?}" "${CONDA_ENV:?}" "${CONTAINER_IMAGE:?}" "${CONTAINER_MOUNTS:?}" "${TEACHER_CKPT:?}"
REPO="$SEG4D_REPO"
CKDIR="$CKPT_ROOT/s4vy"
mkdir -p "$CKDIR"
# --- singleton-pool chain: clones share the job name; among RUNNING same-name jobs the LOWEST jobid proceeds,
#     the rest requeue; POOL pending clones always kept warm so a next link is queued and aging.
[ -f "$CKDIR/DONE" ] && {   # training complete -> drain the pool and stop
  for j in $(squeue -h -u $USER -n "$SLURM_JOB_NAME" -t PD -o %i); do scancel "$j"; done
  echo "[chain] DONE -> pool drained"; exit 0; }
RMIN=$(squeue -h -u $USER -n "$SLURM_JOB_NAME" -t R -o %i | sort -n | head -1)
if [ -n "$RMIN" ] && [ "$RMIN" != "$SLURM_JOB_ID" ]; then
  echo "[chain] link $RMIN owns the mutex -> requeue self"; scontrol requeue "$SLURM_JOB_ID"; sleep 120; exit 0
fi
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
PORT=$((20000 + SLURM_JOB_ID % 20000))
STEPS=${STEPS:-50000}
POOL=${POOL:-5}
NPD=$(squeue -h -u $USER -n "$SLURM_JOB_NAME" -t PD -o %i | wc -l)
for i in $(seq 1 $((POOL - NPD))); do
  (cd "$REPO" && sbatch --nodes=$SLURM_NNODES --job-name="$SLURM_JOB_NAME" --nice=6200 \
     scripts/train_s4vy.sbatch) && echo "[chain] pool +1"
done
promote_next() {   # on exit, promote the oldest parked clone into the run band
  NX=$(squeue -h -u $USER -n "$SLURM_JOB_NAME" -t PD -o %i | sort -n | head -1)
  [ -n "$NX" ] && scontrol update jobid=$NX nice=0 && echo "[chain] promoted $NX to nice=0"
}
trap promote_next TERM EXIT
srun --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" --container-workdir="$REPO" --no-container-remap-root \
  bash -lc "
    source $CONDA_ROOT/etc/profile.d/conda.sh; conda activate $CONDA_ENV
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 NCCL_ASYNC_ERROR_HANDLING=1
    export VGGT_OMEGA_ROOT=$VGGT_OMEGA_ROOT VGGT_OMEGA_CKPT=$VGGT_OMEGA_CKPT
    export SEG_NO_FRAMEPOS=1     # student (+eval) drop decoder frame_pos; teacher keeps it (teacher_framepos=true in cfg)
    export WANDB_RUN_ID=s4vy WANDB_MODE=offline WANDB_DIR=$CKDIR/wandb
    cd $REPO
    torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 --node_rank=\$SLURM_NODEID \
      --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER:$PORT \
      train_segmenter.py --config segomega/configs/s4vy.yaml \
      --opts train.num_workers=16 train.save_every=500 train.ckpt_dir=$CKDIR train.total_steps=$STEPS \
             data.root=$DATA_ROOT train.ema_init_ckpt=$TEACHER_CKPT
  " &
SRUN_PID=$!
wait $SRUN_PID
promote_next
