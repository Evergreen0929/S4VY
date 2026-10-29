#!/bin/bash
#SBATCH --job-name=seg4d29i
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --time=3:55:00
#SBATCH --open-mode=append
#SBATCH --output=slurm-%x.%j.out
# Joint exhaustive + prompt training. 8-node chain (CHAIN_MODE=dep for a single-successor chain).
# RUN: source env.sh && sbatch --nodes=8 scripts/train_s4vy_joint.sbatch
set -u
: "${SEG4D_REPO:?source env.sh first}" "${CKPT_ROOT:?}" "${DATA_ROOT:?}" "${VGGT_OMEGA_ROOT:?}" "${VGGT_OMEGA_CKPT:?}" "${PROMPT_PSEUDO_ROOT:?}"
: "${CONDA_ROOT:?}" "${CONDA_ENV:?}" "${CONTAINER_IMAGE:?}" "${CONTAINER_MOUNTS:?}"
REPO="$SEG4D_REPO"
CKDIR="$CKPT_ROOT/s4vy_joint"
mkdir -p "$CKDIR"
[ -f "$CKDIR/DONE" ] && {
  for j in $(squeue -h -u $USER -n "$SLURM_JOB_NAME" -t PD -o %i); do scancel "$j"; done
  echo "[chain] DONE -> pool drained"; exit 0; }
RMIN=$(squeue -h -u $USER -n "$SLURM_JOB_NAME" -t R -o %i | sort -n | head -1)
if [ -n "$RMIN" ] && [ "$RMIN" != "$SLURM_JOB_ID" ]; then
  echo "[chain] link $RMIN owns the mutex -> requeue self"; scontrol requeue "$SLURM_JOB_ID"; sleep 120; exit 0
fi
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
PORT=$((20000 + SLURM_JOB_ID % 20000))
STEPS=${STEPS:-50000}
# CHAIN_MODE=dep: classic dependency chain (exactly one successor). Default "pool": parallel-clone pool.
if [ "${CHAIN_MODE:-pool}" = dep ]; then
  NPD=$(squeue -h -u $USER -n "$SLURM_JOB_NAME" -t PD -o %i | wc -l)
  [ "$NPD" -eq 0 ] && (cd "$REPO" && sbatch --nodes=$SLURM_NNODES --job-name="$SLURM_JOB_NAME" \
     --dependency=afterany:$SLURM_JOB_ID --export=ALL,CHAIN_MODE=dep --nice=0 \
     scripts/train_s4vy_joint.sbatch) && echo "[chain] dep successor submitted"
else
  POOL=${POOL:-5}
  NPD=$(squeue -h -u $USER -n "$SLURM_JOB_NAME" -t PD -o %i | wc -l)
  for i in $(seq 1 $((POOL - NPD))); do
    (cd "$REPO" && sbatch --nodes=$SLURM_NNODES --job-name="$SLURM_JOB_NAME" --nice=8000 \
       scripts/train_s4vy_joint.sbatch) && echo "[chain] pool +1"
  done
fi
promote_next() {
  NX=$(squeue -h -u $USER -n "$SLURM_JOB_NAME" -t PD -o %i | sort -n | head -1)
  [ -n "$NX" ] && scontrol update jobid=$NX nice=0 && echo "[chain] promoted $NX (oldest) to nice=0"
}
trap promote_next TERM EXIT
srun --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" --container-workdir="$REPO" --no-container-remap-root \
  bash -lc "
    source $CONDA_ROOT/etc/profile.d/conda.sh; conda activate $CONDA_ENV
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 NCCL_ASYNC_ERROR_HANDLING=1
    export VGGT_OMEGA_ROOT=$VGGT_OMEGA_ROOT VGGT_OMEGA_CKPT=$VGGT_OMEGA_CKPT
    export SEG_NO_FRAMEPOS=1     # no decoder frame_pos (match s4vy)
    export WANDB_RUN_ID=s4vy_joint WANDB_MODE=offline WANDB_DIR=$CKDIR/wandb
    cd $REPO
    torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 --node_rank=\$SLURM_NODEID \
      --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER:$PORT \
      train_segmenter.py --config segomega/configs/s4vy_joint.yaml \
      --opts train.num_workers=16 train.save_every=500 train.ckpt_dir=$CKDIR train.total_steps=$STEPS \
             data.root=$DATA_ROOT train.joint_prompt.data.root=$DATA_ROOT \
             train.joint_prompt.data.pseudo_prompt_root=$PROMPT_PSEUDO_ROOT
  " &
SRUN_PID=$!
wait $SRUN_PID
promote_next
