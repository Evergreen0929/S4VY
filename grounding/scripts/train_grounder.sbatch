#!/bin/bash
#SBATCH --job-name=s4vy-grounder
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --open-mode=append
#SBATCH --output=slurm-%x.%j.out
# usage: sbatch -J <name> train_grounder.sbatch <GROUP> [MIX]   MIX=mix|r1|r2 (default mix)
GROUP=$1
MIX=${2:-mix}
NP=${NP:-8}   # VLM training uses 8 GPUs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_load_env.sh"
CONDA=${GROUNDING_CONDA_ROOT:?}
IMG=${GROUNDING_CONTAINER_IMAGE:?}
WORK=$GROUNDING_REPO
D=${GROUNDING_DATA_ROOT:?}
CKROOT=${AGENT_CKPT_ROOT:?}
# per-group extras
WH=0; ME=""
case "$GROUP" in
  g3) WH=200 ;;                                   # head-warmup steps
  g2|g4) ME="$D/mask_embed_init.pt" ;;           # dense: init mask_embed from the segmenter
esac
PORT=$((20000 + SLURM_JOB_ID % 20000))            # unique master port per job (avoid co-located clash)
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)   # multi-node rendezvous host (SLURM_NNODES>1)
# ---- self-requeueing chain (opt-in: SELFCHAIN=1) ----
# At START, submit ONE successor that waits on THIS job (afterany:$SLURM_JOB_ID), giving a rolling
# "1 running + 1 pending" pair; each link submits its own successor, so the chain never runs out.
# It stops when a link sees step >= CHAIN_TARGET. The trainer writes $OUT_DIR/DONE at total_steps --
# without that sentinel a finished link would still see step < CHAIN_TARGET and queue forever.
# (To re-run a finished experiment, rm $OUT_DIR/DONE first.)
# The sbatch is retried: the controller can fail a submit silently, which would break the chain.
_OUT=$CKROOT/${GROUP}_${MIX}${TAG:-}
if [ -f "$_OUT/DONE" ]; then echo "[chain] DONE sentinel at $_OUT (step $(cat $_OUT/DONE)) -> stop"; exit 0; fi
if [ "${SELFCHAIN:-0}" = "1" ]; then
  _STEP=$(python3 -c "import json;print(json.load(open('$_OUT/meta.json')).get('step',0))" 2>/dev/null || echo 0)
  _TGT=${CHAIN_TARGET:-7640}
  # CHAIN_PARTS uses + as its separator and is translated to , below: a comma inside --export=ALL,K=V
  # is the export-list separator, so a comma-joined partition list silently collapses to its first entry.
  CHAIN_PARTS=${CHAIN_PARTS:-${GROUNDING_TRAIN_PARTITIONS//,/+}}
  if [ "${_STEP:-0}" -lt "$_TGT" ]; then
    for _try in 1 2 3 4 5 6; do
      _sj=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID -A "${GROUNDING_ACCOUNT:?}" -J "$SLURM_JOB_NAME" --export=ALL \
            --nodes=${SLURM_NNODES:-1} --time=${CHAIN_TIME:-4:00:00} -p "${CHAIN_PARTS//+/,}" "$0" "$GROUP" "$MIX" 2>/dev/null)
      if [ -n "$_sj" ]; then echo "[selfchain] successor $_sj (afterany:$SLURM_JOB_ID, try $_try, step=$_STEP<$_TGT)"; break
      else echo "[selfchain] submit FAILED try $_try, retrying..."; sleep 30; fi
    done
  else
    echo "[selfchain] DONE (step=$_STEP >= $_TGT) -> no successor"
  fi
fi
srun --container-image="$IMG" --container-mounts="${GROUNDING_CONTAINER_MOUNTS}" --container-workdir="$WORK" --no-container-remap-root \
  bash -lc "
    source $CONDA/etc/profile.d/conda.sh; conda activate $GROUNDING_CONDA_ENV
    export HF_HOME=$GROUNDING_HF_HOME HF_HUB_OFFLINE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
    export MODEL_PATH=${MODEL_PATH:-$QWEN3_VL_4B}
    export GROUNDING_REPO=$GROUNDING_REPO SEG4D_REPO=$SEG4D_REPO GROUNDING_DATA_ROOT=$GROUNDING_DATA_ROOT
    export GROUP=$GROUP MIX=$MIX R2_PROB=0.5 LOWMEM=${LOWMEM:-0} SEL_HEAD=${SEL_HEAD:-0} SEL_W=${SEL_W:-0.1}
    export COT=${COT:-0} COT_W=${COT_W:-1.0} SCHED_MIX=${SCHED_MIX:-0} SCHED_P=${SCHED_P:-0.5} ACTION=${ACTION:-0} ACT_W=${ACT_W:-0.5} REG17=${REG17:-0} R1_SELECT_ONLY=${R1_SELECT_ONLY:-0}
    export LOC=${LOC:-}   # SAM-hint baseline: point|bbox (R2 generates coord text; R1 temporal top-1)
    export SEG_TAG=${SEG_TAG:-s4vy} FIXEDQ=${FIXEDQ:-0} SFRAME=${SFRAME:-0} INST_APPEND=${INST_APPEND:-0} NO_INST_PROJ=${NO_INST_PROJ:-0} SF_NOREG=${SF_NOREG:-0}
    export MFRAME=${MFRAME:-0} MF_NF=${MF_NF:-8} MF_WIN=${MF_WIN:-32} MF_BIDIR=${MF_BIDIR:-0} MF_IMG=${MF_IMG:-0}
    export MANIFEST=${MANIFEST_OVERRIDE:-$D/grounder_train.jsonl}
    export OUT_DIR=$CKROOT/${GROUP}_${MIX}${TAG:-}
    export EPOCHS=${EPOCHS:-2} BS=${BS:-1} GRAD_ACC=${GRAD_ACC:-4} LR=${LR:-2e-4} WARMUP=${WARMUP:-150} NF_VLM=8 REG_MAX_FRAMES=128 VIS_T=20
    export LORA_RANK=${LORA_RANK:-32} LORA_ALPHA=${LORA_ALPHA:-64}
    export WARMUP_HEADS=$WH MAX_STEPS=${MAX_STEPS:-0} FULL_SFT=${FULL_SFT:-0} NO_DEEPSPEED=${NO_DEEPSPEED:-0}
    [ -n \"$ME\" ] && export MASK_EMBED_CKPT=$ME
    cd $WORK
    NPROC=\$((8 * \$SLURM_NNODES)); MRANK=\$SLURM_NODEID   # multi-node: 1 task/node (ntasks-per-node=1), 8 procs/node
    MN=\"--num_machines \$SLURM_NNODES --machine_rank \$MRANK --main_process_ip $MASTER --main_process_port $PORT\"
    if [ \"\$FULL_SFT\" = \"1\" ] && [ \"\$NO_DEEPSPEED\" != \"1\" ]; then
      export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 NCCL_TIMEOUT=1800
      accelerate launch --num_processes \$NPROC \$MN --mixed_precision bf16 \
        --use_deepspeed --deepspeed_multinode_launcher standard --zero_stage 2 \
        --gradient_accumulation_steps \$GRAD_ACC --gradient_clipping 1.0 train_agent.py
    else
      export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 NCCL_TIMEOUT=1800
      accelerate launch --num_processes \$NPROC \$MN --mixed_precision bf16 --multi_gpu train_agent.py
    fi
  "
