#!/bin/bash
# Internal launcher for the dual-stream grounder. Use train_grounder_4b.sh instead.
set -e
SZ=${1:-4B}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_load_env.sh"
W=$GROUNDING_REPO
D=${GROUNDING_DATA_ROOT:?source grounding/env.sh first}
case "$SZ" in
  4B) MODEL=${QWEN3_VL_4B:?} ;;
  *)  echo "ERROR: unsupported grounder size '$SZ' (the released recipe is 4B)"; exit 1 ;;
esac
MAN=${MAN:-${GROUNDER_BIG_MANIFEST:?}}
LR=${LR:-8e-6}; EPOCHS=${EPOCHS:-4}; PARTS=${GROUNDING_TRAIN_PARTITIONS:?}
HEAD_LR=${HEAD_LR:-1e-4}; WARMUP=${WARMUP:-300}
TAG=_joint_${SZ,,}
CKD=${AGENT_CKPT_ROOT:?}/g1_mix$TAG
[ -d "$MODEL" ] || { echo "ERROR: base VLM not found: $MODEL"; exit 1; }
[ -f "$MAN" ] || { echo "ERROR: manifest not found: $MAN"; exit 1; }
[ -f "$CKD/DONE" ] && { echo "ERROR: $CKD/DONE exists -- this run already finished; rm it to re-run"; exit 1; }
NROWS=$(wc -l <"$MAN"); [ "$NROWS" -ge 60000 ] || { echo "ERROR: manifest has only $NROWS rows (expected >= 60000)"; exit 1; }
NLOAD=$(python3 -c "
import json,os,glob as _g
D='$D'; SPPC=os.environ.get('SPPC', D+'/scannetpp_cache_v2')
c=[D+'/s4vy_register_cache', D+'/gt_idmap_cache', D+'/s4vy_train_cache']; seen={}; n=0
VID={os.path.basename(p)[:-4] for p in _g.glob(D+'/vidref_cache_*/*.npz')}
for l in open('$MAN'):
    sc=json.loads(l)['scene']
    if sc not in seen:
        seen[sc]=(sc in VID) or os.path.exists(f'{SPPC}/{sc}.npz') or all(os.path.exists(f'{d}/{sc}.npz') for d in c)
    n+=seen[sc]
print(n)")
CHAIN_TARGET=${CHAIN_TARGET:-$(( (NLOAD / 64) * EPOCHS ))}
_EP=$(( NLOAD / 64 ))
_CAP=400
SAVE_EVERY=${SAVE_EVERY:-$(( _EP < _CAP ? _EP : _CAP ))}
echo "$SZ | manifest $NROWS rows | $NLOAD loadable scenes | ${EPOCHS} epochs -> steps=$CHAIN_TARGET | lr=$LR head_lr=$HEAD_LR | save every $SAVE_EVERY"
J=$(sbatch --parsable -A "${GROUNDING_ACCOUNT:?}" -J s4vy-grounder -p "$PARTS" \
  --export=ALL,MFRAME=1,MF_NF=8,MF_WIN=32,MF_BIDIR=0,NO_INST_PROJ=1,COT=1,COT_W=1.0,BOX_OUT=1,SEG_TAG=s4vy,FULL_SFT=1,EPOCHS=$EPOCHS,LR=$LR,HEAD_LR=$HEAD_LR,WARMUP=$WARMUP,BS=1,GRAD_ACC=8,SAVE_EVERY=$SAVE_EVERY,MODEL_PATH=$MODEL,MANIFEST_OVERRIDE=$MAN,TAG=$TAG,SELFCHAIN=1,CHAIN_TARGET=$CHAIN_TARGET,CHAIN_PARTS=${PARTS//,/+} \
  "$W/scripts/train_grounder.sbatch" g1 mix)
echo "launched: job $J -> $CKD"
