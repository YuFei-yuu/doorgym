#!/usr/bin/env bash
set -euo pipefail
cd /data/zyf/video2door-wc4-act
export VIDEO2DOOR_GPU=${1:?physical training GPU}
_uuid=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=uuid --format=csv,noheader)
exec 9>"/data/zyf/video2door/.repro/locks/$_uuid.lock"
flock -n 9
_contexts=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$_contexts" ]]; then echo "Selected training GPU has foreign compute contexts" >&2; exit 1; fi
source migration/wc4_act/activate.sh act
export WC4_ACT300K_RUN="$PWD/.repro/runs/wc4-act300k-20261030"
export WC4_ACT300K_EVAL_GPUS=${2:?physical eval candidates}
export WANDB_RUN_GROUP=wc4-act300k-20261030
exec python -B migration/wc4_act/act300k/start.py "${@:3}"
