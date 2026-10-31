#!/usr/bin/env bash
set -euo pipefail
cd /data/zyf/video2door-wc4-act
export VIDEO2DOOR_GPU=${1:?physical training GPU}
WC4_A100_UUID=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=uuid --format=csv,noheader)
exec 9>"/data/zyf/video2door/.repro/locks/$WC4_A100_UUID.lock"
flock -n 9
WC4_A100_CONTEXTS=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$WC4_A100_CONTEXTS" ]]; then echo 'Selected training GPU is occupied' >&2; exit 3; fi
source migration/wc4_act/activate.sh act
export CUDA_VISIBLE_DEVICES="$WC4_A100_UUID"
export WC4_ACT100_200K_RUN="$PWD/.repro/runs/wc4-act100-fresh200k-20261031"
export WC4_ACT100_200K_EVAL_GPUS=${2:?evaluation candidates}
export WC4_EXECUTION_RUN="$WC4_ACT100_200K_RUN"
export WANDB_RUN_GROUP=wc4-act100-fresh200k-20261031
export PYTHONDONTWRITEBYTECODE=1
exec python -B migration/wc4_act/act100_200k/start.py "${@:3}"
