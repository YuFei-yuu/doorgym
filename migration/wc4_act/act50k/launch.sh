#!/usr/bin/env bash
set -euo pipefail
cd /data/zyf/video2door-wc4-act
export VIDEO2DOOR_GPU=${1:?physical training GPU}
_uuid=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=uuid --format=csv,noheader)
exec 9>"/data/zyf/video2door/.repro/locks/$_uuid.lock"
flock -n 9
source migration/wc4_act/activate.sh act
export WC4_ACT50K_RUN="$PWD/.repro/runs/wc4-act50k-20261029"
export WC4_ACT50K_EVAL_GPU=${2:?physical eval GPU}
export WANDB_RUN_GROUP=wc4-act50k-20261029
exec python -B migration/wc4_act/act50k/start.py "${@:3}"
