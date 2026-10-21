#!/usr/bin/env bash
set -euo pipefail
cd /data/zyf/video2door-wc4-act
WC4_R9_MODE=${1:?prepare/preflight/attempt}
WC4_R9_INDEX=${2:?index}
export VIDEO2DOOR_GPU=${3:?GPU}
WC4_R9_RUN=$PWD/.repro/runs/wc4-fulltask-expand-r9-20261021
mkdir -p "$WC4_R9_RUN-launch"
WC4_R9_UUID=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=uuid --format=csv,noheader)
exec 9>"/data/zyf/video2door/.repro/locks/$WC4_R9_UUID.lock"
flock -n 9
for WC4_R9_CHECK in 1 2 3; do
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$WC4_R9_RUN-launch/resource-${WC4_R9_MODE}-${WC4_R9_INDEX}-${WC4_R9_CHECK}.txt"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv > "$WC4_R9_RUN-launch/compute-${WC4_R9_MODE}-${WC4_R9_INDEX}-${WC4_R9_CHECK}.txt"
    if grep -q "$WC4_R9_UUID" "$WC4_R9_RUN-launch/compute-${WC4_R9_MODE}-${WC4_R9_INDEX}-${WC4_R9_CHECK}.txt"; then exit 3; fi
    nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{if($1>=1024 || $2>=10)exit 1}'
    sleep 1
done
source migration/wc4_act/activate.sh gym
python -B migration/wc4_act/r9/collect.py --run "$WC4_R9_RUN" --mode "$WC4_R9_MODE" --index "$WC4_R9_INDEX"
