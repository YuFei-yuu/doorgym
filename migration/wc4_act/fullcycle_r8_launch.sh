#!/usr/bin/env bash
set -euo pipefail
cd /data/zyf/video2door-wc4-act
WC4_R8_MODE=${1:?preflight or attempt}
WC4_R8_INDEX=${2:?frozen trial index 0..7}
export VIDEO2DOOR_GPU=${3:?idle GPU index}
WC4_R8_RUN=$PWD/.repro/runs/wc4-fulltask-pilot-r8-20261021
WC4_R8_UUID=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=uuid --format=csv,noheader)
exec 9>"/data/zyf/video2door/.repro/locks/$WC4_R8_UUID.lock"
flock -n 9
for WC4_R8_CHECK in 1 2 3; do
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$WC4_R8_RUN/resource-${WC4_R8_MODE}-${WC4_R8_INDEX}-${WC4_R8_CHECK}.txt"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv > "$WC4_R8_RUN/compute-${WC4_R8_MODE}-${WC4_R8_INDEX}-${WC4_R8_CHECK}.txt"
    if grep -q "$WC4_R8_UUID" "$WC4_R8_RUN/compute-${WC4_R8_MODE}-${WC4_R8_INDEX}-${WC4_R8_CHECK}.txt"; then exit 3; fi
    nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{if($1>=1024 || $2>=10)exit 1}'
    sleep 1
done
source migration/wc4_act/activate.sh gym
python -B migration/wc4_act/run_fullcycle_pilot.py --mode "$WC4_R8_MODE" --index "$WC4_R8_INDEX" --run "$WC4_R8_RUN"
