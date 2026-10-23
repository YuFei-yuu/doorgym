#!/usr/bin/env bash
set -euo pipefail
cd /data/zyf/video2door-wc4-act
export VIDEO2DOOR_GPU=${1:?Choose idle physical GPU}
run=$PWD/.repro/runs/wc4-fulltask-act-r18-s1000-20261023
test -f "$run/prepared-command.json"
uuid=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=uuid --format=csv,noheader)
exec 9>"/data/zyf/video2door/.repro/locks/$uuid.lock"
flock -n 9
for check in 1 2 3; do
 nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$run/gpu-before-$check.txt"
 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv > "$run/apps-before-$check.txt"
 if grep -q "$uuid" "$run/apps-before-$check.txt"; then exit 3; fi
 nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{if($1>=1024 || $2>=10)exit 1}'
 sleep 1
done
export WC4_EXECUTION_RUN=$run
source migration/wc4_act/activate.sh act
export WANDB_RUN_GROUP=wc4-fulltask-act-r18-s1000-20261023
exec python -B migration/wc4_act/r18/launch.py
