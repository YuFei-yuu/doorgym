#!/usr/bin/env bash
set -euo pipefail
cd /data/zyf/video2door-wc4-act
export VIDEO2DOOR_GPU=${1:?physical idle GPU index}
export WC4_EXECUTION_RUN=$PWD/.repro/runs/wc4-fulltask-act-r14-s1000-20261022
WC4_R14_UUID=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=uuid --format=csv,noheader)
exec 9>"/data/zyf/video2door/.repro/locks/$WC4_R14_UUID.lock"
flock -n 9
for WC4_R14_CHECK in 1 2 3; do
 nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv > "$WC4_EXECUTION_RUN/gpu-before-$WC4_R14_CHECK.txt"
 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv > "$WC4_EXECUTION_RUN/compute-before-$WC4_R14_CHECK.txt"
 if grep -q "$WC4_R14_UUID" "$WC4_EXECUTION_RUN/compute-before-$WC4_R14_CHECK.txt"; then exit 3; fi
 nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{if($1>=1024 || $2>=10)exit 1}'
 sleep 1
done
source migration/wc4_act/activate.sh act
export WANDB_RUN_GROUP=wc4-fulltask-act-r14-s1000-20261022
cat /proc/$$/fdinfo/9 > "$WC4_EXECUTION_RUN/lease-fdinfo-before.txt"
exec python -B migration/wc4_act/r14/launch_entry.py
