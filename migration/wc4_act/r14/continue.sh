#!/usr/bin/env bash
set -euo pipefail
cd /data/zyf/video2door-wc4-act
export R14_TRAIN_GPU=${1:?idle train GPU}
export R14_MID_GPU=${2:?idle remaining-mid GPU}
test "$R14_TRAIN_GPU" != "$R14_MID_GPU"
R14_ROOT=$PWD/.repro/runs/wc4-fulltask-act-r14-s1000-20261022
R14_UUID_T=$(nvidia-smi -i "$R14_TRAIN_GPU" --query-gpu=uuid --format=csv,noheader)
R14_UUID_M=$(nvidia-smi -i "$R14_MID_GPU" --query-gpu=uuid --format=csv,noheader)
exec 9>"/data/zyf/video2door/.repro/locks/$R14_UUID_T.lock"
exec 10>"/data/zyf/video2door/.repro/locks/$R14_UUID_M.lock"
flock -n 9
flock -n 10
for R14_CHECK in 1 2 3; do
 nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$R14_ROOT/continuation-resource-$R14_CHECK.txt"
 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv > "$R14_ROOT/continuation-compute-$R14_CHECK.txt"
 for R14_UUID in "$R14_UUID_T" "$R14_UUID_M"; do
  if grep -q "$R14_UUID" "$R14_ROOT/continuation-compute-$R14_CHECK.txt"; then exit 3; fi
 done
 for R14_GPU in "$R14_TRAIN_GPU" "$R14_MID_GPU"; do
  nvidia-smi -i "$R14_GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{if($1>=1024 || $2>=10)exit 1}'
 done
 sleep 1
done
export VIDEO2DOOR_GPU=$R14_TRAIN_GPU
source migration/wc4_act/activate.sh act
export WANDB_RUN_GROUP=wc4-fulltask-act-r14-s1000-20261022
exec python -B migration/wc4_act/r14/continuation_group.py
