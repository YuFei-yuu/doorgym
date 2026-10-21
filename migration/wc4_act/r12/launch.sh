#!/usr/bin/env bash
# Retain this parent lease for training, child exports, Gym and inference worker.
set -euo pipefail
cd /data/zyf/video2door-wc4-act
WC4_R12_MODE=${1:?prepare or train}
export VIDEO2DOOR_GPU=${2:?physical idle GPU index}
WC4_R12_RUN=$PWD/.repro/runs/wc4-fulltask-act-r12-s1000-20261022
WC4_R12_UUID=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=uuid --format=csv,noheader)
exec 9>"/data/zyf/video2door/.repro/locks/$WC4_R12_UUID.lock"
flock -n 9
for WC4_R12_CHECK in 1 2 3; do
 if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$WC4_R12_UUID"; then exit 3; fi
 nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{if($1>=1024 || $2>=10)exit 1}'
 sleep 1
done
if [[ "$WC4_R12_MODE" == prepare ]]; then
 source migration/wc4_act/activate.sh gym
 export CUDA_VISIBLE_DEVICES=''
 exec python -B migration/wc4_act/r12/prepare.py
elif [[ "$WC4_R12_MODE" == train ]]; then
 test -f "$WC4_R12_RUN/frozen-files.json"
 test ! -e "$WC4_R12_RUN/train"
 source migration/wc4_act/activate.sh act
 export WANDB_RUN_GROUP=wc4-fulltask-act-r12-s1000-20261022
 exec python -B migration/wc4_act/r12/launch_train.py
else
 exit 4
fi
