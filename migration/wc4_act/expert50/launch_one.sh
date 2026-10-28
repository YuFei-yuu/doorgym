#!/usr/bin/env bash
set -euo pipefail
cd /data/zyf/video2door-wc4-act
WC4_E50_RUN=${1:?absolute run}
WC4_E50_SEED=${2:?seed}
export VIDEO2DOOR_GPU=${3:?physical GPU}
WC4_E50_OUT="$WC4_E50_RUN/attempts/seed$WC4_E50_SEED"
WC4_E50_UUID=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=uuid --format=csv,noheader)
exec 9>"/data/zyf/video2door/.repro/locks/$WC4_E50_UUID.lock"
flock -n 9
trap 'date -u +%FT%TZ > "$WC4_E50_OUT/lease-released.txt"; flock -u 9' EXIT
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv > "$WC4_E50_OUT/compute-start.txt"
if grep -q "$WC4_E50_UUID" "$WC4_E50_OUT/compute-start.txt"; then exit 3; fi
nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{if($1>=1024 || $2>=10)exit 1}'
source migration/wc4_act/activate.sh gym
export WANDB_MODE=disabled
set +e
nice -n 10 python -B migration/wc4_act/expert50/execute.py --run "$WC4_E50_RUN" --seed "$WC4_E50_SEED" > "$WC4_E50_OUT/attempt_000.log" 2>&1
WC4_E50_RC=$?
set -e
printf '%s\n' "$WC4_E50_RC" > "$WC4_E50_OUT/attempt_000.exit"
exit "$WC4_E50_RC"
