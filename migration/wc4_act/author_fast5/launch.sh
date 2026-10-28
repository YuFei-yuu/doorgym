#!/usr/bin/env bash
set -euo pipefail
cd /data/zyf/video2door-wc4-act
WC4_RANDOM_BATCH=${1:?absolute run path}
WC4_C8_RUN="$WC4_RANDOM_BATCH"
export VIDEO2DOOR_GPU=${2:?GPU index}
test -f "$WC4_RANDOM_BATCH/manifest.json"
test ! -e "$WC4_C8_RUN/scene-claim.json"
test ! -e "$WC4_C8_RUN/attempt_000"
test ! -e "$WC4_C8_RUN/launch-start.txt"
date -u +%FT%TZ > "$WC4_C8_RUN/launch-start.txt"
WC4_C8_UUID=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=uuid --format=csv,noheader)
exec 9>"/data/zyf/video2door/.repro/locks/$WC4_C8_UUID.lock"
flock -n 9
trap 'date -u +%FT%TZ > "$WC4_C8_RUN/lease-released.txt"; flock -u 9' EXIT
for WC4_C8_CHECK in 1 2 3; do
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$WC4_C8_RUN/resource-$WC4_C8_CHECK.txt"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv > "$WC4_C8_RUN/compute-$WC4_C8_CHECK.txt"
    if grep -q "$WC4_C8_UUID" "$WC4_C8_RUN/compute-$WC4_C8_CHECK.txt"; then exit 3; fi
    nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{if($1>=1024 || $2>=10)exit 1}'
    sleep 1
done
df -B1 "$WC4_C8_RUN" > "$WC4_C8_RUN/capacity-start.txt"
test "$(df -B1 --output=avail "$WC4_C8_RUN" | tail -1)" -ge 53687091200
source migration/wc4_act/activate.sh gym
export WANDB_MODE=disabled
set +e
nice -n 10 python -B migration/wc4_act/author_fast5/execute.py --run "$WC4_RANDOM_BATCH" > "$WC4_C8_RUN/attempt_000.log" 2>&1
WC4_C8_EXIT=$?
printf '%s\n' "$WC4_C8_EXIT" > "$WC4_C8_RUN/attempt_000.exit"
df -B1 "$WC4_C8_RUN" > "$WC4_C8_RUN/capacity-end.txt"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$WC4_C8_RUN/resource-end.txt"
exit "$WC4_C8_EXIT"
