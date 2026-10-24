#!/usr/bin/env bash
set -euo pipefail
cd /data/zyf/video2door-wc4-act
WC4_S4_RUN=${1:?absolute independent run path}
export VIDEO2DOOR_GPU=${2:?GPU index}
test -f "$WC4_S4_RUN/manifest.json"
test ! -e "$WC4_S4_RUN/scene-claim.json"
test ! -e "$WC4_S4_RUN/attempt_000"
test ! -e "$WC4_S4_RUN/launch-start.txt"
date -u +%FT%TZ > "$WC4_S4_RUN/launch-start.txt"
WC4_S4_UUID=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=uuid --format=csv,noheader)
exec 9>"/data/zyf/video2door/.repro/locks/$WC4_S4_UUID.lock"
flock -n 9
trap 'date -u +%FT%TZ > "$WC4_S4_RUN/lease-released.txt"; flock -u 9' EXIT
for WC4_S4_CHECK in 1 2 3; do
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$WC4_S4_RUN/resource-$WC4_S4_CHECK.txt"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv > "$WC4_S4_RUN/compute-$WC4_S4_CHECK.txt"
    if grep -q "$WC4_S4_UUID" "$WC4_S4_RUN/compute-$WC4_S4_CHECK.txt"; then exit 3; fi
    nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{if($1>=1024 || $2>=10)exit 1}'
    sleep 1
done
df -B1 "$WC4_S4_RUN" > "$WC4_S4_RUN/capacity-start.txt"
test "$(df -B1 --output=avail "$WC4_S4_RUN" | tail -1)" -ge 53687091200
source migration/wc4_act/activate.sh gym
export WANDB_MODE=disabled
set +e
nice -n 10 python -B migration/wc4_act/newexpert_stage4/execute.py --run "$WC4_S4_RUN" > "$WC4_S4_RUN/attempt_000.log" 2>&1
WC4_S4_EXIT=$?
printf '%s\n' "$WC4_S4_EXIT" > "$WC4_S4_RUN/attempt_000.exit"
df -B1 "$WC4_S4_RUN" > "$WC4_S4_RUN/capacity-end.txt"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$WC4_S4_RUN/resource-end.txt"
exit "$WC4_S4_EXIT"
