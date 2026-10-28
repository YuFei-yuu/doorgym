#!/usr/bin/env bash
set -euo pipefail
cd /data/zyf/video2door-wc4-act
WC4_R5_RUN=${1:?absolute run path}
export VIDEO2DOOR_GPU=${2:?physical GPU}
WC4_R5_UUID=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=uuid --format=csv,noheader)
exec 9>"/data/zyf/video2door/.repro/locks/$WC4_R5_UUID.lock"
flock -n 9
trap 'date -u +%FT%TZ > "$WC4_R5_RUN/lease-released.txt"; flock -u 9' EXIT
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$WC4_R5_RUN/resource-start.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv > "$WC4_R5_RUN/compute-start.txt"
if grep -q "$WC4_R5_UUID" "$WC4_R5_RUN/compute-start.txt"; then exit 3; fi
nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{if($1>=1024 || $2>=10)exit 1}'
source migration/wc4_act/activate.sh gym
export WANDB_MODE=disabled
for WC4_R5_INDEX in 0 1 2 3 4; do
    WC4_R5_TRIAL=$(printf 'trial_%02d' "$WC4_R5_INDEX")
    test ! -e "$WC4_R5_RUN/$WC4_R5_TRIAL/attempt_000"
    set +e
    nice -n 10 python -B migration/wc4_act/legdiag_random5/execute.py --run "$WC4_R5_RUN" --index "$WC4_R5_INDEX" > "$WC4_R5_RUN/$WC4_R5_TRIAL/attempt_000.log" 2>&1
    WC4_R5_RC=$?
    set -e
    printf '%s\n' "$WC4_R5_RC" > "$WC4_R5_RUN/$WC4_R5_TRIAL/attempt_000.exit"
    if [ "$WC4_R5_RC" -ne 0 ]; then exit "$WC4_R5_RC"; fi
done
