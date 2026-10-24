#!/usr/bin/env bash
set -euo pipefail
cd /data/zyf/video2door-wc4-act
WC4_S6_RUN=${1:?independent run path}
export VIDEO2DOOR_GPU=${2:?GPU index}
test -f "$WC4_S6_RUN/preparation-receipt.json"
test ! -e "$WC4_S6_RUN/launch-start.txt"
date -u +%FT%TZ > "$WC4_S6_RUN/launch-start.txt"
WC4_S6_UUID=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=uuid --format=csv,noheader)
exec 9>"/data/zyf/video2door/.repro/locks/$WC4_S6_UUID.lock"
flock -n 9
trap 'date -u +%FT%TZ > "$WC4_S6_RUN/lease-released.txt"; flock -u 9' EXIT
for WC4_S6_CHECK in 1 2 3; do
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$WC4_S6_RUN/resource-$WC4_S6_CHECK.txt"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv > "$WC4_S6_RUN/compute-$WC4_S6_CHECK.txt"
    if grep -q "$WC4_S6_UUID" "$WC4_S6_RUN/compute-$WC4_S6_CHECK.txt"; then exit 3; fi
    nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{if($1>=1024 || $2>=10)exit 1}'
    sleep 1
done
source migration/wc4_act/activate.sh gym
export WANDB_MODE=disabled
df -B1 "$WC4_S6_RUN" > "$WC4_S6_RUN/capacity-start.txt"
test "$(df -B1 --output=avail "$WC4_S6_RUN" | tail -1)" -gt 42949672960
for WC4_S6_INDEX in 0 1 2 3 4 5 6 7; do
    WC4_S6_TRIAL=$(printf '%s/trial_%02d' "$WC4_S6_RUN" "$WC4_S6_INDEX")
    test ! -e "$WC4_S6_TRIAL/scene-claim.json"
    set +e
    nice -n 10 python -B migration/wc4_act/newexpert_stage6/execute.py --run "$WC4_S6_RUN" --index "$WC4_S6_INDEX" > "$WC4_S6_TRIAL/attempt.log" 2>&1
    WC4_S6_EXIT=$?
    printf '%s\n' "$WC4_S6_EXIT" > "$WC4_S6_TRIAL/attempt.exit"
    set -e
    if [[ "$WC4_S6_EXIT" != 0 ]]; then printf '%s\n' "entry failure at $WC4_S6_INDEX" > "$WC4_S6_RUN/SYSTEMIC_STOP.txt"; exit 5; fi
    set +e
    CUDA_VISIBLE_DEVICES= "$DOOR_DP_LEROBOT_PYTHON" -B migration/wc4_act/newexpert_stage6/record_check.py --run "$WC4_S6_TRIAL" > "$WC4_S6_TRIAL/record-check.log" 2>&1
    WC4_S6_CHECK_EXIT=$?
    set -e
    printf '%s\n' "$WC4_S6_CHECK_EXIT" > "$WC4_S6_TRIAL/record-check.exit"
    if [[ "$WC4_S6_CHECK_EXIT" != 0 ]]; then printf '%s\n' "record check failure at $WC4_S6_INDEX" > "$WC4_S6_RUN/SYSTEMIC_STOP.txt"; exit 6; fi
    set +e
    CUDA_VISIBLE_DEVICES= python -B migration/wc4_act/newexpert_stage6/analyze.py --run "$WC4_S6_TRIAL" > "$WC4_S6_TRIAL/motion-audit.log" 2>&1
    WC4_S6_MOTION_EXIT=$?
    set -e
    printf '%s\n' "$WC4_S6_MOTION_EXIT" > "$WC4_S6_TRIAL/motion-audit.exit"
    if [[ "$WC4_S6_MOTION_EXIT" != 0 ]]; then printf '%s\n' "motion evidence failure at $WC4_S6_INDEX" > "$WC4_S6_RUN/SYSTEMIC_STOP.txt"; exit 7; fi
    printf '%s\n' "$WC4_S6_INDEX" >> "$WC4_S6_RUN/completed-indices.txt"
done
date -u +%FT%TZ > "$WC4_S6_RUN/eight-trials-finished.txt"
