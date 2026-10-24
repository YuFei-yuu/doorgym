#!/usr/bin/env bash
set -euo pipefail
cd /data/zyf/video2door-wc4-act
WC4_S7_RUN=${1:?batch path}
export VIDEO2DOOR_GPU=${2:?GPU index}
WC4_S7_WORKER=${3:?worker 0 or 1}
WC4_S7_GROUP="$WC4_S7_RUN/worker_$WC4_S7_WORKER"
mkdir "$WC4_S7_GROUP"
WC4_S7_UUID=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=uuid --format=csv,noheader)
printf '%s\n' "$WC4_S7_UUID" > "$WC4_S7_GROUP/gpu-uuid.txt"
exec 9>"/data/zyf/video2door/.repro/locks/$WC4_S7_UUID.lock"
WC4_S7_LEASE_ACQUIRED=0
wc4_s7_exit() {
    WC4_S7_FINAL_EXIT=$?
    if [[ "$WC4_S7_FINAL_EXIT" != 0 ]]; then
        CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 /data/zyf/video2door/.repro/envs/b1z1-blackwell/bin/python -B migration/wc4_act/newexpert_stage7/set_stop.py --run "$WC4_S7_RUN" --index -1 --stage worker_exit --reason "worker $WC4_S7_WORKER exit $WC4_S7_FINAL_EXIT"
    fi
    if [[ "$WC4_S7_LEASE_ACQUIRED" = 1 ]]; then
        date -u +%FT%TZ > "$WC4_S7_GROUP/lease-released.txt"
        flock -u 9
    fi
}
trap wc4_s7_exit EXIT
flock -n 9
WC4_S7_LEASE_ACQUIRED=1
for WC4_S7_CHECK in 1 2 3; do
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$WC4_S7_GROUP/resource-$WC4_S7_CHECK.txt"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv > "$WC4_S7_GROUP/compute-$WC4_S7_CHECK.txt"
    if grep -q "$WC4_S7_UUID" "$WC4_S7_GROUP/compute-$WC4_S7_CHECK.txt"; then exit 3; fi
    nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{if($1>=1024 || $2>=10)exit 1}'
    sleep 1
done
source migration/wc4_act/activate.sh gym
export WANDB_MODE=disabled
df -B1 "$WC4_S7_RUN" > "$WC4_S7_GROUP/capacity-start.txt"
test "$(df -B1 --output=avail "$WC4_S7_RUN" | tail -1)" -gt 42949672960
date -u +%FT%TZ > "$WC4_S7_GROUP/start.txt"
date -u +%FT%TZ > "$WC4_S7_GROUP/ready.txt"
WC4_S7_OTHER=$((1-WC4_S7_WORKER))
while ! test -f "$WC4_S7_RUN/worker_$WC4_S7_OTHER/ready.txt"; do
    if test -e "$WC4_S7_RUN/SYSTEMIC_STOP.json"; then exit 5; fi
    sleep 1
done
for (( WC4_S7_INDEX=WC4_S7_WORKER; WC4_S7_INDEX<42; WC4_S7_INDEX+=2 )); do
    if test -e "$WC4_S7_RUN/SYSTEMIC_STOP.json"; then exit 5; fi
    while test -e "$WC4_S7_RUN/DIAGNOSTIC_HOLD.json"; do
        if test -e "$WC4_S7_RUN/SYSTEMIC_STOP.json"; then exit 5; fi
        sleep 1
    done
    WC4_S7_TRIAL=$(printf '%s/trial_%02d' "$WC4_S7_RUN" "$WC4_S7_INDEX")
    test ! -e "$WC4_S7_TRIAL/scene-claim.json"
    set +e
    nice -n 10 python -B migration/wc4_act/newexpert_stage7/execute.py --run "$WC4_S7_RUN" --index "$WC4_S7_INDEX" > "$WC4_S7_TRIAL/attempt.log" 2>&1
    WC4_S7_EXIT=$?
    set -e
    printf '%s\n' "$WC4_S7_EXIT" > "$WC4_S7_TRIAL/attempt.exit"
    if [[ "$WC4_S7_EXIT" != 0 ]]; then
        CUDA_VISIBLE_DEVICES= python -B migration/wc4_act/newexpert_stage7/set_stop.py --run "$WC4_S7_RUN" --index "$WC4_S7_INDEX" --stage entry --reason "entry exit $WC4_S7_EXIT"
        exit 5
    fi
    set +e
    CUDA_VISIBLE_DEVICES= "$DOOR_DP_LEROBOT_PYTHON" -B migration/wc4_act/newexpert_stage7/record_check.py --run "$WC4_S7_TRIAL" > "$WC4_S7_TRIAL/record-check.log" 2>&1
    WC4_S7_CHECK_EXIT=$?
    set -e
    printf '%s\n' "$WC4_S7_CHECK_EXIT" > "$WC4_S7_TRIAL/record-check.exit"
    if [[ "$WC4_S7_CHECK_EXIT" != 0 ]]; then
        CUDA_VISIBLE_DEVICES= python -B migration/wc4_act/newexpert_stage7/set_stop.py --run "$WC4_S7_RUN" --index "$WC4_S7_INDEX" --stage raw_records --reason "raw check exit $WC4_S7_CHECK_EXIT"
        exit 6
    fi
    set +e
    CUDA_VISIBLE_DEVICES= python -B migration/wc4_act/newexpert_stage7/review.py --run "$WC4_S7_TRIAL" > "$WC4_S7_TRIAL/task-review.log" 2>&1
    WC4_S7_REVIEW_EXIT=$?
    set -e
    printf '%s\n' "$WC4_S7_REVIEW_EXIT" > "$WC4_S7_TRIAL/task-review.exit"
    if [[ "$WC4_S7_REVIEW_EXIT" != 0 ]]; then
        CUDA_VISIBLE_DEVICES= python -B migration/wc4_act/newexpert_stage7/hold.py --run "$WC4_S7_RUN" --index "$WC4_S7_INDEX" --reason "offline review exit $WC4_S7_REVIEW_EXIT"
        while test -e "$WC4_S7_RUN/DIAGNOSTIC_HOLD.json"; do
            if test -e "$WC4_S7_RUN/SYSTEMIC_STOP.json"; then exit 7; fi
            sleep 1
        done
        test -f "$WC4_S7_TRIAL/audit/task-review.json"
    fi
    printf '%s\n' "$WC4_S7_INDEX" >> "$WC4_S7_GROUP/completed-indices.txt"
done
date -u +%FT%TZ > "$WC4_S7_GROUP/finished.txt"
