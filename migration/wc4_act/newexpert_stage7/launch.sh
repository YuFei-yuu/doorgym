#!/usr/bin/env bash
set -euo pipefail
cd /data/zyf/video2door-wc4-act
WC4_S7_RUN=${1:?batch path}
test -f "$WC4_S7_RUN/preparation-receipt.json"
test ! -e "$WC4_S7_RUN/launch-start.txt"
date -u +%FT%TZ > "$WC4_S7_RUN/launch-start.txt"
set +e
bash migration/wc4_act/newexpert_stage7/launch_worker.sh "$WC4_S7_RUN" 1 0 > "$WC4_S7_RUN/worker-0.log" 2>&1 &
WC4_S7_PID0=$!
bash migration/wc4_act/newexpert_stage7/launch_worker.sh "$WC4_S7_RUN" 2 1 > "$WC4_S7_RUN/worker-1.log" 2>&1 &
WC4_S7_PID1=$!
printf '%s %s\n' "$WC4_S7_PID0" "$WC4_S7_PID1" > "$WC4_S7_RUN/worker-pids.txt"
wait "$WC4_S7_PID0"; WC4_S7_EXIT0=$?
wait "$WC4_S7_PID1"; WC4_S7_EXIT1=$?
printf '%s %s\n' "$WC4_S7_EXIT0" "$WC4_S7_EXIT1" > "$WC4_S7_RUN/worker-exits.txt"
date -u +%FT%TZ > "$WC4_S7_RUN/scheduler-exit.txt"
if [[ "$WC4_S7_EXIT0" != 0 || "$WC4_S7_EXIT1" != 0 ]]; then exit 8; fi
