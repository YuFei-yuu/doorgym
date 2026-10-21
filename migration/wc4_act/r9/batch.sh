#!/usr/bin/env bash
set -euo pipefail
cd /data/zyf/video2door-wc4-act
WC4_R9_RUN=$PWD/.repro/runs/wc4-fulltask-expand-r9-20261021
for WC4_R9_INDEX in "$@"; do
    test ! -e "$WC4_R9_RUN/SYSTEMIC_STOP.json"
    bash migration/wc4_act/r9/launch.sh preflight "$WC4_R9_INDEX" 2
    bash migration/wc4_act/r9/launch.sh attempt "$WC4_R9_INDEX" 2
    if ! bash -c 'source migration/wc4_act/activate.sh act; python -B migration/wc4_act/r9/validate.py --run "$1" --index "$2"' _ "$WC4_R9_RUN" "$WC4_R9_INDEX" > "$WC4_R9_RUN/validation-$WC4_R9_INDEX.log" 2>&1; then
        echo '{"reason":"recording validation failed; inspect logs before further collection"}' > "$WC4_R9_RUN/SYSTEMIC_STOP.json"
        exit 4
    fi
    /data/zyf/video2door/.repro/envs/b1z1-blackwell/bin/python -B migration/wc4_act/r9/review.py "$WC4_R9_INDEX" > "$WC4_R9_RUN/review-$WC4_R9_INDEX.log" 2>&1
    /data/zyf/video2door/.repro/envs/b1z1-blackwell/bin/python -B migration/wc4_act/r9/audit.py --run "$WC4_R9_RUN" --index "$WC4_R9_INDEX" --mode screen
done
