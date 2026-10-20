#!/usr/bin/env bash
# Execution-chat entry point. Never invoked by the design-only review.
set -euo pipefail
cd /data/zyf/video2door-wc4-act
WC4_R7_MODE=${1:?preflight or attempt_000}
WC4_R7_GPU=${2:?select an idle GPU index}
WC4_R7_RECEIPT=${3:?absolute path to reviewed consistency.json}
WC4_R7_RUN=$PWD/.repro/runs/wc4-fullcycle-compact-r7-20261021
if [[ "$WC4_R7_MODE" != preflight && "$WC4_R7_MODE" != attempt_000 ]]; then exit 4; fi
test -f "$WC4_R7_RECEIPT"
test ! -e "$WC4_R7_RUN/$WC4_R7_MODE"
mkdir -p "$WC4_R7_RUN"
export VIDEO2DOOR_GPU="$WC4_R7_GPU"
WC4_R7_UUID=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=uuid --format=csv,noheader)
exec 9>"/data/zyf/video2door/.repro/locks/$WC4_R7_UUID.lock"
flock -n 9
for WC4_R7_CHECK in 1 2 3; do
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$WC4_R7_RUN/resource-${WC4_R7_MODE}-${WC4_R7_CHECK}.txt"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv > "$WC4_R7_RUN/compute-${WC4_R7_MODE}-${WC4_R7_CHECK}.txt"
    if grep -q "$WC4_R7_UUID" "$WC4_R7_RUN/compute-${WC4_R7_MODE}-${WC4_R7_CHECK}.txt"; then exit 3; fi
    nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{if($1>=1024 || $2>=10)exit 1}'
    sleep 1
done
source migration/wc4_act/activate.sh gym
git rev-parse HEAD > "$WC4_R7_RUN/source-head-${WC4_R7_MODE}.txt"
git diff > "$WC4_R7_RUN/worktree-${WC4_R7_MODE}.patch"
cp migration/wc4_act/fullcycle_compact.json "$WC4_R7_RUN/geometry-compact.json"
cp "$WC4_R7_RECEIPT" "$WC4_R7_RUN/consistency.json"
WC4_R7_EXTRA=()
if [[ "$WC4_R7_MODE" == preflight ]]; then WC4_R7_EXTRA=(--preflight); fi
set +e
nice -n 10 python -B migration/wc4_act/fullcycle_expert.py \
    --output "$WC4_R7_RUN/$WC4_R7_MODE" \
    --geometry "$WC4_R7_RUN/geometry-compact.json" --validation "$WC4_R7_RUN/consistency.json" "${WC4_R7_EXTRA[@]}" \
    --rl_device cuda:0 --sim_device cuda:0 --graphics_device_id 0 \
    --num_envs 1 --steps 9000 --seed 62000 --headless \
    --enable_wrist_camera --enable_front_camera --camera_depth --camera_seg --depth_only \
    --dp_fps 25 --camera_fps 25 --camera_depth_clip_lower 0.2 --camera_depth_clip_far 1.5 \
    --camera_intrinsics_mode legacy --no_preview_trajectory_at_spawn \
    --door_cfg "$PWD/high-level/data/cfg/b1z1_opendoor.yaml" --door_name wc4 \
    --asset_root "$PWD/high-level/data/asset/a2wz1" --asset_file urdf/a2wz1.urdf \
    --a2w_float_ik_config "$PWD/high-level/float_ik/config/a2w_float_ik_push_door_parallel.yaml" \
    --depth_aug_config "$PWD/high-level/dp/config/depth_camera_aug_default.yaml" \
    --skill_program_json "$PWD/high-level/float_ik/door_twin/examples/wc4_smoke_skill.json" \
    --arm_ik_solver gym_jacobian --ee_pose_frame robot_base_full \
    --no_ikpush_env_randomization --no_enable_depth_camera_randomization \
    --no_enable_depth_noise --no_enable_depth_gaussian_blur --no_collision_geom_check --no_show_camera_images \
    > "$WC4_R7_RUN/$WC4_R7_MODE.log" 2>&1
WC4_R7_EXIT=$?
printf '%s\n' "$WC4_R7_EXIT" > "$WC4_R7_RUN/$WC4_R7_MODE.exit"
exit "$WC4_R7_EXIT"
