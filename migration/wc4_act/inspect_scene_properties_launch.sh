#!/usr/bin/env bash
set -euo pipefail
cd /data/zyf/video2door-wc4-act
WC4_INSPECT_RUN=$PWD/.repro/runs/wc4-fullcycle-properties-r3-20261020
WC4_INSPECT_PLAN=$PWD/.repro/reviews/wc4-fullcycle-r2-independent-20261020-03/geometry-compact.json
test ! -e "$WC4_INSPECT_RUN/scene-creation-receipt.json"
test "$(sha256sum "$WC4_INSPECT_PLAN" | cut -d' ' -f1)" = 872dd74761d51bd241caa89682ccff285321a70ded53d3bc0988cd80fd16be47
export VIDEO2DOOR_GPU=2
WC4_INSPECT_UUID=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=uuid --format=csv,noheader)
exec 9>"/data/zyf/video2door/.repro/locks/$WC4_INSPECT_UUID.lock"
flock -n 9
for WC4_INSPECT_CHECK in 1 2 3; do
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$WC4_INSPECT_RUN/gpu-before-$WC4_INSPECT_CHECK.txt"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv > "$WC4_INSPECT_RUN/compute-before-$WC4_INSPECT_CHECK.txt"
    if grep -q "$WC4_INSPECT_UUID" "$WC4_INSPECT_RUN/compute-before-$WC4_INSPECT_CHECK.txt"; then exit 3; fi
    nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{if($1>=1024 || $2>=10)exit 1}'
    sleep 1
done
source migration/wc4_act/activate.sh gym
git rev-parse HEAD > "$WC4_INSPECT_RUN/execution-head.txt"
git diff > "$WC4_INSPECT_RUN/execution-worktree.patch"
set +e
nice -n 10 python -B migration/wc4_act/inspect_scene_properties.py \
    --output "$WC4_INSPECT_RUN" --geometry "$WC4_INSPECT_PLAN" \
    --rl_device cuda:0 --sim_device cuda:0 --graphics_device_id 0 --num_envs 1 --steps 0 --seed 62000 --headless \
    --door_cfg "$PWD/high-level/data/cfg/b1z1_opendoor.yaml" --door_name wc4 \
    --asset_root "$PWD/high-level/data/asset/a2wz1" --asset_file urdf/a2wz1.urdf \
    --a2w_float_ik_config "$PWD/high-level/float_ik/config/a2w_float_ik_push_door_parallel.yaml" \
    --depth_aug_config "$PWD/high-level/dp/config/depth_camera_aug_default.yaml" \
    --skill_program_json "$PWD/high-level/float_ik/door_twin/examples/wc4_smoke_skill.json" \
    --arm_ik_solver gym_jacobian --ee_pose_frame robot_base_full \
    --camera_intrinsics_mode legacy --no_ikpush_env_randomization --no_enable_depth_camera_randomization \
    --no_enable_depth_noise --no_enable_depth_gaussian_blur --no_collision_geom_check --no_show_camera_images \
    --no_preview_trajectory_at_spawn > "$WC4_INSPECT_RUN/scene.log" 2>&1
WC4_INSPECT_EXIT=$?
printf '%s\n' "$WC4_INSPECT_EXIT" > "$WC4_INSPECT_RUN/scene.exit"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$WC4_INSPECT_RUN/gpu-after.txt"
exit "$WC4_INSPECT_EXIT"
