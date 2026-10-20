#!/usr/bin/env bash
set -euo pipefail
cd /data/zyf/video2door-wc4-act
WC4_FULL_RUN=$PWD/.repro/runs/wc4-fullcycle-nominal-r4-20261020
WC4_FULL_MODE=${1:?preflight or attempt_000}
export VIDEO2DOOR_GPU=2
WC4_FULL_UUID=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=uuid --format=csv,noheader)
exec 9>"/data/zyf/video2door/.repro/locks/$WC4_FULL_UUID.lock"
flock -n 9
for WC4_FULL_CHECK in 1 2 3; do
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$WC4_FULL_RUN/resource-check-${WC4_FULL_MODE}-${WC4_FULL_CHECK}.txt"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv > "$WC4_FULL_RUN/compute-check-${WC4_FULL_MODE}-${WC4_FULL_CHECK}.txt"
    if grep -q "$WC4_FULL_UUID" "$WC4_FULL_RUN/compute-check-${WC4_FULL_MODE}-${WC4_FULL_CHECK}.txt"; then exit 3; fi
    nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{if($1>=1024 || $2>=10)exit 1}'
    sleep 1
done
source migration/wc4_act/activate.sh gym
python -B -c 'import isaacgym; import torch,sys,numpy; print(sys.executable,sys.version,torch.__version__,numpy.__version__); x=torch.ones(8,device="cuda"); print("CUDA smoke",(x+x).cpu().tolist())' > "$WC4_FULL_RUN/environment-${WC4_FULL_MODE}.txt" 2>&1
git rev-parse HEAD > "$WC4_FULL_RUN/source-head-${WC4_FULL_MODE}.txt"
git diff > "$WC4_FULL_RUN/worktree-${WC4_FULL_MODE}.patch"
WC4_FULL_EXTRA=()
if [[ "$WC4_FULL_MODE" == preflight ]]; then WC4_FULL_EXTRA=(--preflight); fi
if [[ "$WC4_FULL_MODE" != preflight && "$WC4_FULL_MODE" != attempt_000 ]]; then exit 4; fi
set +e
nice -n 10 python -B migration/wc4_act/fullcycle_expert.py \
    --output "$WC4_FULL_RUN/$WC4_FULL_MODE" \
    --geometry "$WC4_FULL_RUN/geometry-compact.json" --validation "$WC4_FULL_RUN/consistency.json" "${WC4_FULL_EXTRA[@]}" \
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
    > "$WC4_FULL_RUN/$WC4_FULL_MODE.log" 2>&1
WC4_FULL_EXIT=$?
printf '%s\n' "$WC4_FULL_EXIT" > "$WC4_FULL_RUN/$WC4_FULL_MODE.exit"
exit "$WC4_FULL_EXIT"
