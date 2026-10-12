# Floating-base Cartesian Diffusion Policy

The default learning interface is now **DP world-frame end-effector pose → IK →
joint position servos**, with the kinematic floating base retained. The original
`DoorDiffusionPolicy` network file is reused unchanged. There is no locomotion
policy and no walking/balance model.

The frozen joint-action experiment reached step 25000 (best 15000). Those
checkpoints and demonstrations belong to `float_push_joint_v1`; they are **not**
Cartesian models or Cartesian labels. Its incomplete independent test and latest
budget are documented in [`docs/FLOAT_PUSH_TRAINING_NEXT.md`](../../../docs/FLOAT_PUSH_TRAINING_NEXT.md)
and the [historical report](../../../docs/FLOAT_PUSH_LONG_TRAINING_REPORT.md).

## Current contract

`FloatEnv`, `rollout.py` and `train.py` default to `float_push_ee_world_v1`.

| Action indices | Meaning | Units / frame |
| --- | --- | --- |
| 0–2 | Floating-base vx, vy, yaw-rate | m/s, m/s, rad/s; current body frame |
| 3–5 | Absolute end-effector target xyz | m; simulator world frame |
| 6–9 | Absolute target orientation xyzw | Unit quaternion; world frame |
| 10 | Gripper joint position target | rad |

The action is **11D**, retaining lateral base velocity in addition to the
original DP's Cartesian pose/gripper semantics. State is **39D**: the historical
28D robot proprioception plus the previous executed 11D Cartesian command.
Actual EE pose in proprioception remains base-relative; command pose is world-relative.
Reset initializes the previous command to zero base velocity and the measured EE
pose/current gripper target. Control stays at 60 Hz, z=0.60 m, observation / prediction /
execution windows 4/16/4, and 25 inference steps. Vision remains four channels of
two-camera handle mask/masked depth at 54×96, using Gym segmentation.

`step()` normalizes target quaternions, keeps their signs continuous, clips base
velocity and gripper position, moves the base, then runs IK against the current
robot state. Near-zero/nonfinite quaternions and incompatible shapes are rejected.
World xyz is not silently converted to base-relative coordinates or replaced with
a door/handle target. Quaternion normalization is not reported as bound clipping.

The general IK functions are reused directly from the original floating scripts:

- `isaacgym_visualize_b1z1_basearn.setup_ik_controller`
- `isaacgym_float_ik_b1z1_basearn_push_door.set_ik_target`
- `isaacgym_float_ik_b1z1_basearn_push_door.update_arm_ik_targets`

These implement damped Jacobian IK with current-state feedback, then position
servos (`gym.set_actor_dof_position_targets`). Existing floating defaults are
retained: damping 0.08, position gain 1, rotation gain 0.7, rotation weight 0.1,
IK solve increment limit 0.06 rad, servo stiffness/damping 80/8. These are the
floating helper's settings, not the whole-body environment's 400/40 servo gains.
The subsequent joint-limit projection may correct an already out-of-range actual
joint by more than 0.06 rad; that number is not a universal final-command jump bound.
IK cannot guarantee feasible, smooth or successful motion for arbitrary DP targets.

## Expert labels and independence

Expert, policy and action replay all execute through the same Cartesian `step()`.
Only `expert_action()` invokes scripted trajectory planning. Policy mode rejects
that method and has no expert phase, episode clock or door GT in network inputs.
Generic IK is allowed: it tracks only the submitted pose, without choosing a
handle target, phase or recovery trajectory.

Cartesian expert labels are recorded **before IK**, as normalized/applied pose
commands. The old expert sometimes disabled orientation tracking; the fixed pose
contract represents those steps by holding the current orientation after commanded
base rotation. For return/hold-home, the old interpolated joint target is converted
by FK to a Cartesian teacher pose and passed through the same IK; there is no
joint-space return bypass. These changes define a new teacher/execution version,
not a claim that the old expert trajectory is numerically unchanged.

The short Cartesian validation run further corrects the scripted teacher: when
the door-open world contact target freezes, base advance pauses until the
existing return-home phase. Before IK, `expert_pose_reference` limits translation
to 6 mm per control tick, clips the desired reference to a 6 cm neighborhood of
the measured EE, and limits orientation steps to 0.035 rad. Sequential position
limits and subsequent physics mean 6 cm is not a guaranteed post-step tracking
error bound. These limits remove abrupt teacher labels; they are not called by
policy execution. Model, replay and expert still submit the resulting Cartesian
commands to the same generic IK/servo executor. The source snapshot identifies
this teacher revision; do not mix its data with earlier expert versions merely
because their action schema is the same.

Recollect demonstrations. Old executed joint angles do not uniquely recover the
intended pre-IK pose, and simply relabeling the old files is invalid. New NPZs embed
schema and feature names; training validates dimensions, unit quaternions and
previous-action alignment before computing train-only normalization. Mixed schemas
and cross-schema resume are rejected. The new state/action dimensions require new
model parameters and normalization, not resuming the old 15000/25000 checkpoint.

## Entries and records

- `rollout.py`: expert, independent policy, or action replay. Output must be new.
  `--record` saves synchronized state/images/Cartesian action/next_state and raw Gym
  depth; `--video` records every second frame at 30 fps; `--nominal` disables reset
  randomization. `--diagnostic-steps 60` additionally records first-second histories.
- `train.py`: original network and noise-MSE objective, episode manifest split,
  successful demonstrations only, train-only normalization, immutable checkpoints,
  optimizer/sampler/RNG resume within the same schema. Requires ≥8 train and ≥2 val
  episodes. Existing training outputs require explicit `--resume`.
- `check_policy.py`: cached inference versus full network forward for either schema.
- `analyze.py`: records Cartesian targets separately from post-IK joint commands.
- `test_ee_contract.py`: CPU contract, label isolation, FK pose, gradient and model
  sampling tests. `check_ee_sim.py`: bounded simulator probe with expert planning
  blocked, same-seed reset, IK feedback and an untrained policy. Neither is a
  trained-policy performance evaluation.

`trajectory.npz` / recorded episodes retain both the high-level `action` and the
post-IK `joint_position_target`, plus `actual_joint_position`. Metrics additionally
record EE command position/orientation tracking error and the IK-to-current-joint
increment. The previous-action state stores the executed high-level command,
not the post-IK joint targets. Full raw sensor depth may contain Gym -inf values;
processed policy images remain finite.

Use `/data/zyf/video2door/.repro/blackwell/activate.sh` on pro5000. Set
`VIDEO2DOOR_GPU` before activation, after three resource samples two seconds apart
(minimum free memory 16 GiB, mean utilization below 70%). One GPU/process, two CPU
threads, `nice -n 10`, bounded wall time and a ledger. Example arguments, to be run
inside an allocated budget with new output directories:

```bash
python high-level/float_ik/learning/rollout.py \
  --schema float_push_ee_world_v1 --mode expert --seed 7100 --record \
  --output experiments/float-push/<new-run>/data/episode_000
python high-level/float_ik/learning/train.py \
  --schema float_push_ee_world_v1 --manifest <new-manifest.json> \
  --output experiments/float-push/<new-run>/training --steps <approved-limit> \
  --seconds <allocated-seconds>
python high-level/float_ik/learning/rollout.py \
  --schema float_push_ee_world_v1 --mode policy --checkpoint <new-ee-model.pt> \
  --seed 7200 --video --output experiments/float-push/<new-run>/validation/seed_7200
```

Manifest entries keep `episode` (NPZ path), `seed`, `split` (train/val/test),
`success` and `status`. Allocate splits by episode before success filtering and
retain every attempted episode. Example seed values above are not an assertion
that those seeds remain unused; inspect experiment records before selection.

## Historical joint interface

Use explicit `--schema float_push_joint_v1` for old 10D joint-action models/data;
`FloatEnv(schema='float_push_joint_v1')` retains the legacy executor. The frozen
`long_tools/` and `smoke_tools/` drivers now explicitly request this schema.
Their fixed schedules, `checks.py`, audit/report templates and budgets are for
historical joint runs and must not be repurposed as a Cartesian training driver.
`schema.py`'s legacy `SCHEMA/ACTION_NAMES/STATE_NAMES` exports remain for those
historical tools; new callers use `DEFAULT_SCHEMA` and `contract(schema)`.

The task metric still means unlocked + final signed angle ≥80° + rear clearance
≥0.52 m. It does not require sustained grasp, collision-free motion or lateral
whole-body doorway containment. Legacy `open_80deg` actually defaults to 20°.
Keep numerical task success separate from motion-quality assessment.

## Implementation verification

Server evidence: `experiments/float-push/20261012-ee-ik-adapter-check` (separate
600-second implementation-check cap, not the old training ledger). It contains
CPU checks, full nominal Cartesian expert/replay episodes, policy interface
probes, video, source hashes and comparison results. That check verified the
adapter without training a model. The subsequent 2000-step run is described below.

`verification.json` confirms 4 CPU tests and 10 simulator probe checks passed.
Nominal seed 7000 expert and independent replay both completed 2850 steps with
reach/grasp/unlock/task success and a final door angle of 90 degrees. Cartesian
actions, states, joint targets and actual joints match exactly across all frames.
The 47.5-second video is `expert_nominal_final/rollout_h264.mp4`; it shows the
expert, not a trained DP. Total bounded check time was 136.384194 seconds with
zero optimizer updates; the frozen experiment ledger is unchanged.

## Subsequent 2000-step validation

See the [short-run report](../../../docs/FLOAT_PUSH_EE_SHORT_REPORT.md) and
[frozen short-run tools](short_tools/README.md). New expert quality screening
accepted 16 train and 3 validation episodes out of 20 attempts. The fixed final
2000-step model is **not a reliable door-opening policy**: direct execution
diverged and stopped at 368 steps with PhysX CUDA 700. A separate diagnostic
wrapper applying generic pose-reference limits completed three paired runs but
achieved no reach/grasp, and the base drifted far from the expert path. This
wrapper is explicit, is not the default executor, and its results must not replace
the direct-execution failure. World-reference rate limits alone do not keep up
with the allowed moving-base velocity; see the report before using this profile.
Teacher-observation position prediction RMSE was still about 0.221 m. All
attempts, model hashes and videos are retained; the old joint experiment is intact.
