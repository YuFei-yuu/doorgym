# Floating-base push learning adapter

This directory adapts the existing floating-base expert; it does not use the
locomotion controller in `door_rl` or the old DP recording wrapper. The original
expert and `DoorDiffusionPolicy` network are unchanged.

Use `/data/zyf/video2door/.repro/blackwell/activate.sh` on ps1. Set
`VIDEO2DOOR_GPU` before activation, after checking three resource samples.
Run one GPU job at a time, with two CPU threads and `nice -n 10`.

## Contract

`schema.py` defines `float_push_joint_v1`: body-frame base vx/vy/yaw-rate,
six arm joint position targets and one gripper position target. This 10D schema
is **incompatible** with the original DP 10D Cartesian action. The checkpoint
loader checks names, dimensions and schema. The base is kinematic at z=0.60 m;
there is no walking or balance model.

`FloatEnv.reset(seed, config)`, `observe()`, `step(action)` and `close()` provide
60 Hz control. Policy construction disables expert trajectory and IK functions.
Door angle/handle pose/phase are diagnostic only; the original physical lock is
retained and automatic opening torque is zero.

Policy input: 38D proprioception, four frames of two-camera handle masks and
masked depth (54x96). Gym segmentation is declared simulator perception input.
No episode clock, expert phase or door state enters the network. Seven joint
positions and velocities use radians. Quaternion order is xyzw.

## Entries

- `rollout.py`: expert, independent policy, or interface-only action replay.
  Output directories must be new. Every launched episode has a summary,
  including failures. `--record` saves synchronized NPZ; `--video` saves every
  second simulation frame at 30 fps. `--nominal` disables reset randomization.
- `train.py`: successful train episodes only, episode-level manifest split,
  train-only normalization, fixed validation noise, immutable step checkpoints
  and `--resume` with optimizer/sampler/RNG state.
- `check_policy.py`: cached inference equals full model forward.
- `checks.py`: synchronization, replay, reset, boundaries and data isolation.
- `offline_eval.py`: action error/timing on recorded validation observations;
  this is not a closed-loop score.
- `analyze.py`: CPU charts, aggregates, labeled H.264 export and full decode.

Early smoke episodes store processed metric `depth_m` (finite replacement and
0–2 m clipping), not untouched sensor depth, and lack an explicit terminal
`next_state`. The corrected recorder also saves `gym_depth_m` with original Gym
negative depth/-inf values and `next_state` for every transition. The separate
`recording-audit` episode checks these fields; it does not replace historical
training data or enter model selection.

The original metric name `open_80deg` defaults to **20 degrees**. The new task
success is unlocked + final signed opening >=80 degrees + rear projection
>=0.52 m; lateral doorway containment and leg contacts are not checked.
`signed_open_ratio` preserves the expert's absolute-ratio implementation;
direction is checked separately. Report all attempted episodes, not only the
successful demonstrations.

The first smoke run is `experiments/float-push/20261011-224552-smoke`.
Its bounded driver and wall-time ledger live under that run's `tools/` and
`budget.jsonl`. The first stage has a 7200-second total cap; formal training
requires user confirmation and a new run. See the delivered Chinese report
`/Users/yu/Data/Video2Door/FLOAT_PUSH_SMOKE_REPORT.md` for measured outcomes,
limitations, commands and artifact paths.
