# Floating-base DP support tools

The active design is [upstream whole-body DP with a floating base](../../../docs/FLOAT_PUSH_REPRODUCTION_PLAN.md). Implement its execution adapter before collecting v3 data. Training and inference use `high-level/dp/train_door_dp.py` and `DoorDPPolicyController`.

Retained reusable tools:

- `run_tools/launch.py`, `bounded.py`: GPU sampling, single-process limits and cumulative budget.
- `checks.py`, `dataset_contract.py`: named Cartesian fields, frame alignment, split isolation and original checkpoint train-only statistics.
- `offline_eval.py`: expert-observation errors through the original controller, without error-based task gates. Reads the original config/model/stats checkpoint format.
- `analyze.py`, `short_tools/deliver.py`: trajectory plots, task tables and paired video delivery. Recorded `action_names` determine coordinates.
- `upstream_reuse.py`: NPZ container adapter inheriting original sequence indexing/statistics, when the LeRobot path is unavailable.
- `kinematics.py`, `artifacts.py`: URDF FK, portable paths and strict JSON.

The retired float executors, 39D/11D trainers, reference limiters and experiment schedulers were removed. The source repository's original float scripts and whole-body DP modules remain unchanged.

Metrics and plot definitions: [recording specification](../../../docs/FLOAT_PUSH_METRICS.md). Run `python -m unittest test_analysis` here; run `python run_tools/test_budget.py` for process accounting.
