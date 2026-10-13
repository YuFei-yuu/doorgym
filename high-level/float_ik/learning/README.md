# Floating-base DP support tools

The active design is [upstream whole-body DP with a floating base](../../../docs/FLOAT_PUSH_REPRODUCTION_PLAN.md). `wholebody_float.py` supplies the source split-asset floating backend to the original whole-body task loop. Use `--float_execution source` for the frozen source pose API / CPU tensor pipeline with GPU PhysX. The nominal expert and exact command replay reached 90 degrees and completed the original pass flow. All 20 expert collection attempts succeeded. Original uncached training completed 5000 updates; frozen-model evaluation completed without retraining: validation opened 1/3 and passed 0/3, test opened 0/3 and passed 0/3. The model has not met the task objective. See [results](../../../docs/FLOAT_PUSH_V3_RESULTS.md) and [v3 progress](../../../docs/FLOAT_PUSH_V3_PROGRESS.md).

Retained reusable tools:

- `run_tools/launch.py`, `bounded.py`: GPU sampling, single-process limits and cumulative budget.
- `checks.py`, `dataset_contract.py`: named Cartesian fields, frame alignment, split isolation and original checkpoint train-only statistics.
- `offline_eval.py`: expert-observation errors through the original controller, without error-based task gates. Reads the original config/model/stats checkpoint format.
- `analyze.py`, `short_tools/deliver.py`: trajectory plots, task tables and paired video delivery. Recorded `action_names` determine coordinates.
- `upstream_reuse.py`: NPZ container adapter inheriting original sequence indexing/statistics, when the LeRobot path is unavailable.
- `kinematics.py`, `artifacts.py`: URDF FK, portable paths and strict JSON.
- `analyze_v3.py`: plots, CSV metrics and FK/Jacobian audits for the new source-task backend.
- `resume_eval_v3.py`: evaluation-only continuation from a verified frozen 5000-update checkpoint; never invokes training.
- `run_tools/activate_train.sh`: isolated pro5000 Conda training environment; disables user-site packages.

The retired float executors, 39D/11D trainers, reference limiters and experiment schedulers were removed. The source repository's original float scripts and whole-body DP modules remain unchanged.

Metrics and plot definitions: [recording specification](../../../docs/FLOAT_PUSH_METRICS.md). Run `python -m unittest test_analysis` here; run `python run_tools/test_budget.py` for process accounting.
