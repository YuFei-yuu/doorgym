# Frozen Cartesian short-run tools

Results and interpretation: [short-run report](../../../../docs/FLOAT_PUSH_EE_SHORT_REPORT.md).
Run: `experiments/float-push/20261012-ee-short-refined` on pro5000. It trained
one fixed 2000-step model from 16 new train and 3 accepted validation episodes.

- `../ee_short_run.py`: the frozen collection/training/direct-validation schedule,
  including source guards, resource samples, a 3600-second cumulative cap and
  immutable attempts. Its plan and predecessor are specific to this experiment.
- `../ee_quality.py`: CPU expert-data quality screening and trajectory plots.
- `limited_rollout.py`: **explicit diagnostic execution profile**, reusing the
  teacher's pure pose-reference arithmetic before generic IK in both expert and
  policy modes. No door state or expert planner is used by the policy executor.
  It annotates config/summary with `ee_reference_limited_v1` and its source hash.
  This wrapper is not the default `FloatEnv` executor or a new trained checkpoint.
- `run_limited.py`: fixed post-failure seeds 8500–8502 and frozen checkpoint;
  same budget ledger, separate `execution_diagnostic/` directories. Stops on error.
- `run_offline.py`: bounded dispatch of the existing teacher-observation diagnostic.
- `deliver.py`: CPU checkpoint/data/source checks, plots, synchronized videos,
  full-file decoding and artifact hashes. Uses the frozen run path.

The direct-execution 8500 failure and unstarted 8501/8502 remain separate from
the added reference-limited diagnostics. Do not merge their outcomes or interpret
numerical door/clearance success without reach/grasp and motion inspection.
These scripts preserve this run; they are not authorization or a reusable seed
allocation for a new training experiment. Original executed copies are in the
server experiment directory. Training data/model/video outputs are not in Git.
