# Cartesian diagnostic and delivery tools

These entries now accept explicit run paths. They do not depend on deleted
joint-action experiments or assert that old files still exist.

- `deliver.py`: delegates to the CPU `analyze.py` CLI. Requires `--root` and a new
  `--output`; supports repeated `--episode`, explicit `--pair EXPERT POLICY`,
  and `--videos`. Output includes CSV/JSON tables, PNG/PDF plots and paired H.264
  video with full decode checks. Source profiles and initial observations must
  match; failed/incomplete rollouts remain identified.
- `run_offline.py`: resource/budget wrapper for `offline_eval.py`. Requires root,
  output, unique stage, timeout and total-seconds. Extra evaluator args follow `--`.
- `limited_rollout.py`: explicit `ee_reference_limited_v1` ablation. It remains
  unsuitable as a default executor: world reference rate limits alone do not
  coordinate the moving base. No expert planner or door GT is added to policy mode.
- `run_limited.py`: takes root, seed, timeout and total-seconds; verifies the run's
  frozen model hash and dispatches expert/policy under the same explicit limiter.

Original short-run evidence remains in `20261012-ee-short-refined`. New analysis
outputs must use a new directory; do not overwrite its historical delivery files.
Source snapshots may contain historical mixed-interface source for provenance;
they are not active entry points. Removed legacy tool subdirectories are listed
in cleanup manifests; old all-file manifests describe the pre-cleanup snapshot.
