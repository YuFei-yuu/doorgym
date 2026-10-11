# Archived first-stage orchestration

These files were synchronized from
`/data/zyf/video2door/experiments/float-push/20261011-224552-smoke/tools/`
after the completed first-stage experiment. They preserve the actual budget
guard, sequential driver, metric dictionary, diagnostic and delivery tools.

They are **not a ready-to-run formal-training launcher**. `bounded.py` has the
original 7200-second total limit. `driver.py` contains first-stage stage limits,
seed ranges and checkpoint/evaluation decisions. `build_report.py` targets the
original local report directory. Some provenance files and baseline wrappers
remain under the server's existing `.repro/` setup.

For a new run, first read `docs/FLOAT_PUSH_TRAINING_NEXT.md`, create a new run
directory, and explicitly configure its budgets, manifests and seeds. Do not
edit the historical run's ledger or overwrite checkpoints to extend it.

For faithful first-stage reproduction, the original layout expects these
scripts under `<run>/tools/`. Commands and dependencies are recorded in the
server run and local result delivery. Large datasets/checkpoints/videos and
generated experiment outputs are intentionally not committed to Git.
