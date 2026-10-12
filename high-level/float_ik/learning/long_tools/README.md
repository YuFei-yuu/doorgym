# Controlled continuation (historical joint schema)

These tools remain pinned to `float_push_joint_v1`. The current standalone
`rollout.py` / `train.py` default to `float_push_ee_world_v1`; driver dispatches
explicitly pass the legacy schema. Do not use this fixed schedule or its audit/
report templates for new Cartesian data or checkpoints. See the parent README
for the new EE → IK → servo contract.

Current handoff: [`docs/FLOAT_PUSH_TRAINING_NEXT.md`](../../../../docs/FLOAT_PUSH_TRAINING_NEXT.md).
The completed controlled run reached step 25000; best is 15000 and last is
25000. Independent testing stopped after a PhysX CUDA error (30 started,
29 completed, 90 unstarted). Its selection is frozen. The resource-stop
continuation command below describes historical recovery and must not reopen
this run for more optimizer updates. See the report in `docs/` before any work.

Use `driver.py --root <new-run> --old <smoke-run>` on pro5000. The original smoke
driver is archived and must not launch this experiment. This driver holds a
process lock, samples resources, accounts every child process, and reserves
final testing before dispatching more training. The total is 21600 seconds,
including initialization, checkpoint saves and termination. A supervisor loss
charges its entire assigned timeout after confirming the child is dead.

The formal branch preserves the exact old manifest, normalization, model,
AdamW state and RNG. It advances 5000 -> 7000 -> ... -> 25000, validating all
seeds 4000–4009 after every 2000 new steps. Three consecutive complete points
without improved success count or mean final angle (>0.001 degree) stop it.
Loss improvement does not reset patience. No training episodes are added.

Paired 25/100-step sampling uses the old 5000 checkpoint on 4000–4002; formal
continuation/test stay at 25. Test seeds 5000–5029 are accessed only after
selection is frozen. Every started attempt is retained, including errors.

`audit.py` checks train-only normalization and early-loop synchronization,
coordinates, previous action, history/queue, raw depth, and input distributions.
`test_budget.py` exercises the actual timeout, reserve, lock and recovery paths.
CPU-only delivery is separate from the GPU driver. Preserve all old results.

Historical resource-stop recovery (before independent testing) uses
`driver.py --root <same-run> --old <smoke-run> --resume-continuation 21000`.
It seals the original ledger prefix and the 15000/21000 checkpoint hashes,
archives the provisional selection, verifies no independent test has started,
reconstructs patience from all complete validation points, and evaluates an
existing pending checkpoint without repeating its optimizer updates.
`test_resume.py` validates these recovery guards on CPU.
Repeated explicitly authorized resource-stop recoveries archive each previous
audit under `resume_events/`, preserve checkpoint and ledger-prefix hashes,
and reuse completed validation points. Once an independent test is started,
the continuation entry point rejects reopening model selection.
