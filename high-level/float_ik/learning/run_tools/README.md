# Shared bounded execution

`bounded.py` and `test_budget.py` were preserved from the retired orchestration
and moved here without changing their accounting/lock/timeout behavior.
`launch.py` samples GPU resources three times, selects one eligible GPU, activates
the repository environment and runs one child through that accounting layer.
No old experiment schedule, seed range or fixed budget is inherited.

```bash
python high-level/float_ik/learning/run_tools/launch.py \
  --root <new-run> --stage <unique-stage> --timeout <seconds> \
  --total-seconds <run-cap> -- python <script> <arguments>
```

`ee_short_run.py --root <new-run> --plan <plan.json> init` requires an explicit
plan with schema, total_seconds, optimizer_steps, train_seconds, train_seeds,
offline_val_seeds, rollout_val_seeds, pilot_seed, inference_steps, teacher and
checkpoint_selection. Seed sets must be disjoint. The other stages are pilot,
collect, train and evaluate; pilot/dataset quality gates must pass before
collection/training. A plan is a recorded configuration, not a renewed resource
authorization. Source changes after initialization require a new run.

The existing `ee_reference_limited_v1` profile remains diagnostic only; fixing
moving-base reference coordination is separate work. See the current handoff.
