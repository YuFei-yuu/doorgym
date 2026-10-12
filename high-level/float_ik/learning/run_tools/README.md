# Shared resource and budget launcher

`launch.py` samples GPUs three times, two seconds apart, requires at least16GiB
free memory and mean utilization below70%, activates the project environment,
and dispatches one child through `bounded.py`. Children use one GPU, nice10 and
two math/PhysX threads. The ledger includes failed and interrupted children.

```bash
python high-level/float_ik/learning/run_tools/launch.py \
  --root <run> --stage <unique-stage> --timeout <seconds> \
  --total-seconds <approved-cap> -- python <script> <arguments>
```

At a new run, copy the retained cumulative budget ledger before launching.
Current task cap7200s; prior charges176.587700s. Preserved ledger location is in
[the handoff](../../../../docs/FLOAT_PUSH_TRAINING_NEXT.md).

`python test_budget.py` checks timeout, accounting, recovery and mutual exclusion.
