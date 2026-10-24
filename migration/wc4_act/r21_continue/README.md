# R21 authorized150155 continuation

Only remaining49845 updates and200k16 trials. Original R21 sources/run/STOPPED/FINAL preserved. Sampling copies original125000-row schedule byte-for-byte and starts DataLoader at originalrow75155. Per-update audit retains originalordinal completed−75000.

ProgressWatch accumulates only active-training lag greater than existing100-step reporting cadence; evaluation(childPID bound to trainer) clears timer, resume starts a fresh600s window. Six API failures still stop even while evaluating. Disk40GiB and UUID conflict checks remain unconditional. Existing per-key explicit-step W&B calls default to pending rows; continuation wrapper explicitly commits the complete metric row once at each existing log callback. Local optimizer journal remains authoritative.

Author checkpoint restoration is checked at load, then all model/optimizer/parameter-groups/normalizer/RNG checks repeat immediately before first new optimizer update. Committed model RNG is reapplied after new DataLoader startup, before first forward; this is disclosed engineering restoration, not a worker/prefetch cursor continuity claim. No optimizer reload/reset at this check.
