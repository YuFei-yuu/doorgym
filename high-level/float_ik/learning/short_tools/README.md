# Analysis and delivery

`deliver.py --root <run> --output <new-directory>` delegates to `analyze.py`.
Use `--episode <relative-folder>` to choose trajectories and
`--pair <expert-folder> <policy-folder> --videos` for a paired H.264 video.

`run_offline.py --root <run> --output <json> --stage <name> --timeout <seconds>
--total-seconds <cap> -- --checkpoint <original-checkpoint>` dispatches
`offline_eval.py` through the shared resource/budget launcher.

The execution adapter writes the metrics described in
[the recording specification](../../../../docs/FLOAT_PUSH_METRICS.md).
