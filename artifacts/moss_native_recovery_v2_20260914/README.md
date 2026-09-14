# MOSS native recovery v2 artifacts

These are pulled from `/root/Evo_stream_moss_native_recovery_v2_20260914` on
the training host. Checkpoints are intentionally not included.

- `bridge/train_metrics.jsonl`: Bridge training metrics.
- `bridge/fixed_replay.json`: fixed-observation replay and native parity gate.
- `joint/train_metrics.jsonl`: Joint training metrics.
- `evaluations/mt50_scale025/summary_report.json`: completed MT50 report, 450/500 (90.0%).
- `evaluations/mt50_scale025/episodes.jsonl.gz`: per-episode MT50 ledger.
- `evaluations/*/summary_report.json`: selected 4-task and history-scale diagnostics.

The MT50 run used the raw FabriVLA 93K checkpoint, FastAttn/FA2, window 16,
history scale 0.25, 400 environment steps, execution horizon 5, and 50 flow
steps. The report records the full provenance and task-level results.
