# ORA0 archive

Saved from `/root/libero_spatial_ora0_v1` on 2026-09-15.

This archive keeps the reproducibility metadata only:

- `logs/`: training logs (about 5.2 MB)
- `eval/`: the 30-trial LIBERO spatial evaluation and per-task records
- `config/`, launch scripts, and top-level logs
- `checkpoint_manifest.tsv`: remote ORA0 checkpoint names, sizes, and timestamps

The large datasets, long-trajectory tensors, core dumps, and 149 GB checkpoint pool remain on the server; they are not needed for the current FabriVLA/MOSS checkpoint upload. The ORA0 evaluation record at `eval/merged.json` is an old diagnostic run: 0/30 success, 0.0%, while expert replay passed at step 80. It should not be used as the current MetaWorld result.
