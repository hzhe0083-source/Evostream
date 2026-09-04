# Baseline archive: Streaming Flow action expert

Archived before replacing the external flow action expert with ephemeral
action queries. Restore with `git checkout streaming-flow-baseline`.

- commit: 5722abf (`feat: add streaming MOSS-Action VLA`)
- tag: `streaming-flow-baseline`
- contract suite: `python3 -m pytest test_contract.py -q` -> **15 passed** (torch 2.8.0+cpu, transformers 4.57.1, h5py 3.16.0)
- no LIBERO rollout numbers were produced on this machine (CPU-only torch, no MOSS-VL checkpoint present)

## What this baseline contained

- `StreamingFlowActionExpert`: external transformer decoder cross-attending to
  three MOSS taps (layers 14/18/23), integrating one action per control tick.
- Per-tick persistent `StreamingActionState(action, step)`.
- Dataset emitted `flow_centers` / `flow_times` for the two-phase flow loss.
- Collator packed a whole episode into one causal sequence and supervised every
  frame-end token.
