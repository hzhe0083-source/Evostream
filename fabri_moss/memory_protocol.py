"""memory_replay_v1 protocol helper and specification for NativeTrainingDataset."""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


MEMORY_REPLAY_V1_CONFIG: Dict[str, Any] = {
    "protocol_name": "memory_replay_v1",
    "version": "1.2",
    "recent_frames": 4,
    "consolidate_every": 4,
    "memory_slots": 16,
    "spatial_grid": 4,
    "anchor_policy": "snapshot_latest",
    "protect_decision_frames": True,
    "arrival_chunk_choices": [1, 2, 4, 8, 16],
    "merge_rule": "protected_original_frame_intervals",
    "bank_capacity": "episode_growing_protected_anchors_and_interval_summaries",
    "train_weights": {
        "epoch0": {"dense": 0.50, "memory": 0.50, "current_only": 0.0},
        "mature": {"dense": 0.25, "memory": 0.65, "current_only": 0.10},
    },
    "period_choices": [1.0 / 30.0, 2.0 / 30.0, 4.0 / 30.0, 8.0 / 30.0],
    "nonquery_drop_probability": 0.10,
    "bank_update": "detached_projected_visual",
    "prefix": "episode_start",
    "float_tolerance": 1e-8,
}


def get_memory_protocol_contract() -> Dict[str, Any]:
    """Return JSON-serializable stable contract specification for memory_replay_v1."""
    contract = json.loads(json.dumps(MEMORY_REPLAY_V1_CONFIG))
    contract["algorithm_specification"] = {
        "rng_seed_formula": (
            "((seed & 0xFFFFFFFF) ^ ((effective_epoch * 1000003) & 0xFFFFFFFF) ^ "
            "((ep_idx * 7919) & 0xFFFFFFFF) ^ ((target_start_row * 31337) & 0xFFFFFFFF) ^ "
            "salt) & 0xFFFFFFFF, with salt=(20260909 ^ 31) and effective_epoch=(1 if split == 'val' else epoch)"
        ),
        "mode_selection": "Threshold sampling using train_weights.epoch0 or train_weights.mature.",
        "dense_mode": (
            "Identical max 16 old window [max(0, target_end_row - history_frames) .. target_end_row), "
            "with optional single group covering all targets. memory_replay=False."
        ),
        "current_only_mode": (
            "obs_rows=target_rows, each target in its own independent replay group. memory_replay=False."
        ),
        "memory_mode": (
            "Complete causal episode prefix from row 0 to target_end_row - 1 at chosen real period. "
            "Build raw lattice advancing clock even on dropped ticks, then drop non-query ticks with probability 0.10. "
            "Union with ALL target_rows so every target current observation exists. "
            "Decision frames partitioned from prefix [0, first_target_idx) using arrival chunks (1, 2, 4, 8, 16) "
            "and all target pool indices. All decision frames are protected under snapshot_latest policy. "
            "Do not restrict to 16 observations total; entire prefix is retained. "
            "No interleaved subgroups for memory; memory_replay=True tells training wrapper to maintain memory bank."
        ),
        "calendar_and_retirement": (
            "Calendar audit tracks observation frame counts per target query, derived counts, "
            "retired_anchor_count (retired decision frames), retired_summary_bin_count (unique partitions of non-decision retired frames), "
            "and protected_frame_ids (decision frame IDs <= query)."
        ),
        "bank_architecture": (
            "memory_slots=16 is legacy without anchors and does not limit version 1.2; "
            "summary bank retains visual tokens plus protected snapshot decision tokens plus recent frames "
            "using detached bank updates."
        ),
    }
    return contract


def compute_memory_replay_layout(
    seed: int,
    epoch: int,
    ep_idx: int,
    target_start_row: int,
    target_end_row: int,
    all_timestamps: Sequence[float],
    history_frames: int = 16,
    split: str = "train",
    all_frame_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Compute deterministic memory_replay_v1 layout and metadata for a segment.

    Parameters
    ----------
    seed : int
        Dataset seed.
    epoch : int
        Current epoch (non-negative).
    ep_idx : int
        Episode index (non-negative).
    target_start_row : int
        Start row of targets in episode dataframe.
    target_end_row : int
        End row of targets in episode dataframe.
    all_timestamps : Sequence[float]
        Validated timestamps for all rows in the episode.
    history_frames : int
        Dataset max history frames (positive, default 16).
    split : str
        'train' or 'val'. For 'val', always fixed mature phase (epoch 1).
    all_frame_ids : Optional[Sequence[int]]
        Original frame IDs for all rows in the episode. If None, defaults to
        list(range(len(all_timestamps))) for test compatibility. Must be strictly
        increasing, non-negative, and match len(all_timestamps).

    Returns
    -------
    dict with:
        "obs_rows": List[int] (sorted unique row indices)
        "replay_groups": Optional[List[Dict[str, Any]]]
        "stream_layout": Dict[str, Any]
        "memory_replay": bool
    """
    if type(history_frames) is not int or history_frames <= 0:
        raise ValueError(f"history_frames must be a positive integer, got {history_frames}")
    if type(epoch) is not int or epoch < 0:
        raise ValueError(f"epoch must be a non-negative integer, got {epoch}")
    if type(ep_idx) is not int or ep_idx < 0:
        raise ValueError(f"ep_idx must be a non-negative integer, got {ep_idx}")
    if type(target_start_row) is not int or type(target_end_row) is not int:
        raise ValueError("target_start_row and target_end_row must be integers")
    if target_start_row < 0 or target_end_row < target_start_row:
        raise ValueError(f"Invalid target range [{target_start_row}, {target_end_row})")
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    N_total = len(all_timestamps)
    if all_frame_ids is None:
        frame_ids_seq: Sequence[int] = list(range(N_total))
    else:
        frame_ids_seq = all_frame_ids
        if len(frame_ids_seq) != N_total:
            raise ValueError(
                f"all_frame_ids length {len(frame_ids_seq)} does not match all_timestamps length {N_total}"
            )
        for idx, fid in enumerate(frame_ids_seq):
            if type(fid) is not int and not (hasattr(fid, "__index__") and isinstance(int(fid), int)):
                raise ValueError(f"all_frame_ids contains non-integer value at index {idx}: {fid!r}")
            fid_int = int(fid)
            if fid_int < 0:
                raise ValueError(f"all_frame_ids contains negative value at index {idx}: {fid_int}")
            if idx > 0 and fid_int <= int(frame_ids_seq[idx - 1]):
                raise ValueError(
                    f"all_frame_ids must be strictly increasing, but index {idx} ({fid_int}) <= index {idx - 1} ({frame_ids_seq[idx - 1]})"
                )

    target_rows = list(range(target_start_row, target_end_row))
    M = len(target_rows)
    if M <= 0:
        raise ValueError(f"Empty target rows [{target_start_row}, {target_end_row})")
    if M > history_frames:
        raise ValueError(f"Target count M={M} exceeds history_frames={history_frames}")
    if target_end_row > len(all_timestamps):
        raise ValueError(
            f"target_end_row {target_end_row} exceeds all_timestamps length {len(all_timestamps)}"
        )

    # Local RNG independent of global state, worker, or rank
    # Salt formula: 20260909 ^ 31 (stable extra salt 31)
    effective_epoch = 1 if split == "val" else epoch
    salt = 20260909 ^ 31
    rng_seed = (
        (seed & 0xFFFFFFFF)
        ^ ((effective_epoch * 1000003) & 0xFFFFFFFF)
        ^ ((ep_idx * 7919) & 0xFFFFFFFF)
        ^ ((target_start_row * 31337) & 0xFFFFFFFF)
        ^ salt
    ) & 0xFFFFFFFF
    local_rng = random.Random(rng_seed)

    # Determine mode using weights from MEMORY_REPLAY_V1_CONFIG
    weights = (
        MEMORY_REPLAY_V1_CONFIG["train_weights"]["epoch0"]
        if effective_epoch == 0
        else MEMORY_REPLAY_V1_CONFIG["train_weights"]["mature"]
    )
    r_mode = local_rng.random()
    p_dense = weights["dense"]
    p_memory = weights["memory"]
    if r_mode < p_dense:
        mode = "dense"
    elif r_mode < p_dense + p_memory:
        mode = "memory"
    else:
        mode = "current_only"

    if mode == "dense":
        context_start_row = max(0, target_end_row - history_frames)
        obs_rows = list(range(context_start_row, target_end_row))
        obs_to_idx = {r: i for i, r in enumerate(obs_rows)}
        target_positions = list(range(M))
        group_obs_indices = [obs_to_idx[r] for r in obs_rows]
        prefix_len = len(obs_rows) - M
        replay_groups = [
            {
                "observation_indices": group_obs_indices,
                "target_positions": target_positions,
            }
        ]
        stream_layout = {
            "mode": "dense",
            "requested_period": None,
            "full_prefix_span_seconds": float(all_timestamps[obs_rows[-1]] - all_timestamps[obs_rows[0]]),
            "num_observations": len(obs_rows),
            "groups": [
                {
                    "target_positions": target_positions,
                    "target_rows": target_rows,
                    "obs_rows": obs_rows,
                    "visible_frame_counts": [prefix_len + k + 1 for k in range(M)],
                    "history_count": prefix_len,
                    "prefix_n": prefix_len,
                    "delta_counts": [1] * M,
                    "selected_capacity": len(obs_rows),
                    "start_time": float(all_timestamps[obs_rows[0]]),
                    "end_time": float(all_timestamps[obs_rows[-1]]),
                    "span_seconds": float(all_timestamps[obs_rows[-1]] - all_timestamps[obs_rows[0]]),
                    "actual_span_seconds": float(all_timestamps[obs_rows[-1]] - all_timestamps[obs_rows[0]]),
                }
            ],
        }
        return {
            "obs_rows": obs_rows,
            "replay_groups": replay_groups,
            "stream_layout": stream_layout,
            "memory_replay": False,
        }

    if mode == "current_only":
        obs_rows = list(target_rows)
        obs_to_idx = {r: i for i, r in enumerate(obs_rows)}
        replay_groups = []
        groups_layout = []
        for pos, r in enumerate(target_rows):
            replay_groups.append({
                "observation_indices": [obs_to_idx[r]],
                "target_positions": [pos],
            })
            groups_layout.append({
                "target_positions": [pos],
                "target_rows": [r],
                "obs_rows": [r],
                "visible_frame_counts": [1],
                "history_count": 0,
                "prefix_n": 0,
                "delta_counts": [1],
                "selected_capacity": 1,
                "start_time": float(all_timestamps[r]),
                "end_time": float(all_timestamps[r]),
                "span_seconds": 0.0,
                "actual_span_seconds": 0.0,
            })
        stream_layout = {
            "mode": "current_only",
            "requested_period": None,
            "full_prefix_span_seconds": float(all_timestamps[obs_rows[-1]] - all_timestamps[obs_rows[0]]),
            "num_observations": len(obs_rows),
            "groups": groups_layout,
        }
        return {
            "obs_rows": obs_rows,
            "replay_groups": replay_groups,
            "stream_layout": stream_layout,
            "memory_replay": False,
        }

    # mode == "memory"
    period = local_rng.choice(MEMORY_REPLAY_V1_CONFIG["period_choices"])
    drop_prob = MEMORY_REPLAY_V1_CONFIG["nonquery_drop_probability"]
    tol = MEMORY_REPLAY_V1_CONFIG["float_tolerance"]
    target_rows_set = set(target_rows)

    # 1. Complete causal episode prefix from row 0 to target_end_row - 1 at chosen real period
    # Build raw lattice advancing clock even on dropped ticks
    raw_lattice_rows: List[int] = []
    last_lattice_time: Optional[float] = None
    for r in range(0, target_end_row):
        t = all_timestamps[r]
        if last_lattice_time is None:
            raw_lattice_rows.append(r)
            last_lattice_time = t
        elif (t - last_lattice_time) >= (period - tol):
            raw_lattice_rows.append(r)
            last_lattice_time = t

    # 2. Drop non-query ticks with probability nonquery_drop_probability
    selected_lattice_rows: List[int] = []
    for r in raw_lattice_rows:
        if r in target_rows_set:
            selected_lattice_rows.append(r)
        else:
            if local_rng.random() >= drop_prob:
                selected_lattice_rows.append(r)

    # 3. Union with ALL target_rows so every target current observation exists.
    obs_set = set(selected_lattice_rows)
    obs_set.update(target_rows)
    obs_rows = sorted(obs_set)

    # 4. Compute decision frames in the observation pool
    # Find first target index in obs_rows
    obs_to_idx = {r: i for i, r in enumerate(obs_rows)}
    target_pool_indices = [obs_to_idx[r] for r in target_rows]
    first_idx = target_pool_indices[0]

    # Partition prefix [0, first_idx) using arrival chunks sampled from arrival_chunk_choices
    chunk_choices = MEMORY_REPLAY_V1_CONFIG["arrival_chunk_choices"]
    prefix_decision_indices: List[int] = []
    curr_start = 0
    while curr_start < first_idx:
        c = local_rng.choice(chunk_choices)
        chunk_end = min(curr_start + c, first_idx)
        # Last index of this chunk is decision frame
        prefix_decision_indices.append(chunk_end - 1)
        curr_start = chunk_end

    # All target pool indices are decision frames
    all_decision_indices = sorted(set(prefix_decision_indices + target_pool_indices))

    # stream_layout fields for decision frames
    decision_frame_ids = [int(frame_ids_seq[obs_rows[i]]) for i in all_decision_indices]
    decision_delta_counts: List[int] = []
    for idx_pos, d_idx in enumerate(all_decision_indices):
        if idx_pos == 0:
            decision_delta_counts.append(d_idx + 1)
        else:
            decision_delta_counts.append(d_idx - all_decision_indices[idx_pos - 1])

    # 5. Derived calendar information per target query
    # visible_counts up to each target
    visible_counts = [sum(1 for r in obs_rows if r <= tgt_r) for tgt_r in target_rows]
    deltas: List[int] = []
    recent_frames = MEMORY_REPLAY_V1_CONFIG["recent_frames"]
    consolidate_every = MEMORY_REPLAY_V1_CONFIG["consolidate_every"]

    calendar_queries: List[Dict[str, Any]] = []
    for i_tgt, tgt_r in enumerate(target_rows):
        # Observed prefix rows up to this target query
        prefix_obs_rows = [r for r in obs_rows if r <= tgt_r]
        vis = len(prefix_obs_rows)
        if i_tgt == 0:
            d = vis
        else:
            d = vis - visible_counts[i_tgt - 1]
        deltas.append(d)

        # Retirement triggers: initial retirement when count reaches recent_frames + consolidate_every (8),
        # and subsequently every consolidate_every (4) frames.
        if vis < recent_frames + consolidate_every:
            retirements = 0
            retired_frames = 0
            recent_count = vis
        else:
            retirements = 1 + (vis - (recent_frames + consolidate_every)) // consolidate_every
            retired_frames = retirements * consolidate_every
            recent_count = vis - retired_frames

        # The oldest `retired_frames` frames in the observed prefix are retired
        # Pool indices for prefix_obs_rows are 0, 1, ..., vis - 1
        # Decision pool indices <= vis - 1:
        # Decisions in prefix:
        decisions_in_prefix = [d_idx for d_idx in all_decision_indices if d_idx < vis]
        decisions_in_prefix_set = set(decisions_in_prefix)

        # Retired indices are pool indices 0 .. retired_frames - 1
        retired_decisions = [d_idx for d_idx in decisions_in_prefix if d_idx < retired_frames]
        retired_anchor_count = len(retired_decisions)

        # retired_summary_bin_count: number of unique partitions of non-decision retired frames
        # Partition of index i is max {d in all_decision_indices | d < i}, or -1 if none
        # (i.e. partitioned by the nearest preceding decision index)
        import bisect
        bins = set()
        for i in range(retired_frames):
            if i not in decisions_in_prefix_set:
                pos = bisect.bisect_left(all_decision_indices, i)
                bin_id = all_decision_indices[pos - 1] if pos > 0 else -1
                bins.add(bin_id)
        retired_summary_bin_count = len(bins)

        # protected_frame_ids: original frame IDs for decisions in the prefix up to this target query (d_idx < vis)
        protected_frame_ids = [int(frame_ids_seq[obs_rows[d_idx]]) for d_idx in decisions_in_prefix]

        calendar_queries.append({
            "target_position": i_tgt,
            "target_row": tgt_r,
            "target_time": float(all_timestamps[tgt_r]),
            "visible_observations": vis,
            "delta_from_previous": d,
            "retirement_triggers": retirements,
            "retired_frames_count": retired_frames,
            "recent_frames_count": recent_count,
            "retired_anchor_count": retired_anchor_count,
            "retired_summary_bin_count": retired_summary_bin_count,
            "protected_frame_ids": protected_frame_ids,
        })

    span_seconds = float(all_timestamps[obs_rows[-1]] - all_timestamps[obs_rows[0]])
    full_prefix_span = float(all_timestamps[obs_rows[-1]] - all_timestamps[0])

    stream_layout = {
        "mode": "memory",
        "requested_period": float(period),
        "full_prefix_span_seconds": full_prefix_span,
        "span_seconds": span_seconds,
        "num_observations": len(obs_rows),
        "raw_lattice_ticks": len(raw_lattice_rows),
        "selected_lattice_ticks": len(selected_lattice_rows),
        "target_rows": target_rows,
        "visible_frame_counts": visible_counts,
        "delta_counts": deltas,
        "decision_indices": all_decision_indices,
        "decision_frame_ids": decision_frame_ids,
        "decision_delta_counts": decision_delta_counts,
        "protected_policy": MEMORY_REPLAY_V1_CONFIG["anchor_policy"],
        "calendar_queries": calendar_queries,
    }

    # In memory mode, no interleaved subgroups; global targets 8 each queried at its own memory bank state
    # Replay groups is None (or omitted), memory_replay=True tells training wrapper
    return {
        "obs_rows": obs_rows,
        "replay_groups": None,
        "decision_indices": all_decision_indices,
        "stream_layout": stream_layout,
        "memory_replay": True,
    }
