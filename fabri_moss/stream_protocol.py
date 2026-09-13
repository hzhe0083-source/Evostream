"""stream_replay_v1 protocol helper and specification for NativeTrainingDataset."""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


STREAM_REPLAY_V1_CONFIG: Dict[str, Any] = {
    "protocol_name": "stream_replay_v1",
    "version": "1.0",
    "epoch0_weights": {"dense": 0.50, "stream": 0.50, "current_only": 0.0},
    "mature_weights": {"dense": 0.25, "stream": 0.65, "current_only": 0.10},
    "query_strides": [2, 4],
    "candidate_periods": [1.0 / 30.0, 2.0 / 30.0, 4.0 / 30.0, 8.0 / 30.0],
    "nonquery_drop_probability": 0.10,
    "capacities": [4, 8, 16],
    "float_tolerance": 1e-8,
}


def get_stream_protocol_contract() -> Dict[str, Any]:
    """Return JSON-serializable stable contract specification for stream_replay_v1."""
    contract = json.loads(json.dumps(STREAM_REPLAY_V1_CONFIG))
    contract["algorithm_specification"] = {
        "rng_seed_formula": (
            "((seed & 0xFFFFFFFF) ^ ((effective_epoch * 1000003) & 0xFFFFFFFF) ^ "
            "((ep_idx * 7919) & 0xFFFFFFFF) ^ ((target_start_row * 31337) & 0xFFFFFFFF) ^ "
            "salt) & 0xFFFFFFFF, with salt=20260909 and effective_epoch=(1 if split == 'val' else epoch)"
        ),
        "mode_selection": "Threshold sampling using epoch0_weights or mature_weights from STREAM_REPLAY_V1_CONFIG.",
        "target_partition": "Strided interleave by query_stride (offsets 0..query_stride-1), target_positions 0-indexed per segment.",
        "lattice_construction": (
            "Periodic forward pass from row 0 to target_end_row - 1 constructing raw period lattice. "
            "A tick arrives if (t - last_lattice_time) >= (period - tol). "
            "Non-query ticks are subsequently dropped with probability nonquery_drop_probability. "
            "Because the raw period lattice timestamps are determined prior to dropping, "
            "dropped ticks advance the lattice clock without retrying on adjacent rows."
        ),
        "group_capacity_selection": (
            "Capacity is chosen per group (not per segment) from candidates [c for c in capacities if c <= history_frames], "
            "appending history_frames when needed. Valid candidates must be >= len(interval_obs). "
            "Prefix lattice history is sampled uniformly up to chosen capacity. "
            "Union frame pool collects unique observations across all replay groups."
        ),
    }
    return contract


def compute_stream_replay_v1_layout(
    seed: int,
    epoch: int,
    ep_idx: int,
    target_start_row: int,
    target_end_row: int,
    all_timestamps: Sequence[float],
    history_frames: int = 16,
    split: str = "train",
) -> Dict[str, Any]:
    """Compute deterministic stream_replay_v1 layout and group assignment for a segment.

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

    Returns
    -------
    dict with:
        "obs_rows": List[int] (sorted unique row indices in union frame pool)
        "replay_groups": List[Dict[str, Any]]
        "stream_layout": Dict[str, Any]
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
    effective_epoch = 1 if split == "val" else epoch
    salt = 20260909
    rng_seed = (
        (seed & 0xFFFFFFFF)
        ^ ((effective_epoch * 1000003) & 0xFFFFFFFF)
        ^ ((ep_idx * 7919) & 0xFFFFFFFF)
        ^ ((target_start_row * 31337) & 0xFFFFFFFF)
        ^ salt
    ) & 0xFFFFFFFF
    local_rng = random.Random(rng_seed)

    # Determine mode using weights from STREAM_REPLAY_V1_CONFIG
    weights = (
        STREAM_REPLAY_V1_CONFIG["epoch0_weights"]
        if effective_epoch == 0
        else STREAM_REPLAY_V1_CONFIG["mature_weights"]
    )
    r_mode = local_rng.random()
    p_dense = weights["dense"]
    p_stream = weights["stream"]
    if r_mode < p_dense:
        mode = "dense"
    elif r_mode < p_dense + p_stream:
        mode = "stream"
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
            "query_stride": None,
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
            "query_stride": 1,
            "groups": groups_layout,
        }
        return {
            "obs_rows": obs_rows,
            "replay_groups": replay_groups,
            "stream_layout": stream_layout,
        }

    # mode == "stream"
    query_stride = local_rng.choice(STREAM_REPLAY_V1_CONFIG["query_strides"])
    period = local_rng.choice(STREAM_REPLAY_V1_CONFIG["candidate_periods"])
    drop_prob = STREAM_REPLAY_V1_CONFIG["nonquery_drop_probability"]
    tol = STREAM_REPLAY_V1_CONFIG["float_tolerance"]

    # Partition targets into query_stride interleaved subsequences
    group_target_positions: List[List[int]] = []
    group_target_rows: List[List[int]] = []
    for offset in range(query_stride):
        pos_list = [p for p in range(offset, M, query_stride)]
        if pos_list:
            group_target_positions.append(pos_list)
            group_target_rows.append([target_rows[p] for p in pos_list])

    target_rows_set = set(target_rows)

    # 1. Raw periodic lattice tick construction:
    # Forward pass from row 0 to target_end_row - 1
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
    # Dropping advances lattice clock because raw ticks were fixed in step 1.
    selected_lattice_rows: List[int] = []
    for r in raw_lattice_rows:
        if r in target_rows_set:
            selected_lattice_rows.append(r)
        else:
            if local_rng.random() >= drop_prob:
                selected_lattice_rows.append(r)

    # Capacities: candidates <= history_frames, appending history_frames when needed
    candidates = [c for c in STREAM_REPLAY_V1_CONFIG["capacities"] if c <= history_frames]
    if not candidates or history_frames not in candidates:
        candidates.append(history_frames)
    cap_candidates = sorted(set(candidates))

    group_obs_rows_list: List[List[int]] = []
    group_chosen_caps: List[int] = []
    group_prefix_lens: List[int] = []
    group_visible_counts: List[List[int]] = []
    group_delta_counts: List[List[int]] = []

    for g_targets in group_target_rows:
        first_t = g_targets[0]
        last_t = g_targets[-1]

        # Interval rows: lattice rows in [first_t, last_t] + all g_targets (forced query observations)
        interval_rows_set = {r for r in selected_lattice_rows if first_t <= r <= last_t}
        interval_rows_set.update(g_targets)
        interval_obs = sorted(interval_rows_set)

        if len(interval_obs) > history_frames:
            raise RuntimeError(
                f"Interval observations ({len(interval_obs)}) exceed history_frames ({history_frames})"
            )

        # Preceding lattice rows: selected_lattice_rows < first_t
        preceding_lattice = [r for r in selected_lattice_rows if r < first_t]

        # Valid capacities >= len(interval_obs), never > history_frames
        valid_caps = [c for c in cap_candidates if c >= len(interval_obs)]
        if valid_caps:
            chosen_capacity = local_rng.choice(valid_caps)
        else:
            chosen_capacity = history_frames

        # Prefix history count uniform from 0 .. chosen_capacity - len(interval_obs)
        max_prefix_len = max(0, chosen_capacity - len(interval_obs))
        prefix_len = local_rng.randint(0, min(max_prefix_len, len(preceding_lattice)))
        if prefix_len > 0:
            prefix_rows = preceding_lattice[-prefix_len:]
        else:
            prefix_rows = []

        g_obs_rows = sorted(prefix_rows + interval_obs)
        if not (len(g_obs_rows) <= chosen_capacity <= history_frames) or g_obs_rows[-1] > last_t or not set(g_targets) <= set(g_obs_rows):
            raise RuntimeError("Constructed replay group violates capacity or causal target coverage")

        group_obs_rows_list.append(g_obs_rows)
        group_chosen_caps.append(chosen_capacity)
        group_prefix_lens.append(prefix_len)

        # Visible frame counts per target in this group (causal: obs_row <= target)
        vis_counts = [sum(1 for obs_r in g_obs_rows if obs_r <= tgt_r) for tgt_r in g_targets]
        group_visible_counts.append(vis_counts)

        # Delta counts across query prefixes
        deltas: List[int] = []
        for i_tgt, vis in enumerate(vis_counts):
            if i_tgt == 0:
                deltas.append(vis - prefix_len)
            else:
                deltas.append(vis - vis_counts[i_tgt - 1])
        group_delta_counts.append(deltas)

    # Frame pool: unique sorted union of all obs_rows across all groups
    pool_rows = sorted(set(r for g_obs in group_obs_rows_list for r in g_obs))
    pool_to_idx = {r: i for i, r in enumerate(pool_rows)}

    replay_groups: List[Dict[str, Any]] = []
    groups_layout: List[Dict[str, Any]] = []

    for g_idx, g_targets in enumerate(group_target_rows):
        g_pos = group_target_positions[g_idx]
        g_obs = group_obs_rows_list[g_idx]
        obs_indices = [pool_to_idx[r] for r in g_obs]
        p_len = group_prefix_lens[g_idx]
        replay_groups.append({
            "observation_indices": obs_indices,
            "target_positions": g_pos,
        })
        groups_layout.append({
            "target_positions": g_pos,
            "target_rows": g_targets,
            "obs_rows": g_obs,
            "visible_frame_counts": group_visible_counts[g_idx],
            "history_count": p_len,
            "prefix_n": p_len,
            "delta_counts": group_delta_counts[g_idx],
            "selected_capacity": group_chosen_caps[g_idx],
            "start_time": float(all_timestamps[g_obs[0]]),
            "end_time": float(all_timestamps[g_obs[-1]]),
            "span_seconds": float(all_timestamps[g_obs[-1]] - all_timestamps[g_obs[0]]),
            "actual_span_seconds": float(all_timestamps[g_obs[-1]] - all_timestamps[g_obs[0]]),
        })

    stream_layout = {
        "mode": "stream",
        "requested_period": float(period),
        "query_stride": int(query_stride),
        "groups": groups_layout,
    }

    return {
        "obs_rows": pool_rows,
        "replay_groups": replay_groups,
        "stream_layout": stream_layout,
    }
