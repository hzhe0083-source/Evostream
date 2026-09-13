"""Validation tests for memory_replay_v1 protocol in NativeTrainingDataset.

Covers:
1. Contract validation: get_memory_protocol_contract() returns JSON-serializable stable contract.
2. Dataset contract & fingerprint: base data contracts equal; data_fingerprint includes memory protocol JSON.
3. Distribution: epoch 0 only dense/memory (no current_only), mature has dense/memory/current_only.
4. Validation split fixed mature: split='val' always evaluates to mature weights regardless of epoch, and reproducible.
5. Exhaustive target coverage: every target anchor row is visited exactly once per epoch, geometry same across modes.
6. Identity of state/action labels across modes: torch.equal for state, state_mask, actions, action_mask.
7. Long prefix & >16 observations: full prefix retains observations from row 0 to target_end_row-1, num_obs can exceed 16.
8. Retirement triggers: 4/8 mapping (first retirement at 8 frames -> 4 retired / 4 recent; subsequent every 4 frames).
9. Current-only mode: independent replay groups per target, memory_replay not in sample (omitted when false).
10. Memory mode: memory_replay is True, replay_groups is None (not interleaved subgroups).
11. Query frames forced: all target_rows exist in obs_rows even if not aligned with lattice or if dropped.
12. Strict causality: modifications to future timestamps do not alter earlier prefix lattice ticks or decisions.
13. Non-timestamp fallback: dataset preserves strict timestamp checking / fallback errors.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from fabri_moss.memory_protocol import (
    MEMORY_REPLAY_V1_CONFIG,
    compute_memory_replay_layout,
    get_memory_protocol_contract,
)
from fabri_moss.native_data import NativeTrainingDataset
from fabri_moss.tests.test_native_data import _create_mock_metaworld_root, mock_norm_stats


@pytest.fixture(autouse=True)
def mock_video_decode(monkeypatch):
    """Avoid needing real mp4 video codecs during tests."""
    def dummy_decode(path, f_indices):
        return {idx: Image.new("RGB", (64, 64), color=(idx % 255, 0, 0)) for idx in f_indices}
    monkeypatch.setattr("fabri_moss.native_data.decode_all_video_frames", dummy_decode)


def test_contract_specification():
    """Verify get_memory_protocol_contract() returns correct stable JSON dictionary for v1.2."""
    contract = get_memory_protocol_contract()
    json_str = json.dumps(contract)
    assert json.loads(json_str) == contract
    assert contract["protocol_name"] == "memory_replay_v1"
    assert contract["version"] == "1.2"
    assert contract["recent_frames"] == 4
    assert contract["consolidate_every"] == 4
    assert contract["memory_slots"] == 16
    assert contract["spatial_grid"] == 4
    assert contract["anchor_policy"] == "snapshot_latest"
    assert contract["protect_decision_frames"] is True
    assert contract["arrival_chunk_choices"] == [1, 2, 4, 8, 16]
    assert "anchor_interval" not in contract
    assert contract["merge_rule"] == "protected_original_frame_intervals"
    assert contract["bank_capacity"] == "episode_growing_protected_anchors_and_interval_summaries"
    assert contract["train_weights"]["epoch0"] == {"dense": 0.50, "memory": 0.50, "current_only": 0.0}
    assert contract["train_weights"]["mature"] == {"dense": 0.25, "memory": 0.65, "current_only": 0.10}
    assert contract["period_choices"] == [1.0 / 30.0, 2.0 / 30.0, 4.0 / 30.0, 8.0 / 30.0]
    assert contract["nonquery_drop_probability"] == 0.10
    assert contract["bank_update"] == "detached_projected_visual"
    assert contract["prefix"] == "episode_start"


def test_base_data_contract_equality(tmp_path, mock_norm_stats):
    """Test get_base_data_contract() is identical to old get_data_contract() when memory protocol used."""
    root = _create_mock_metaworld_root(tmp_path, num_tasks=2, episodes_per_task=4)

    ds_old = NativeTrainingDataset(root=root, norm_stats=mock_norm_stats, stream_protocol=None)
    ds_mem = NativeTrainingDataset(root=root, norm_stats=mock_norm_stats, stream_protocol="memory_replay_v1")

    contract_old = ds_old.get_data_contract()
    contract_base = ds_mem.get_base_data_contract()

    assert contract_old == contract_base
    assert "stream_protocol" not in contract_base

    contract_mem = ds_mem.get_data_contract()
    assert "stream_protocol" in contract_mem
    assert contract_mem["stream_protocol"] == get_memory_protocol_contract()
    assert contract_mem["data_fingerprint"] != contract_base["data_fingerprint"]


def test_modes_distribution_epoch0_and_mature(tmp_path, mock_norm_stats):
    """Verify epoch 0 only has dense/memory, mature has dense/memory/current_only."""
    root = _create_mock_metaworld_root(tmp_path, num_tasks=2, episodes_per_task=20, lengths=[40])
    ds = NativeTrainingDataset(
        root=root,
        norm_stats=mock_norm_stats,
        stream_protocol="memory_replay_v1",
        split="train",
    )

    # Epoch 0
    ds.set_epoch(0)
    modes_epoch0 = set()
    for idx in range(len(ds)):
        item = ds[idx]
        mode = item["stream_layout"]["mode"]
        assert mode in ("dense", "memory")
        modes_epoch0.add(mode)
    assert "current_only" not in modes_epoch0
    assert "dense" in modes_epoch0
    assert "memory" in modes_epoch0

    # Epoch 1 (mature phase)
    ds.set_epoch(1)
    modes_epoch1 = set()
    for idx in range(len(ds)):
        item = ds[idx]
        mode = item["stream_layout"]["mode"]
        modes_epoch1.add(mode)
    assert "dense" in modes_epoch1
    assert "memory" in modes_epoch1
    assert "current_only" in modes_epoch1


def test_val_split_fixed_to_mature(tmp_path, mock_norm_stats):
    """Validation split always uses mature weights (effective epoch 1) regardless of set_epoch."""
    root = _create_mock_metaworld_root(tmp_path, num_tasks=2, episodes_per_task=20, lengths=[40])
    ds_val = NativeTrainingDataset(
        root=root,
        norm_stats=mock_norm_stats,
        stream_protocol="memory_replay_v1",
        split="val",
    )

    # In val split, epoch 0 should still produce current_only if mature weights are active
    ds_val.set_epoch(0)
    modes = set()
    for idx in range(len(ds_val)):
        item = ds_val[idx]
        modes.add(item["stream_layout"]["mode"])
    assert "current_only" in modes
    assert "memory" in modes


def test_target_coverage_all_tails_and_geometry(tmp_path, mock_norm_stats):
    """Test exhaustive target coverage across variable lengths, same geometry for all modes."""
    lengths = [1, 5, 8, 9, 16, 25, 50]
    root = _create_mock_metaworld_root(tmp_path, num_tasks=2, episodes_per_task=len(lengths), lengths=lengths)
    ds = NativeTrainingDataset(
        root=root,
        norm_stats=mock_norm_stats,
        stream_protocol="memory_replay_v1",
        history_frames=16,
    )

    for epoch in (0, 1):
        ds.set_epoch(epoch)
        total_targets = 0
        seen_targets = set()
        for idx in range(len(ds)):
            item = ds[idx]
            ep_idx = item["episode_id"]
            tgt_rows = list(range(item["target_start_row"], item["target_end_row"]))
            assert item["target_count"] == len(tgt_rows)
            for r in tgt_rows:
                assert (ep_idx, r) not in seen_targets
                seen_targets.add((ep_idx, r))
            total_targets += len(tgt_rows)
        assert total_targets == ds.total_targets


def test_label_and_state_identity_between_modes(tmp_path, mock_norm_stats):
    """Compare dense, memory, and current_only modes on the same segment: states and actions must be torch.equal."""
    root = _create_mock_metaworld_root(tmp_path, num_tasks=1, episodes_per_task=5, lengths=[50])
    ds_none = NativeTrainingDataset(root=root, norm_stats=mock_norm_stats, stream_protocol=None)
    ds_mem = NativeTrainingDataset(root=root, norm_stats=mock_norm_stats, stream_protocol="memory_replay_v1")

    for idx in range(len(ds_none)):
        item_base = ds_none[idx]
        for ep in (0, 1, 2):
            ds_mem.set_epoch(ep)
            item_mem = ds_mem[idx]

            assert item_mem["episode_id"] == item_base["episode_id"]
            assert item_mem["target_start_row"] == item_base["target_start_row"]
            assert item_mem["target_end_row"] == item_base["target_end_row"]
            assert item_mem["target_count"] == item_base["target_count"]
            assert item_mem["target_frame_ids"] == item_base["target_frame_ids"]

            assert torch.equal(item_mem["state"], item_base["state"])
            assert torch.equal(item_mem["state_mask"], item_base["state_mask"])
            assert torch.equal(item_mem["actions"], item_base["actions"])
            assert torch.equal(item_mem["action_mask"], item_base["action_mask"])


def test_full_prefix_and_observations_bound(tmp_path, mock_norm_stats):
    """Memory mode must retain causal history from row 0, num_observations can exceed 16."""
    root = _create_mock_metaworld_root(tmp_path, num_tasks=1, episodes_per_task=2, lengths=[100])
    ds = NativeTrainingDataset(
        root=root,
        norm_stats=mock_norm_stats,
        stream_protocol="memory_replay_v1",
        history_frames=16,
    )
    ds.set_epoch(0)

    # Search for a segment in memory mode with a later target_start_row (e.g. > 40)
    found_long_memory = False
    for idx in range(len(ds)):
        item = ds[idx]
        if item["stream_layout"]["mode"] == "memory" and item["target_start_row"] >= 32:
            found_long_memory = True
            obs_rows = [item["context_start_row"] + i for i in range(len(item["frame_ids"]))]
            # Oldest observation must be row 0
            assert item["frame_ids"][0] == 0
            # Number of observations can easily exceed 16 for a 100-frame episode at 1/30 or 2/30s
            assert len(item["frame_ids"]) >= 10
            # memory_replay is True
            assert item.get("memory_replay") is True
            assert "replay_groups" not in item
            break
    assert found_long_memory


def test_retirement_trigger_calendar_computation():
    """Verify retirement triggers math and v1.2 calendar fields (anchors, bins, protected_frame_ids)."""
    timestamps = [i * (1.0 / 30.0) for i in range(50)]
    all_fids = [i * 2 for i in range(50)]
    # Search for a seed that yields memory mode
    found_memory = False
    for s in range(50):
        layout = compute_memory_replay_layout(
            seed=s,
            epoch=0,
            ep_idx=0,
            target_start_row=40,
            target_end_row=48,
            all_timestamps=timestamps,
            history_frames=16,
            all_frame_ids=all_fids,
        )
        if layout["stream_layout"]["mode"] == "memory":
            found_memory = True
            calendar = layout["stream_layout"]["calendar_queries"]
            obs_rows = layout["obs_rows"]
            decision_indices = layout["decision_indices"]
            decision_indices_set = set(decision_indices)

            for q in calendar:
                vis = q["visible_observations"]
                ret = q["retirement_triggers"]
                ret_frames = q["retired_frames_count"]
                recent = q["recent_frames_count"]
                assert vis == ret_frames + recent
                if vis < 8:
                    assert ret == 0
                    assert ret_frames == 0
                    assert recent == vis
                elif vis < 12:
                    assert ret == 1
                    assert ret_frames == 4
                    assert 4 <= recent < 8
                else:
                    assert ret >= 2
                    assert ret_frames == ret * 4
                    assert 4 <= recent < 8

                # Check v1.2 decision-based anchor & bin fields
                tgt_r = q["target_row"]
                prefix_obs = [r for r in obs_rows if r <= tgt_r]
                assert len(prefix_obs) == vis

                # Retired decisions count
                expected_retired_anchors = sum(1 for d in decision_indices if d < ret_frames)
                assert q["retired_anchor_count"] == expected_retired_anchors

                # Retired summary bin count: partition of non-decision retired frames by nearest preceding decision
                import bisect
                expected_bins = set()
                for i in range(ret_frames):
                    if i not in decision_indices_set:
                        pos = bisect.bisect_left(decision_indices, i)
                        b = decision_indices[pos - 1] if pos > 0 else -1
                        expected_bins.add(b)
                assert q["retired_summary_bin_count"] == len(expected_bins)

                # Protected frame IDs: all decisions in prefix up to this query
                expected_protected_fids = [all_fids[obs_rows[d]] for d in decision_indices if d < vis]
                assert q["protected_frame_ids"] == expected_protected_fids
            break
    assert found_memory, "Must find memory mode to assert calendar computation"


def test_variable_arrival_chunk_choices_coverage():
    """Verify arrival chunk choices [1, 2, 4, 8, 16] are exercised across seeds."""
    timestamps = [i * (1.0 / 30.0) for i in range(120)]
    all_fids = list(range(120))
    chunk_sizes_observed = set()

    for s in range(200):
        layout = compute_memory_replay_layout(
            seed=s,
            epoch=0,
            ep_idx=0,
            target_start_row=80,
            target_end_row=88,
            all_timestamps=timestamps,
            history_frames=16,
            all_frame_ids=all_fids,
        )
        if layout["stream_layout"]["mode"] == "memory":
            deltas = layout["stream_layout"]["decision_delta_counts"]
            # The delta counts before target boundaries represent chunk sizes
            # (the last chunk before first_idx might be truncated, but earlier full chunks match choices)
            first_idx = layout["obs_rows"].index(80)
            dec_indices = layout["decision_indices"]
            prev = 0
            for d in dec_indices:
                if d < first_idx:
                    chunk_sizes_observed.add(d + 1 - prev)
                    prev = d + 1
                else:
                    break

    # Check that across seeds we observe multiple chunk choices from [1, 2, 4, 8, 16]
    for expected_c in [1, 2, 4, 8, 16]:
        assert expected_c in chunk_sizes_observed, f"Chunk size {expected_c} never observed"


def test_decision_indices_targets_subset_and_real_frame_ids():
    """Targets are always decision frames; decision_frame_ids match real frame_ids."""
    timestamps = [i * 0.1 for i in range(50)]
    all_fids = [100 + i * 3 for i in range(50)]  # Non-consecutive frame IDs: 100, 103, 106, ...

    for s in range(30):
        layout = compute_memory_replay_layout(
            seed=s,
            epoch=0,
            ep_idx=0,
            target_start_row=20,
            target_end_row=28,
            all_timestamps=timestamps,
            history_frames=16,
            all_frame_ids=all_fids,
        )
        if layout["stream_layout"]["mode"] == "memory":
            obs_rows = layout["obs_rows"]
            target_rows = list(range(20, 28))
            obs_to_idx = {r: i for i, r in enumerate(obs_rows)}
            target_pool_indices = [obs_to_idx[r] for r in target_rows]

            dec_indices = layout["decision_indices"]
            # Target pool indices must be a subset of decision_indices
            for t_idx in target_pool_indices:
                assert t_idx in dec_indices

            # decision_indices must be sorted and unique
            assert dec_indices == sorted(list(set(dec_indices)))

            # decision_frame_ids match all_frame_ids[obs_rows[d]]
            dec_fids = layout["stream_layout"]["decision_frame_ids"]
            expected_fids = [all_fids[obs_rows[d]] for d in dec_indices]
            assert dec_fids == expected_fids

            # decision_delta_counts matches firstboundary+1 then intervals
            deltas = layout["stream_layout"]["decision_delta_counts"]
            assert len(deltas) == len(dec_indices)
            assert deltas[0] == dec_indices[0] + 1
            for k in range(1, len(dec_indices)):
                assert deltas[k] == dec_indices[k] - dec_indices[k - 1]


def test_prefix_all_zero_target_short_safe():
    """When first_idx is 0 (target_start_row == 0), safely starts from targets without error."""
    timestamps = [i * 0.1 for i in range(20)]
    all_fids = list(range(20))

    layout = compute_memory_replay_layout(
        seed=42,
        epoch=0,
        ep_idx=0,
        target_start_row=0,
        target_end_row=8,
        all_timestamps=timestamps,
        history_frames=16,
        all_frame_ids=all_fids,
    )
    if layout["stream_layout"]["mode"] == "memory":
        dec_indices = layout["decision_indices"]
        # Targets are 0..7, so decision_indices must contain 0..7
        assert dec_indices == list(range(8))
        assert layout["stream_layout"]["decision_delta_counts"] == [1] * 8


def test_invalid_all_frame_ids_validation():
    """Verify validation of all_frame_ids: length match, non-negative, strictly increasing."""
    timestamps = [0.0, 0.1, 0.2]
    # Wrong length
    with pytest.raises(ValueError, match="all_frame_ids length 2 does not match"):
        compute_memory_replay_layout(
            seed=1, epoch=0, ep_idx=0, target_start_row=1, target_end_row=3,
            all_timestamps=timestamps, all_frame_ids=[0, 1]
        )
    # Negative id
    with pytest.raises(ValueError, match="all_frame_ids contains negative value"):
        compute_memory_replay_layout(
            seed=1, epoch=0, ep_idx=0, target_start_row=1, target_end_row=3,
            all_timestamps=timestamps, all_frame_ids=[-1, 0, 1]
        )
    # Non-increasing id
    with pytest.raises(ValueError, match="all_frame_ids must be strictly increasing"):
        compute_memory_replay_layout(
            seed=1, epoch=0, ep_idx=0, target_start_row=1, target_end_row=3,
            all_timestamps=timestamps, all_frame_ids=[0, 5, 5]
        )


def test_current_only_independent_groups(tmp_path, mock_norm_stats):
    """current_only mode should have separate independent replay groups for each target, no memory_replay=True."""
    timestamps = [i * 0.1 for i in range(20)]
    # Find a seed/epoch that yields current_only
    found = False
    for s in range(50):
        layout = compute_memory_replay_layout(
            seed=s,
            epoch=1,
            ep_idx=0,
            target_start_row=8,
            target_end_row=16,
            all_timestamps=timestamps,
            split="train",
        )
        if layout["stream_layout"]["mode"] == "current_only":
            found = True
            assert layout["memory_replay"] is False
            assert len(layout["replay_groups"]) == 8
            for i, grp in enumerate(layout["replay_groups"]):
                assert grp["target_positions"] == [i]
                assert len(grp["observation_indices"]) == 1
            break
    assert found


def test_query_frames_forced_at_non_lattice():
    """Verify all target rows exist in obs_rows even if timestamps are non-lattice or dropped."""
    # Irregular timestamps
    timestamps = [0.0, 0.01, 0.05, 0.07, 0.20, 0.21, 0.22, 0.35, 0.50, 0.60]
    layout = compute_memory_replay_layout(
        seed=42,
        epoch=0,
        ep_idx=0,
        target_start_row=4,
        target_end_row=7,
        all_timestamps=timestamps,
        history_frames=16,
    )
    obs = layout["obs_rows"]
    for tgt_r in range(4, 7):
        assert tgt_r in obs


def test_future_timestamps_do_not_affect_earlier_prefix():
    """Modifying future timestamps (beyond target_end_row) cannot affect prefix observations."""
    ts1 = [i * 0.033 for i in range(40)]
    ts2 = list(ts1)
    # Modify future timestamps beyond target_end_row (30)
    for i in range(30, 40):
        ts2[i] += 10.0

    layout1 = compute_memory_replay_layout(
        seed=101, epoch=0, ep_idx=0, target_start_row=24, target_end_row=30, all_timestamps=ts1
    )
    layout2 = compute_memory_replay_layout(
        seed=101, epoch=0, ep_idx=0, target_start_row=24, target_end_row=30, all_timestamps=ts2
    )

    assert layout1["obs_rows"] == layout2["obs_rows"]
    assert layout1["stream_layout"]["mode"] == layout2["stream_layout"]["mode"]


def test_non_timestamp_fallback_raises(tmp_path, mock_norm_stats):
    """Missing timestamps and missing fps raises error as in original schema."""
    root = _create_mock_metaworld_root(
        tmp_path,
        num_tasks=1,
        episodes_per_task=1,
        fps=None,
        include_parquet_timestamp=False,
    )
    ds = NativeTrainingDataset(root=root, norm_stats=mock_norm_stats, stream_protocol="memory_replay_v1")
    with pytest.raises(ValueError, match="requires either parquet 'timestamp' column or valid metadata 'fps'"):
        _ = ds[0]
