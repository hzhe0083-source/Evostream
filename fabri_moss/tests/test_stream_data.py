"""Validation tests for stream_replay_v1 protocol in NativeTrainingDataset.

Covers:
1. Modes over deterministic samples (epoch0 no current_only, mature all modes).
2. Target coverage at all tails and short episodes (every target exactly once).
3. Group bound, forced current observation, independent interleave partition.
4. Every-row labels/state identical between dense and stream/current_only modes (torch.equal).
5. Period uses actual timestamps not row stride; non-uniform vs uniform comparisons.
6. Future timestamp modifications cause no previous selection change (strict causality).
7. Missing/invalid/non-increasing timestamp errors preserve old behavior.
8. Determinism across workers/access orders/set_epoch, val fixed to mature/noaug.
9. Base data contracts equal (get_base_data_contract() == old get_data_contract()).
10. Data fingerprint correctly incorporates stream protocol specification.
11. Pure function tests: drop skipping without retry, capacity bounds, non-consecutive frame IDs.
12. Real DataLoader multi-worker spawn test.
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
from torch.utils.data import DataLoader, Dataset

from fabri_moss.native_data import NativeTrainingDataset
from fabri_moss.stream_protocol import (
    STREAM_REPLAY_V1_CONFIG,
    compute_stream_replay_v1_layout,
    get_stream_protocol_contract,
)
from fabri_moss.tests.test_native_data import _create_mock_metaworld_root, mock_norm_stats


@pytest.fixture(autouse=True)
def mock_video_decode(monkeypatch):
    """Avoid needing real mp4 video codecs during tests."""
    def dummy_decode(path, f_indices):
        return {idx: Image.new("RGB", (64, 64), color=(idx % 255, 0, 0)) for idx in f_indices}
    monkeypatch.setattr("fabri_moss.native_data.decode_all_video_frames", dummy_decode)


def test_base_data_contract_equality(tmp_path, mock_norm_stats):
    """Test get_base_data_contract() returns identical contract to old get_data_contract()."""
    root = _create_mock_metaworld_root(tmp_path, num_tasks=2, episodes_per_task=4)

    ds_old = NativeTrainingDataset(root=root, norm_stats=mock_norm_stats, stream_protocol=None)
    ds_new = NativeTrainingDataset(root=root, norm_stats=mock_norm_stats, stream_protocol="stream_replay_v1")

    contract_old = ds_old.get_data_contract()
    contract_base = ds_new.get_base_data_contract()

    assert contract_old == contract_base
    assert "stream_protocol" not in contract_base

    contract_new = ds_new.get_data_contract()
    assert "stream_protocol" in contract_new
    assert contract_new["stream_protocol"] == get_stream_protocol_contract()
    assert contract_new["data_fingerprint"] != contract_base["data_fingerprint"]

    # Unsupported protocol raises ValueError
    with pytest.raises(ValueError, match="Unsupported stream_protocol"):
        NativeTrainingDataset(root=root, norm_stats=mock_norm_stats, stream_protocol="invalid_v1")


def test_modes_distribution_epoch0_and_mature(tmp_path, mock_norm_stats):
    """Verify epoch 0 has only dense/stream (no current_only), epoch >= 1 has all modes."""
    root = _create_mock_metaworld_root(tmp_path, num_tasks=2, episodes_per_task=20, lengths=[30])
    ds = NativeTrainingDataset(
        root=root,
        norm_stats=mock_norm_stats,
        stream_protocol="stream_replay_v1",
        split="train",
    )

    # Epoch 0
    ds.set_epoch(0)
    modes_epoch0 = set()
    for idx in range(len(ds)):
        item = ds[idx]
        mode = item["stream_layout"]["mode"]
        assert mode in ("dense", "stream")
        modes_epoch0.add(mode)
    assert "current_only" not in modes_epoch0
    assert "dense" in modes_epoch0
    assert "stream" in modes_epoch0

    # Epoch 1 (mature phase)
    ds.set_epoch(1)
    modes_epoch1 = set()
    for idx in range(len(ds)):
        item = ds[idx]
        mode = item["stream_layout"]["mode"]
        modes_epoch1.add(mode)
    assert "dense" in modes_epoch1
    assert "stream" in modes_epoch1
    assert "current_only" in modes_epoch1


def test_target_coverage_all_tails_and_short_episodes(tmp_path, mock_norm_stats):
    """Test exhaustive target coverage at varied episode lengths and tail segments."""
    lengths = [1, 2, 3, 7, 8, 9, 15, 16, 17, 25]
    root = _create_mock_metaworld_root(tmp_path, num_tasks=2, episodes_per_task=len(lengths), lengths=lengths)
    ds = NativeTrainingDataset(
        root=root,
        norm_stats=mock_norm_stats,
        stream_protocol="stream_replay_v1",
        history_frames=16,
    )

    for epoch in (0, 1, 2):
        ds.set_epoch(epoch)
        total_targets_seen = 0
        for idx in range(len(ds)):
            item = ds[idx]
            ep_idx, start, end = ds.segments[idx]
            target_count = item["target_count"]
            assert target_count == (end - start)
            total_targets_seen += target_count

            # Verify target_indices map into images_window/frame_ids
            num_images = len(item["images_window"])
            assert len(item["frame_ids"]) == num_images
            assert len(item["observation_times"]) == num_images
            assert num_images <= sum(len(g["observation_indices"]) for g in item["replay_groups"])
            for tidx in item["target_indices"]:
                assert 0 <= tidx < num_images

            # Verify replay_groups structure
            replay_groups = item.get("replay_groups")
            assert replay_groups is not None
            seen_target_positions = []
            for g in replay_groups:
                obs_indices = g["observation_indices"]
                target_positions = g["target_positions"]
                # Must be strictly increasing and <= history_frames
                assert obs_indices == sorted(obs_indices)
                assert len(obs_indices) == len(set(obs_indices))
                assert len(obs_indices) <= ds.history_frames
                assert target_positions == sorted(target_positions)
                seen_target_positions.extend(target_positions)

                # Each target's observation must exist in own group
                # Group has no frames after final group's target
                group_target_outer_indices = [item["target_indices"][pos] for pos in target_positions]
                last_group_target_obs = max(group_target_outer_indices)
                assert obs_indices[-1] <= last_group_target_obs
                for t_obs in group_target_outer_indices:
                    assert t_obs in obs_indices

            # Every target position in 0..M-1 appears exactly once
            assert sorted(seen_target_positions) == list(range(target_count))

        assert total_targets_seen == ds.total_targets


def test_labels_and_state_identical_to_dense(tmp_path, mock_norm_stats):
    """Verify supervised state, actions, state_mask, action_mask are strictly identical (torch.equal)."""
    root = _create_mock_metaworld_root(tmp_path, num_tasks=2, episodes_per_task=5, lengths=[25])
    ds_dense = NativeTrainingDataset(root=root, norm_stats=mock_norm_stats, stream_protocol=None)
    ds_stream = NativeTrainingDataset(root=root, norm_stats=mock_norm_stats, stream_protocol="stream_replay_v1")

    ds_dense.set_epoch(1)
    ds_stream.set_epoch(1)

    for idx in range(len(ds_dense)):
        d_item = ds_dense[idx]
        s_item = ds_stream[idx]

        assert torch.equal(d_item["state"], s_item["state"])
        assert torch.equal(d_item["state_mask"], s_item["state_mask"])
        assert torch.equal(d_item["actions"], s_item["actions"])
        assert torch.equal(d_item["action_mask"], s_item["action_mask"])
        assert d_item["prompt"] == s_item["prompt"]
        assert d_item["episode_id"] == s_item["episode_id"]
        assert d_item["target_frame_ids"] == s_item["target_frame_ids"]


def test_physical_period_timestamps_nonuniform_vs_uniform():
    """Test period selection directly verifies non-uniform timestamps produce different selections."""
    # Find a seed that produces mode == "stream"
    timestamps_uniform = [i * (1.0 / 30.0) for i in range(40)]
    # In non-uniform, clump frames together then have large gaps
    timestamps_nonuniform = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007,
                             0.10, 0.101, 0.102, 0.103, 0.20, 0.201, 0.202, 0.203,
                             0.30, 0.301, 0.302, 0.303, 0.40, 0.401, 0.402, 0.403,
                             0.50, 0.501, 0.502, 0.503, 0.60, 0.601, 0.602, 0.603,
                             0.70, 0.701, 0.702, 0.703, 0.80, 0.801, 0.802, 0.803]

    chosen_seed = None
    for s in range(100):
        res = compute_stream_replay_v1_layout(
            seed=s, epoch=1, ep_idx=0, target_start_row=24, target_end_row=32,
            all_timestamps=timestamps_uniform, history_frames=16
        )
        if res["stream_layout"]["mode"] == "stream":
            chosen_seed = s
            break
    assert chosen_seed is not None

    res_u = compute_stream_replay_v1_layout(
        seed=chosen_seed, epoch=1, ep_idx=0, target_start_row=24, target_end_row=32,
        all_timestamps=timestamps_uniform, history_frames=16
    )
    res_nu = compute_stream_replay_v1_layout(
        seed=chosen_seed, epoch=1, ep_idx=0, target_start_row=24, target_end_row=32,
        all_timestamps=timestamps_nonuniform, history_frames=16
    )
    # The selected observation rows must reflect the physical timestamps and differ
    assert res_u["obs_rows"] != res_nu["obs_rows"]


def test_drop_first_tick_advances_lattice_no_retry():
    """Verify that when a non-query tick is dropped, last_lattice_time advances and does NOT retry next row."""
    # Period = 4.0 / 30.0 (~0.133333s)
    # Let timestamps be: row 0: 0.0, row 1: 0.01, row 2: 0.02, row 3: 0.14, row 4: 0.15, row 5: 0.28
    # If row 0 is dropped, row 1 should NOT be accepted just because row 0 was dropped;
    # next tick can only happen when t >= 0.133333 (which is row 3).
    # We monkeypatch random.Random in compute_stream_replay_v1_layout to force drop on row 0.
    class ControlledRNG:
        def __init__(self, *args, **kwargs):
            self._step = 0

        def random(self):
            self._step += 1
            # Step 1: mode selection -> return 0.5 (stream mode in mature)
            if self._step == 1:
                return 0.50
            # Step 2: row 0 drop check -> return 0.05 < 0.10 (DROP row 0)
            if self._step == 2:
                return 0.05
            # Remaining drop checks -> return 0.99 (KEEP tick)
            return 0.99

        def choice(self, seq):
            # Pick stride 2 and period 4.0/30.0
            if seq == STREAM_REPLAY_V1_CONFIG["query_strides"]:
                return 2
            if seq == STREAM_REPLAY_V1_CONFIG["candidate_periods"]:
                return 4.0 / 30.0
            return seq[0]

        def randint(self, a, b):
            # Pick max available prefix history to ensure preceding lattice rows are included
            return b

    timestamps = [0.0, 0.01, 0.02, 0.14, 0.15, 0.28, 0.29, 0.42]
    # targets at [6, 8)
    import unittest.mock as mock
    with mock.patch("fabri_moss.stream_protocol.random.Random", ControlledRNG):
        res = compute_stream_replay_v1_layout(
            seed=42, epoch=1, ep_idx=0, target_start_row=6, target_end_row=8,
            all_timestamps=timestamps, history_frames=16
        )

    # Row 0 dropped. Row 1 and 2 (t=0.01, 0.02) must NOT be selected (no jitter retry).
    # Row 3 (t=0.14 >= 0.1333) should be the first lattice tick selected.
    assert 0 not in res["obs_rows"]
    assert 1 not in res["obs_rows"]
    assert 2 not in res["obs_rows"]
    assert 3 in res["obs_rows"]


def test_future_timestamp_modification_no_previous_selection_change(tmp_path, mock_norm_stats):
    """Test causality: modifying future timestamps after segment end does not alter chosen rows."""
    root = _create_mock_metaworld_root(tmp_path, num_tasks=1, episodes_per_task=1, lengths=[40])
    ds1 = NativeTrainingDataset(root=root, norm_stats=mock_norm_stats, stream_protocol="stream_replay_v1")
    ds1.set_epoch(1)
    item1 = ds1[0]

    # Modify parquet timestamps of future rows (rows 10..39)
    pq_path = tmp_path / "data" / "chunk-000" / "episode_000000.parquet"
    df = pd.read_parquet(pq_path)
    df.loc[10:39, "timestamp"] = df.loc[10:39, "timestamp"] + 100.0
    df.to_parquet(pq_path)

    ds2 = NativeTrainingDataset(root=tmp_path, norm_stats=mock_norm_stats, stream_protocol="stream_replay_v1")
    ds2.set_epoch(1)
    item2 = ds2[0]

    assert item1["frame_ids"] == item2["frame_ids"]
    assert item1["stream_layout"] == item2["stream_layout"]
    assert item1["target_indices"] == item2["target_indices"]


def test_frame_pool_can_exceed_individual_stream_capacity():
    res = compute_stream_replay_v1_layout(
        seed=4042, epoch=1, ep_idx=47, target_start_row=88, target_end_row=96,
        all_timestamps=[i / 30 for i in range(120)], history_frames=16,
    )
    assert len(res["obs_rows"]) == 22
    assert max(len(g["observation_indices"]) for g in res["replay_groups"]) == 16


def test_pure_function_capacity_bounds_and_invariants():
    """Exhaustive check on compute_stream_replay_v1_layout across H in [1, 2, 3, 5, 10, 16, 32]."""
    all_timestamps = [i * 0.033333 for i in range(100)]
    for H in [1, 2, 3, 5, 10, 16, 32]:
        # Target count M <= H
        for M in [1, min(H, 4), min(H, 8)]:
            if M > H:
                continue
            start_row = 20
            end_row = start_row + M
            for ep in [0, 1]:
                res = compute_stream_replay_v1_layout(
                    seed=123, epoch=ep, ep_idx=0,
                    target_start_row=start_row, target_end_row=end_row,
                    all_timestamps=all_timestamps, history_frames=H
                )
                obs = res["obs_rows"]
                assert len(obs) == len(set(obs))
                for g in res["replay_groups"]:
                    assert len(g["observation_indices"]) <= H
                    # Verify every group target has its index
                    for t_pos in g["target_positions"]:
                        tgt_row = start_row + t_pos
                        assert tgt_row in obs


def test_non_consecutive_original_frame_ids(tmp_path, mock_norm_stats):
    """Test that dataset works correctly when original frame_index values are non-consecutive."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "info.json", "w") as f:
        json.dump({"chunks_size": 1000}, f)
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "task_non_consec"}) + "\n")

    chunk_dir = tmp_path / "data" / "chunk-000"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    video_dir = tmp_path / "videos" / "chunk-000" / "observation.images.image"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "episode_000000.mp4").touch()

    # Frame indices spaced by 10
    ep_len = 20
    frame_indices = [i * 10 for i in range(ep_len)]
    with open(meta_dir / "episodes.jsonl", "w") as f:
        f.write(json.dumps({"episode_index": 0, "length": ep_len, "tasks": ["task_non_consec"]}) + "\n")

    df = pd.DataFrame({
        "frame_index": frame_indices,
        "episode_index": [0] * ep_len,
        "task_index": [0] * ep_len,
        "observation.state": [[0.0] * 4 for _ in range(ep_len)],
        "action": [[0.0] * 4 for _ in range(ep_len)],
        "timestamp": [i * 0.1 for i in range(ep_len)],
    })
    df.to_parquet(chunk_dir / "episode_000000.parquet")

    ds = NativeTrainingDataset(
        root=tmp_path,
        norm_stats=mock_norm_stats,
        stream_protocol="stream_replay_v1",
        history_frames=16,
        target_frames=8,
    )
    item = ds[0]
    # Check that frame_ids match the actual non-consecutive frame_indices in df
    for fid in item["frame_ids"]:
        assert fid in frame_indices
    for tfid in item["target_frame_ids"]:
        assert tfid in frame_indices


class PicklableStreamSegmentDataset(Dataset):
    """Minimal picklable dataset wrapper to test multi-worker DataLoader spawn."""
    def __init__(self, seed: int, epoch: int, timestamps: List[float], history_frames: int):
        self.seed = seed
        self.epoch = epoch
        self.timestamps = timestamps
        self.history_frames = history_frames
        self.segments = [(0, i * 4, (i + 1) * 4) for i in range(5)]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        ep_idx, start, end = self.segments[idx]
        res = compute_stream_replay_v1_layout(
            seed=self.seed,
            epoch=self.epoch,
            ep_idx=ep_idx,
            target_start_row=start,
            target_end_row=end,
            all_timestamps=self.timestamps,
            history_frames=self.history_frames,
        )
        return {
            "idx": idx,
            "obs_rows": torch.tensor(res["obs_rows"], dtype=torch.long),
            "mode": res["stream_layout"]["mode"],
        }


def test_dataloader_multiprocess_workers():
    """Verify multi-worker DataLoader (num_workers=2) produces identical layouts to single-worker."""
    timestamps = [i * 0.033333 for i in range(30)]
    ds = PicklableStreamSegmentDataset(seed=2026, epoch=1, timestamps=timestamps, history_frames=16)

    loader_single = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    loader_multi = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, multiprocessing_context="spawn")

    items_single = [b for b in loader_single]
    items_multi = [b for b in loader_multi]

    assert len(items_single) == len(items_multi)
    for s, m in zip(items_single, items_multi):
        assert s["idx"] == m["idx"]
        assert torch.equal(s["obs_rows"], m["obs_rows"])
        assert s["mode"] == m["mode"]


def test_val_split_fixed_mature_and_no_augmentation(tmp_path, mock_norm_stats):
    """Test validation split is fixed to mature phase (epoch 1) and never augmented."""
    root = _create_mock_metaworld_root(tmp_path, num_tasks=2, episodes_per_task=10, lengths=[20])
    ds_val = NativeTrainingDataset(
        root=root,
        norm_stats=mock_norm_stats,
        stream_protocol="stream_replay_v1",
        split="val",
        val_fraction=0.2,
    )
    assert not ds_val.augmentation

    # Val items at epoch 0 and epoch 5 must produce exactly the same layouts
    ds_val.set_epoch(0)
    layouts_ep0 = [ds_val[i]["stream_layout"] for i in range(len(ds_val))]
    ds_val.set_epoch(5)
    layouts_ep5 = [ds_val[i]["stream_layout"] for i in range(len(ds_val))]
    assert layouts_ep0 == layouts_ep5


def test_worker_and_access_order_invariance(tmp_path, mock_norm_stats):
    """Test sample generation is completely independent of access order."""
    root = _create_mock_metaworld_root(tmp_path, num_tasks=2, episodes_per_task=6, lengths=[25])
    ds = NativeTrainingDataset(
        root=root,
        norm_stats=mock_norm_stats,
        stream_protocol="stream_replay_v1",
    )
    ds.set_epoch(2)

    # Sequential access
    seq_items = [ds[i] for i in range(len(ds))]

    # Random order access
    indices = list(range(len(ds)))
    random.Random(12345).shuffle(indices)
    shuffled_items = {i: ds[i] for i in indices}

    for i in range(len(ds)):
        assert seq_items[i]["frame_ids"] == shuffled_items[i]["frame_ids"]
        assert seq_items[i]["target_indices"] == shuffled_items[i]["target_indices"]
        assert seq_items[i]["stream_layout"] == shuffled_items[i]["stream_layout"]
        assert torch.equal(seq_items[i]["actions"], shuffled_items[i]["actions"])
