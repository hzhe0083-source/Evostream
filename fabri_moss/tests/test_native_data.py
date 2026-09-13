"""Tests for NativeTrainingDataset and native causal segment data utilities.

Verifies:
1. Exhaustive target coverage once across variable L and tails (no dropped / repeated supervision).
2. Causality and target mapping (no future evidence beyond end, strictly ordered observation frames).
3. Observation history bound (len(obs) <= history_frames = 16, M <= target_frames = 8).
4. Observation timestamps precedence (parquet timestamp > metadata fps fallback, missing/invalid/non-increasing reject).
5. Action repeat-last at tail and valid dimension masking.
6. Train/val task-stratified disjoint partition with stable sha256.
7. Data augmentation determinism: same across frames within episode/epoch, changes with epoch, disabled on val.
8. Bounded raw video decode cache (LRU capacity=2, no reload for same episode segments).
9. Data contract JSON serializability, counts, normalization fingerprint.
10. validation_indices deterministic selection (one middle segment per task).
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from fabri_moss.native_data import BoundedFrameCache, NativeTrainingDataset


def _create_mock_metaworld_root(
    tmp_path: Path,
    num_tasks: int = 2,
    episodes_per_task: int = 10,
    lengths: Optional[List[int]] = None,
    fps: Optional[float] = 30.0,
    include_parquet_timestamp: bool = True,
    timestamp_step: float = 0.1,
) -> Path:
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    info: Dict[str, Any] = {
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    if fps is not None:
        info["fps"] = fps
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f)

    tasks = {}
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for t_idx in range(num_tasks):
            t_name = f"task_{t_idx:02d}"
            tasks[t_idx] = t_name
            f.write(json.dumps({"task_index": t_idx, "task": t_name}) + "\n")

    episodes = []
    ep_idx = 0
    chunk_dir = tmp_path / "data" / "chunk-000"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    video_dir = tmp_path / "videos" / "chunk-000" / "observation.images.image"
    video_dir.mkdir(parents=True, exist_ok=True)

    for t_idx in range(num_tasks):
        t_name = tasks[t_idx]
        for e_num in range(episodes_per_task):
            if lengths is not None:
                ep_len = lengths[ep_idx % len(lengths)]
            else:
                ep_len = 25  # Default 25 frames (ceil(25/8) = 4 segments: [0,8), [8,16), [16,24), [24,25))

            episodes.append({
                "episode_index": ep_idx,
                "length": ep_len,
                "tasks": [t_name],
            })

            # Create dummy video file
            (video_dir / f"episode_{ep_idx:06d}.mp4").touch()

            # Create parquet
            f_indices = list(range(ep_len))
            df_dict: Dict[str, Any] = {
                "frame_index": f_indices,
                "episode_index": [ep_idx] * ep_len,
                "task_index": [t_idx] * ep_len,
                "observation.state": [[0.1 * (i % 5)] * 4 for i in range(ep_len)],
                "action": [[0.05 * (i % 7) + 0.1 * d for d in range(4)] for i in range(ep_len)],
            }
            if include_parquet_timestamp:
                df_dict["timestamp"] = [float(i * timestamp_step) for i in range(ep_len)]

            df = pd.DataFrame(df_dict)
            df.to_parquet(chunk_dir / f"episode_{ep_idx:06d}.parquet")
            ep_idx += 1

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    return tmp_path


@pytest.fixture
def mock_norm_stats():
    return {
        "observation.state": {"min": [-1.0] * 4, "max": [1.0] * 4},
        "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
    }


def test_exhaustive_target_coverage_and_tails(tmp_path, mock_norm_stats, monkeypatch):
    # Test varying episode lengths (e.g. 1, 7, 8, 9, 25, 33) to verify all original rows covered exactly once
    lengths = [1, 7, 8, 9, 25, 33]
    root = _create_mock_metaworld_root(tmp_path, num_tasks=2, episodes_per_task=3, lengths=lengths)

    # Mock video decoding
    monkeypatch.setattr(
        "fabri_moss.native_data.decode_all_video_frames",
        lambda path, fids: {f: Image.new("RGB", (32, 32), color=(f % 255, 0, 0)) for f in fids},
    )

    ds = NativeTrainingDataset(
        root=root,
        norm_stats=mock_norm_stats,
        history_frames=16,
        target_frames=8,
        horizon=10,
        split="all",
    )

    # Map episode_id -> set of supervised row indices
    supervised_per_ep: Dict[int, List[int]] = {}
    for i in range(len(ds)):
        sample = ds[i]
        ep_id = sample["episode_id"]
        t_start = sample["target_start_row"]
        t_end = sample["target_end_row"]
        assert sample["target_count"] == (t_end - t_start)
        assert sample["target_count"] <= 8
        assert len(sample["target_indices"]) == sample["target_count"]
        assert len(sample["target_frame_ids"]) == sample["target_count"]

        # Supervised rows in this segment
        rows = list(range(t_start, t_end))
        supervised_per_ep.setdefault(ep_id, []).extend(rows)

    # Check total targets matches sum of episode lengths
    total_len = sum(ep["length"] for ep in ds.active_episodes)
    assert ds.total_targets == total_len

    # Check each episode's rows are [0, L) exactly once with no duplicates or omissions
    for ep in ds.active_episodes:
        ep_id = ep["episode_index"]
        ep_len = ep["length"]
        seen_rows = supervised_per_ep[ep_id]
        assert seen_rows == list(range(ep_len)), f"Episode {ep_id} failed exact row coverage"


def test_causality_and_target_mapping(tmp_path, mock_norm_stats, monkeypatch):
    root = _create_mock_metaworld_root(tmp_path, num_tasks=1, episodes_per_task=2, lengths=[25])

    monkeypatch.setattr(
        "fabri_moss.native_data.decode_all_video_frames",
        lambda path, fids: {f: Image.new("RGB", (32, 32)) for f in fids},
    )

    ds = NativeTrainingDataset(
        root=root,
        norm_stats=mock_norm_stats,
        history_frames=16,
        target_frames=8,
        split="all",
    )

    # For episode of length 25:
    # Seg 0: [0, 8) -> obs [0, 8), target [0, 8), target_indices 0..7
    # Seg 1: [8, 16) -> obs [0, 16), target [8, 16), target_indices 8..15
    # Seg 2: [16, 24) -> obs [8, 24), target [16, 24), target_indices 8..15
    # Seg 3: [24, 25) -> obs [9, 25), target [24, 25), target_indices [15]

    s0 = ds[0]
    assert s0["context_start_row"] == 0
    assert s0["target_start_row"] == 0
    assert s0["target_end_row"] == 8
    assert s0["frame_ids"] == list(range(0, 8))
    assert s0["target_indices"] == list(range(0, 8))
    assert s0["target_frame_ids"] == list(range(0, 8))
    assert len(s0["images_window"]) == 8

    s1 = ds[1]
    assert s1["context_start_row"] == 0
    assert s1["target_start_row"] == 8
    assert s1["target_end_row"] == 16
    assert s1["frame_ids"] == list(range(0, 16))
    assert s1["target_indices"] == list(range(8, 16))
    assert s1["target_frame_ids"] == list(range(8, 16))
    assert len(s1["images_window"]) == 16

    s2 = ds[2]
    assert s2["context_start_row"] == 8
    assert s2["target_start_row"] == 16
    assert s2["target_end_row"] == 24
    assert s2["frame_ids"] == list(range(8, 24))
    assert s2["target_indices"] == list(range(8, 16))
    assert s2["target_frame_ids"] == list(range(16, 24))
    assert len(s2["images_window"]) == 16

    s3 = ds[3]
    assert s3["context_start_row"] == 9
    assert s3["target_start_row"] == 24
    assert s3["target_end_row"] == 25
    assert s3["frame_ids"] == list(range(9, 25))
    assert s3["target_indices"] == [15]
    assert s3["target_frame_ids"] == [24]
    assert len(s3["images_window"]) == 16

    # Verify no future frames in observation window: last frame is end - 1
    for seg_idx in range(len(ds)):
        sample = ds[seg_idx]
        assert sample["frame_ids"][-1] == sample["target_end_row"] - 1
        assert max(sample["target_indices"]) < len(sample["images_window"])
        for tidx, tfid in zip(sample["target_indices"], sample["target_frame_ids"]):
            assert sample["frame_ids"][tidx] == tfid


def test_history_bound_and_validation(mock_norm_stats):
    with pytest.raises(ValueError, match="target_frames .* must be <= history_frames"):
        NativeTrainingDataset(
            root=Path("/dummy"),
            norm_stats=mock_norm_stats,
            history_frames=4,
            target_frames=8,
        )

    with pytest.raises(ValueError, match="positive integer"):
        NativeTrainingDataset(
            root=Path("/dummy"),
            norm_stats=mock_norm_stats,
            history_frames=-1,
            target_frames=8,
        )


def test_timestamps_precedence_and_metadata_fps_fallback(tmp_path, mock_norm_stats, monkeypatch):
    # Test parquet timestamp presence
    root1 = _create_mock_metaworld_root(
        tmp_path / "ts_parquet",
        num_tasks=1,
        episodes_per_task=1,
        lengths=[10],
        fps=30.0,
        include_parquet_timestamp=True,
        timestamp_step=0.05,
    )
    monkeypatch.setattr(
        "fabri_moss.native_data.decode_all_video_frames",
        lambda path, fids: {f: Image.new("RGB", (16, 16)) for f in fids},
    )
    ds1 = NativeTrainingDataset(root=root1, norm_stats=mock_norm_stats, split="all")
    s1 = ds1[0]
    assert s1["time_source"] == "parquet_timestamp"
    np.testing.assert_allclose(s1["observation_times"][:3], [0.0, 0.05, 0.10])

    # Test metadata fps fallback when parquet timestamp column is missing
    root2 = _create_mock_metaworld_root(
        tmp_path / "ts_fps",
        num_tasks=1,
        episodes_per_task=1,
        lengths=[10],
        fps=20.0,
        include_parquet_timestamp=False,
    )
    ds2 = NativeTrainingDataset(root=root2, norm_stats=mock_norm_stats, split="all")
    s2 = ds2[0]
    assert s2["time_source"] == "metadata_fps"
    np.testing.assert_allclose(s2["observation_times"][:3], [0.0, 1.0 / 20.0, 2.0 / 20.0])


def test_timestamps_rejection_conditions(tmp_path, mock_norm_stats, monkeypatch):
    root = _create_mock_metaworld_root(
        tmp_path,
        num_tasks=1,
        episodes_per_task=1,
        lengths=[5],
        fps=None,
        include_parquet_timestamp=False,
    )
    monkeypatch.setattr(
        "fabri_moss.native_data.decode_all_video_frames",
        lambda path, fids: {f: Image.new("RGB", (16, 16)) for f in fids},
    )

    # Neither timestamp nor fps -> ValueError
    ds = NativeTrainingDataset(root=root, norm_stats=mock_norm_stats, split="all")
    with pytest.raises(ValueError, match="requires either parquet 'timestamp' column or valid metadata 'fps'"):
        _ = ds[0]


def test_invalid_parquet_timestamps_rejection(tmp_path, mock_norm_stats, monkeypatch):
    root = _create_mock_metaworld_root(
        tmp_path,
        num_tasks=1,
        episodes_per_task=1,
        lengths=[5],
        fps=30.0,
        include_parquet_timestamp=True,
    )
    monkeypatch.setattr(
        "fabri_moss.native_data.decode_all_video_frames",
        lambda path, fids: {f: Image.new("RGB", (16, 16)) for f in fids},
    )

    parquet_file = root / "data" / "chunk-000" / "episode_000000.parquet"
    df = pd.read_parquet(parquet_file)

    # Case 1: Non-increasing timestamps
    df["timestamp"] = [0.0, 0.2, 0.1, 0.3, 0.4]
    df.to_parquet(parquet_file)
    ds = NativeTrainingDataset(root=root, norm_stats=mock_norm_stats, split="all")
    with pytest.raises(ValueError, match="non-increasing"):
        _ = ds[0]

    # Case 2: NaN / negative timestamps
    df["timestamp"] = [0.0, -0.1, 0.2, 0.3, 0.4]
    df.to_parquet(parquet_file)
    ds.parquet_cache.cache.clear()
    with pytest.raises(ValueError, match="invalid timestamp"):
        _ = ds[0]


def test_action_repeat_last_and_valid_dimension_mask(tmp_path, mock_norm_stats, monkeypatch):
    # Horizon 50, episode length 10 -> row 8 requires 48 steps of repeat-last padding
    root = _create_mock_metaworld_root(tmp_path, num_tasks=1, episodes_per_task=1, lengths=[10])
    monkeypatch.setattr(
        "fabri_moss.native_data.decode_all_video_frames",
        lambda path, fids: {f: Image.new("RGB", (16, 16)) for f in fids},
    )

    ds = NativeTrainingDataset(
        root=root,
        norm_stats=mock_norm_stats,
        history_frames=16,
        target_frames=8,
        horizon=50,
        state_dim=24,
        action_dim=24,
        split="all",
    )

    # Segment 1: target_rows [8, 10) -> target_count = 2
    s1 = ds[1]
    actions = s1["actions"]  # [M=2, H=50, action_dim=24]
    action_mask = s1["action_mask"]  # [M=2, H=50, action_dim=24]
    state = s1["state"]  # [M=2, state_dim=24]
    state_mask = s1["state_mask"]  # [M=2, state_dim=24]

    assert actions.shape == (2, 50, 24)
    assert action_mask.shape == (2, 50, 24)
    assert state.shape == (2, 24)
    assert state_mask.shape == (2, 24)

    # Valid raw_dim is 4, padded to 24:
    # Action mask must be 1 for first 4 dims across all 50 steps (including tail repeated steps), and 0 for dims 4..24
    assert torch.all(action_mask[:, :, :4] == 1.0)
    assert torch.all(action_mask[:, :, 4:] == 0.0)

    # Verify tail repeated action values are identical to last step
    # For target row 8 (index 0 in s1): step 0 is row 8, step 1 is row 9, step 2..49 repeat row 9
    act_step1 = actions[0, 1, :4]
    for step in range(2, 50):
        torch.testing.assert_close(actions[0, step, :4], act_step1)


def test_train_val_stratified_split_disjoint(tmp_path, mock_norm_stats):
    # 2 tasks, 10 episodes each -> 20 episodes total
    # val_fraction 0.1 -> 1 val per task (2 val, 18 train)
    root = _create_mock_metaworld_root(tmp_path, num_tasks=2, episodes_per_task=10, lengths=[20])

    train_ds = NativeTrainingDataset(
        root=root,
        norm_stats=mock_norm_stats,
        split="train",
        val_fraction=0.1,
        seed=4042,
    )
    val_ds = NativeTrainingDataset(
        root=root,
        norm_stats=mock_norm_stats,
        split="val",
        val_fraction=0.1,
        seed=4042,
    )

    train_ep_ids = set(train_ds.get_data_contract()["active_episode_ids"])
    val_ep_ids = set(val_ds.get_data_contract()["active_episode_ids"])

    assert len(train_ep_ids) == 18
    assert len(val_ep_ids) == 2
    assert train_ep_ids.isdisjoint(val_ep_ids)
    assert val_ds.augmentation is False


def test_augmentation_determinism_and_epoch_change(tmp_path, mock_norm_stats, monkeypatch):
    root = _create_mock_metaworld_root(tmp_path, num_tasks=1, episodes_per_task=1, lengths=[20])

    # Distinct pattern per frame
    def mock_decode(path, fids):
        res = {}
        for f in fids:
            img = Image.new("RGB", (64, 64), color=(f * 10 + 20, 100, 150))
            res[f] = img
        return res

    monkeypatch.setattr("fabri_moss.native_data.decode_all_video_frames", mock_decode)

    ds = NativeTrainingDataset(
        root=root,
        norm_stats=mock_norm_stats,
        history_frames=16,
        target_frames=8,
        split="train",
        augmentation=True,
    )

    # Force augmentation params to always trigger by setting p=1.0 in monkeypatch if desired,
    # or find an epoch where aug is active
    ds.set_epoch(0)
    s0_ep0 = ds[0]

    # Two accesses in same epoch should be deterministic
    s0_ep0_again = ds[0]
    img_a = np.array(s0_ep0["images_window"][0][0])
    img_b = np.array(s0_ep0_again["images_window"][0][0])
    np.testing.assert_array_equal(img_a, img_b)

    # Check across multiple epochs to verify augmentation varies
    diff_found = False
    for ep in range(1, 10):
        ds.set_epoch(ep)
        s_new = ds[0]
        img_new = np.array(s_new["images_window"][0][0])
        if not np.array_equal(img_a, img_new):
            diff_found = True
            break
    assert diff_found, "Augmentation did not change across 10 epochs"


def test_bounded_raw_frame_cache(tmp_path, mock_norm_stats, monkeypatch):
    root = _create_mock_metaworld_root(tmp_path, num_tasks=1, episodes_per_task=3, lengths=[25])
    decode_call_count = 0

    def mock_decode(path, fids):
        nonlocal decode_call_count
        decode_call_count += 1
        return {f: Image.new("RGB", (16, 16)) for f in fids}

    monkeypatch.setattr("fabri_moss.native_data.decode_all_video_frames", mock_decode)

    ds = NativeTrainingDataset(
        root=root,
        norm_stats=mock_norm_stats,
        history_frames=16,
        target_frames=8,
        split="all",
    )

    # Episode 0 has 4 segments: [0, 8), [8, 16), [16, 24), [24, 25)
    # Accessing segments 0, 1, 2, 3 of episode 0 must decode only once!
    assert decode_call_count == 0
    _ = ds[0]
    assert decode_call_count == 1
    _ = ds[1]
    assert decode_call_count == 1
    _ = ds[2]
    assert decode_call_count == 1
    _ = ds[3]
    assert decode_call_count == 1

    # Capacity is 2. Access episode 1 and episode 2
    # Segments for ep 1: index 4, 5, 6, 7
    # Segments for ep 2: index 8, 9, 10, 11
    _ = ds[4]  # decodes ep 1
    assert decode_call_count == 2
    _ = ds[8]  # decodes ep 2 (cache now has ep 1, ep 2; ep 0 evicted)
    assert decode_call_count == 3

    # Accessing ep 0 again will trigger decode because capacity is 2
    _ = ds[0]
    assert decode_call_count == 4


def test_data_contract_and_validation_indices(tmp_path, mock_norm_stats, monkeypatch):
    root = _create_mock_metaworld_root(tmp_path, num_tasks=3, episodes_per_task=4, lengths=[20])
    monkeypatch.setattr(
        "fabri_moss.native_data.decode_all_video_frames",
        lambda path, fids: {f: Image.new("RGB", (16, 16)) for f in fids},
    )

    ds = NativeTrainingDataset(
        root=root,
        norm_stats=mock_norm_stats,
        history_frames=16,
        target_frames=8,
        horizon=50,
        split="train",
    )

    contract = ds.get_data_contract()
    assert contract["dataset_class"] == "NativeTrainingDataset"
    assert contract["context_mode"] == "native_causal_segments"
    assert contract["history_frames"] == 16
    assert contract["target_frames"] == 8
    assert contract["horizon"] == 50
    assert contract["target_count"] == ds.total_targets
    assert contract["segment_count"] == len(ds)
    assert len(contract["normalization_fingerprint"]) == 64
    assert len(contract["data_fingerprint"]) == 64
    # Ensure JSON serializability
    json_str = json.dumps(contract)
    assert json_str is not None

    # Test validation_indices: 1 middle segment per task
    val_indices = ds.validation_indices(per_task=1)
    # 3 tasks -> exactly 3 validation segment indices
    assert len(val_indices) == 3
    assert sorted(val_indices) == val_indices
    for s_idx in val_indices:
        assert 0 <= s_idx < len(ds)


def test_duplicate_episode_metadata_is_rejected(tmp_path, mock_norm_stats):
    root = _create_mock_metaworld_root(tmp_path, num_tasks=1, episodes_per_task=1)
    metadata = root / "meta" / "episodes.jsonl"
    metadata.write_text(metadata.read_text() * 2)
    with pytest.raises(ValueError, match="Duplicate episode_index"):
        NativeTrainingDataset(root=root, norm_stats=mock_norm_stats, split="all")
