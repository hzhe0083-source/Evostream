import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from fabri_moss.data import (
    MetaWorldWindows,
    compute_history_row_indices,
    normalize_and_mask,
    partition_episode_consume_chunks,
)


def test_compute_history_row_indices():
    assert compute_history_row_indices(0, window=2, stride=5) == [0]
    assert compute_history_row_indices(2, window=2, stride=5) == [0, 2]
    assert compute_history_row_indices(5, window=2, stride=5) == [0, 5]
    assert compute_history_row_indices(12, window=2, stride=5) == [7, 12]
    assert compute_history_row_indices(10, window=1, stride=5) == [10]
    assert compute_history_row_indices(10, window=3, stride=4) == [2, 6, 10]


def test_official_denominator_formula_and_no_smallrange_replace():
    min_val = [0.0, 1.0]
    max_val = [10.0, 1.0000001]
    raw = np.array([5.0, 1.0000001], dtype=np.float32)

    padded, mask = normalize_and_mask(raw, min_val, max_val, target_dim=4)

    assert abs(padded[0] - 0.0) < 1e-4
    expected = np.clip(2 * (raw - np.asarray(min_val, dtype=np.float32)) / (np.asarray(max_val, dtype=np.float32) - np.asarray(min_val, dtype=np.float32) + 1e-8) - 1, -1, 1)
    np.testing.assert_array_equal(padded[:2], expected)
    assert padded[1] > 0.5
    assert mask[0] == 1.0 and mask[1] == 1.0
    assert mask[2] == 0.0 and mask[3] == 0.0


def test_strict_dimension_and_target_dim():
    min_val = [0.0, 0.0]
    max_val = [1.0, 1.0]

    with pytest.raises(ValueError, match="Expected 1D data of dim 2"):
        normalize_and_mask(np.array([0.5, 0.5, 0.5]), min_val, max_val, target_dim=4)

    with pytest.raises(ValueError, match="target_dim .* cannot be smaller"):
        normalize_and_mask(np.array([0.5, 0.5]), min_val, max_val, target_dim=1)


def test_non_finite_rejection():
    min_val = [0.0]
    max_val = [1.0]

    with pytest.raises(ValueError, match="non-finite"):
        normalize_and_mask(np.array([np.nan]), min_val, max_val, target_dim=2)

    with pytest.raises(ValueError, match="non-finite"):
        normalize_and_mask(np.array([np.inf]), min_val, max_val, target_dim=2)


def test_real_server_templates_and_end_to_end_dataset(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        meta_dir = root / "meta"
        meta_dir.mkdir(parents=True)

        info = {
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        }
        with open(meta_dir / "info.json", "w") as f:
            json.dump(info, f)

        with open(meta_dir / "tasks.jsonl", "w") as f:
            f.write(json.dumps({"task_index": 0, "task": "Pick up nut"}) + "\n")
            f.write(json.dumps({"task_index": 1, "task": "Close drawer"}) + "\n")

        episodes = [
            {"episode_index": 0, "length": 15, "tasks": ["Pick up nut"]},
            {"episode_index": 1, "length": 20, "tasks": ["Close drawer"]},
        ]
        with open(meta_dir / "episodes.jsonl", "w") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")

        chunk0_dir = root / "data" / "chunk-000"
        chunk0_dir.mkdir(parents=True)

        video_dir = root / "videos" / "chunk-000" / "observation.images.image"
        video_dir.mkdir(parents=True)
        (video_dir / "episode_000000.mp4").touch()
        (video_dir / "episode_000001.mp4").touch()

        def make_df(ep_idx, n_rows, task_idx):
            return pd.DataFrame({
                "frame_index": list(range(n_rows)),
                "episode_index": [ep_idx] * n_rows,
                "task_index": [task_idx] * n_rows,
                "observation.state": [[0.5] * 4 for _ in range(n_rows)],
                "action": [[0.1, -0.2, 0.3, 0.4] for _ in range(n_rows)],
            })

        df0 = make_df(0, 15, 0)
        df1 = make_df(1, 20, 1)
        df0.to_parquet(chunk0_dir / "episode_000000.parquet")
        df1.to_parquet(chunk0_dir / "episode_000001.parquet")

        norm_stats = {
            "observation.state": {"min": [0.0] * 4, "max": [1.0] * 4},
            "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
        }

        from PIL import Image
        import fabri_moss.data
        monkeypatch.setattr(
            fabri_moss.data,
            "decode_exact_video_frames",
            lambda path, frame_indices: {
                int(i): Image.new("RGB", (448, 448)) for i in frame_indices
            },
        )

        ds_train = MetaWorldWindows(
            root=root,
            norm_stats=norm_stats,
            split="train",
            val_fraction=0.5,
            seed=4042,
        )
        ds_val = MetaWorldWindows(
            root=root,
            norm_stats=norm_stats,
            split="val",
            val_fraction=0.5,
            seed=4042,
        )

        train_eps = {ep["episode_index"] for ep in ds_train.active_episodes}
        val_eps = {ep["episode_index"] for ep in ds_val.active_episodes}
        assert len(train_eps.intersection(val_eps)) == 0
        assert len(train_eps) == 1 and len(val_eps) == 1

        sample = ds_train[0]
        assert sample["prompt"] in ["Pick up nut", "Close drawer"]
        assert sample["state"].shape == (1, 24)
        assert sample["state_mask"].shape == (1, 24)
        assert sample["state_mask"][0, :4].sum() == 4
        assert sample["state_mask"][0, 4:].sum() == 0
        assert sample["actions"].shape == (1, 50, 24)
        assert sample["action_mask"].shape == (1, 50, 24)
        valid_len = sample["valid_action_lengths"][0]
        assert sample["action_mask"][0, :, :4].sum() == valid_len * 4
        assert sample["action_mask"][0, :, 4:].sum() == 0
        assert len(sample["frame_ids"]) == 1
        assert sample["frame_ids"][0] == 0

        # Sample at row 12 with window=2, stride=5 -> rows [7, 12] -> frame_ids [7, 12]
        sample_12 = ds_train[12]
        assert sample_12["frame_ids"] == [7, 12]
        # Repeat-last padding test: row 12 in 15-frame episode has 3 rows left (12, 13, 14),
        # remaining 47 rows repeat row 14 action
        act = sample_12["actions"][0]
        assert torch.allclose(act[3], act[49])


def test_partition_episode_consume_chunks():
    import random

    ep_len = 100
    frame_stride = 3
    window = 5
    min_context = 2
    sampled_rows = list(range(0, ep_len, frame_stride))

    # Fixed seed determinism
    rng1 = random.Random(4042)
    chunks1 = partition_episode_consume_chunks(
        ep_len=ep_len,
        frame_stride=frame_stride,
        window=window,
        min_context_frames=min_context,
        rng=rng1,
    )

    rng2 = random.Random(4042)
    chunks2 = partition_episode_consume_chunks(
        ep_len=ep_len,
        frame_stride=frame_stride,
        window=window,
        min_context_frames=min_context,
        rng=rng2,
    )
    assert chunks1 == chunks2

    # Different seed producing different chunk sequence
    rng3 = random.Random(9999)
    chunks3 = partition_episode_consume_chunks(
        ep_len=ep_len,
        frame_stride=frame_stride,
        window=window,
        min_context_frames=min_context,
        rng=rng3,
    )
    assert chunks1 != chunks3

    # Check that chunks do not overlap and their concatenation exactly covers sampled_rows
    concatenated = []
    for idx, c in enumerate(chunks1):
        assert len(c) > 0
        if idx < len(chunks1) - 1:
            assert min_context <= len(c) <= window
        else:
            assert len(c) <= window
        concatenated.extend(c)
    assert concatenated == sampled_rows

    concatenated3 = []
    for c in chunks3:
        concatenated3.extend(c)
    assert concatenated3 == sampled_rows


def test_causal_chunk_returns_all_targets_and_no_future(monkeypatch):
    """Causal chunks expose the full chunk but each target sees only its prefix."""
    from PIL import Image
    import fabri_moss.data

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        meta_dir = root / "meta"
        meta_dir.mkdir(parents=True)
        (root / "data" / "chunk-000").mkdir(parents=True)
        (root / "videos" / "chunk-000" / "observation.images.image").mkdir(parents=True)
        json.dump({
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "fps": 10.0,
        }, open(meta_dir / "info.json", "w"))
        (meta_dir / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "test"}) + "\n")
        (meta_dir / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 7}) + "\n")
        (root / "videos" / "chunk-000" / "observation.images.image" / "episode_000000.mp4").touch()
        pd.DataFrame({
            "frame_index": list(range(7)), "episode_index": [0] * 7,
            "task_index": [0] * 7,
            "observation.state": [[float(i)] * 4 for i in range(7)],
            "action": [[float(i)] * 4 for i in range(7)],
        }).to_parquet(root / "data" / "chunk-000" / "episode_000000.parquet")
        stats = {
            "observation.state": {"min": [0.0] * 4, "max": [10.0] * 4},
            "action": {"min": [0.0] * 4, "max": [10.0] * 4},
        }
        monkeypatch.setattr(
            fabri_moss.data, "decode_exact_video_frames",
            lambda path, ids: {int(i): Image.new("RGB", (8, 8), color=(int(i), 0, 0)) for i in ids},
        )
        ds = MetaWorldWindows(
            root=root, norm_stats=stats, split="all", context_mode="causal",
            window=3, min_context_frames=2, frame_stride=1, horizon=2,
            state_dim=4, action_dim=4, seed=7,
        )
        assert len(ds) >= 2
        for sample in (ds[0], ds[1]):
            rows = sample["target_rows"]
            indices = sample["target_indices"]
            visible = sample["visible_counts"]
            assert rows == sorted(rows)
            assert indices == list(range(len(rows)))
            assert visible == [i + 1 for i in indices]
            assert sample["frame_ids"] == rows
            assert sample["target_frame_ids"] == rows
            assert sample["state"].shape[0] == len(rows)
            assert sample["actions"].shape[0] == len(rows)
            for group, count in zip(sample["replay_groups"], visible):
                assert group["observation_indices"] == list(range(count))
                assert max(group["observation_indices"]) < len(sample["frame_ids"])

        cadence = MetaWorldWindows(
            root=root,
            norm_stats=stats,
            split="all",
            context_mode="causal",
            window=3,
            min_context_frames=2,
            frame_stride=5,
            decision_stride=5,
            horizon=5,
            state_dim=4,
            action_dim=4,
            seed=7,
        )
        all_frames = []
        all_targets = []
        for sample in (cadence[i] for i in range(len(cadence))):
            all_frames.extend(sample["frame_ids"])
            all_targets.extend(sample["target_frame_ids"])
            for group in sample["replay_groups"]:
                target_pos = group["target_positions"][0]
                assert max(group["observation_indices"]) <= sample["target_indices"][target_pos]
        assert sorted(all_frames) == list(range(7))
        assert sorted(all_targets) == [0, 5]
        tail = next(cadence[i] for i in range(len(cadence)) if 5 in cadence[i]["target_frame_ids"])
        target_pos = tail["target_frame_ids"].index(5)
        assert tail["valid_action_lengths"][target_pos] == 2
        assert tail["action_time_mask"][target_pos].tolist() == [1, 1, 0, 0, 0]


@pytest.mark.parametrize(
    "timestamps,info_fps,expected_times,expected_source",
    [
        # Priority: parquet timestamp takes priority over info_fps
        ([0.0, 0.2, 0.7], 30.0, [0.0, 0.2, 0.7], "parquet_timestamp"),
        # When parquet timestamp is missing (None), fallback to metadata fps
        (None, 30.0, [0.0, 1.0 / 30.0, 2.0 / 30.0], "metadata_fps"),
        # When both are missing, neither field should be present
        (None, None, None, None),
    ],
)
def test_metaworld_windows_timestamp_resolution(
    monkeypatch,
    timestamps,
    info_fps,
    expected_times,
    expected_source,
):
    from PIL import Image
    import fabri_moss.data

    monkeypatch.setattr(
        fabri_moss.data,
        "decode_exact_video_frames",
        lambda path, frame_indices: {
            int(i): Image.new("RGB", (448, 448)) for i in frame_indices
        },
    )

    df_dict = {
        "frame_index": [0, 1, 2],
        "episode_index": [0, 0, 0],
        "task_index": [0, 0, 0],
        "observation.state": [[0.5] * 4 for _ in range(3)],
        "action": [[0.1, -0.2, 0.3, 0.4] for _ in range(3)],
    }
    if timestamps is not None:
        df_dict["timestamp"] = timestamps
    df = pd.DataFrame(df_dict)

    ds = object.__new__(MetaWorldWindows)
    ds.anchors = [({"episode_index": 0, "length": 3}, 2)]
    ds.context_mode = "window"
    ds.window = 3
    ds.frame_stride = 1
    ds._get_episode_dataframe = lambda ep: df
    ds.tasks = {0: "task"}
    ds.info = {"fps": info_fps} if info_fps is not None else {}
    ds._locate_video_path = lambda ep_idx: Path("/dummy/video.mp4")

    ds.state_mins = [0.0] * 4
    ds.state_maxs = [1.0] * 4
    ds.action_mins = [-1.0] * 4
    ds.action_maxs = [1.0] * 4
    ds.raw_dim = 4
    ds.horizon = 2
    ds.state_dim = 24
    ds.action_dim = 24

    sample = ds[0]

    if expected_times is not None:
        assert "observation_times" in sample
        np.testing.assert_allclose(sample["observation_times"], expected_times)
        assert sample["time_source"] == expected_source
    else:
        assert "observation_times" not in sample
        assert "time_source" not in sample


@pytest.mark.parametrize(
    "invalid_ts",
    [
        [0.0, float("nan"), 0.2],
        [0.0, float("inf"), 0.2],
        [0.0, -0.1, 0.2],
    ],
)
def test_metaworld_windows_invalid_parquet_timestamp_rejection(monkeypatch, invalid_ts):
    from PIL import Image
    import fabri_moss.data

    monkeypatch.setattr(
        fabri_moss.data,
        "decode_exact_video_frames",
        lambda path, frame_indices: {
            int(i): Image.new("RGB", (448, 448)) for i in frame_indices
        },
    )

    df = pd.DataFrame({
        "frame_index": [0, 1, 2],
        "episode_index": [0, 0, 0],
        "task_index": [0, 0, 0],
        "observation.state": [[0.5] * 4 for _ in range(3)],
        "action": [[0.1, -0.2, 0.3, 0.4] for _ in range(3)],
        "timestamp": invalid_ts,
    })

    ds = object.__new__(MetaWorldWindows)
    ds.anchors = [({"episode_index": 0, "length": 3}, 2)]
    ds.context_mode = "window"
    ds.window = 3
    ds.frame_stride = 1
    ds._get_episode_dataframe = lambda ep: df
    ds.tasks = {0: "task"}
    ds.info = {"fps": 30.0}
    ds._locate_video_path = lambda ep_idx: Path("/dummy/video.mp4")

    ds.state_mins = [0.0] * 4
    ds.state_maxs = [1.0] * 4
    ds.action_mins = [-1.0] * 4
    ds.action_maxs = [1.0] * 4
    ds.raw_dim = 4
    ds.horizon = 2
    ds.state_dim = 24
    ds.action_dim = 24

    with pytest.raises(ValueError, match="invalid timestamp"):
        _ = ds[0]
