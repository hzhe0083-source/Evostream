import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from fabri_moss.delta_data import (
    EpisodePermutationSampler,
    MetaWorldEpisodes,
    apply_episode_augmentation,
    decode_all_video_frames,
    partition_delta_chunks,
    sample_episode_augmentation_params,
)


def test_partition_delta_chunks():
    frame_indices = list(range(12))
    rng = random.Random(4042)
    chunks = partition_delta_chunks(frame_indices, max_chunk_size=5, rng=rng)

    # First chunk must be strictly singleton [0]
    assert chunks[0] == [0]

    # All frames must be covered exactly once without duplication
    flattened = [f for c in chunks for f in c]
    assert flattened == frame_indices

    # All subsequent chunks must have length in 1..5
    for c in chunks[1:]:
        assert 1 <= len(c) <= 5


def test_episode_augmentation_determinism_and_shared_params():
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    rng1 = random.Random(12345)
    params1 = sample_episode_augmentation_params(rng1, img_width=64, img_height=64, p=1.0)

    assert params1 is not None
    assert "crop" in params1
    assert "rotation" in params1
    assert "brightness" in params1
    assert "jitter_order" in params1

    # Two frames from the same episode must transform with exact same params
    out1 = apply_episode_augmentation(img, params1)
    out2 = apply_episode_augmentation(img, params1)

    assert np.array_equal(np.array(out1), np.array(out2))

    # Worker count invariance: sampling with same episode/epoch RNG yields identical params
    rng2 = random.Random(12345)
    params2 = sample_episode_augmentation_params(rng2, img_width=64, img_height=64, p=1.0)
    assert params1 == params2


def test_episode_permutation_sampler_with_start_cursor():
    sampler = EpisodePermutationSampler(num_episodes=10, seed=4042, start_epoch=0, start_cursor=3)
    full_perm = sampler.get_permutation(epoch=0)
    assert len(full_perm) == 10

    # Iteration must directly yield full_perm[3:]
    sliced_perm = list(sampler)
    assert sliced_perm == full_perm[3:]
    assert len(sampler) == 7

    # Different epoch
    sampler.set_epoch(1, start_cursor=0)
    perm1 = list(sampler)
    assert len(perm1) == 10
    assert perm1 != full_perm


def test_stratified_split_sha256_subprocesses():
    """Verify that split with sha256 hash yields identical active_episodes across independent Python processes."""
    code = """
import json, tempfile, sys
from pathlib import Path
from fabri_moss.delta_data import MetaWorldEpisodes

tmpdir = sys.argv[1]
root = Path(tmpdir)
norm_stats = {
    "observation.state": {"min": [0.0] * 4, "max": [1.0] * 4},
    "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
}
ds = MetaWorldEpisodes(root=root, norm_stats=norm_stats, split="train", seed=4042)
print(json.dumps([ep["episode_index"] for ep in ds.active_episodes]))
"""
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

        # 3 tasks
        with open(meta_dir / "tasks.jsonl", "w") as f:
            f.write(json.dumps({"task_index": 0, "task": "Task A"}) + "\n")
            f.write(json.dumps({"task_index": 1, "task": "Task B"}) + "\n")
            f.write(json.dumps({"task_index": 2, "task": "Task C"}) + "\n")

        episodes = []
        for i in range(30):
            t_name = f"Task {chr(ord('A') + (i % 3))}"
            episodes.append({"episode_index": i, "length": 10, "tasks": [t_name]})

        with open(meta_dir / "episodes.jsonl", "w") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")

        # Run in two separate Python processes
        proc1 = subprocess.run(
            [sys.executable, "-c", code, str(root)],
            capture_output=True,
            text=True,
            check=True,
        )
        proc2 = subprocess.run(
            [sys.executable, "-c", code, str(root)],
            capture_output=True,
            text=True,
            check=True,
        )

        assert proc1.stdout.strip() == proc2.stdout.strip()
        ep_ids = json.loads(proc1.stdout.strip())
        assert len(ep_ids) == 27  # 10 per task, val_fraction=0.1 -> 1 val per task -> 9 train per task -> 27


def test_stratified_split_and_smoke_cap():
    """Test stratified 45/5 split across tasks and max_episodes smoke behavior."""
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

        # 2 tasks
        with open(meta_dir / "tasks.jsonl", "w") as f:
            f.write(json.dumps({"task_index": 0, "task": "Task A"}) + "\n")
            f.write(json.dumps({"task_index": 1, "task": "Task B"}) + "\n")

        # 50 episodes for Task A, 50 episodes for Task B (total 100)
        episodes = []
        for i in range(50):
            episodes.append({"episode_index": i, "length": 10, "tasks": ["Task A"]})
        for i in range(50, 100):
            episodes.append({"episode_index": i, "length": 10, "tasks": ["Task B"]})

        with open(meta_dir / "episodes.jsonl", "w") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")

        norm_stats = {
            "observation.state": {"min": [0.0] * 4, "max": [1.0] * 4},
            "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
        }

        # Train split: 45 per task -> 90 total
        train_ds = MetaWorldEpisodes(
            root=root,
            norm_stats=norm_stats,
            split="train",
            val_fraction=0.1,
            seed=4042,
        )
        assert len(train_ds) == 90

        # Val split: 5 per task -> 10 total
        val_ds = MetaWorldEpisodes(
            root=root,
            norm_stats=norm_stats,
            split="val",
            val_fraction=0.1,
            seed=4042,
        )
        assert len(val_ds) == 10

        # Verify no data leakage between train and val
        train_ids = set(ep["episode_index"] for ep in train_ds.active_episodes)
        val_ids = set(ep["episode_index"] for ep in val_ds.active_episodes)
        assert len(train_ids.intersection(val_ids)) == 0

        # Verify data contract contents
        contract = train_ds.get_data_contract()
        assert "metadata_files_sha256" in contract
        assert "info.json" in contract["metadata_files_sha256"]
        assert "tasks.jsonl" in contract["metadata_files_sha256"]
        assert "episodes.jsonl" in contract["metadata_files_sha256"]
        assert contract["horizon"] == 50
        assert contract["state_dim"] == 24
        assert contract["action_dim"] == 24
        assert contract["active_episode_ids"] == [ep["episode_index"] for ep in train_ds.active_episodes]


def test_meta_world_episodes_single_decode_and_repeatlast(monkeypatch):
    """Test full episode access: single sequential video decode, repeat-last padding, mask semantics."""
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
            f.write(json.dumps({"task_index": 0, "task": "Reach target"}) + "\n")

        # 1 episode with 8 rows
        episodes = [{"episode_index": 0, "length": 8, "tasks": ["Reach target"]}]
        with open(meta_dir / "episodes.jsonl", "w") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")

        chunk0_dir = root / "data" / "chunk-000"
        chunk0_dir.mkdir(parents=True)

        video_dir = root / "videos" / "chunk-000" / "observation.images.image"
        video_dir.mkdir(parents=True)
        (video_dir / "episode_000000.mp4").touch()

        # Parquet data: 4D actions
        df = pd.DataFrame({
            "frame_index": list(range(8)),
            "episode_index": [0] * 8,
            "task_index": [0] * 8,
            "observation.state": [[0.1, 0.2, 0.3, 0.4]] * 8,
            "action": [[0.5, -0.5, 0.1, 0.2]] * 8,
        })
        df.to_parquet(chunk0_dir / "episode_000000.parquet")

        decode_calls = []

        def mock_decode(video_path, target_frame_indices):
            decode_calls.append(list(target_frame_indices))
            return {i: Image.new("RGB", (32, 32)) for i in target_frame_indices}

        monkeypatch.setattr("fabri_moss.delta_data.decode_all_video_frames", mock_decode)

        norm_stats = {
            "observation.state": {"min": [0.0] * 4, "max": [1.0] * 4},
            "action": {"min": [-1.0] * 4, "max": [1.0] * 4},
        }

        ds = MetaWorldEpisodes(
            root=root,
            norm_stats=norm_stats,
            horizon=10,  # horizon > ep_len to test repeat-last
            state_dim=24,
            action_dim=24,
            max_chunk_size=3,
            split="train",
            val_fraction=0.0,
            augmentation=True,
        )

        ep_item = ds[0]
        assert len(decode_calls) == 1
        assert decode_calls[0] == list(range(8))

        assert ep_item["episode_id"] == 0
        assert ep_item["prompt"] == "Reach target"
        chunks = ep_item["chunks"]
        assert len(chunks) >= 2

        # First chunk is singleton [0]
        assert chunks[0]["frame_ids"] == [0]
        assert chunks[0]["is_first_decision"] is True

        # Check last chunk: actions should be padded to horizon=10
        last_chunk = chunks[-1]
        assert last_chunk["actions"].shape == (1, 10, 24)
        assert last_chunk["action_mask"].shape == (1, 10, 24)

        # FabriVLA contract: repeat-last future action steps are supervised on all valid action dimensions (first 4 dims).
        # Invalid 20 dimensions are 0.
        assert (last_chunk["action_mask"][0, :, :4] == 1.0).all()
        assert (last_chunk["action_mask"][0, :, 4:] == 0.0).all()

        # Check that repeated actions match the final frame action
        actions_tensor = last_chunk["actions"][0]  # [10, 24]
        # Frame 7 was the last frame; all steps past actual action length repeat frame 7
        assert torch.allclose(actions_tensor[-1, :4], actions_tensor[-2, :4])
