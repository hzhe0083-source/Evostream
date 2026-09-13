"""Unit tests for fabri_moss.verify_native_cache verification probe."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from fabri_moss.async_pipeline import Observation
from fabri_moss.native_cache import (
    NativeCacheAdapter,
    NativeCacheConfig,
    NativeEmbeddingBlock,
    NativeKVState,
)
from fabri_moss.tests.test_native_cache import make_tiny_native_adapter
from fabri_moss.verify_native_cache import (
    compute_tensor_diff,
    convert_samples_to_observations,
    count_vit_calls,
    find_episode_samples,
    rebind_block_owner,
    run_section_a_single_frame,
    run_section_b_history,
    run_section_c_eviction,
    run_section_d_async,
)


def test_compute_tensor_diff():
    """Verify compute_tensor_diff handles exact equal and divergent tensors."""
    t1 = torch.zeros((2, 10))
    t2 = torch.zeros((2, 10))
    diff = compute_tensor_diff(t1, t2)
    assert diff["max_diff"] == 0.0
    assert diff["allclose_1e5"] is True
    assert diff["allclose_1e4"] is True

    t3 = torch.full((2, 10), 2e-5)
    diff2 = compute_tensor_diff(t1, t3)
    assert diff2["max_diff"] > 1e-5
    assert diff2["allclose_1e5"] is False
    assert diff2["allclose_1e4"] is True


def test_find_episode_samples_and_mock_dataset():
    """Verify find_episode_samples indexes dataset strictly 7 times for target episode."""
    dataset = MagicMock()
    dataset.anchors = [
        ({"episode_index": 147}, 0),
        ({"episode_index": 148}, 10),
        ({"episode_index": 148}, 11),
        ({"episode_index": 148}, 12),
        ({"episode_index": 148}, 13),
        ({"episode_index": 148}, 14),
        ({"episode_index": 148}, 15),
        ({"episode_index": 148}, 16),
        ({"episode_index": 148}, 17),  # 8th frame, should not be fetched
        ({"episode_index": 149}, 20),
    ]

    def mock_getitem(idx):
        return {
            "frame_ids": torch.tensor([idx]),
            "images": [torch.zeros((3, 32, 32))],
            "state": torch.zeros(24),
            "state_mask": torch.ones(24),
            "action_mask": torch.ones((50, 24)),
            "prompt": "pick cube",
        }

    dataset.__getitem__ = MagicMock(side_effect=mock_getitem)

    ep_id, samples = find_episode_samples(dataset, target_episode_id=148, required_frames=7)
    assert ep_id == 148
    assert len(samples) == 7
    # Verify exactly 7 __getitem__ calls were made (no full dataset scanning/decoding)
    assert dataset.__getitem__.call_count == 7
    fids = [int(s["frame_ids"][-1]) for s in samples]
    assert fids == list(range(1, 8))


def test_find_episode_samples_missing_episode():
    """Verify find_episode_samples raises ValueError if episode is missing."""
    dataset = MagicMock()
    dataset.anchors = [({"episode_index": 99}, 0)]
    with pytest.raises(ValueError, match="Episode 148 not found"):
        find_episode_samples(dataset, target_episode_id=148, required_frames=7)


def test_convert_samples_to_observations():
    """Verify observations are on CPU and have strict shapes."""
    samples = [
        {
            "frame_ids": torch.tensor([i]),
            "images_window": [[torch.zeros((3, 16, 16))]],
            "state": torch.zeros(24),
            "state_mask": torch.ones(24),
            "action_mask": torch.ones((50, 24)),
            "prompt": "test prompt",
        }
        for i in range(3)
    ]
    obs_list = convert_samples_to_observations(samples)
    assert len(obs_list) == 3
    for i, obs in enumerate(obs_list):
        assert obs.frame_id == i
        assert obs.state.shape == (1, 24)
        assert obs.state_mask.shape == (1, 24)
        assert obs.action_mask.shape == (1, 24)
        assert obs.state.device.type == "cpu"
        assert obs.images[0].device.type == "cpu"


class MockActionHeadConfig:
    def __init__(self, horizon: int = 50, per_action_dim: int = 24):
        self.horizon = horizon
        self.per_action_dim = per_action_dim
        self.num_inference_timesteps = 50


class MockStrictActionHead(nn.Module):
    """Mock ActionHead that strictly matches FabriVLAActionHead contract."""
    def __init__(self, horizon: int = 50, per_action_dim: int = 24):
        super().__init__()
        self.config = MockActionHeadConfig(horizon, per_action_dim)

    def sample(
        self,
        fused_tokens: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        state_mask: Optional[torch.Tensor] = None,
        shallow_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = fused_tokens.shape[0]
        # Seeded noise deterministic output with dims >= 4 zeroed
        out = torch.rand((b, self.config.horizon, self.config.per_action_dim), dtype=torch.float32)
        out[:, :, 4:] = 0.0
        return out


def test_run_section_a_parity_and_parameter_identity():
    """Verify section A enforces 0 trainable params and catches action mismatches."""
    adapter = make_tiny_native_adapter(max_frames=4, shallow_layer=1)
    adapter.policy.action_head = MockStrictActionHead()
    prompt = "test prompt"

    def mock_get_vl_embeddings(images, image_mask, prompt, return_cls_only=False, shallow_layer_index=None):
        blk = adapter.encode_frame(images=images, frame_id=0, prompt=prompt)
        deep, shallow, _ = adapter.read_blocks([blk], prompt=prompt)
        return deep, shallow

    adapter.policy.get_vl_embeddings = mock_get_vl_embeddings

    obs = Observation(
        frame_id=0,
        capture_time=1.0,
        images=["img0"],
        state=torch.zeros((1, 24)),
        state_mask=torch.ones((1, 24)),
        action_mask=torch.ones((1, 24)),
    )
    blk0 = adapter.encode_frame(images=obs.images, frame_id=0, prompt=prompt)

    # Clean run
    res = run_section_a_single_frame(
        policy=adapter.policy,
        adapter=adapter,
        block0=blk0,
        obs0=obs,
        prompt=prompt,
        seed=42,
        flow_steps=50,
    )
    assert res["passed"] is True
    assert res["trainable_parameters"] == 0
    assert res["action_padding_max"] == 0.0

    # Test false positive gate: inject action head mismatch
    original_sample = adapter.policy.action_head.sample
    call_idx = [0]

    def mismatch_sample(*args, **kwargs):
        out = original_sample(*args, **kwargs)
        if call_idx[0] == 1:
            out = out + 1.0  # Mismatch on adapter sample
        call_idx[0] += 1
        return out

    adapter.policy.action_head.sample = mismatch_sample
    with pytest.raises(AssertionError, match="Section A action diff"):
        run_section_a_single_frame(
            policy=adapter.policy,
            adapter=adapter,
            block0=blk0,
            obs0=obs,
            prompt=prompt,
            seed=42,
            flow_steps=50,
        )


def test_run_section_b_history():
    """Verify Section B incremental caching vs fresh replay and diagnostic one-shot."""
    adapter = make_tiny_native_adapter(max_frames=16, shallow_layer=1)
    prompt = "test prompt"
    blocks = [
        adapter.encode_frame(images=[f"img_{i}"], frame_id=i, prompt=prompt)
        for i in range(7)
    ]

    res = run_section_b_history(adapter, blocks, prompt)
    assert res["passed"] is True
    assert res["fresh_replay_deep_diff"]["allclose_1e5"] is True
    assert res["fresh_replay_shallow_diff"]["allclose_1e5"] is True
    assert res["fresh_replay_max_kv_diff"] <= 1e-5
    assert res["frame_count"] == 7
    assert res["retained_blocks"] == 7


def test_run_section_c_eviction():
    """Verify Section C sliding window eviction with W=2."""
    adapter = make_tiny_native_adapter(max_frames=4, shallow_layer=1)
    prompt = "test prompt"
    blocks = [
        adapter.encode_frame(images=[f"img_{i}"], frame_id=i, prompt=prompt)
        for i in range(3)
    ]

    res = run_section_c_eviction(adapter.policy, blocks, prompt)
    assert res["passed"] is True
    assert res["rebuild_count"] == 1
    assert res["retained_frame_ids"] == [1, 2]
    assert res["rebuild_vs_fresh_deep_diff"]["allclose_1e5"] is True
    assert res["rebuild_vs_fresh_max_kv_diff"] <= 1e-5


def test_run_section_d_async():
    """Verify Section D async pipeline with 5 + 2 frames, concurrency, and reset."""
    adapter = make_tiny_native_adapter(max_frames=16, shallow_layer=1)
    adapter.policy.action_head = MockStrictActionHead()
    prompt = "test prompt"

    observations = [
        Observation(
            frame_id=i,
            capture_time=1.0 + i * 0.01,
            images=[torch.zeros((3, 16, 16))],
            state=torch.zeros((1, 24)),
            state_mask=torch.ones((1, 24)),
            action_mask=torch.ones((1, 24)),
        )
        for i in range(7)
    ]

    res = run_section_d_async(
        adapter=adapter,
        observations=observations,
        prompt=prompt,
        flow_steps=50,
    )
    assert res["passed"] is True
    assert res["total_frames_committed"] == 7
    assert res["retained_blocks_count"] == 7
    assert res["plan2_deep_diff"]["allclose_1e5"] is True
    assert res["stats_after_reset"]["memory_frame_count"] == 0
    assert res["stats_post_reset_one_frame"]["memory_frame_count"] == 1


def test_cli_guards(monkeypatch, tmp_path):
    """Verify main() CLI rejects CPU device, non-positive threads, small history, and non-empty dir."""
    from fabri_moss.verify_native_cache import main
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    # 1. Non-cuda device rejection
    monkeypatch.setattr("sys.argv", ["verify_native_cache.py", "--device", "cpu", "--checkpoint", "dummy", "--vlm", "dummy", "--fabri-root", "dummy", "--data-root", "dummy", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="strictly requires a functional CUDA device"):
        main()

    # 2. Threads <= 0 rejection
    monkeypatch.setattr("sys.argv", ["verify_native_cache.py", "--device", "cuda:0", "--threads", "0", "--checkpoint", "dummy", "--vlm", "dummy", "--fabri-root", "dummy", "--data-root", "dummy", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="threads must be > 0"):
        main()

    # 3. History frames < 7 rejection
    monkeypatch.setattr("sys.argv", ["verify_native_cache.py", "--device", "cuda:0", "--threads", "2", "--history-frames", "5", "--checkpoint", "dummy", "--vlm", "dummy", "--fabri-root", "dummy", "--data-root", "dummy", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="history_frames must be >= 7"):
        main()

    # 4. Flow steps <= 0 rejection
    monkeypatch.setattr("sys.argv", ["verify_native_cache.py", "--device", "cuda:0", "--threads", "2", "--history-frames", "16", "--flow-steps", "0", "--checkpoint", "dummy", "--vlm", "dummy", "--fabri-root", "dummy", "--data-root", "dummy", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="flow_steps must be > 0"):
        main()

    # 5. Non-empty output dir rejection
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("hello")
    monkeypatch.setattr("sys.argv", ["verify_native_cache.py", "--device", "cuda:0", "--checkpoint", "dummy", "--vlm", "dummy", "--fabri-root", "dummy", "--data-root", "dummy", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="is not empty"):
        main()


def test_main_dataset_constructor_contract():
    """Verify that MetaWorldWindows is constructed with root and norm_stats rather than data_root."""
    from fabri_moss.data import MetaWorldWindows
    import inspect
    sig = inspect.signature(MetaWorldWindows.__init__)
    assert "root" in sig.parameters
    assert "norm_stats" in sig.parameters
    assert "data_root" not in sig.parameters
    assert "window_size" not in sig.parameters
