"""Unit tests for verify_delta.py and bounded delta verification functions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn

from fabri_moss.async_pipeline import Observation
from fabri_moss.core import MossConfig, MossInternVL
from fabri_moss.tests.test_core import TinyFabriVLAPolicy
from fabri_moss.verify_delta import (
    DEFAULT_VERIFY_SEED,
    build_parser,
    compute_interval_overlap,
    compute_tensor_diff,
    convert_samples_to_observations,
    count_wrapped_calls,
    find_episode_samples,
    run_verification,
    verify_async_delta_pipeline,
    verify_backward_probe,
    verify_direct_batches,
)


def make_test_obs(frame_id: int, capture_time: float = 0.0) -> Observation:
    return Observation(
        frame_id=frame_id,
        capture_time=capture_time,
        images=[torch.zeros((1, 3, 16, 16))],
        state=torch.full((1, 24), float(frame_id), dtype=torch.float32),
        state_mask=torch.ones((1, 24), dtype=torch.bool),
        action_mask=torch.ones((1, 24), dtype=torch.bool),
    )


class TinyActionHeadWithSample(nn.Module):
    def __init__(self, hidden_size: int = 128, horizon: int = 50, action_dim: int = 24):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.proj = nn.Linear(hidden_size, horizon * 4)

    def sample(
        self,
        deep_tokens: torch.Tensor,
        state: torch.Tensor,
        state_mask: torch.Tensor,
        action_mask: torch.Tensor,
        shallow_tokens: torch.Tensor,
    ) -> torch.Tensor:
        B = deep_tokens.shape[0]
        rep = deep_tokens.mean(dim=1) + shallow_tokens.mean(dim=1)
        # Produce 4 active dimensions and pad remainder to 24 with zeros
        out_active = self.proj(rep).view(B, self.horizon, 4)
        out_padded = torch.zeros(
            (B, self.horizon, self.action_dim),
            dtype=out_active.dtype,
            device=out_active.device,
        )
        out_padded[:, :, :4] = out_active
        return out_padded


class FakeDatasetWithAnchors:
    def __init__(self, anchors: List[Tuple[Dict[str, Any], int]], samples_map: Dict[Tuple[int, int], Dict[str, Any]]):
        self.anchors = anchors
        self.samples_map = samples_map
        self.getitem_call_indices: List[int] = []

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self.getitem_call_indices.append(idx)
        anchor = self.anchors[idx]
        ep_dict, row = anchor
        ep_id = ep_dict["episode_index"]
        return self.samples_map[(ep_id, row)]


def test_cli_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([
        "--data-root", "/tmp/fake_data",
        "--output-dir", "/tmp/fake_output",
    ])
    assert args.data_root == "/tmp/fake_data"
    assert args.output_dir == "/tmp/fake_output"
    assert args.device == "cuda:0"
    assert args.flow_steps == 50
    assert args.threads == 4
    assert args.episode_id == 148
    assert args.seed == DEFAULT_VERIFY_SEED
    assert args.adapter_stage == "bridge"


def test_compute_interval_overlap():
    assert compute_interval_overlap((0.0, 1.0), (0.5, 1.5)) == pytest.approx(0.5)
    assert compute_interval_overlap((0.0, 1.0), (1.5, 2.5)) == 0.0
    assert compute_interval_overlap((1.0, 3.0), (0.5, 2.0)) == pytest.approx(1.0)
    assert compute_interval_overlap((1.0, 2.0), (0.0, 3.0)) == pytest.approx(1.0)


def test_compute_tensor_diff():
    a = torch.ones(2, 4)
    b = torch.ones(2, 4)
    diff = compute_tensor_diff(a, b)
    assert diff["max_diff"] == 0.0
    assert diff["allclose_1e5"] is True

    c = a + 5e-5
    diff2 = compute_tensor_diff(a, c)
    assert diff2["max_diff"] == pytest.approx(5e-5, rel=1e-2)
    assert diff2["allclose_1e5"] is False
    assert diff2["allclose_1e4"] is True


def test_find_episode_samples_anchors_strict_and_no_fallback():
    # Construct a dataset with 2 episodes: ep 100 has 50 frames, ep 148 has 15 frames, ep 200 has 50 frames
    anchors: List[Tuple[Dict[str, Any], int]] = []
    samples_map: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for ep in [100, 148, 200]:
        num_frames = 15 if ep == 148 else 50
        for r in range(num_frames):
            anchors.append(({"episode_index": ep}, r))
            samples_map[(ep, r)] = {
                "episode_id": ep,
                "frame_ids": [r],
                "images_window": [[torch.zeros(3, 16, 16)]],
                "state": torch.zeros(24),
                "state_mask": torch.ones(24),
                "action_mask": torch.ones(24),
                "prompt": f"prompt for ep {ep}",
            }

    fake_dataset = FakeDatasetWithAnchors(anchors, samples_map)

    # 1. Target 148, exactly 10 frames
    ep_id, samples = find_episode_samples(fake_dataset, target_episode_id=148, required_frames=10)
    assert ep_id == 148
    assert len(samples) == 10
    # Assert EXACTLY 10 getitem calls
    assert len(fake_dataset.getitem_call_indices) == 10
    # Assert NO other episode samples were ever accessed
    for called_idx in fake_dataset.getitem_call_indices:
        assert fake_dataset.anchors[called_idx][0]["episode_index"] == 148

    # 2. Target non-existent episode raises ValueError and does not fallback
    fake_dataset_2 = FakeDatasetWithAnchors(anchors, samples_map)
    with pytest.raises(ValueError, match="not found or has insufficient frames"):
        find_episode_samples(fake_dataset_2, target_episode_id=999, required_frames=10)
    assert len(fake_dataset_2.getitem_call_indices) == 0

    # 3. Target episode with fewer than 10 frames raises ValueError and does not fallback
    insufficient_anchors = [({"episode_index": 50}, r) for r in range(4)]
    insufficient_samples = {
        (50, r): {
            "episode_id": 50,
            "frame_ids": [r],
            "images_window": [[torch.zeros(3, 16, 16)]],
            "state": torch.zeros(24),
            "state_mask": torch.ones(24),
            "action_mask": torch.ones(24),
            "prompt": "short ep",
        }
        for r in range(4)
    }
    fake_dataset_3 = FakeDatasetWithAnchors(insufficient_anchors, insufficient_samples)
    with pytest.raises(ValueError, match="not found or has insufficient frames"):
        find_episode_samples(fake_dataset_3, target_episode_id=50, required_frames=10)
    assert len(fake_dataset_3.getitem_call_indices) == 0


def test_convert_samples_to_observations():
    samples = [
        {
            "episode_id": 148,
            "frame_ids": [i],
            "images_window": [[torch.zeros(3, 16, 16)]],
            "state": torch.zeros(24),
            "state_mask": torch.ones(24),
            "action_mask": torch.ones(24),
            "prompt": "pick cube",
        }
        for i in range(10)
    ]
    obs_list = convert_samples_to_observations(samples)
    assert len(obs_list) == 10
    assert isinstance(obs_list[0], Observation)
    assert obs_list[0].frame_id == 0
    assert obs_list[9].frame_id == 9


def test_count_wrapped_calls_exception_cleanup():
    policy = TinyFabriVLAPolicy()
    model = MossInternVL(
        policy,
        config=MossConfig(cross_layers=(1, 2), max_frames=5, memory_mode="delta", shallow_layer=2),
    )
    orig_encode = model.encode_image
    orig_project = model.project_frame

    try:
        with count_wrapped_calls(model) as counts:
            assert model.encode_image != orig_encode
            raise RuntimeError("Forced exception to test unwrapping cleanup")
    except RuntimeError:
        pass

    assert model.encode_image == orig_encode
    assert model.project_frame == orig_project


def test_direct_batches_counts_and_invariants():
    policy = TinyFabriVLAPolicy()
    config = MossConfig(
        cross_layers=(1, 2),
        max_frames=5,
        memory_mode="delta",
        shallow_layer=2,
    )
    model = MossInternVL(policy, config=config)

    for block in model.cross_blocks.values():
        with torch.no_grad():
            block.attn_gate.fill_(0.1)
            block.mlp_gate.fill_(0.1)

    observations = [make_test_obs(fid, capture_time=fid * 0.05) for fid in range(10)]
    prompt = "close drawer"

    direct_res = verify_direct_batches(
        model=model,
        observations=observations,
        prompt=prompt,
        synthetic_open_gate=True,
    )

    assert direct_res["overall_passed"] is True
    assert direct_res["parity_pass"] is True
    assert direct_res["s1_immutability_preserved"] is True
    assert direct_res["state_frame_count"] == 10
    assert direct_res["state_last_frame_id"] == 9
    assert direct_res["history_impact_observed"] is True
    # Test counting assertions
    assert direct_res["counts_after_initial_10"] is True
    assert direct_res["queries_did_not_increment_counts"] is True
    assert direct_res["counts_total_correct"] is True
    assert direct_res["counts"]["encode_image"] == 20
    assert direct_res["counts"]["project_frame"] == 20
    # Test state dict excludes raw s1/s2 and has only deep/shallow
    assert "s1" not in direct_res["batch1_outputs"]
    assert "s2" not in direct_res["batch2_outputs"]
    assert "deep" in direct_res["batch1_outputs"]
    assert "shallow" in direct_res["batch1_outputs"]


def test_async_delta_pipeline_reset_and_parity():
    policy = TinyFabriVLAPolicy()
    policy.action_head = TinyActionHeadWithSample(hidden_size=128, horizon=50, action_dim=24)

    config = MossConfig(
        cross_layers=(1, 2),
        max_frames=5,
        memory_mode="delta",
        shallow_layer=2,
    )
    model = MossInternVL(policy, config=config)

    for block in model.cross_blocks.values():
        with torch.no_grad():
            block.attn_gate.fill_(0.1)
            block.mlp_gate.fill_(0.1)

    observations = [make_test_obs(fid, capture_time=fid * 0.05) for fid in range(10)]
    prompt = "close drawer"

    direct_res = verify_direct_batches(
        model=model,
        observations=observations,
        prompt=prompt,
        synthetic_open_gate=True,
    )

    async_res = verify_async_delta_pipeline(
        model=model,
        observations=observations,
        prompt=prompt,
        direct_batch2_outputs=direct_res["batch2_outputs"],
        wait_timeout=10.0,
    )

    assert async_res["overall_passed"] is True
    assert async_res["p1_cutoff_correct"] is True
    assert async_res["p2_cutoff_correct"] is True
    assert async_res["batches_disjoint"] is True
    assert async_res["memory_invariants_pass"] is True
    assert async_res["reset_verified"] is True
    assert async_res["stats_after_reset"]["memory_frame_count"] == 0
    assert async_res["stats_after_reset"]["memory_bytes"] == 0
    assert async_res["stats_after_close"]["memory_frame_count"] == 0
    assert async_res["stats_after_close"]["memory_bytes"] == 0
    assert async_res["p1_next_mem_none"] is True
    assert async_res["p2_next_mem_none"] is True
    assert async_res["async_parity_pass"] is True
    assert async_res["actions_finite"] is True
    assert async_res["actions_shape_valid"] is True
    assert async_res["actions_padded_zero"] is True


def test_backward_probe_per_layer_requirements():
    policy = TinyFabriVLAPolicy()
    config = MossConfig(
        cross_layers=(1, 2),
        max_frames=5,
        memory_mode="delta",
        shallow_layer=2,
    )
    model = MossInternVL(policy, config=config)

    for block in model.cross_blocks.values():
        with torch.no_grad():
            block.attn_gate.fill_(0.1)
            block.mlp_gate.fill_(0.1)

    observations = [make_test_obs(fid) for fid in range(10)]
    prompt = "close drawer"

    backward_res = verify_backward_probe(
        model=model,
        observations=observations,
        prompt=prompt,
        adapter_loaded=False,
    )

    assert backward_res["overall_passed"] is True
    assert backward_res["policy_grads_none"] is True
    assert backward_res["all_layers_valid"] is True
    assert backward_res["readout_valid"] is True
    assert "1" in backward_res["per_layer_checks"]
    assert "2" in backward_res["per_layer_checks"]
    for lay_id in ("1", "2"):
        lay_chk = backward_res["per_layer_checks"][lay_id]
        assert lay_chk["k_proj"] is True
        assert lay_chk["v_proj"] is True
        assert lay_chk["write_logits"] is True
        assert lay_chk["memory_gate"] is True

    # Adapter loaded case
    skipped_res = verify_backward_probe(
        model=model,
        observations=observations,
        prompt=prompt,
        adapter_loaded=True,
    )
    assert skipped_res["skipped"] is True
    assert "保留加载策略用于推理验收" in skipped_res["reason"]
    assert skipped_res["overall_passed"] is True


def test_run_verification_error_reporting_on_cuda_validation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / "output"
        parser = build_parser()
        # Test that specifying cpu raises ValueError and writes report.json with passed=False
        args = parser.parse_args([
            "--data-root", tmp_dir,
            "--output-dir", str(out_dir),
            "--device", "cpu",
        ])

        with pytest.raises(ValueError, match="Verification requires a CUDA device"):
            run_verification(args)

        report_file = out_dir / "report.json"
        assert report_file.exists()
        saved = json.loads(report_file.read_text(encoding="utf-8"))
        assert saved["passed"] is False
        assert saved["overall_pass"] is False
        assert "Verification requires a CUDA device" in saved["error"]


def test_run_verification_refuses_non_empty_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / "output"
        out_dir.mkdir()
        (out_dir / "existing_file.txt").write_text("hello")

        parser = build_parser()
        args = parser.parse_args([
            "--data-root", tmp_dir,
            "--output-dir", str(out_dir),
            "--device", "cuda:0",
        ])

        with pytest.raises(RuntimeError, match="exists and is not empty"):
            run_verification(args)
