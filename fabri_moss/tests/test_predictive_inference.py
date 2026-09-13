"""Unit and integration tests for fabri_moss.predictive_inference.

Tests cover:
- snapshot_checkpoint: hardlink / exclusive copy, rejection on dest exists, immutable under source replacement.
- load_predictive_inference for both adapter v1 and joint v1 formats.
- Provenance checks: checkpoint sha256 mismatch rejection, parent sha256 mismatch rejection.
- Temporal contract and RoPE-only assertion: rejecting non-compact or text_timestamps.
- Rejection on missing or malformed writer, invalid shapes, non-finite values.
- Online teacher omission: teacher/optimizer/scheduler never instantiated or loaded into inference policy.
- PredictiveEpisodeHistory: strictly increasing frame_id, non-decreasing time, deepcopies/clones,
  sample_for_decision and commit_decision lifecycle, no future leakage.
- Real policy integration: running predict_actions with PredictiveEpisodeHistory over 3 decisions,
  observing CausalMemoryWriter activation via hooks.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
import types
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

from fabri_moss.async_pipeline import Observation
from fabri_moss.compact_protocol import get_compact_protocol_contract
from fabri_moss.predictive_inference import (
    PredictiveEpisodeHistory,
    load_predictive_inference,
    snapshot_checkpoint,
)
from fabri_moss.predictive_memory import CausalMemoryWriter, WriterConfig
from fabri_moss.predictive_policy import FutureLatentHead, PredictiveMemoryPolicy
from fabri_moss.runtime import compute_file_sha256
from fabri_moss.tests.test_native_training import make_tiny_training_policy


def _make_toy_policy(shallow_layer: int = 1) -> PredictiveMemoryPolicy:
    """Create a tiny PredictiveMemoryPolicy with toy dimensions."""
    base_seq = make_tiny_training_policy(
        shallow_layer=shallow_layer,
        gradient_checkpointing=False,
    )
    base_policy = base_seq.policy

    # Ensure ViT and mlp1 layout
    model = base_policy.embedder.model
    if "projector" in model._modules:
        proj = model._modules.pop("projector")
        model._modules["mlp1"] = proj
    if hasattr(model, "projector"):
        delattr(model, "projector")

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.mlp1(self.vision_model.encoder(pixel_values))

    model.extract_feature = types.MethodType(extract_feature, model)

    writer_cfg = WriterConfig(
        input_dim=64,
        hidden_dim=32,
        num_heads=2,
        num_layers=1,
        grid=1,
        intermediate_grid=1,
        recent_frames=2,
        consolidate_every=2,
        tbptt_decisions=2,
    )

    policy = PredictiveMemoryPolicy(
        policy=base_policy,
        writer_config=writer_cfg,
        shallow_layer=shallow_layer,
        gradient_checkpointing=False,
    )
    return policy


def _create_toy_checkpoints(
    tmp_path: Path,
) -> Tuple[Path, Path, PredictiveMemoryPolicy, Dict[str, Any]]:
    """Generate mock parent, adapter, and joint checkpoint files."""
    policy = _make_toy_policy()
    with torch.no_grad():
        policy.writer.out_proj.weight.fill_(0.013)
    writer_cfg = policy.writer_config
    tc = {
        "temporal": get_compact_protocol_contract()["temporal"],
        "text_timestamps": False,
        "shallow_layer": 1,
        "writer_config": dataclasses.asdict(writer_cfg),
    }

    norm_stats = {
        "observation.state": {
            "min": torch.zeros(7),
            "max": torch.ones(7),
        },
        "action": {
            "min": torch.zeros(4),
            "max": torch.ones(4),
        },
    }

    # 1. Parent checkpoint
    parent_path = tmp_path / "parent.pt"
    parent_payload = {
        "model": policy.policy.state_dict(),
        "config": {"horizon": 2, "state_dim": 7, "action_dim": 4, "image_size": 448},
        "norm_stats": norm_stats,
        "step": 100,
    }
    torch.save(parent_payload, parent_path)
    parent_sha = compute_file_sha256(parent_path)
    tc.update(parent_path=str(parent_path), parent_sha256=parent_sha)

    # 2. Adapter checkpoint
    adapter_path = tmp_path / "adapter.pt"
    adapter_payload = {
        "format": "predictive_memory_adapter_v1",
        "parent_path": str(parent_path),
        "parent_sha256": parent_sha,
        "parent_global_step": 100,
        "update": 10,
        "global_step": 110,
        "writer": {k: v.clone() for k, v in policy.writer.state_dict().items()},
        "future_head": {k: v.clone() for k, v in policy.future_head.state_dict().items()},
        "writer_config": dataclasses.asdict(writer_cfg),
        "training_contract": tc,
    }
    torch.save(adapter_payload, adapter_path)

    # 3. Joint checkpoint
    joint_path = tmp_path / "joint.pt"
    joint_payload = {
        "format": "predictive_memory_joint_v1",
        "model": policy.policy.state_dict(),
        "config": {"horizon": 2, "state_dim": 7, "action_dim": 4, "image_size": 448},
        "norm_stats": norm_stats,
        "writer": {k: v.clone() for k, v in policy.writer.state_dict().items()},
        "future_head": {k: v.clone() for k, v in policy.future_head.state_dict().items()},
        "teacher": {"mock_teacher_key": torch.ones(5)},
        "optimizer": {"mock_opt": 1},
        "scheduler": {"mock_sched": 1},
        "update": 20,
        "global_step": 120,
        "parent_global_step": 100,
        "writer_config": dataclasses.asdict(writer_cfg),
        "training_contract": tc,
    }
    torch.save(joint_payload, joint_path)

    return parent_path, adapter_path, joint_path, policy, norm_stats, tc


def test_snapshot_checkpoint_behavior(tmp_path: Path):
    """Test snapshot creation, hardlink/copy, and race/tamper resistance."""
    src = tmp_path / "source.pt"
    src.write_bytes(b"initial_data_12345")
    src_sha = compute_file_sha256(src)

    dst = tmp_path / "snap" / "checkpoint.pt"
    res = snapshot_checkpoint(src, dst)
    assert res == dst
    assert dst.exists()
    assert compute_file_sha256(dst) == src_sha

    # Destination already exists must raise FileExistsError
    with pytest.raises(FileExistsError):
        snapshot_checkpoint(src, dst)

    # Atomic replace of source (simulating training save_checkpoint update_last: tmp link + os.replace)
    # The existing snapshot must remain immutable pointing to initial inode / contents
    tmp_replacement = tmp_path / "replacement.pt"
    tmp_replacement.write_bytes(b"tampered_data_67890")
    tmp_replacement.replace(src)
    assert compute_file_sha256(dst) == src_sha


def test_load_predictive_inference_adapter_and_joint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test successful loading of both adapter_v1 and joint_v1 checkpoints."""
    parent_path, adapter_path, joint_path, toy_policy, expected_norm, tc = _create_toy_checkpoints(tmp_path)

    # Monkeypatch load_native_checkpoint to return a copy of toy_policy.policy
    def mock_load_native(*args, **kwargs):
        assert kwargs["trainable"] is True
        assert Path(kwargs["checkpoint_path"]).parent in (tmp_path / "snap1", tmp_path / "snap2")
        p_copy = copy.deepcopy(toy_policy.policy)
        stored = torch.load(kwargs["checkpoint_path"], weights_only=False, mmap=True)
        p_copy.load_state_dict(stored["model"], strict=True)
        return p_copy, stored["config"], stored["norm_stats"], {"backend": "cpu"}

    monkeypatch.setattr("fabri_moss.predictive_inference.load_native_checkpoint", mock_load_native)

    # 1. Test adapter format load
    snap_dir1 = tmp_path / "snap1"
    policy1, norm1, meta1 = load_predictive_inference(
        adapter_path,
        snapshot_dir=snap_dir1,
        device="cpu",
    )
    assert isinstance(policy1, PredictiveMemoryPolicy)
    assert not policy1.policy.training
    assert not policy1.writer.training
    assert not policy1.future_head.training
    assert meta1["format"] == "predictive_memory_adapter_v1"
    assert meta1["teacher_instantiated"] is False
    assert meta1["writer_params_loaded"] > 0
    assert meta1["future_head_params_loaded"] > 0
    for name in expected_norm:
        for key in ("min", "max"):
            torch.testing.assert_close(norm1[name][key], expected_norm[name][key], rtol=0, atol=0)
    for name, tensor in toy_policy.writer.state_dict().items():
        assert torch.equal(policy1.writer.state_dict()[name], tensor)
    assert not any(p.requires_grad for p in policy1.parameters())

    # 2. Test joint format load
    snap_dir2 = tmp_path / "snap2"
    policy2, norm2, meta2 = load_predictive_inference(
        joint_path,
        snapshot_dir=snap_dir2,
        device="cpu",
    )
    assert isinstance(policy2, PredictiveMemoryPolicy)
    assert meta2["format"] == "predictive_memory_joint_v1"
    assert meta2["teacher_instantiated"] is False
    assert "teacher" in meta2["training_only_omitted"]
    assert not hasattr(policy2, "future_teacher")
    for module in ("policy", "writer", "future_head"):
        for name, tensor in getattr(toy_policy, module).state_dict().items():
            assert torch.equal(getattr(policy2, module).state_dict()[name], tensor)


def test_load_predictive_inference_provenance_rejections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test rejection on bad sha256, bad parent sha256, bad format, or non-RoPE temporal contract."""
    parent_path, adapter_path, joint_path, toy_policy, expected_norm, tc = _create_toy_checkpoints(tmp_path)

    def mock_load_native(*args, **kwargs):
        p_copy = copy.deepcopy(toy_policy.policy)
        return p_copy, {}, expected_norm, {}

    monkeypatch.setattr("fabri_moss.predictive_inference.load_native_checkpoint", mock_load_native)

    # Bad expected_sha256
    with pytest.raises(ValueError, match="Checkpoint SHA256 mismatch"):
        load_predictive_inference(
            adapter_path,
            snapshot_dir=tmp_path / "snap_err1",
            device="cpu",
            expected_sha256="deadbeef" * 8,
        )

    # Tampered parent sha256 in adapter
    bad_adapter = tmp_path / "bad_adapter.pt"
    data = torch.load(adapter_path, weights_only=False)
    data["parent_sha256"] = "0000" * 16
    torch.save(data, bad_adapter)
    with pytest.raises(ValueError, match="Parent checkpoint SHA256 mismatch"):
        load_predictive_inference(bad_adapter, snapshot_dir=tmp_path / "snap_err2", device="cpu")

    # Unrecognized format
    bad_fmt = tmp_path / "bad_fmt.pt"
    data = torch.load(adapter_path, weights_only=False)
    data["format"] = "unknown_v9"
    torch.save(data, bad_fmt)
    with pytest.raises(ValueError, match="Unrecognized or missing checkpoint format"):
        load_predictive_inference(bad_fmt, snapshot_dir=tmp_path / "snap_err3", device="cpu")

    # Tampered temporal contract (text_timestamps = True)
    bad_tc = tmp_path / "bad_tc.pt"
    data = torch.load(adapter_path, weights_only=False)
    data["training_contract"]["text_timestamps"] = True
    torch.save(data, bad_tc)
    with pytest.raises(ValueError, match="text_timestamps must be False"):
        load_predictive_inference(bad_tc, snapshot_dir=tmp_path / "snap_err4", device="cpu")

    # Missing writer
    missing_writer = tmp_path / "missing_writer.pt"
    data = torch.load(adapter_path, weights_only=False)
    del data["writer"]
    torch.save(data, missing_writer)
    with pytest.raises(KeyError, match="missing required 'writer'"):
        load_predictive_inference(missing_writer, snapshot_dir=tmp_path / "snap_err5", device="cpu")


def test_predictive_episode_history_lifecycle():
    """Test PredictiveEpisodeHistory append, deepcopy, sampling, and committing."""
    history = PredictiveEpisodeHistory("pick up block")
    assert history.frame_ids == ()
    assert history.observation_times == ()
    assert history.decision_indices == ()

    img0 = np.zeros((448, 448, 3), dtype=np.uint8)
    st0 = torch.zeros(1, 7)
    sm0 = torch.ones(1, 7, dtype=torch.bool)
    am0 = torch.ones(1, 2, 4, dtype=torch.bool)

    obs0 = Observation(
        frame_id=0,
        capture_time=0.0,
        observation_time=0.0,
        images=[img0],
        state=st0,
        state_mask=sm0,
        action_mask=am0,
    )
    history.append(obs0)
    assert history.frame_ids == (0,)
    assert history.observation_times == (0.0,)

    # Mutating external img0 / st0 should not corrupt history
    img0[0, 0, 0] = 255
    st0[0, 0] = 99.0
    s_test = history.sample_for_decision("cpu")
    assert s_test["images_window"][0][0][0, 0, 0] == 0
    assert s_test["state"][0, 0] == 0.0

    # Sampling decision at step 0
    sample0 = history.sample_for_decision("cpu")
    assert sample0["target_indices"] == [0]
    assert sample0["decision_indices"] == [0]
    assert sample0["memory_replay"] is True
    assert sample0["state"].shape == (1, 7)

    # Committing decision
    history.commit_decision()
    assert history.decision_indices == (0,)

    # Sampling again without new observation must fail (duplicate decision)
    with pytest.raises(RuntimeError, match="already committed index 0"):
        history.sample_for_decision("cpu")

    # Non-monotonic frame_id rejection: append frame_id 0 again
    obs_bad_fid = Observation(
        frame_id=0,
        capture_time=0.1,
        observation_time=0.1,
        images=[img0],
        state=st0,
        state_mask=sm0,
        action_mask=am0,
    )
    with pytest.raises(ValueError, match="must be strictly greater"):
        history.append(obs_bad_fid)

    # Non-monotonic time rejection: append frame 1 at 0.5s, then frame 2 at 0.2s
    history.append(
        Observation(
            frame_id=1,
            capture_time=0.5,
            observation_time=0.5,
            images=[img0],
            state=st0,
            state_mask=sm0,
            action_mask=am0,
        )
    )
    obs_backwards = Observation(
        frame_id=2,
        capture_time=0.2,
        observation_time=0.2,
        images=[img0],
        state=st0,
        state_mask=sm0,
        action_mask=am0,
    )
    with pytest.raises(ValueError, match="is backwards relative to last time"):
        history.append(obs_backwards)

    # Add frame 2 at 0.6s
    history.append(
        Observation(
            frame_id=2,
            capture_time=0.6,
            observation_time=0.6,
            images=[np.zeros((448, 448, 3), dtype=np.uint8)],
            state=torch.zeros(1, 7),
            state_mask=sm0,
            action_mask=am0,
        )
    )

    # Sample decision at frame 2
    sample2 = history.sample_for_decision("cpu")
    assert sample2["target_indices"] == [2]
    assert sample2["decision_indices"] == [0, 2]
    assert len(sample2["images_window"]) == 3
    history.commit_decision()
    assert history.decision_indices == (0, 2)

    # Reset clears history
    history.reset("new task")
    assert history.frame_ids == ()
    assert history.decision_indices == ()


@pytest.mark.parametrize("case", ["missing_text", "bad_shape", "nonfinite", "writer_config"])
def test_loader_rejects_incomplete_or_mismatched_modules(tmp_path, monkeypatch, case):
    _, adapter, _, toy, norm, _ = _create_toy_checkpoints(tmp_path)
    monkeypatch.setattr("fabri_moss.predictive_inference.load_native_checkpoint", lambda **kw: (copy.deepcopy(toy.policy), {}, norm, {}))
    data = torch.load(adapter, weights_only=False)
    if case == "missing_text":
        del data["training_contract"]["text_timestamps"]
    elif case == "bad_shape":
        data["writer"]["out_proj.weight"] = torch.zeros(1, 1)
    elif case == "nonfinite":
        data["future_head"]["learned_queries"].fill_(float("nan"))
    else:
        data["writer_config"] = dict(data["writer_config"], hidden_dim=64)
    bad = tmp_path / "bad.pt"
    torch.save(data, bad)
    with pytest.raises((ValueError, RuntimeError)):
        load_predictive_inference(bad, snapshot_dir=tmp_path / "snapshot", device="cpu")


def test_history_requires_physical_time_and_isolates_samples():
    history = PredictiveEpisodeHistory("task")
    kwargs = dict(frame_id=0, capture_time=500.0, images=[Image.new("RGB", (8, 8))],
                  state=torch.zeros(1, 4), state_mask=torch.ones(1, 4, dtype=torch.bool),
                  action_mask=torch.ones(1, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="Observation time"):
        history.append(Observation(**kwargs))
    history.append(Observation(**kwargs, observation_time=0.0))
    sample = history.sample_for_decision("cpu")
    sample["state"].fill_(7)
    sample["state_mask"].fill_(False)
    again = history.sample_for_decision("cpu")
    assert torch.count_nonzero(again["state"]) == 0
    assert again["state_mask"].all()
    history.append(Observation(**dict(kwargs, frame_id=1), observation_time=0.1))
    with pytest.raises(RuntimeError, match="No pending"):
        history.commit_decision()


def test_closed_loop_writer_activation_over_3_plans():
    """Run PredictiveMemoryPolicy.predict_actions over 3 plans and verify writer hook count."""
    policy = _make_toy_policy(shallow_layer=1)
    policy.eval()
    def forbidden_future(*args, **kwargs):
        raise AssertionError("Future head must not be used by action inference")
    policy.future_head.forward = forbidden_future

    writer_call_count = 0

    def writer_hook(module, args, kwargs):
        nonlocal writer_call_count
        writer_call_count += 1

    hook_handle = policy.writer.register_forward_hook(writer_hook)

    history = PredictiveEpisodeHistory("open microwave")
    sm0 = torch.ones(1, 7, dtype=torch.bool)
    am0 = torch.ones(1, 2, 4, dtype=torch.bool)

    try:
        # Plan 1: frame 0
        history.append(
            Observation(
                frame_id=0,
                capture_time=0.0,
                observation_time=0.0,
                images=[np.zeros((448, 448, 3), dtype=np.uint8)],
                state=torch.zeros(1, 7),
                state_mask=sm0,
                action_mask=am0,
            )
        )
        sample1 = history.sample_for_decision("cpu")
        act1 = policy.predict_actions(sample1)
        assert act1.ndim in (2, 3)
        history.commit_decision()

        # Plan 2: frames 1, 2
        for fid, t in [(1, 0.1), (2, 0.2)]:
            history.append(
                Observation(
                    frame_id=fid,
                    capture_time=t,
                    observation_time=t,
                    images=[np.zeros((448, 448, 3), dtype=np.uint8)],
                    state=torch.zeros(1, 7),
                    state_mask=sm0,
                    action_mask=am0,
                )
            )
        sample2 = history.sample_for_decision("cpu")
        act2 = policy.predict_actions(sample2)
        assert act2.ndim in (2, 3)
        history.commit_decision()

        # Plan 3: frames 3, 4
        for fid, t in [(3, 0.3), (4, 0.4)]:
            history.append(
                Observation(
                    frame_id=fid,
                    capture_time=t,
                    observation_time=t,
                    images=[np.zeros((448, 448, 3), dtype=np.uint8)],
                    state=torch.zeros(1, 7),
                    state_mask=sm0,
                    action_mask=am0,
                )
            )
        sample3 = history.sample_for_decision("cpu")
        act3 = policy.predict_actions(sample3)
        assert act3.ndim in (2, 3)
        history.commit_decision()

        # Consolidate_every=2, so by plan 3 with 5 frames and 3 decisions, writer must have been called
        assert writer_call_count > 0, f"Expected writer to be called, got count {writer_call_count}"

    finally:
        hook_handle.remove()
