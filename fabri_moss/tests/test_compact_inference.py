"""Unit and integration tests for fabri_moss.compact_inference and compact evaluation."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

from fabri_moss.compact_inference import (
    CompactInferencePolicy,
    load_compact_inference,
)
from fabri_moss.compact_memory import CompactMemoryConfig
from fabri_moss.compact_protocol import get_compact_protocol_contract
from fabri_moss.evaluate_predictive import (
    build_parser,
    run_episode,
    run_evaluation,
)
from fabri_moss.runtime import compute_file_sha256
from fabri_moss.tests.test_compact_training import _build_test_compact_policy
from fabri_moss.tests.test_evaluate_predictive import (
    _make_toy_metadata_workspace,
    FakeGymnasiumEnv,
    FakeGymWrapper,
)
from fabri_moss.tests.test_native_training import make_sample


def _create_toy_compact_checkpoint(
    tmp_path: Path,
    base_policy: nn.Module,
    shallow_layer_index: int = 1,
) -> Tuple[Path, str, Dict[str, Any], Dict[str, Any]]:
    """Create a mock native compact memory checkpoint file."""
    raw_config = {
        "horizon": 2,
        "state_dim": 24,
        "action_dim": 4,
        "image_size": 448,
        "shallow_layer_index": shallow_layer_index,
    }
    norm_stats = {
        "observation.state": {
            "min": torch.zeros(24),
            "max": torch.ones(24),
        },
        "action": {
            "min": torch.zeros(4),
            "max": torch.ones(4),
        },
    }
    tc = {
        "format": "native_compact_memory_replay_v1",
        "stream_protocol": get_compact_protocol_contract(),
        "use_timestamps": True,
    }
    ckpt_path = tmp_path / "compact_checkpoint.pt"
    payload = {
        "format": "native_compact_memory_replay_v1",
        "model": base_policy.state_dict(),
        "config": raw_config,
        "norm_stats": norm_stats,
        "training_contract": tc,
        "global_step": 14741,
    }
    torch.save(payload, ckpt_path)
    sha256 = compute_file_sha256(ckpt_path)
    return ckpt_path, sha256, raw_config, norm_stats


def test_compact_inference_policy_equality_and_frozen():
    """Verify CompactInferencePolicy matches reference compact.features + head.sample, has no writer, and is frozen."""
    ref_compact, base_seq = _build_test_compact_policy(shallow_layer=1, gradient_checkpointing=False)
    base_policy = base_seq.policy

    # Ensure action head configuration attributes
    base_policy.action_head.config.horizon = 2
    base_policy.action_head.config.state_dim = 24
    base_policy.action_head.config.per_action_dim = 4
    base_policy.action_head.config.shallow_layer_index = 1

    compact_inf = CompactInferencePolicy(
        policy=base_policy,
        shallow_layer=1,
        use_timestamps=True,
        gradient_checkpointing=False,
        compact_config=ref_compact.compact_config,
    )
    compact_inf.requires_grad_(False)
    compact_inf.eval()

    # Parameter count must match underlying policy exactly
    assert len(list(compact_inf.parameters())) == len(list(base_policy.parameters()))
    for p1, p2 in zip(compact_inf.parameters(), base_policy.parameters()):
        assert p1 is p2

    # Strictly no .writer or .future_head
    assert not hasattr(compact_inf, "writer")
    assert not hasattr(compact_inf, "future_head")
    assert all(not p.requires_grad for p in compact_inf.parameters())
    assert not compact_inf.training

    # Rejection of sample without memory_replay
    sample_dense = make_sample(N=4, target_indices=[3], horizon=2, action_dim=4)
    with pytest.raises(ValueError, match="memory_replay must be True"):
        compact_inf.predict_actions(sample_dense)

    # Valid sample with memory_replay
    sample = make_sample(N=4, target_indices=[3], horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = [1, 3]

    # Verify features equivalence
    d_inf, sh_inf = compact_inf.features(sample)
    d_ref, sh_ref = ref_compact.features(sample)
    torch.testing.assert_close(d_inf, d_ref)
    torch.testing.assert_close(sh_inf, sh_ref)

    # Verify predict_actions equivalence with direct action_head.sample
    torch.manual_seed(42)
    pred_actions = compact_inf.predict_actions(sample)

    torch.manual_seed(42)
    ah_config = getattr(compact_inf.policy.action_head, "config", None)
    shallow_fusion = getattr(ah_config, "shallow_fusion", "none")
    pass_shallow = sh_ref[-1:] if shallow_fusion != "none" else None
    ref_pred = compact_inf.policy.action_head.sample(
        d_ref[-1:],
        state=sample.get("state")[-1:] if sample.get("state") is not None else None,
        state_mask=sample.get("state_mask")[-1:] if sample.get("state_mask") is not None else None,
        action_mask=sample.get("action_mask")[-1:] if sample.get("action_mask") is not None else None,
        shallow_tokens=pass_shallow,
    )
    torch.testing.assert_close(pred_actions, ref_pred)
    assert pred_actions.shape == (1, 2, 4)


def test_run_episode_compact_temporal(monkeypatch):
    """Test actual run_episode using FakeGymnasiumEnv and real CompactInferencePolicy without writer."""
    ref_compact, base_seq = _build_test_compact_policy(shallow_layer=1, gradient_checkpointing=False)
    base_policy = base_seq.policy

    base_policy.action_head.config.horizon = 2
    base_policy.action_head.config.state_dim = 24
    base_policy.action_head.config.per_action_dim = 4
    base_policy.action_head.config.shallow_layer_index = 1

    compact_policy = CompactInferencePolicy(
        policy=base_policy,
        shallow_layer=1,
        use_timestamps=True,
        gradient_checkpointing=False,
        compact_config=ref_compact.compact_config,
    )
    compact_policy.requires_grad_(False)
    compact_policy.eval()

    env = FakeGymnasiumEnv(dt=0.1)
    norm_stats = {
        "observation.state": {
            "min": np.zeros(24, dtype=np.float32),
            "max": np.ones(24, dtype=np.float32),
        },
        "action": {
            "min": np.full(4, -1.0, dtype=np.float32),
            "max": np.full(4, 1.0, dtype=np.float32),
        },
    }

    result = run_episode(
        env=env,
        model=compact_policy,
        prompt="push the block to the goal",
        norm_stats=norm_stats,
        episode_horizon=12,
        exec_horizon=2,
        observation_stride=1,
        seed=1001,
        step_seconds=0.1,
        state_dim=24,
        action_dim=4,
        image_size=448,
        policy_kind="compact-temporal",
    )

    assert result["policy_kind"] == "compact-temporal"
    assert result["memory_kind"] == "fixed_compact_temporal"
    assert result["learned_writer_enabled"] is False
    assert result["total_writer_calls"] == 0
    assert result["steps"] == 12
    assert result["executed_actions_count"] == 12
    assert result["plans_completed"] == 6  # 12 steps / 2 exec_horizon = 6 plans

    for p in result["plans"]:
        assert p["writer_calls"] == 0
        assert p["learned_writer_enabled"] is False
        assert p["memory_kind"] == "fixed_compact_temporal"
        assert p["source_step"] in (0, 2, 4, 6, 8, 10)


def test_run_episode_compact_temporal_rejects_writer():
    """Verify run_episode rejects model with writer when policy_kind='compact-temporal'."""
    ref_compact, base_seq = _build_test_compact_policy(shallow_layer=1, gradient_checkpointing=False)
    base_policy = base_seq.policy
    base_policy.action_head.config.horizon = 2
    base_policy.action_head.config.state_dim = 24
    base_policy.action_head.config.per_action_dim = 4

    compact_policy = CompactInferencePolicy(
        policy=base_policy,
        shallow_layer=1,
        use_timestamps=True,
    )
    # Add fake writer to violate compact constraint
    compact_policy.writer = nn.Identity()

    env = FakeGymnasiumEnv(dt=0.1)
    norm_stats = {}
    with pytest.raises(ValueError, match="strictly forbids learned writer"):
        run_episode(
            env=env,
            model=compact_policy,
            prompt="task",
            norm_stats=norm_stats,
            exec_horizon=2,
            policy_kind="compact-temporal",
        )


def test_load_compact_inference_success(tmp_path: Path, monkeypatch):
    """Test load_compact_inference successfully loads checkpoint, verifies FP32, eval, and metadata."""
    ref_compact, base_seq = _build_test_compact_policy(shallow_layer=1, gradient_checkpointing=False)
    base_policy = base_seq.policy
    base_policy.action_head.config.horizon = 2
    base_policy.action_head.config.state_dim = 24
    base_policy.action_head.config.per_action_dim = 4
    base_policy.action_head.config.shallow_layer_index = 1

    ckpt_path, sha256, raw_config, norm_stats = _create_toy_compact_checkpoint(
        tmp_path=tmp_path,
        base_policy=base_policy,
        shallow_layer_index=1,
    )

    snap_dir = tmp_path / "snapshots"

    # Monkeypatch load_native_checkpoint at compact_inference boundary
    def fake_load_native(*args, **kwargs):
        copied_policy = copy.deepcopy(base_policy)
        return copied_policy, raw_config, norm_stats, {"backend": "test_eager"}

    monkeypatch.setattr("fabri_moss.compact_inference.load_native_checkpoint", fake_load_native)

    model, loaded_norm, metadata = load_compact_inference(
        checkpoint_path=ckpt_path,
        snapshot_dir=snap_dir,
        expected_sha256=sha256,
        device="cpu",
    )

    assert isinstance(model, CompactInferencePolicy)
    assert not hasattr(model, "writer")
    assert not hasattr(model, "future_head")
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    assert all(p.dtype == torch.float32 for p in model.parameters())

    assert metadata["format"] == "native_compact_memory_replay_v1"
    assert metadata["checkpoint_sha256"] == sha256
    assert metadata["global_step"] == 14741
    assert metadata["update"] is None
    assert metadata["parent"] is None
    assert metadata["memory_kind"] == "fixed_compact_temporal"
    assert metadata["learned_writer_enabled"] is False
    assert metadata["new_trainable_params"] == 0
    assert metadata["writer_params_loaded"] == 0
    assert metadata["future_head_params_loaded"] == 0
    assert metadata["teacher_instantiated"] is False
    assert metadata["shallow_layer"] == 1
    assert metadata["precision"] == "float32"
    assert metadata["horizon"] == 2
    assert metadata["state_dim"] == 24
    assert metadata["action_dim"] == 4
    assert metadata["compact_protocol_contract"] == get_compact_protocol_contract()


def test_load_compact_inference_rejections(tmp_path: Path, monkeypatch):
    """Test load_compact_inference rejections: SHA, wrong format, predictive joint, timestamps, stream_protocol."""
    ref_compact, base_seq = _build_test_compact_policy(shallow_layer=1, gradient_checkpointing=False)
    base_policy = base_seq.policy
    ckpt_path, sha256, raw_config, norm_stats = _create_toy_compact_checkpoint(
        tmp_path=tmp_path,
        base_policy=base_policy,
        shallow_layer_index=1,
    )

    snap_dir = tmp_path / "snapshots"

    # 1. SHA256 mismatch rejection
    with pytest.raises(ValueError, match="Checkpoint SHA256 mismatch"):
        load_compact_inference(
            checkpoint_path=ckpt_path,
            snapshot_dir=snap_dir,
            expected_sha256="deadbeef" * 8,
            device="cpu",
        )

    # 2. Predictive joint format rejection before base load
    joint_ckpt = tmp_path / "joint.pt"
    payload_joint = {
        "format": "predictive_memory_joint_v1",
        "model": base_policy.state_dict(),
        "config": raw_config,
        "norm_stats": norm_stats,
        "writer": {},
        "future_head": {},
    }
    torch.save(payload_joint, joint_ckpt)
    snap_joint = tmp_path / "snap_joint"
    with pytest.raises(ValueError, match="Expected native compact checkpoint, but got predictive format"):
        load_compact_inference(
            checkpoint_path=joint_ckpt,
            snapshot_dir=snap_joint,
            device="cpu",
        )

    # 3. Top-level writer or future_head present
    writer_ckpt = tmp_path / "with_writer.pt"
    payload_writer = {
        "format": "native_compact_memory_replay_v1",
        "model": base_policy.state_dict(),
        "config": raw_config,
        "norm_stats": norm_stats,
        "training_contract": {
            "format": "native_compact_memory_replay_v1",
            "stream_protocol": get_compact_protocol_contract(),
            "use_timestamps": True,
        },
        "writer": {},
    }
    torch.save(payload_writer, writer_ckpt)
    snap_writer = tmp_path / "snap_writer"
    with pytest.raises(ValueError, match="strictly forbids learned writer modules"):
        load_compact_inference(
            checkpoint_path=writer_ckpt,
            snapshot_dir=snap_writer,
            device="cpu",
        )

    # 4. text_timestamps / use_timestamps is False in training_contract
    no_ts_ckpt = tmp_path / "no_ts.pt"
    payload_no_ts = {
        "format": "native_compact_memory_replay_v1",
        "model": base_policy.state_dict(),
        "config": raw_config,
        "norm_stats": norm_stats,
        "training_contract": {
            "format": "native_compact_memory_replay_v1",
            "stream_protocol": get_compact_protocol_contract(),
            "use_timestamps": False,
        },
    }
    torch.save(payload_no_ts, no_ts_ckpt)
    snap_no_ts = tmp_path / "snap_no_ts"
    with pytest.raises(ValueError, match="use_timestamps must be True"):
        load_compact_inference(
            checkpoint_path=no_ts_ckpt,
            snapshot_dir=snap_no_ts,
            device="cpu",
        )

    # 5. stream_protocol mismatch
    bad_proto_ckpt = tmp_path / "bad_proto.pt"
    payload_bad_proto = {
        "format": "native_compact_memory_replay_v1",
        "model": base_policy.state_dict(),
        "config": raw_config,
        "norm_stats": norm_stats,
        "training_contract": {
            "format": "native_compact_memory_replay_v1",
            "stream_protocol": {"wrong": 1},
            "use_timestamps": True,
        },
    }
    torch.save(payload_bad_proto, bad_proto_ckpt)
    snap_bad_proto = tmp_path / "snap_bad_proto"
    with pytest.raises(ValueError, match="stream_protocol mismatch"):
        load_compact_inference(
            checkpoint_path=bad_proto_ckpt,
            snapshot_dir=snap_bad_proto,
            device="cpu",
        )


def test_compact_inference_prompts_contain_frame_timestamps():
    """Verify that compact inference prompts contain 'Frame i (t=...s)' timestamps unlike predictive inference."""
    ref_compact, base_seq = _build_test_compact_policy(shallow_layer=1, gradient_checkpointing=False)
    base_policy = base_seq.policy
    base_policy.action_head.config.horizon = 2
    base_policy.action_head.config.state_dim = 24
    base_policy.action_head.config.per_action_dim = 4
    base_policy.action_head.config.shallow_layer_index = 1

    compact_inf = CompactInferencePolicy(
        policy=base_policy,
        shallow_layer=1,
        use_timestamps=True,
        gradient_checkpointing=False,
        compact_config=ref_compact.compact_config,
    )
    compact_inf.requires_grad_(False)
    compact_inf.eval()

    captured_prompts: List[str] = []
    orig_fuse = base_policy.embedder._prepare_batch_and_fuse_embeddings

    def spy_prepare_batch(prompts, *args, **kwargs):
        captured_prompts.extend(prompts)
        return orig_fuse(prompts, *args, **kwargs)

    base_policy.embedder._prepare_batch_and_fuse_embeddings = spy_prepare_batch

    sample = make_sample(N=3, target_indices=[2], horizon=2, action_dim=4)
    sample["memory_replay"] = True
    sample["decision_indices"] = [0, 2]
    sample["frame_ids"] = [10, 11, 12]
    sample["observation_times"] = [0.0, 0.1, 0.2]

    _ = compact_inf.predict_actions(sample)

    assert len(captured_prompts) > 0
    # Every captured prompt in compact inference should have explicit Frame ID and timestamp
    for p in captured_prompts:
        assert "Frame " in p
        assert "time " in p
        assert " s." in p


def test_run_evaluation_compact_temporal(tmp_path: Path, monkeypatch):
    """Test full run_evaluation with --policy-kind compact-temporal end-to-end."""
    fabri_root, _ = _make_toy_metadata_workspace(tmp_path / "fabri")
    out_dir = tmp_path / "eval_out"

    ref_compact, base_seq = _build_test_compact_policy(shallow_layer=1, gradient_checkpointing=False)
    base_policy = base_seq.policy
    base_policy.action_head.config.horizon = 2
    base_policy.action_head.config.state_dim = 24
    base_policy.action_head.config.per_action_dim = 4
    base_policy.action_head.config.shallow_layer_index = 1

    compact_policy = CompactInferencePolicy(
        policy=base_policy,
        shallow_layer=1,
        use_timestamps=True,
        compact_config=ref_compact.compact_config,
    )
    compact_policy.requires_grad_(False)
    compact_policy.eval()

    norm_stats = {
        "observation.state": {"min": np.zeros(24), "max": np.ones(24)},
        "action": {"min": np.zeros(4), "max": np.ones(4)},
    }
    ckpt_meta = {
        "format": "native_compact_memory_replay_v1",
        "image_size": 448,
        "horizon": 2,
        "state_dim": 24,
        "action_dim": 4,
    }

    # Mock load_compact_inference
    monkeypatch.setattr(
        "fabri_moss.compact_inference.load_compact_inference",
        lambda **kwargs: (compact_policy, norm_stats, ckpt_meta),
    )

    env1 = FakeGymnasiumEnv(dt=0.1)
    env2 = FakeGymnasiumEnv(dt=0.1)
    wrapper = FakeGymWrapper([env1, env2])

    parser = build_parser()
    args = parser.parse_args([
        "--checkpoint", str(tmp_path / "dummy_compact.pt"),
        "--output-dir", str(out_dir),
        "--fabri-root", str(fabri_root),
        "--tasks", "reach-v2", "push-v2",
        "--episodes", "1",
        "--episode-horizon", "2",
        "--exec-horizon", "2",
        "--device", "cpu",
        "--policy-kind", "compact-temporal",
    ])

    summary = run_evaluation(args, device="cpu", env_factory=lambda seed: wrapper)

    assert summary["status"] == "completed"
    assert summary["provenance"]["policy_kind"] == "compact-temporal"
    assert summary["provenance"]["protocol_name"] == "compact_sync_full_prefix_v1"
    assert summary["provenance"]["learned_writer_enabled"] is False
    assert summary["provenance"]["memory_kind"] == "fixed_compact_temporal"
    assert summary["counts"]["completed_episodes"] == 2

    # Check episodes.jsonl records
    episodes_file = out_dir / "episodes.jsonl"
    assert episodes_file.exists()
    with episodes_file.open("r", encoding="utf-8") as f:
        ep_lines = [json.loads(line) for line in f if line.strip()]
    assert len(ep_lines) == 2
    for ep in ep_lines:
        assert ep["policy_kind"] == "compact-temporal"
        assert ep["memory_kind"] == "fixed_compact_temporal"
        assert ep["learned_writer_enabled"] is False
        assert ep["total_writer_calls"] == 0
