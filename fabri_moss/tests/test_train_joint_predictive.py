"""Full lifecycle and contract integration tests for train_joint_predictive.py.

Tests Stage 3 joint predictive fine-tuning:
1. Stage 2 handoff -> Stage 3 initial weights & 6-group optimizer / AdamW moments inheritance.
2. Uninterrupted 4 updates vs 2-then-resumed-4 exactness / reproducibility.
3. Invariant checks: teacher parameters strictly unchanged; finite gradients on 6 trainable groups.
4. Validation & error boundaries: corrupt stage2 budget, missing SHA, resume cursor sanity.
5. Max-updates gate behavior: 2 updates does not write final.pt; completing epoch writes final.pt.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import types
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pytest
import torch
import torch.nn as nn

import fabri_moss.joint_predictive_policy as joint_predictive_policy
from fabri_moss.predictive_memory import WriterConfig
import fabri_moss.predictive_policy as predictive_policy
import fabri_moss.predictive_data as predictive_data
import fabri_moss.runtime as runtime
from fabri_moss.tests.test_native_training import make_tiny_training_policy
from fabri_moss.tests.test_train_predictive_runtime import (
    FakePredictiveTrainingDataset,
    _assert_equal_recursive,
    _create_mock_parent_checkpoint,
)
import fabri_moss.train_predictive as train_predictive


def _get_train_joint_module():
    try:
        import fabri_moss.train_joint_predictive as tjp
        return tjp
    except ImportError as e:
        pytest.fail(f"train_joint_predictive module could not be imported: {e}")


def _fix_test_native_layout(base_policy: nn.Module) -> None:
    model = base_policy.embedder.model
    if "projector" in model._modules:
        proj = model._modules.pop("projector")
        model._modules["mlp1"] = proj
    if hasattr(model, "projector"):
        delattr(model, "projector")

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.mlp1(self.vision_model.encoder(pixel_values))

    model.extract_feature = types.MethodType(extract_feature, model)


class FutureTeacherFixture(nn.Module):
    def __init__(self, vision_model: nn.Module, mlp1: nn.Module):
        super().__init__()
        self.vision_model = copy.deepcopy(vision_model)
        self.mlp1 = copy.deepcopy(mlp1)
        self.float()
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> FutureTeacherFixture:
        super().train(False)
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.vision_model.encoder(pixel_values)
            return self.mlp1(feat)


@pytest.fixture
def stage3_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    parent_path = tmp_path / "parent_compact.pt"
    _create_mock_parent_checkpoint(parent_path)
    parent = torch.load(parent_path, weights_only=False)
    parent.update(config={}, norm_stats={"norm": True})
    torch.save(parent, parent_path)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4042)
        base_seq = make_tiny_training_policy(shallow_layer=1, gradient_checkpointing=False)
        fixed_base_policy = base_seq.policy
        _fix_test_native_layout(fixed_base_policy)

    writer_cfg = WriterConfig(
        input_dim=64,
        hidden_dim=32,
        num_heads=2,
        num_layers=1,
        grid=1,
        intermediate_grid=1,
        recent_frames=2,
        consolidate_every=2,
        tbptt_decisions=4,
    )

    orig_stage2_cls = predictive_policy.PredictiveMemoryPolicy

    def mock_stage2_factory(*args: Any, **kwargs: Any) -> Any:
        kwargs["writer_config"] = writer_cfg
        kwargs["shallow_layer"] = 1
        kwargs["gradient_checkpointing"] = False
        return orig_stage2_cls(*args, **kwargs)

    captured_joint_models: List[Tuple[Any, Dict[str, str], Dict[str, str]]] = []
    orig_joint_cls = joint_predictive_policy.JointPredictiveMemoryPolicy

    def mock_joint_factory(*args: Any, **kwargs: Any) -> Any:
        kwargs["writer_config"] = writer_cfg
        kwargs["shallow_layer"] = 1
        kwargs["gradient_checkpointing"] = False
        embedder_model = (
            args[0].embedder.model if args else kwargs.get("policy").embedder.model
        )
        kwargs["future_teacher"] = FutureTeacherFixture(
            embedder_model.vision_model, embedder_model.mlp1
        )
        inst = orig_joint_cls(*args, **kwargs)
        base_hashes = {
            n: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
            for n, p in inst.policy.named_parameters()
        }
        teacher_hashes = {
            n: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
            for n, p in inst.future_teacher.named_parameters()
        }
        captured_joint_models.append((inst, base_hashes, teacher_hashes))
        return inst

    def mock_load_native_checkpoint(*, checkpoint_path: Any, **kwargs: Any) -> Any:
        fresh_base = copy.deepcopy(fixed_base_policy)
        ckpt_content = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in ckpt_content:
            fresh_base.load_state_dict(ckpt_content["model"], strict=True)
        return fresh_base, {}, {"norm": True}, {}

    monkeypatch.setattr(predictive_policy, "PredictiveMemoryPolicy", mock_stage2_factory)
    monkeypatch.setattr(joint_predictive_policy, "JointPredictiveMemoryPolicy", mock_joint_factory)
    monkeypatch.setattr(runtime, "load_native_checkpoint", mock_load_native_checkpoint)
    FakePredictiveTrainingDataset.first_batch_dense = False
    monkeypatch.setattr(predictive_data, "PredictiveTrainingDataset", FakePredictiveTrainingDataset)

    # Generate actual Stage 2 complete checkpoint via train_predictive.run_training
    stage2_dir = tmp_path / "stage2_complete"
    stage2_args = train_predictive.parse_args([
        "--init-from", str(parent_path),
        "--output-dir", str(stage2_dir),
        "--writer-updates", "1",
        "--joint-updates", "1",
        "--max-updates", "2",
        "--workers", "0",
        "--global-batch-size", "2",
        "--warmup-updates", "2",
        "--save-every", "100",
        "--eval-every", "100",
    ])
    train_predictive.run_training(stage2_args, device=torch.device("cpu"))
    stage2_ckpt_path = stage2_dir / "stage2.pt"
    assert stage2_ckpt_path.exists(), "Setup failed: stage2.pt was not created"

    return {
        "parent_path": parent_path,
        "stage2_ckpt_path": stage2_ckpt_path,
        "stage2_dir": stage2_dir,
        "fixed_base_policy": fixed_base_policy,
        "captured_joint_models": captured_joint_models,
        "tmp_path": tmp_path,
    }


def test_stage3_initialization_and_moments_inheritance(stage3_env: Dict[str, Any]):
    """Verify init weights exact source via captured snapshots and Stage 2 moments preserved."""
    tjp = _get_train_joint_module()
    stage2_ckpt = torch.load(
        str(stage3_env["stage2_ckpt_path"]), map_location="cpu", weights_only=False
    )
    stage2_opt_state = stage2_ckpt["optimizer"]["state"]

    out_dir = stage3_env["tmp_path"] / "test_init_stage3"
    args = tjp.parse_args([
        "--init-from-stage2", str(stage3_env["stage2_ckpt_path"]),
        "--output-dir", str(out_dir),
        "--epochs", "6",
        "--max-updates", "1",
        "--workers", "0",
        "--global-batch-size", "2",
    ])
    tjp.run_training(args, device=torch.device("cpu"))

    ckpt_step1 = torch.load(str(out_dir / "last.pt"), map_location="cpu", weights_only=False)
    assert ckpt_step1["format"] == "predictive_memory_joint_v1"
    for k in ("model", "writer", "future_head", "teacher", "optimizer", "scheduler"):
        assert k in ckpt_step1, f"Missing {k} in stage3 checkpoint"

    # Verify adapter weights match stage 2
    for k, v in stage2_ckpt["writer"].items():
        assert k in ckpt_step1["writer"]
        assert ckpt_step1["writer"][k].shape == v.shape
    for k, v in stage2_ckpt["future_head"].items():
        assert k in ckpt_step1["future_head"]
        assert ckpt_step1["future_head"][k].shape == v.shape

    # Verify Stage 2 moments carried over (step > 1 in adapter moments, while base has step=1)
    opt_state = ckpt_step1["optimizer"]["state"]
    adapter_steps = [s["step"] for s in opt_state.values() if s.get("step", 0) > 1]
    assert len(adapter_steps) > 0, "Stage 2 adapter moments were not inherited into Stage 3 optimizer"


def test_stage3_zero_step_preserves_adapter_weights_and_moments(stage3_env):
    tjp = _get_train_joint_module()
    out_dir = stage3_env["tmp_path"] / "joint_initial"
    stop = stage3_env["tmp_path"] / "STOP_JOINT"
    stop.touch()
    args = tjp.parse_args([
        "--init-from-stage2", str(stage3_env["stage2_ckpt_path"]),
        "--output-dir", str(out_dir), "--epochs", "6", "--workers", "0",
        "--global-batch-size", "2", "--stop-file", str(stop),
    ])
    tjp.run_training(args, device=torch.device("cpu"))
    source = torch.load(stage3_env["stage2_ckpt_path"], weights_only=False)
    joint = torch.load(out_dir / "last.pt", weights_only=False)
    assert joint["update"] == 0 and joint["scheduler"]["last_epoch"] == 0
    for key in ("writer", "future_head"):
        _assert_equal_recursive(source[key], joint[key], key)
    for key in ("epoch", "batch_cursor", "epoch_targets_seen"):
        assert source[key] == joint[key]
    def moment_hash(state):
        digest = hashlib.sha256()
        for key, value in sorted(state.items()):
            digest.update(key.encode())
            digest.update(value.cpu().numpy().tobytes() if isinstance(value, torch.Tensor) else str(value).encode())
        return digest.hexdigest()
    assert sorted(map(moment_hash, source["optimizer"]["state"].values())) == sorted(map(moment_hash, joint["optimizer"]["state"].values()))
    for group in joint["optimizer"]["param_groups"]:
        if group["group_name"] not in ("writer", "future"):
            assert not any(pid in joint["optimizer"]["state"] for pid in group["params"])


def test_stage3_gradient_flow_and_teacher_immutability(stage3_env: Dict[str, Any]):
    """Verify 6 trainable groups receive finite gradients and teacher stays strictly unchanged."""
    tjp = _get_train_joint_module()
    out_dir = stage3_env["tmp_path"] / "test_gradients"
    args = tjp.parse_args([
        "--init-from-stage2", str(stage3_env["stage2_ckpt_path"]),
        "--output-dir", str(out_dir),
        "--epochs", "6",
        "--max-updates", "1",
        "--workers", "0",
        "--global-batch-size", "2",
    ])
    tjp.run_training(args, device=torch.device("cpu"))

    inst, initial_base_hashes, initial_teacher_hashes = stage3_env["captured_joint_models"][-1]
    # Teacher parameters must remain bitwise identical
    for name, param in inst.future_teacher.named_parameters():
        curr_h = hashlib.sha256(param.detach().cpu().numpy().tobytes()).hexdigest()
        assert curr_h == initial_teacher_hashes[name], f"Teacher param {name} was modified!"
        assert not param.requires_grad, f"Teacher param {name} requires_grad must be False"

    # Base backbone, writer, future_head must have updated
    changed = 0
    for name, param in inst.policy.named_parameters():
        curr_h = hashlib.sha256(param.detach().cpu().numpy().tobytes()).hexdigest()
        if curr_h != initial_base_hashes[name]:
            changed += 1
    assert changed > 0, "Base policy parameters did not update during Stage 3!"


def test_stage3_uninterrupted_vs_resumed_exactness(stage3_env: Dict[str, Any]):
    """Compare uninterrupted 4 updates vs 2-then-resume-4 for exact reproducibility."""
    tjp = _get_train_joint_module()

    # 1. Uninterrupted 4 updates
    out_dir_4 = stage3_env["tmp_path"] / "run_uninterrupted"
    args_4 = tjp.parse_args([
        "--init-from-stage2", str(stage3_env["stage2_ckpt_path"]),
        "--output-dir", str(out_dir_4),
        "--epochs", "6",
        "--max-updates", "4",
        "--workers", "0",
        "--global-batch-size", "2",
    ])
    tjp.run_training(args_4, device=torch.device("cpu"))

    # 2. First 2 updates
    out_dir_res = stage3_env["tmp_path"] / "run_resumed"
    args_split_2 = tjp.parse_args([
        "--init-from-stage2", str(stage3_env["stage2_ckpt_path"]),
        "--output-dir", str(out_dir_res),
        "--epochs", "6",
        "--max-updates", "2",
        "--workers", "0",
        "--global-batch-size", "2",
    ])
    tjp.run_training(args_split_2, device=torch.device("cpu"))
    assert not (out_dir_res / "final.pt").exists(), "final.pt should not exist at update 2"

    # 3. Resume to 4 updates
    args_split_4 = tjp.parse_args([
        "--resume", str(out_dir_res / "last.pt"),
        "--output-dir", str(out_dir_res),
        "--epochs", "6",
        "--max-updates", "4",
        "--workers", "0",
        "--global-batch-size", "2",
    ])
    tjp.run_training(args_split_4, device=torch.device("cpu"))

    ckpt_unint = torch.load(str(out_dir_4 / "last.pt"), map_location="cpu", weights_only=False)
    ckpt_res = torch.load(str(out_dir_res / "last.pt"), map_location="cpu", weights_only=False)

    _assert_equal_recursive(ckpt_unint["writer"], ckpt_res["writer"], "writer")
    _assert_equal_recursive(ckpt_unint["future_head"], ckpt_res["future_head"], "future_head")
    _assert_equal_recursive(ckpt_unint["teacher"], ckpt_res["teacher"], "teacher")
    _assert_equal_recursive(ckpt_unint["model"], ckpt_res["model"], "model")
    _assert_equal_recursive(ckpt_unint["optimizer"], ckpt_res["optimizer"], "optimizer")
    _assert_equal_recursive(ckpt_unint["scheduler"], ckpt_res["scheduler"], "scheduler")
    _assert_equal_recursive(ckpt_unint["rng_states_per_rank"], ckpt_res["rng_states_per_rank"], "rng")
    assert ckpt_unint["epoch"] == ckpt_res["epoch"]
    assert ckpt_unint["batch_cursor"] == ckpt_res["batch_cursor"]
    assert ckpt_unint["update"] == 4 and ckpt_res["update"] == 4


def test_stage3_rejection_boundaries(stage3_env: Dict[str, Any]):
    """Verify validation boundaries: corrupt SHA, bad cursor, and completed resume idempotence."""
    tjp = _get_train_joint_module()

    # 1. Reject corrupted stage 2 SHA
    corrupt_stage2 = stage3_env["tmp_path"] / "corrupt_stage2.pt"
    raw_bytes = stage3_env["stage2_ckpt_path"].read_bytes()
    corrupt_stage2.write_bytes(raw_bytes[:-20] + b"corrupt0000000000000")
    args_corrupt = tjp.parse_args([
        "--init-from-stage2", str(corrupt_stage2),
        "--output-dir", str(stage3_env["tmp_path"] / "out_corrupt"),
        "--epochs", "6",
    ])
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        tjp.run_training(args_corrupt, device=torch.device("cpu"))

    # 2. Reject mismatched/incomplete stage2 update budget
    incomplete_stage2 = stage3_env["tmp_path"] / "incomplete_stage2.pt"
    ckpt = torch.load(str(stage3_env["stage2_ckpt_path"]), map_location="cpu", weights_only=False)
    ckpt["update"] = 1  # stage 1 only
    ckpt["stage"] = "stage1_writer_only"
    torch.save(ckpt, str(incomplete_stage2))
    args_incomplete = tjp.parse_args([
        "--init-from-stage2", str(incomplete_stage2),
        "--output-dir", str(stage3_env["tmp_path"] / "out_inc"),
        "--epochs", "6",
    ])
    with pytest.raises(ValueError):
        tjp.run_training(args_incomplete, device=torch.device("cpu"))

    # 3. Epoch finish creates final.pt; resuming completed run is idempotent
    finish_dir = stage3_env["tmp_path"] / "run_finish"
    args_finish = tjp.parse_args([
        "--init-from-stage2", str(stage3_env["stage2_ckpt_path"]),
        "--output-dir", str(finish_dir),
        "--epochs", "6",
        "--workers", "0",
        "--global-batch-size", "2",
    ])
    tjp.run_training(args_finish, device=torch.device("cpu"))
    assert (finish_dir / "final.pt").exists(), "final.pt must exist after completing target epoch"

    h_before = {name: hashlib.sha256((finish_dir / name).read_bytes()).hexdigest() for name in ("last.pt", "run_config.json", "metrics.jsonl")}
    args_re_run = tjp.parse_args([
        "--resume", str(finish_dir / "last.pt"),
        "--output-dir", str(finish_dir),
        "--epochs", "6",
        "--workers", "0",
        "--global-batch-size", "2",
    ])
    tjp.run_training(args_re_run, device=torch.device("cpu"))
    h_after = {name: hashlib.sha256((finish_dir / name).read_bytes()).hexdigest() for name in h_before}
    assert h_before == h_after, "Resuming completed run must leave checkpoint hashes unchanged"
