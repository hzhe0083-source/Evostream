"""Tests for two-stage predictive trainer (train_predictive.py).

Verifies boundary schedules, phase 1/2 optimizer updates, strict checkpoint roundtrip,
contract validation and corruption rejection without module mutation, cursor validation,
evaluation loss aggregation, and CLI argument validation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from pathlib import Path
import random
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
import numpy as np
import pytest
import torch
import torch.nn as nn

from fabri_moss.predictive_memory import CausalMemoryWriter, WriterConfig
from fabri_moss.predictive_policy import FutureLatentHead
from fabri_moss.train_predictive import (
    assert_base_frozen_invariants,
    build_predictive_checkpoint_payload,
    compute_future_loss_weight,
    compute_parameter_subset_hashes,
    create_predictive_optimizer_and_scheduler,
    evaluate_predictive,
    load_predictive_checkpoint,
    parse_args,
    save_predictive_checkpoint,
    validate_cli_arguments,
    validate_predictive_cursor,
)


def _assert_tensors_equal(t1: Any, t2: Any) -> None:
    """Compare nested structures containing torch.Tensor or primitives."""
    if isinstance(t1, torch.Tensor):
        assert isinstance(t2, torch.Tensor), f"Expected Tensor, got {type(t2)}"
        assert torch.equal(t1, t2), f"Tensor mismatch: {t1} vs {t2}"
    elif isinstance(t1, dict):
        assert isinstance(t2, dict), f"Expected dict, got {type(t2)}"
        assert set(t1.keys()) == set(t2.keys()), f"Dict keys mismatch: {t1.keys()} vs {t2.keys()}"
        for k in t1:
            _assert_tensors_equal(t1[k], t2[k])
    elif isinstance(t1, (list, tuple)):
        assert isinstance(t2, (list, tuple)), f"Expected sequence, got {type(t2)}"
        assert len(t1) == len(t2), f"Length mismatch: {len(t1)} vs {len(t2)}"
        for item1, item2 in zip(t1, t2):
            _assert_tensors_equal(item1, item2)
    elif isinstance(t1, np.ndarray):
        assert isinstance(t2, np.ndarray), f"Expected ndarray, got {type(t2)}"
        assert np.array_equal(t1, t2), f"Array mismatch: {t1} vs {t2}"
    else:
        assert t1 == t2, f"Value mismatch: {t1} vs {t2}"


def make_full_training_contract(
    writer_config: WriterConfig,
    parent_path: str = "/fake/parent.pt",
    parent_sha256: str = "deadbeef12345678",
    parent_global_step: int = 14000,
    writer_updates: int = 500,
    joint_updates: int = 2000,
    lambda_future: float = 0.001,
    future_ramp_updates: int = 200,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    warmup_updates: int = 50,
    world_size: int = 1,
    seed: int = 4042,
    global_batch_size: int = 8,
) -> Dict[str, Any]:
    """Helper returning canonical predictive adapter training contract dictionary."""
    return {
        "format": "predictive_memory_adapter_v1",
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "parent_global_step": int(parent_global_step),
        "writer_updates": int(writer_updates),
        "joint_updates": int(joint_updates),
        "lambda_future": float(lambda_future),
        "future_ramp_updates": int(future_ramp_updates),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "grad_clip": float(grad_clip),
        "warmup_updates": int(warmup_updates),
        "world_size": int(world_size),
        "seed": int(seed),
        "global_batch_size": int(global_batch_size),
        "writer_config": dataclasses.asdict(writer_config),
    }


# ---------------------------------------------------------------------------
# 1. compute_future_loss_weight Boundary & Schedule Tests
# ---------------------------------------------------------------------------


def test_compute_future_loss_weight_boundaries_and_joint_ramp():
    """Verify stage 1 zero future weight and stage 2 linear ramp schedule."""
    writer_updates = 500
    joint_updates = 2000
    future_weight = 0.001
    ramp_updates = 200

    for u in (0, 1, 100, 250, 498, 499):
        w, stage = compute_future_loss_weight(u, writer_updates, joint_updates, future_weight, ramp_updates)
        assert w == 0.0
        assert stage == "stage1_writer_only"

    w_500, stage_500 = compute_future_loss_weight(500, writer_updates, joint_updates, future_weight, ramp_updates)
    assert stage_500 == "stage2_joint_future"
    assert abs(w_500 - future_weight * (1.0 / float(ramp_updates))) < 1e-9

    w_599, stage_599 = compute_future_loss_weight(599, writer_updates, joint_updates, future_weight, ramp_updates)
    assert stage_599 == "stage2_joint_future"
    assert abs(w_599 - future_weight * 0.5) < 1e-9

    w_699, stage_699 = compute_future_loss_weight(699, writer_updates, joint_updates, future_weight, ramp_updates)
    assert stage_699 == "stage2_joint_future"
    assert abs(w_699 - future_weight) < 1e-9

    for u in (700, 1000, 1500, 2499, 2500):
        w_post, stage_post = compute_future_loss_weight(u, writer_updates, joint_updates, future_weight, ramp_updates)
        assert stage_post == "stage2_joint_future"
        assert abs(w_post - future_weight) < 1e-9

    w_noramp, stage_noramp = compute_future_loss_weight(500, writer_updates, joint_updates, future_weight, ramp_updates=0)
    assert stage_noramp == "stage2_joint_future"
    assert abs(w_noramp - future_weight) < 1e-9


# ---------------------------------------------------------------------------
# 2. Optimizer Phase 1 vs Phase 2 Autograd Gradient & Weight Behavior
# ---------------------------------------------------------------------------


def test_optimizer_phase1_futurehead_no_grads_and_phase2_real_update():
    """Verify phase 1 trains only writer without head optimizer states, while phase 2 updates head."""
    torch.manual_seed(4042)
    w_cfg = WriterConfig(input_dim=64, hidden_dim=32, num_heads=2, num_layers=1, grid=1)
    writer = CausalMemoryWriter(w_cfg)
    head = FutureLatentHead(input_dim=64, hidden_dim=32, num_queries=2, num_heads=2)

    optimizer, scheduler = create_predictive_optimizer_and_scheduler(
        nn.ModuleList([writer, head]), lr=1e-3, weight_decay=1e-4, warmup_steps=5
    )

    writer_params_init = [p.detach().clone() for p in writer.parameters()]
    head_params_init = [p.detach().clone() for p in head.parameters()]

    optimizer.zero_grad(set_to_none=True)
    dummy_input = torch.randn(2, 1, 64)
    writer_out = writer.out_proj(writer.input_proj(dummy_input))
    writer_loss = writer_out.sum()
    writer_loss.backward()

    assert any(p.grad is not None for p in writer.parameters())
    assert not any(p.grad is not None for p in head.parameters())

    optimizer.step()
    scheduler.step()

    for p, p_init in zip(head.parameters(), head_params_init):
        assert torch.equal(p, p_init)
    for p in head.parameters():
        assert p not in optimizer.state or len(optimizer.state[p]) == 0
    assert any(not torch.equal(p, p_init) for p, p_init in zip(writer.parameters(), writer_params_init))

    optimizer.zero_grad(set_to_none=True)
    deep_ctx = torch.randn(2, 4, 64)
    shallow_ctx = torch.randn(2, 4, 64)
    dt_tensor = torch.tensor([[0.1, 0.3], [0.1, 0.3]], dtype=torch.float32)
    pred_latents = head(deep_ctx, shallow_ctx, dt_tensor)
    head_loss = ((pred_latents - torch.randn_like(pred_latents)) ** 2).mean()
    head_loss.backward()

    assert sum(p.grad is not None for p in head.parameters()) > 0

    optimizer.step()
    scheduler.step()

    assert any(not torch.equal(p, p_init) for p, p_init in zip(head.parameters(), head_params_init))
    assert all(
        p in optimizer.state and "exp_avg" in optimizer.state[p]
        for p in head.parameters() if p.grad is not None
    )


# ---------------------------------------------------------------------------
# 3. Checkpoint Save, Build Payload & Full Load State Restoration (Stage 1, update 1)
# ---------------------------------------------------------------------------


def test_save_build_and_load_predictive_checkpoint_strict_restoration(tmp_path: Path):
    """Verify stage 1 update 1 roundtrip strictly restores weights, optimizer moments, scheduler, and RNGs."""
    torch.manual_seed(12345)
    random.seed(12345)
    np.random.seed(12345)

    w_cfg = WriterConfig(input_dim=64, hidden_dim=32, num_heads=2, num_layers=1, grid=1)
    writer = CausalMemoryWriter(w_cfg)
    head = FutureLatentHead(input_dim=64, hidden_dim=32, num_queries=2, num_heads=2)

    optimizer, scheduler = create_predictive_optimizer_and_scheduler(
        nn.ModuleList([writer, head]), lr=1e-4, weight_decay=1e-4, warmup_steps=50
    )

    # Lightweight Stage 1 update: only train writer via out_proj(input_proj)
    dummy_in = torch.randn(2, 1, 64)
    loss = writer.out_proj(writer.input_proj(dummy_in)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    assert scheduler.last_epoch == 1
    # Verify no optimizer states created for future head in stage 1
    for p in head.parameters():
        assert p not in optimizer.state or len(optimizer.state[p]) == 0

    parent_path = "/fake/parent.pt"
    parent_sha = "deadbeef12345678"
    parent_step = 14000
    tc = make_full_training_contract(
        w_cfg,
        parent_path=parent_path,
        parent_sha256=parent_sha,
        parent_global_step=parent_step,
        writer_updates=500,
        joint_updates=2000,
    )
    train_dc = {"data_root": "/fake/data", "dataset_class": "PredictiveTrainingDataset"}
    val_dc = {"data_root": "/fake/data", "split": "val"}
    fingerprints = {"writer_init": "zero_out_proj", "protocol": "predictive_v1"}

    saved_path = save_predictive_checkpoint(
        output_dir=tmp_path,
        filename="test_update_1.pt",
        writer_module=writer,
        future_head_module=head,
        optimizer=optimizer,
        scheduler=scheduler,
        update=1,
        stage_name="stage1_writer_only",
        epoch=2,
        batch_cursor=16,
        epoch_targets_seen=32,
        parent_path=parent_path,
        parent_sha256=parent_sha,
        parent_global_step=parent_step,
        writer_config_dict=dataclasses.asdict(w_cfg),
        train_contract=tc,
        train_data_contract=train_dc,
        val_data_contract=val_dc,
        source_module_fingerprints=fingerprints,
        update_last=True,
    )

    assert saved_path.exists()
    assert (tmp_path / "last.pt").exists()

    # Capture expected post-save RNG states
    expected_py_state = random.getstate()
    expected_np_state = np.random.get_state()
    expected_torch_state = torch.get_rng_state()

    # Advance local RNGs before loading to prove load restores saved state
    _ = torch.randn(20)
    _ = random.random()
    _ = np.random.rand(20)

    # Fresh receiver modules and fresh optimizer/scheduler
    new_writer = CausalMemoryWriter(w_cfg)
    new_head = FutureLatentHead(input_dim=64, hidden_dim=32, num_queries=2, num_heads=2)
    new_optimizer, new_scheduler = create_predictive_optimizer_and_scheduler(
        nn.ModuleList([new_writer, new_head]), lr=1e-4, weight_decay=1e-4, warmup_steps=50
    )

    resume_info = load_predictive_checkpoint(
        checkpoint_path=saved_path,
        writer_module=new_writer,
        future_head_module=new_head,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        expected_parent_sha256=parent_sha,
        expected_training_contract=tc,
        expected_train_data_contract=train_dc,
        expected_val_data_contract=val_dc,
        device=torch.device("cpu"),
        expected_source_module_fingerprints=fingerprints,
    )

    assert resume_info["update"] == 1
    assert resume_info["stage"] == "stage1_writer_only"
    assert resume_info["epoch"] == 2
    assert resume_info["batch_cursor"] == 16
    assert resume_info["epoch_targets_seen"] == 32
    assert resume_info["parent_global_step"] == 14000
    assert resume_info["parent_path"] == parent_path

    # Parameter equality
    for p_orig, p_loaded in zip(writer.parameters(), new_writer.parameters()):
        assert torch.equal(p_orig, p_loaded)
    for p_orig, p_loaded in zip(head.parameters(), new_head.parameters()):
        assert torch.equal(p_orig, p_loaded)

    # Scheduler state equality
    assert new_scheduler.state_dict() == scheduler.state_dict()

    # Optimizer moments recursive equality
    _assert_tensors_equal(new_optimizer.state_dict(), optimizer.state_dict())

    # RNG state equality
    _assert_tensors_equal(random.getstate(), expected_py_state)
    _assert_tensors_equal(np.random.get_state(), expected_np_state)
    _assert_tensors_equal(torch.get_rng_state(), expected_torch_state)


# ---------------------------------------------------------------------------
# 4. Strict Rejection Parametrizations & Corruption Checks Without Mutation
# ---------------------------------------------------------------------------


def _setup_valid_checkpoint_fixture(tmp_path: Path) -> Tuple[Path, WriterConfig, Dict[str, Any], Dict[str, Any], Dict[str, Any], str]:
    """Helper creating a valid update 1 stage 1 checkpoint fixture."""
    w_cfg = WriterConfig(input_dim=64, hidden_dim=32, num_heads=2, num_layers=1, grid=1)
    writer = CausalMemoryWriter(w_cfg)
    head = FutureLatentHead(input_dim=64, hidden_dim=32, num_queries=2, num_heads=2)
    opt, sched = create_predictive_optimizer_and_scheduler(nn.ModuleList([writer, head]))

    # Step once with lightweight writer loss
    dummy_in = torch.randn(2, 1, 64)
    loss = writer.out_proj(writer.input_proj(dummy_in)).sum()
    loss.backward()
    opt.step()
    sched.step()

    parent_sha = "valid_parent_sha_12345"
    parent_path = "/valid/parent.pt"
    parent_step = 5000
    tc = make_full_training_contract(
        w_cfg,
        parent_path=parent_path,
        parent_sha256=parent_sha,
        parent_global_step=parent_step,
        writer_updates=500,
        joint_updates=2000,
        global_batch_size=8,
    )
    train_dc = {"data_root": "/valid/train/path"}
    val_dc = {"data_root": "/valid/val/path"}

    ckpt_file = save_predictive_checkpoint(
        output_dir=tmp_path,
        filename="valid_ckpt.pt",
        writer_module=writer,
        future_head_module=head,
        optimizer=opt,
        scheduler=sched,
        update=1,
        stage_name="stage1_writer_only",
        epoch=1,
        batch_cursor=8,
        epoch_targets_seen=16,
        parent_path=parent_path,
        parent_sha256=parent_sha,
        parent_global_step=parent_step,
        writer_config_dict=dataclasses.asdict(w_cfg),
        train_contract=tc,
        train_data_contract=train_dc,
        val_data_contract=val_dc,
        source_module_fingerprints={},
    )
    return ckpt_file, w_cfg, tc, train_dc, val_dc, parent_sha


def _make_receiver(w_cfg: WriterConfig):
    rec_w = CausalMemoryWriter(w_cfg)
    rec_h = FutureLatentHead(input_dim=64, hidden_dim=32, num_queries=2, num_heads=2)
    rec_opt, rec_sched = create_predictive_optimizer_and_scheduler(nn.ModuleList([rec_w, rec_h]))
    w_snap = {k: v.clone() for k, v in rec_w.state_dict().items()}
    h_snap = {k: v.clone() for k, v in rec_h.state_dict().items()}
    return rec_w, rec_h, rec_opt, rec_sched, w_snap, h_snap


def _assert_unmutated(rec_w, rec_h, w_snap, h_snap):
    for k, v in rec_w.state_dict().items():
        assert torch.equal(v, w_snap[k]), f"Writer mutated on key {k} before validation!"
    for k, v in rec_h.state_dict().items():
        assert torch.equal(v, h_snap[k]), f"FutureHead mutated on key {k} before validation!"


def test_load_checkpoint_parent_sha_and_data_contracts(tmp_path: Path):
    """Verify parent SHA, train data, val data, and format rejections without mutation."""
    ckpt_file, w_cfg, tc, train_dc, val_dc, parent_sha = _setup_valid_checkpoint_fixture(tmp_path)

    # Parent SHA mismatch
    rec_w, rec_h, rec_opt, rec_sched, w_snap, h_snap = _make_receiver(w_cfg)
    with pytest.raises(ValueError, match="Parent SHA mismatch"):
        load_predictive_checkpoint(
            ckpt_file, rec_w, rec_h, rec_opt, rec_sched,
            expected_parent_sha256="wrong_parent_sha",
            expected_training_contract=tc,
            expected_train_data_contract=train_dc,
            expected_val_data_contract=val_dc,
        )
    _assert_unmutated(rec_w, rec_h, w_snap, h_snap)

    # Train data contract mismatch
    rec_w, rec_h, rec_opt, rec_sched, w_snap, h_snap = _make_receiver(w_cfg)
    with pytest.raises(ValueError, match="Resume train_data_contract mismatch"):
        load_predictive_checkpoint(
            ckpt_file, rec_w, rec_h, rec_opt, rec_sched,
            expected_parent_sha256=parent_sha,
            expected_training_contract=tc,
            expected_train_data_contract={"data_root": "/other/train"},
            expected_val_data_contract=val_dc,
        )
    _assert_unmutated(rec_w, rec_h, w_snap, h_snap)

    # Val data contract mismatch
    rec_w, rec_h, rec_opt, rec_sched, w_snap, h_snap = _make_receiver(w_cfg)
    with pytest.raises(ValueError, match="Resume val_data_contract mismatch"):
        load_predictive_checkpoint(
            ckpt_file, rec_w, rec_h, rec_opt, rec_sched,
            expected_parent_sha256=parent_sha,
            expected_training_contract=tc,
            expected_train_data_contract=train_dc,
            expected_val_data_contract={"data_root": "/other/val"},
        )
    _assert_unmutated(rec_w, rec_h, w_snap, h_snap)

    # Format mismatch
    rec_w, rec_h, rec_opt, rec_sched, w_snap, h_snap = _make_receiver(w_cfg)
    bad_fmt_path = tmp_path / "bad_fmt.pt"
    payload = torch.load(str(ckpt_file), weights_only=False)
    payload["format"] = "wrong_format_v2"
    torch.save(payload, str(bad_fmt_path))
    with pytest.raises(ValueError, match="Expected format 'predictive_memory_adapter_v1'"):
        load_predictive_checkpoint(
            bad_fmt_path, rec_w, rec_h, rec_opt, rec_sched,
            expected_parent_sha256=parent_sha,
            expected_training_contract=tc,
            expected_train_data_contract=train_dc,
            expected_val_data_contract=val_dc,
        )
    _assert_unmutated(rec_w, rec_h, w_snap, h_snap)


@pytest.mark.parametrize(
    "corrupt_fn,match_err",
    [
        (lambda p: p.pop("writer"), "Missing required checkpoint field: 'writer'"),
        (lambda p: p.pop("optimizer"), "Missing required checkpoint field: 'optimizer'"),
        (lambda p: p.pop("scheduler"), "Missing required checkpoint field: 'scheduler'"),
        (lambda p: p.pop("global_step"), "Missing required checkpoint field: 'global_step'"),
        (lambda p: p["training_contract"].pop("lr"), "Training contract mismatch on 'lr'"),
        (lambda p: p["training_contract"].update({"extra_key": 123}), "Training contract mismatch on 'extra_key'"),
        (lambda p: p["training_contract"].update({"lr": 2e-4}), "Training contract mismatch on 'lr'"),
        (lambda p: p["training_contract"].update({"future_ramp_updates": 300}), "Training contract mismatch on 'future_ramp_updates'"),
        (lambda p: p["training_contract"].update({"warmup_updates": 100}), "Training contract mismatch on 'warmup_updates'"),
        (lambda p: p["training_contract"].update({"world_size": 2}), "Training contract mismatch on 'world_size'"),
        (lambda p: p["training_contract"].update({"writer_updates": 100}), "Training contract mismatch on 'writer_updates'"),
        (lambda p: p["training_contract"].update({"joint_updates": 100}), "Training contract mismatch on 'joint_updates'"),
        (lambda p: p.update({"writer_config": {"input_dim": 32, "hidden_dim": 16}}), "Writer config mismatch"),
        (lambda p: p["scheduler"].update({"last_epoch": 99}), "Scheduler last_epoch 99 != update 1"),
        (lambda p: p.update({"stage": "stage2_joint_future"}), "Checkpoint stage mismatch: got 'stage2_joint_future', expected 'stage1_writer_only'"),
        (lambda p: p.update({"global_step": 99999}), "Checkpoint global_step 99999 != parent_global_step 5000 + update 1"),
        (lambda p: p.update({"update": -1}), "Checkpoint field 'update' must be a non-negative integer"),
        (lambda p: p["writer"].update({"out_proj.weight": torch.randn(10, 10)}), "writer parameter 'out_proj.weight' shape mismatch"),
        (lambda p: p["writer"].update({"out_proj.weight": p["writer"]["out_proj.weight"].to(torch.float64)}), "writer parameter 'out_proj.weight' dtype mismatch"),
    ],
)
def test_load_checkpoint_strict_rejections_without_mutation(tmp_path: Path, corrupt_fn, match_err):
    """Parametrized rejection of corrupted payloads asserting zero target module mutation."""
    ckpt_file, w_cfg, tc, train_dc, val_dc, parent_sha = _setup_valid_checkpoint_fixture(tmp_path)
    corrupted_path = tmp_path / f"corrupted_{abs(hash(match_err))}.pt"
    payload = torch.load(str(ckpt_file), weights_only=False)
    corrupt_fn(payload)
    torch.save(payload, str(corrupted_path))

    rec_w, rec_h, rec_opt, rec_sched, w_snap, h_snap = _make_receiver(w_cfg)
    with pytest.raises(ValueError, match=re.escape(match_err)):
        load_predictive_checkpoint(
            corrupted_path, rec_w, rec_h, rec_opt, rec_sched,
            expected_parent_sha256=parent_sha,
            expected_training_contract=tc,
            expected_train_data_contract=train_dc,
            expected_val_data_contract=val_dc,
        )
    _assert_unmutated(rec_w, rec_h, w_snap, h_snap)


# ---------------------------------------------------------------------------
# 5. Base Frozen Invariants & Subset Hash Sampling
# ---------------------------------------------------------------------------


def test_base_frozen_invariants_and_hash_sampling():
    """Verify assert_base_frozen_invariants strictly enforces requires_grad=False and grad=None."""
    p_frozen1 = nn.Parameter(torch.randn(4, 4), requires_grad=False)
    p_frozen2 = nn.Parameter(torch.randn(4, 4), requires_grad=False)
    named = [("layer.0.weight", p_frozen1), ("layer.1.weight", p_frozen2)]

    assert_base_frozen_invariants(named)
    hashes = compute_parameter_subset_hashes(named, num_samples=2)
    assert len(hashes) == 2
    for _, h in hashes.items():
        assert len(h) == 64

    p_bad_grad = nn.Parameter(torch.randn(4, 4), requires_grad=True)
    with pytest.raises(RuntimeError, match="requires_grad=True"):
        assert_base_frozen_invariants([("bad.weight", p_bad_grad)])

    p_bad_grad_tensor = nn.Parameter(torch.randn(4, 4), requires_grad=False)
    p_bad_grad_tensor.grad = torch.zeros_like(p_bad_grad_tensor)
    with pytest.raises(RuntimeError, match="non-None grad"):
        assert_base_frozen_invariants([("bad.grad", p_bad_grad_tensor)])


# ---------------------------------------------------------------------------
# 6. evaluate_predictive Arithmetic and RNG Restoration
# ---------------------------------------------------------------------------


class DummyEvalModel(nn.Module):
    """Dummy model returning preset per-sample losses for aggregation testing."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, sample: Dict[str, Any], future_weight: float = 0.0) -> Dict[str, Any]:
        return {
            "action_loss": sample["act_val"],
            "future_loss": sample["fut_val"],
            "future_valid_count": sample["fval_cnt"],
        }


def test_evaluate_predictive_arithmetic_and_rng_restoration():
    """Verify target-weighted loss aggregation with unequal target and future counts, and RNG restoration."""
    torch.manual_seed(9999)
    random.seed(9999)
    np.random.seed(9999)

    pre_py_rng = random.getstate()
    pre_np_rng = np.random.get_state()
    pre_cpu_rng = torch.get_rng_state()

    dataset = [
        {"target_count": 10, "act_val": 2.0, "fut_val": 4.0, "fval_cnt": 5},
        {"target_count": 30, "act_val": 1.0, "fut_val": 2.0, "fval_cnt": 25},
    ]

    model = DummyEvalModel()
    future_weight = 0.1
    metrics = evaluate_predictive(model, dataset, [0, 1], future_weight=future_weight, device=torch.device("cpu"))

    # Expected target-weighted means:
    # action: (10*2.0 + 30*1.0) / 40 = 50 / 40 = 1.25
    # future: (10*4.0 + 30*2.0) / 40 = 100 / 40 = 2.5
    # loss: 1.25 + 0.1 * 2.5 = 1.5
    assert abs(metrics["action_loss"] - 1.25) < 1e-9
    assert abs(metrics["future_loss"] - 2.5) < 1e-9
    assert abs(metrics["loss"] - 1.5) < 1e-9
    assert metrics["targets"] == 40.0
    assert metrics["future_valid_count"] == 30.0

    # Verify RNG restoration
    _assert_tensors_equal(random.getstate(), pre_py_rng)
    _assert_tensors_equal(np.random.get_state(), pre_np_rng)
    _assert_tensors_equal(torch.get_rng_state(), pre_cpu_rng)


# ---------------------------------------------------------------------------
# 7. validate_predictive_cursor Cursor Alignment and Target Count Tests
# ---------------------------------------------------------------------------


def test_validate_predictive_cursor():
    """Verify validate_predictive_cursor strictly checks boundary alignment and target counts."""
    dataset = SimpleNamespace(
        segments=[
            (0, 0, 10),
            (0, 10, 25),
            (0, 25, 45),
            (0, 45, 60),
            (0, 60, 70),
            (0, 70, 85),
            (0, 85, 100),
            (0, 100, 120),
        ]
    )
    total_segments = len(dataset.segments)
    global_batch_size = 4
    seed = 4042
    epoch = 0
    cursor = 4

    epoch_generator = torch.Generator()
    epoch_generator.manual_seed(seed + epoch * 37)
    perm = torch.randperm(total_segments, generator=epoch_generator).tolist()
    expected_targets = sum(dataset.segments[idx][2] - dataset.segments[idx][1] for idx in perm[:cursor])

    # Valid cursor call passes
    validate_predictive_cursor(total_segments, global_batch_size, seed, epoch, cursor, expected_targets, dataset)

    # Wrong target count rejected
    with pytest.raises(ValueError, match="Cursor state mismatch"):
        validate_predictive_cursor(total_segments, global_batch_size, seed, epoch, cursor, expected_targets + 5, dataset)

    # Unaligned cursor rejected
    with pytest.raises(ValueError, match="must align with global_batch_size"):
        validate_predictive_cursor(total_segments, global_batch_size, seed, epoch, cursor + 1, expected_targets, dataset)


# ---------------------------------------------------------------------------
# 8. CLI Argument Validation Tests (Full Defaults Modified)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attr,bad_val,err_pattern",
    [
        ("lr", float("nan"), "must be positive and finite"),
        ("lr", float("inf"), "must be positive and finite"),
        ("weight_decay", -1e-4, "must be finite and non-negative"),
        ("max_updates", 0, "must be positive"),
        ("workers", -1, "must be finite and non-negative"),
        ("device", "cpu", "Formal training requires CUDA"),
    ],
)
def test_validate_cli_arguments_rejections(attr: str, bad_val: Any, err_pattern: str):
    """Verify parse_args with full defaults followed by specific modifications strictly fails validation."""
    args = parse_args(["--output-dir", "/tmp/test_dir", "--init-from", "/fake/parent.pt"])
    validate_cli_arguments(args)

    setattr(args, attr, bad_val)
    with pytest.raises(ValueError, match=err_pattern):
        validate_cli_arguments(args)


def test_validate_cli_arguments_missing_init_and_resume():
    """Verify rejection when neither init_from nor resume is provided."""
    args = parse_args(["--output-dir", "/tmp/test_dir"])
    with pytest.raises(ValueError, match="Either --init-from or --resume"):
        validate_cli_arguments(args)
