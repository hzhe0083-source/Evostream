"""Unit and integration tests for fabri_moss.train_native.

Covers:
1. True parameter clipping vs gradient tensors
2. Classification of parameters into 4 groups & AdamW parameter grouping / no decay rules
3. Optimizer warmup and cosine scheduler budget calculations
4. Pure CPU 2-rank Gloo vs 1-rank serial bucketing equality, non-uniform M, unused gradients
5. Seeded validation RNG preservation and deterministic repeatability
6. Strict contract equality validation on resume & rejection of changed contracts/shas
7. Full checkpoint save/load round-trip including model, optimizer, scheduler, RNGs
8. Remaining sampler exact epoch coverage, cursor handling, and tail partitions
9. CLI early validation guards (non-positive args, invalid fractions)
10. Completed resume / max-updates early exit without file overwriting
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple
import unittest.mock as mock

# Safe fallback for transformers/torchvision if running in minimal test environment
if "transformers" not in sys.modules:
    try:
        import transformers  # noqa: F401
    except ImportError:
        mock_tf = mock.MagicMock()
        sys.modules["transformers"] = mock_tf
        sys.modules["transformers.cache_utils"] = mock_tf

if "torchvision" not in sys.modules:
    try:
        import torchvision  # noqa: F401
    except ImportError:
        mock_tv = mock.MagicMock()
        sys.modules["torchvision"] = mock_tv
        sys.modules["torchvision.transforms"] = mock_tv
        sys.modules["torchvision.transforms.functional"] = mock_tv

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from fabri_moss.train_native import (
    SegmentSequenceSampler,
    bucketed_gradient_allreduce,
    classify_parameter,
    clip_parameter_groups_norm,
    compute_epoch_batches,
    create_native_optimizer_and_scheduler,
    evaluate_native,
    load_native_training_checkpoint,
    main,
    parse_args,
    run_training,
    save_native_checkpoint,
    validate_cli_arguments,
    validate_resume_state,
)


# ============================================================================
# Pure CPU Test Fixtures: TinyNamedPolicy with all 4 semantic groups
# ============================================================================

class TinyVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(16, 32)
        self.norm = nn.LayerNorm(32)
        self.bias_param = nn.Parameter(torch.zeros(32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(x)) + self.bias_param


class TinyProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Linear(32, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp1(x)


class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Linear(64, 64)
        self.layer_norm = nn.LayerNorm(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.language_model(x))


class TinyActionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_head_linear = nn.Linear(64, 24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.action_head_linear(x)


class TinyNamedPolicy(nn.Module):
    """Test policy containing the exact 4 semantic parameter namespaces.

    - embedder.model.vision_model
    - embedder.model.mlp1
    - embedder.model.language_model
    - action_head
    """
    def __init__(self):
        super().__init__()
        self.embedder = nn.Module()
        self.embedder.model = nn.Module()
        self.embedder.model.vision_model = TinyVisionModel()
        self.embedder.model.mlp1 = nn.Linear(32, 64)
        self.embedder.model.language_model = nn.Linear(64, 64)
        self.action_head = TinyActionHead()
        self.action_head.config = argparse.Namespace(per_action_dim=24, horizon=50, shallow_layer_index=6)
        self.config = self.action_head.config

    def forward(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Simple differentiable forward computing loss_sum
        x = sample["inputs"]  # [M, 16]
        v = self.embedder.model.vision_model(x)
        p = self.embedder.model.mlp1(v)
        l = self.embedder.model.language_model(p)
        a = self.action_head(l)
        target = sample["targets"]  # [M, 24]
        loss_sum = ((a - target) ** 2).sum()
        return {"loss_sum": loss_sum}


class DummyDataset:
    """Minimal dataset for testing loader and evaluation."""
    def __init__(self, num_segments: int = 20, target_count_per_segment: int = 4):
        self.segments = [(0, i * target_count_per_segment, (i + 1) * target_count_per_segment) for i in range(num_segments)]
        self._total_targets = num_segments * target_count_per_segment
        self.epoch = 0

    @property
    def total_targets(self) -> int:
        return self._total_targets

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s, e = self.segments[idx][1], self.segments[idx][2]
        count = e - s
        torch.manual_seed(idx + 100)
        return {
            "inputs": torch.randn(count, 16),
            "targets": torch.randn(count, 24),
            "target_count": count,
        }

    def validation_indices(self, per_task: int = 1) -> List[int]:
        return list(range(min(4, len(self.segments))))


class PureDummyDataset:
    """Minimal dataset that does not reseed torch or python RNG inside __getitem__."""
    def __init__(self, num_segments: int = 10, target_count_per_segment: int = 4):
        self.segments = [(0, i * target_count_per_segment, (i + 1) * target_count_per_segment) for i in range(num_segments)]
        self._total_targets = num_segments * target_count_per_segment

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s, e = self.segments[idx][1], self.segments[idx][2]
        count = e - s
        return {
            "inputs": torch.randn(count, 16),
            "targets": torch.randn(count, 24),
            "target_count": count,
        }


class StochasticToyModel(nn.Module):
    """Model with stochastic noise during evaluation forward."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 24)

    def forward(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        x = sample["inputs"]
        noise = torch.randn(x.shape[0], 24, device=x.device)
        out = self.linear(x) + noise
        target = sample["targets"]
        loss_sum = ((out - target) ** 2).sum()
        return {"loss_sum": loss_sum}


class MockNativeSequencePolicy(nn.Module):
    """Mock wrapper conforming to NativeSequencePolicy interface for CPU toy tests."""
    def __init__(self, policy: nn.Module, shallow_layer: int = 6, use_timestamps: bool = True, gradient_checkpointing: bool = True):
        super().__init__()
        self.policy = policy

    def forward(self, sample_dev: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self.policy(sample_dev)


class MockNativeTrainingDataset:
    """Mock NativeTrainingDataset for toy end-to-end integration tests."""
    def __init__(
        self,
        root: str,
        norm_stats: Any,
        history_frames: int,
        target_frames: int,
        horizon: int,
        state_dim: int,
        action_dim: int,
        split: str = "train",
        seed: int = 42,
        val_fraction: float = 0.1,
        max_episodes: Optional[int] = None,
        augmentation: bool = True,
    ):
        self.split = split
        self.seed = seed
        self.history_frames = history_frames
        self.target_frames = target_frames
        self.epoch = 0
        # 9 segments for train (not divisible by global_batch_size=4), 4 for val
        num_segments = 9 if split == "train" else 4
        self.segments = [(0, i * 2, (i + 1) * 2) for i in range(num_segments)]
        self._total_targets = num_segments * 2

    @property
    def total_targets(self) -> int:
        return self._total_targets

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s, e = self.segments[idx][1], self.segments[idx][2]
        count = e - s
        return {
            "inputs": torch.randn(count, 16),
            "targets": torch.randn(count, 24),
            "target_count": count,
        }

    def get_data_contract(self) -> Dict[str, Any]:
        return {
            "format": "native_multiframe_v1",
            "split": self.split,
            "target_count": self._total_targets,
            "data_fingerprint": f"dfp_{self.split}",
            "seed": self.seed,
            "val_fraction": 0.1,
            "history_frames": self.history_frames,
            "target_frames": self.target_frames,
        }

    def validation_indices(self, per_task: int = 1) -> List[int]:
        return list(range(min(4, len(self.segments))))


# ============================================================================
# Test 1: Gradient norm clipping actually clips parameters
# ============================================================================

def test_clip_parameter_groups_norm_actually_clips():
    p1 = nn.Parameter(torch.ones(10, 10))
    p2 = nn.Parameter(torch.ones(10, 10))
    p1.grad = torch.full((10, 10), 2.0)
    p2.grad = torch.full((10, 10), 2.0)

    # Total norm before = sqrt(100*4 + 100*4) = sqrt(800) ~ 28.284
    initial_norm = math.sqrt(800.0)
    max_norm = 1.0

    returned_norm = clip_parameter_groups_norm([p1, p2], max_norm=max_norm)
    assert math.isclose(returned_norm, initial_norm, rel_tol=1e-4)

    # MUST FIX 1: Gradients must be scaled down in-place
    p1_grad_norm = p1.grad.norm().item()
    p2_grad_norm = p2.grad.norm().item()
    total_clipped = math.sqrt(p1_grad_norm**2 + p2_grad_norm**2)
    assert math.isclose(total_clipped, max_norm, rel_tol=1e-4)
    assert p1.grad[0, 0].item() < 1.0


def test_clip_parameter_groups_norm_error_on_nonfinite():
    p = nn.Parameter(torch.ones(2, 2))
    p.grad = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
    with pytest.raises(RuntimeError):
        clip_parameter_groups_norm([p], max_norm=1.0)


# ============================================================================
# Test 2: Parameter classification and optimizer grouping rules
# ============================================================================

def test_parameter_classification_and_decay_rules():
    policy = TinyNamedPolicy()

    # Verify classification of each parameter
    for name, p in policy.named_parameters():
        grp = classify_parameter(name)
        assert grp in ("vision", "projector", "llm", "head")

    with pytest.raises(ValueError):
        classify_parameter("unknown.submodule.weight")

    optimizer, scheduler = create_native_optimizer_and_scheduler(
        policy=policy,
        lr_vision=1e-6,
        lr_projector=5e-6,
        lr_llm=2e-6,
        lr_head=1e-5,
        weight_decay=1e-4,
        total_steps=100,
        warmup_steps=10,
    )

    # Verify each group has correct lr and decay vs no-decay rules
    found_groups = set()
    for g in optimizer.param_groups:
        g_name = g["group_name"]
        found_groups.add(g_name)
        is_nd = g["no_decay"]
        if is_nd:
            assert g["weight_decay"] == 0.0
        else:
            assert g["weight_decay"] == 1e-4

        # Base initial_lr stored in param group
        base_lr = g["initial_lr"]
        if g_name == "vision":
            assert base_lr == 1e-6
            assert math.isclose(g["lr"], 1e-6 * 0.1, rel_tol=1e-5)
        elif g_name == "projector":
            assert base_lr == 5e-6
            assert math.isclose(g["lr"], 5e-6 * 0.1, rel_tol=1e-5)
        elif g_name == "llm":
            assert base_lr == 2e-6
            assert math.isclose(g["lr"], 2e-6 * 0.1, rel_tol=1e-5)
        elif g_name == "head":
            assert base_lr == 1e-5
            assert math.isclose(g["lr"], 1e-5 * 0.1, rel_tol=1e-5)

    assert found_groups == {"vision", "projector", "llm", "head"}


# ============================================================================
# Test 3: Warmup first update non-zero and fixed total budget
# ============================================================================

def test_warmup_first_update_nonzero_and_cosine_schedule():
    policy = TinyNamedPolicy()
    total_steps = 100
    warmup_steps = 10
    optimizer, scheduler = create_native_optimizer_and_scheduler(
        policy=policy,
        lr_head=1e-4,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    # First step (index 0) passed to lr_lambda
    lrs_step0 = [g["lr"] for g in optimizer.param_groups]
    optimizer.step()
    scheduler.step()
    # At step 1 (second update): ratio = 2/10
    head_grp = next(g for g in optimizer.param_groups if g["group_name"] == "head")
    assert head_grp["lr"] > 0.0
    assert math.isclose(head_grp["lr"], 1e-4 * (2.0 / 10.0), rel_tol=1e-5)

    # Advance to end of warmup
    for _ in range(8):
        optimizer.step()
        scheduler.step()
    head_grp = next(g for g in optimizer.param_groups if g["group_name"] == "head")
    assert math.isclose(head_grp["lr"], 1e-4, rel_tol=1e-5)


# ============================================================================
# Test 4: Epoch batch generator and sampler exact coverage
# ============================================================================

def test_epoch_batch_generator_coverage_and_cursor():
    total_segments = 25
    global_batch_size = 8
    world_size = 2
    seed = 42

    perm, batches, per_rank_indices = compute_epoch_batches(
        total_segments=total_segments,
        global_batch_size=global_batch_size,
        world_size=world_size,
        seed=seed,
        epoch=0,
    )

    # Verify every segment appears exactly once in permutation
    assert sorted(perm) == list(range(total_segments))

    # Batches: 8 + 8 + 8 + 1 = 25
    assert len(batches) == 4
    assert [len(b) for b in batches] == [8, 8, 8, 1]

    # Check rank distribution across batches
    r0_total = len(per_rank_indices[0])
    r1_total = len(per_rank_indices[1])
    assert r0_total + r1_total == total_segments

    # Sampler iteration
    sampler = SegmentSequenceSampler(per_rank_indices[0])
    assert list(sampler) == per_rank_indices[0]
    assert len(sampler) == len(per_rank_indices[0])


# ============================================================================
# Test 5: Seeded validation RNG restoration and deterministic repeatability
# ============================================================================

def test_seeded_validation_repeatability_and_rng_restoration():
    model = StochasticToyModel()
    dataset = PureDummyDataset(num_segments=10, target_count_per_segment=4)
    eval_indices = [1, 3, 5]
    device = torch.device("cpu")

    # Capture all RNG states before validation
    py_state_before = random.getstate()
    np_state_before = np.random.get_state()
    cpu_state_before = torch.get_rng_state()

    loss1, targets1 = evaluate_native(model, dataset, eval_indices, device=device, base_seed=12345)

    # Capture all RNG states after validation
    py_state_after = random.getstate()
    np_state_after = np.random.get_state()
    cpu_state_after = torch.get_rng_state()

    # Check that validation preserved Python, numpy, and PyTorch CPU RNG states
    assert py_state_after == py_state_before
    assert np.array_equal(np_state_after[1], np_state_before[1])
    assert torch.equal(cpu_state_after, cpu_state_before)

    # Perturb the external training RNG
    _ = [random.random() for _ in range(50)]
    _ = np.random.randn(50)
    _ = torch.randn(50)

    # Repeat validation after training RNG perturbation
    loss2, targets2 = evaluate_native(model, dataset, eval_indices, device=device, base_seed=12345)

    assert targets1 == targets2
    assert math.isclose(loss1, loss2, rel_tol=1e-6)


# ============================================================================
# Test 6: Checkpoint payload roundtrip and strict resume contracts
# ============================================================================

def test_checkpoint_roundtrip_and_strict_contract_validation(tmp_path: Path):
    policy = TinyNamedPolicy()
    optimizer, scheduler = create_native_optimizer_and_scheduler(policy, total_steps=100, warmup_steps=10)

    # Advance scheduler to step 10 to match global_step=10
    for _ in range(10):
        optimizer.step()
        scheduler.step()

    source_meta = {"checkpoint_sha256": "abc123sha", "step": 10}
    train_contract = {
        "format": "native_multiframe_v1",
        "epochs": 5,
        "global_batch_size": 4,
        "world_size": 1,
        "seed": 42,
        "use_timestamps": True,
        "history_frames": 16,
        "target_frames": 8,
        "lr_vision": 1e-6,
        "lr_projector": 5e-6,
        "lr_llm": 2e-6,
        "lr_head": 1e-5,
        "warmup_updates": 10,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "total_scheduler_updates": 100,
        "gradient_checkpointing": True,
        "augmentation": True,
    }
    train_data_dc = {"target_count": 100, "data_fingerprint": "dfp_train", "seed": 42}
    val_data_dc = {"target_count": 20, "data_fingerprint": "dfp_val", "seed": 42}

    ckpt_file = save_native_checkpoint(
        output_dir=tmp_path,
        filename="last.pt",
        policy=policy,
        config={"dim": 64},
        norm_stats={},
        training_contract=train_contract,
        train_data_contract=train_data_dc,
        val_data_contract=val_data_dc,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=10,
        epoch=1,
        batch_cursor=8,
        epoch_targets_seen=32,
        source_metadata=source_meta,
        device=torch.device("cpu"),
        update_last=False,
    )

    assert ckpt_file.exists()

    # Normal resume success
    resume_info = load_native_training_checkpoint(
        checkpoint_path=ckpt_file,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_source_metadata=source_meta,
        expected_training_contract=train_contract,
        expected_train_data_contract=train_data_dc,
        expected_val_data_contract=val_data_dc,
        device=torch.device("cpu"),
    )
    assert resume_info["global_step"] == 10
    assert resume_info["epoch"] == 1
    assert resume_info["batch_cursor"] == 8

    # Rejection 1: Source SHA mismatch
    bad_source = copy.deepcopy(source_meta)
    bad_source["checkpoint_sha256"] = "different_sha"
    with pytest.raises(ValueError, match="Source checkpoint SHA256 mismatch"):
        load_native_training_checkpoint(
            checkpoint_path=ckpt_file,
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_source_metadata=bad_source,
            expected_training_contract=train_contract,
            expected_train_data_contract=train_data_dc,
            expected_val_data_contract=val_data_dc,
        )

    # Rejection 2: Training contract change (e.g. learning rate)
    bad_contract = copy.deepcopy(train_contract)
    bad_contract["lr_head"] = 9.99e-5
    with pytest.raises(ValueError, match="Training contract mismatch"):
        load_native_training_checkpoint(
            checkpoint_path=ckpt_file,
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_source_metadata=source_meta,
            expected_training_contract=bad_contract,
            expected_train_data_contract=train_data_dc,
            expected_val_data_contract=val_data_dc,
        )

    # Rejection 3: Data contract change (target_count)
    bad_data = copy.deepcopy(train_data_dc)
    bad_data["target_count"] = 999
    with pytest.raises(ValueError, match="train_data_contract mismatch on 'target_count'"):
        load_native_training_checkpoint(
            checkpoint_path=ckpt_file,
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_source_metadata=source_meta,
            expected_training_contract=train_contract,
            expected_train_data_contract=bad_data,
            expected_val_data_contract=val_data_dc,
        )

    # Rejection 4: Scheduler last_epoch / global_step mismatch
    bad_sched_ckpt = torch.load(str(ckpt_file), weights_only=False)
    bad_sched_ckpt["scheduler"]["last_epoch"] = 999
    bad_sched_path = tmp_path / "bad_sched.pt"
    torch.save(bad_sched_ckpt, str(bad_sched_path))
    with pytest.raises(ValueError, match="Scheduler last_epoch .* mismatch with checkpoint global_step"):
        load_native_training_checkpoint(
            checkpoint_path=bad_sched_path,
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_source_metadata=source_meta,
            expected_training_contract=train_contract,
            expected_train_data_contract=train_data_dc,
            expected_val_data_contract=val_data_dc,
        )


# ============================================================================
# Test 7: CLI Argument Validation Early Guards
# ============================================================================

def test_cli_argument_validation_guards():
    # Negative epochs
    args = parse_args(["--output-dir", "/tmp/test", "--epochs", "0"])
    with pytest.raises(ValueError, match="--epochs must be a positive integer"):
        validate_cli_arguments(args)

    # Target frames > history frames
    args = parse_args(["--output-dir", "/tmp/test", "--history-frames", "4", "--target-frames", "8"])
    with pytest.raises(ValueError, match="must be <= --history-frames"):
        validate_cli_arguments(args)

    # Negative warmup
    args = parse_args(["--output-dir", "/tmp/test", "--warmup-updates", "-1"])
    with pytest.raises(ValueError, match="--warmup-updates must be non-negative"):
        validate_cli_arguments(args)

    # Invalid val fraction
    args = parse_args(["--output-dir", "/tmp/test", "--val-fraction", "1.5"])
    with pytest.raises(ValueError, match="--val-fraction must be in"):
        validate_cli_arguments(args)


# ============================================================================
# Test 8: Validate resume state helper
# ============================================================================

def test_validate_resume_state():
    dataset = DummyDataset(num_segments=10, target_count_per_segment=4)
    # Cursor 4 corresponds to first 4 segments: 4 * 4 = 16 targets
    # total_segments=10, B=4 -> ceil(10/4)=3 steps/epoch. epoch 0, cursor 4 -> ceil(4/4)=1 -> step 1
    validate_resume_state(
        total_segments=10,
        global_batch_size=4,
        seed=42,
        epoch=0,
        cursor=4,
        step=1,
        epoch_targets_seen=16,
        train_dataset=dataset,
        max_epochs=5,
    )

    # Step mismatch raises error
    with pytest.raises(ValueError, match="Resume step mismatch"):
        validate_resume_state(
            total_segments=10,
            global_batch_size=4,
            seed=42,
            epoch=0,
            cursor=4,
            step=2,  # wrong step
            epoch_targets_seen=16,
            train_dataset=dataset,
        )

    # Target mismatch raises error
    with pytest.raises(ValueError, match="Resume state mismatch"):
        validate_resume_state(
            total_segments=10,
            global_batch_size=4,
            seed=42,
            epoch=0,
            cursor=4,
            step=1,
            epoch_targets_seen=15,  # wrong targets
            train_dataset=dataset,
        )

    # Negative epoch raises error
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_resume_state(
            total_segments=10,
            global_batch_size=4,
            seed=42,
            epoch=-1,
            cursor=0,
            step=0,
            epoch_targets_seen=0,
            train_dataset=dataset,
        )

    # Completed run (epoch == max_epochs) requires cursor=0 and targets=0
    with pytest.raises(ValueError, match="Completed run at epoch"):
        validate_resume_state(
            total_segments=10,
            global_batch_size=4,
            seed=42,
            epoch=5,
            cursor=4,
            step=16,
            epoch_targets_seen=16,
            train_dataset=dataset,
            max_epochs=5,
        )


# ============================================================================
# Test 9: Pure CPU 2-Rank Gloo vs 1-Rank Serial Autograd & AdamW Equality
# ============================================================================

def _run_gloo_bucket_test_worker(
    rank: int,
    world_size: int,
    init_file: str,
    results_dir: str,
):
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=f"file://{init_file}")

    torch.manual_seed(42)
    p_shared = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))
    p_unused = nn.Parameter(torch.tensor([5.0, 6.0], dtype=torch.float32))

    optimizer = torch.optim.AdamW([p_shared, p_unused], lr=0.1, weight_decay=0.01)

    # --- Update 1: Variable M across ranks (rank 0 has M=3, rank 1 has M=5, total=8) ---
    x0 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    y0 = torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    x1 = torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [0.5, 1.0], [1.0, 0.5]], dtype=torch.float32)
    y1 = torch.tensor([[1.0, 1.0], [0.0, 0.5], [0.5, 0.0], [1.0, 0.5], [0.5, 1.0]], dtype=torch.float32)

    optimizer.zero_grad(set_to_none=True)
    if rank == 0:
        loss1 = ((x0 @ p_shared - y0) ** 2).sum()
    else:
        loss1 = ((x1 @ p_shared - y1) ** 2).sum()

    loss1.backward()

    # Synchronize across ranks with global target count 8
    bucketed_gradient_allreduce([p_shared, p_unused], global_target_count=8, process_group=None)
    clip_parameter_groups_norm([p_shared, p_unused], max_norm=10.0)
    optimizer.step()

    p_shared_step1 = p_shared.detach().clone().tolist()
    p_unused_step1 = p_unused.detach().clone().tolist()
    p_unused_grad1_is_none = (p_unused.grad is None)

    # --- Update 2: ONLY rank 0 has data (M0=2), rank 1 has NO data (M1=0, grad is None) ---
    x0_2 = torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float32)
    y0_2 = torch.tensor([[0.1, 0.9], [0.5, 0.5]], dtype=torch.float32)

    optimizer.zero_grad(set_to_none=True)
    if rank == 0:
        loss2 = ((x0_2 @ p_shared - y0_2) ** 2).sum()
        loss2.backward()

    # Synchronize across ranks with global target count 2
    bucketed_gradient_allreduce([p_shared, p_unused], global_target_count=2, process_group=None)
    clip_parameter_groups_norm([p_shared, p_unused], max_norm=10.0)
    optimizer.step()

    shared_state = optimizer.state.get(p_shared, {})
    exp_avg = shared_state["exp_avg"].tolist() if "exp_avg" in shared_state else None
    exp_avg_sq = shared_state["exp_avg_sq"].tolist() if "exp_avg_sq" in shared_state else None

    res = {
        "p_shared_step1": p_shared_step1,
        "p_unused_step1": p_unused_step1,
        "p_unused_grad1_is_none": p_unused_grad1_is_none,
        "p_shared_step2": p_shared.detach().clone().tolist(),
        "p_unused_step2": p_unused.detach().clone().tolist(),
        "p_unused_grad2_is_none": (p_unused.grad is None),
        "exp_avg": exp_avg,
        "exp_avg_sq": exp_avg_sq,
    }
    with open(os.path.join(results_dir, f"rank_{rank}.json"), "w") as f:
        json.dump(res, f)

    dist.destroy_process_group()


def test_bucketed_gradient_allreduce_2rank_gloo_vs_serial(tmp_path: Path):
    init_file = str(tmp_path / "gloo_init")
    results_dir = str(tmp_path / "gloo_res")
    os.makedirs(results_dir, exist_ok=True)

    # Compute serial ground truth reference
    p_serial = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))
    p_unused_serial = nn.Parameter(torch.tensor([5.0, 6.0], dtype=torch.float32))
    opt_serial = torch.optim.AdamW([p_serial, p_unused_serial], lr=0.1, weight_decay=0.01)

    # Step 1: full combined batch
    x0 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    y0 = torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    x1 = torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [0.5, 1.0], [1.0, 0.5]], dtype=torch.float32)
    y1 = torch.tensor([[1.0, 1.0], [0.0, 0.5], [0.5, 0.0], [1.0, 0.5], [0.5, 1.0]], dtype=torch.float32)

    opt_serial.zero_grad(set_to_none=True)
    l1 = ((x0 @ p_serial - y0) ** 2).sum() + ((x1 @ p_serial - y1) ** 2).sum()
    l1.backward()
    p_serial.grad.div_(8.0)
    clip_parameter_groups_norm([p_serial, p_unused_serial], max_norm=10.0)
    opt_serial.step()

    ref_p_shared_step1 = p_serial.detach().clone().tolist()
    ref_p_unused_step1 = p_unused_serial.detach().clone().tolist()

    # Step 2: rank 0 data only (target count 2)
    x0_2 = torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float32)
    y0_2 = torch.tensor([[0.1, 0.9], [0.5, 0.5]], dtype=torch.float32)
    opt_serial.zero_grad(set_to_none=True)
    l2 = ((x0_2 @ p_serial - y0_2) ** 2).sum()
    l2.backward()
    p_serial.grad.div_(2.0)
    clip_parameter_groups_norm([p_serial, p_unused_serial], max_norm=10.0)
    opt_serial.step()

    ref_p_shared_step2 = p_serial.detach().clone().tolist()
    ref_p_unused_step2 = p_unused_serial.detach().clone().tolist()
    ref_exp_avg = opt_serial.state[p_serial]["exp_avg"].tolist()
    ref_exp_avg_sq = opt_serial.state[p_serial]["exp_avg_sq"].tolist()

    # Run 2-rank Gloo in parallel
    mp.spawn(
        _run_gloo_bucket_test_worker,
        args=(2, init_file, results_dir),
        nprocs=2,
        join=True,
    )

    with open(os.path.join(results_dir, "rank_0.json")) as f:
        res0 = json.load(f)
    with open(os.path.join(results_dir, "rank_1.json")) as f:
        res1 = json.load(f)

    # Compare rank 0 and rank 1 with serial ground truth
    for res in (res0, res1):
        assert np.allclose(res["p_shared_step1"], ref_p_shared_step1, atol=1e-5)
        assert np.allclose(res["p_shared_step2"], ref_p_shared_step2, atol=1e-5)
        assert res["p_unused_step1"] == ref_p_unused_step1
        assert res["p_unused_step2"] == ref_p_unused_step2
        assert res["p_unused_grad1_is_none"] is True
        assert res["p_unused_grad2_is_none"] is True
        assert np.allclose(res["exp_avg"], ref_exp_avg, atol=1e-5)
        assert np.allclose(res["exp_avg_sq"], ref_exp_avg_sq, atol=1e-5)

    # Cross-rank state equality
    assert np.allclose(res0["p_shared_step2"], res1["p_shared_step2"])
    assert np.allclose(res0["exp_avg"], res1["exp_avg"])
    assert np.allclose(res0["exp_avg_sq"], res1["exp_avg_sq"])


# ============================================================================
# Test 10: Optimizer step updates representative parameters across 4 groups
# ============================================================================

def test_optimizer_first_step_parameter_updates():
    policy = TinyNamedPolicy()
    optimizer, scheduler = create_native_optimizer_and_scheduler(policy)

    # Simulate backward pass
    loss_out = policy({"inputs": torch.randn(4, 16), "targets": torch.randn(4, 24)})
    loss_out["loss_sum"].backward()

    # Track params before update
    params_before = {name: p.clone().detach() for name, p in policy.named_parameters() if p.requires_grad}

    # Verify gradients exist across all 4 groups
    for name, p in policy.named_parameters():
        if p.requires_grad:
            assert p.grad is not None

    clip_parameter_groups_norm(list(policy.parameters()), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    # Check that representative parameters changed in each group
    group_updated = {g: False for g in ("vision", "projector", "llm", "head")}
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        grp = classify_parameter(name)
        if (p.detach() - params_before[name]).abs().max() > 0:
            group_updated[grp] = True

    assert all(group_updated.values()), f"Some groups were not updated: {group_updated}"


# ============================================================================
# Test 11: Non-empty output dir without resume fails early
# ============================================================================

def test_nonempty_output_dir_without_resume_fails_early(tmp_path: Path):
    out_dir = tmp_path / "train_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_file = out_dir / "existing_file.txt"
    existing_file.write_text("prior work")

    # When resume is None and output_dir is non-empty, main must reject before CUDA setup
    with pytest.raises(FileExistsError, match="already exists and is non-empty"):
        main(["--output-dir", str(out_dir)])

    # Assert contents remain strictly unchanged
    assert existing_file.exists()
    assert existing_file.read_text() == "prior work"
    assert list(out_dir.iterdir()) == [existing_file]


# ============================================================================
# Test 12: Toy Training Run 2 -> Resume 4 Parity and Completed Exit
# ============================================================================

def test_toy_training_run2_resume4_parity_and_completed_exit(monkeypatch, tmp_path: Path):
    base_meta = {"checkpoint_sha256": "fake_sha_toy"}
    norm_stats = {}

    def make_base_policy():
        torch.manual_seed(9999)
        pol = TinyNamedPolicy()
        return pol

    import fabri_moss.train_native as tn
    monkeypatch.setattr(tn, "require_training_device", lambda p, d: None)
    monkeypatch.setattr("fabri_moss.runtime.load_native_checkpoint", lambda **kw: (make_base_policy(), {"dim": 64}, norm_stats, base_meta))
    monkeypatch.setattr("fabri_moss.runtime.assert_native_fa2", lambda p: {"native_fa2": True})
    monkeypatch.setattr("fabri_moss.native_training.NativeSequencePolicy", lambda **kw: MockNativeSequencePolicy(kw["policy"]))
    monkeypatch.setattr("fabri_moss.native_data.NativeTrainingDataset", MockNativeTrainingDataset)

    base_args = [
        "--output-dir", "",
        "--epochs", "2",
        "--global-batch-size", "4",
        "--workers", "0",
        "--seed", "42",
        "--val-fraction", "0.1",
        "--lr-vision", "1e-4",
        "--lr-projector", "1e-4",
        "--lr-llm", "1e-4",
        "--lr-head", "1e-4",
        "--warmup-updates", "2",
        "--save-every", "10",
        "--eval-every", "10",
        "--val-per-task", "1",
        "--grad-clip", "1.0",
    ]

    # --- Run A: Continuous 4 updates ---
    dir_cont = tmp_path / "continuous"
    args_cont = base_args.copy()
    args_cont[1] = str(dir_cont)
    args_cont.extend(["--max-updates", "4"])
    main(args_cont, device=torch.device("cpu"))

    # --- Run B1: Partial epoch stop at 2 updates (before epoch end at step 3) ---
    dir_res = tmp_path / "resumed"
    args_part1 = base_args.copy()
    args_part1[1] = str(dir_res)
    args_part1.extend(["--max-updates", "2"])
    main(args_part1, device=torch.device("cpu"))

    # Verify partial epoch checkpoint created at step 2, epoch 0, cursor 8
    ckpt_part1 = torch.load(str(dir_res / "last.pt"), weights_only=False)
    assert ckpt_part1["global_step"] == 2
    assert ckpt_part1["epoch"] == 0
    assert ckpt_part1["batch_cursor"] == 8
    assert ckpt_part1["epoch_targets_seen"] == 16
    assert not (dir_res / "checkpoint_epoch_001.pt").exists()

    # --- Run B2: Resume from last.pt to 4 updates ---
    args_part2 = base_args.copy()
    args_part2[1] = str(dir_res)
    args_part2.extend(["--max-updates", "4", "--resume", str(dir_res / "last.pt")])
    main(args_part2, device=torch.device("cpu"))

    # Verify parity with continuous 4 updates
    ckpt_cont = torch.load(str(dir_cont / "last.pt"), weights_only=False)
    ckpt_res = torch.load(str(dir_res / "last.pt"), weights_only=False)

    assert ckpt_cont["global_step"] == 4
    assert ckpt_res["global_step"] == 4
    assert ckpt_cont["epoch"] == ckpt_res["epoch"]
    assert ckpt_cont["batch_cursor"] == ckpt_res["batch_cursor"]
    assert ckpt_cont["epoch_targets_seen"] == ckpt_res["epoch_targets_seen"]

    for k in ckpt_cont["model"]:
        assert torch.allclose(ckpt_cont["model"][k], ckpt_res["model"][k], atol=1e-5), f"Model weight mismatch on {k}"

    assert ckpt_cont["scheduler"] == ckpt_res["scheduler"]

    # Verify train_metrics.jsonl lines match exactly
    with open(dir_cont / "train_metrics.jsonl") as f:
        metrics_cont = [json.loads(line) for line in f if "loss" in json.loads(line)]
    with open(dir_res / "train_metrics.jsonl") as f:
        metrics_res = [json.loads(line) for line in f if "loss" in json.loads(line)]
    assert len(metrics_cont) == 4
    assert len(metrics_res) == 4
    for mc, mr in zip(metrics_cont, metrics_res):
        assert mc["global_step"] == mr["global_step"]
        assert math.isclose(mc["loss"], mr["loss"], rel_tol=1e-5)
        assert mc["targets"] == mr["targets"]

    # --- Run B3: Completed resume does not mutate files or add resume events ---
    mtime_before = (dir_res / "last.pt").stat().st_mtime_ns
    cfg_before = (dir_res / "run_config.json").read_text()

    # Call main with same max_updates=4 (already completed)
    main(args_part2, device=torch.device("cpu"))

    mtime_after = (dir_res / "last.pt").stat().st_mtime_ns
    cfg_after = (dir_res / "run_config.json").read_text()
    assert mtime_before == mtime_after
    assert cfg_before == cfg_after

    # --- Run C: 10 epochs toy run terminates exact & all targets (N=9 not divisible by B=4) ---
    dir_10ep = tmp_path / "toy_10ep"
    args_10ep = base_args.copy()
    args_10ep[1] = str(dir_10ep)
    args_10ep[3] = "10"  # 10 epochs
    main(args_10ep, device=torch.device("cpu"))

    assert (dir_10ep / "checkpoint_epoch_010.pt").exists()
    ckpt_10ep = torch.load(str(dir_10ep / "last.pt"), weights_only=False)
    assert ckpt_10ep["epoch"] == 10
    assert ckpt_10ep["batch_cursor"] == 0
    assert ckpt_10ep["epoch_targets_seen"] == 0
    assert ckpt_10ep["global_step"] == 30  # 10 epochs * ceil(9/4) = 30 steps


# ============================================================================
# Test 13: Full Model, Optimizer, Scheduler, RNG Next Update Roundtrip
# ============================================================================

def test_full_model_optimizer_scheduler_rng_next_update_roundtrip(tmp_path: Path):
    torch.manual_seed(42)
    policy1 = TinyNamedPolicy()
    opt1, sched1 = create_native_optimizer_and_scheduler(policy1, total_steps=50, warmup_steps=5)

    # Step 1 on policy1
    sample1 = {"inputs": torch.randn(4, 16), "targets": torch.randn(4, 24)}
    loss1 = policy1(sample1)["loss_sum"]
    loss1.backward()
    clip_parameter_groups_norm(list(policy1.parameters()), max_norm=1.0)
    opt1.step()
    sched1.step()

    # Save checkpoint at step 1
    src_meta = {"checkpoint_sha256": "fake_sha"}
    tr_contract = {
        "format": "native_multiframe_v1",
        "epochs": 2,
        "global_batch_size": 4,
        "world_size": 1,
        "seed": 42,
        "use_timestamps": True,
        "history_frames": 16,
        "target_frames": 8,
        "lr_vision": 1e-6,
        "lr_projector": 5e-6,
        "lr_llm": 2e-6,
        "lr_head": 1e-5,
        "warmup_updates": 5,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "total_scheduler_updates": 50,
        "gradient_checkpointing": True,
        "augmentation": True,
    }
    train_dc = {"target_count": 40, "data_fingerprint": "dfp1"}
    val_dc = {"target_count": 10, "data_fingerprint": "dfp2"}

    ckpt_path = save_native_checkpoint(
        output_dir=tmp_path,
        filename="last.pt",
        policy=policy1,
        config={},
        norm_stats={},
        training_contract=tr_contract,
        train_data_contract=train_dc,
        val_data_contract=val_dc,
        optimizer=opt1,
        scheduler=sched1,
        global_step=1,
        epoch=0,
        batch_cursor=4,
        epoch_targets_seen=16,
        source_metadata=src_meta,
        device=torch.device("cpu"),
    )

    # Now step 2 on policy1 directly - WITHOUT manual reseed
    sample2 = {"inputs": torch.randn(4, 16), "targets": torch.randn(4, 24)}
    opt1.zero_grad()
    loss2 = policy1(sample2)["loss_sum"]
    loss2.backward()
    clip_parameter_groups_norm(list(policy1.parameters()), max_norm=1.0)
    opt1.step()
    sched1.step()

    # Resume on fresh policy2
    policy2 = TinyNamedPolicy()
    opt2, sched2 = create_native_optimizer_and_scheduler(policy2, total_steps=50, warmup_steps=5)
    load_native_training_checkpoint(
        checkpoint_path=ckpt_path,
        policy=policy2,
        optimizer=opt2,
        scheduler=sched2,
        expected_source_metadata=src_meta,
        expected_training_contract=tr_contract,
        expected_train_data_contract=train_dc,
        expected_val_data_contract=val_dc,
        device=torch.device("cpu"),
    )

    # Step 2 on policy2 - WITHOUT manual reseed (restored RNG from checkpoint)
    sample2_resume = {"inputs": torch.randn(4, 16), "targets": torch.randn(4, 24)}
    assert torch.equal(sample2["inputs"], sample2_resume["inputs"]), "Inputs mismatch; RNG was not properly restored!"
    assert torch.equal(sample2["targets"], sample2_resume["targets"]), "Targets mismatch; RNG was not properly restored!"

    opt2.zero_grad()
    loss2_resume = policy2(sample2_resume)["loss_sum"]
    loss2_resume.backward()
    clip_parameter_groups_norm(list(policy2.parameters()), max_norm=1.0)
    opt2.step()
    sched2.step()

    # Verify identical weights after step 2
    for (n1, p1), (n2, p2) in zip(policy1.named_parameters(), policy2.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1, p2, atol=1e-7), f"Mismatch on parameter {n1} after resumed step 2!"
