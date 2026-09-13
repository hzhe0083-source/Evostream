"""Unit and integration tests for stream replay protocol transition and execution.

Tests cover:
1. CLI validation: --transition-from mutually exclusive with --resume,
   --transition-from requires --stream-protocol stream_replay_v1,
   --transition-from requires new empty output directory.
2. Exact resume strictly rejects changed protocol (dense vs stream).
3. Explicit transition validates source format (only native_multiframe_v1 -> stream_replay_v1)
   and strictly rejects before model/optimizer mutation if any other contract changes
   (lr, budget, source sha, base data contracts, target geometry).
4. Stage lineage tracking: parent path, SHA256 (rank 0 hash & broadcast), from/to protocol,
   parent global_step, epoch, cursor; preserved on subsequent exact resume and saves.
   Original 93k source checkpoint SHA256 remains unchanged.
5. Direct transition optimizer moments, step, scheduler last_epoch, and per-rank RNG restoration
   using actual autograd loss on TinyNamedPolicy; lineage preserved on subsequent exact resume.
6. Parameterized contract mutation rejection tests for exact resume before model/optimizer mutation.
7. Matched-history diagnostic test with NativeSequencePolicy: single-obs current-only group isolation
   matching separate queries, strict validation of required fields and label/count mismatch failures.
8. Gloo 2-rank collective matched-history diagnostic execution: partitioned evaluation, all-gather
   aggregation, output file written only by rank 0.
9. Stop-file polling: before first update, at partial epoch boundary, and at epoch boundary
   (cursor == total_segments without prematurely advancing epoch before full val).
   Stop file is not automatically deleted.
10. Uninterrupted vs resumed newstage CPU execution parity using deterministic toy model & data,
    verifying model weights, optimizer moments/step, scheduler state, and per-rank RNG states.
11. Launcher script run_stream_10epochs.sh wraps run_native_10epochs.sh with --stream-protocol stream_replay_v1.
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
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import transformers
import torchvision

from fabri_moss.stream_protocol import get_stream_protocol_contract
from fabri_moss.tests.test_native_training import (
    make_sample,
    make_tiny_training_policy,
)
from fabri_moss.tests.test_train_native import (
    MockNativeSequencePolicy,
    TinyNamedPolicy,
)
from fabri_moss.train_native import (
    check_stop_file_requested,
    classify_parameter,
    clip_parameter_groups_norm,
    compute_file_sha256,
    create_native_optimizer_and_scheduler,
    evaluate_matched_history_diagnostics,
    evaluate_native,
    load_native_training_checkpoint,
    main,
    parse_args,
    save_native_checkpoint,
    transition_native_checkpoint,
    validate_cli_arguments,
)


# ============================================================================
# Test Dataset Fixtures
# ============================================================================

class MockStreamTrainingDataset:
    """Mock dataset providing get_base_data_contract and get_data_contract.

    Uses a local torch.Generator(seed=self.seed + idx) for each segment fetch
    to produce deterministic, identical samples across dense and stream fetches.
    """

    def __init__(
        self,
        root: str = "",
        norm_stats: Any = None,
        history_frames: int = 16,
        target_frames: int = 8,
        horizon: int = 50,
        state_dim: int = 24,
        action_dim: int = 24,
        split: str = "train",
        seed: int = 42,
        val_fraction: float = 0.1,
        max_episodes: Optional[int] = None,
        augmentation: bool = True,
        stream_protocol: Optional[str] = None,
        **kwargs: Any,
    ):
        self.split = split
        self.seed = seed
        self.history_frames = history_frames
        self.target_frames = target_frames
        self.horizon = horizon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.stream_protocol = stream_protocol
        self.epoch = 0
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
        g = torch.Generator()
        g.manual_seed(self.seed + idx)
        inputs = torch.randn(count, 16, generator=g)
        targets = torch.randn(count, 24, generator=g)
        state = torch.zeros(count, 24)
        state_mask = torch.ones(count, 24)
        action_mask = torch.ones(count, 24)
        images_window = [torch.randn(1, 16, generator=g) for _ in range(count)]
        frame_ids = list(range(count))
        observation_times = [float(t) for t in range(count)]
        target_indices = list(range(count))
        replay_groups = [
            {"observation_indices": [i], "target_positions": [i]}
            for i in range(count)
        ]
        return {
            "inputs": inputs,
            "targets": targets,
            "actions": targets,
            "target_count": count,
            "state": state,
            "state_mask": state_mask,
            "action_mask": action_mask,
            "prompt": "pick up the red object",
            "episode_id": 0,
            "target_frame_ids": list(range(count)),
            "images_window": images_window,
            "frame_ids": frame_ids,
            "observation_times": observation_times,
            "target_indices": target_indices,
            "stream_layout_mode": "stream" if self.stream_protocol is not None else "dense",
            "replay_groups": replay_groups,
        }

    def get_base_data_contract(self) -> Dict[str, Any]:
        return {
            "dataset_class": "NativeTrainingDataset",
            "context_mode": "native_causal_segments",
            "split": self.split,
            "target_count": self._total_targets,
            "data_fingerprint": f"base_dfp_{self.split}",
            "seed": self.seed,
            "val_fraction": 0.1,
            "history_frames": self.history_frames,
            "target_frames": self.target_frames,
        }

    def get_data_contract(self) -> Dict[str, Any]:
        contract = self.get_base_data_contract()
        if self.stream_protocol is not None:
            contract["stream_protocol"] = get_stream_protocol_contract()
            contract["data_fingerprint"] = f"stream_dfp_{self.split}"
        return contract

    def validation_indices(self, per_task: int = 1) -> List[int]:
        return list(range(min(4, len(self.segments))))


# ============================================================================
# 1. CLI Validation Tests
# ============================================================================

def test_cli_transition_and_resume_mutually_exclusive():
    args = parse_args([
        "--output-dir", "/tmp/test",
        "--resume", "/tmp/ckpt1.pt",
        "--transition-from", "/tmp/ckpt2.pt",
    ])
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_cli_arguments(args)


def test_cli_transition_requires_stream_protocol():
    args = parse_args([
        "--output-dir", "/tmp/test",
        "--transition-from", "/tmp/ckpt.pt",
    ])
    with pytest.raises(ValueError, match="--stream-protocol stream_replay_v1"):
        validate_cli_arguments(args)


def test_cli_stream_protocol_valid_and_invalid():
    # Valid
    args_valid = parse_args([
        "--output-dir", "/tmp/test",
        "--stream-protocol", "stream_replay_v1",
    ])
    assert args_valid.stream_protocol == "stream_replay_v1"

    # Invalid choice rejected by argparse
    with pytest.raises(SystemExit):
        parse_args([
            "--output-dir", "/tmp/test",
            "--stream-protocol", "invalid_proto",
        ])


def test_cli_transition_requires_empty_outdir(tmp_path: Path):
    out_dir = tmp_path / "non_empty_dir"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "some_file.txt").write_text("existing")

    args = [
        "--output-dir", str(out_dir),
        "--stream-protocol", "stream_replay_v1",
        "--transition-from", str(tmp_path / "ckpt.pt"),
    ]
    with pytest.raises(FileExistsError, match="NEW EMPTY output directory"):
        main(args, device=torch.device("cpu"))


# ============================================================================
# 2. Exact Resume Rejects Changed Protocol
# ============================================================================

def test_exact_resume_rejects_protocol_mismatch(tmp_path: Path):
    policy = TinyNamedPolicy()
    opt, sched = create_native_optimizer_and_scheduler(policy, total_steps=10)

    src_meta = {"checkpoint_sha256": "fake_sha_93k"}
    dense_tr_contract = {
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
        "warmup_updates": 2,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "total_scheduler_updates": 10,
        "gradient_checkpointing": True,
        "augmentation": True,
    }
    dense_dc = {
        "dataset_class": "NativeTrainingDataset",
        "context_mode": "native_causal_segments",
        "target_count": 18,
        "data_fingerprint": "dfp",
        "split": "train",
        "seed": 42,
        "val_fraction": 0.1,
        "history_frames": 16,
        "target_frames": 8,
    }

    ckpt_path = save_native_checkpoint(
        output_dir=tmp_path,
        filename="dense_ckpt.pt",
        policy=policy,
        config={},
        norm_stats={},
        training_contract=dense_tr_contract,
        train_data_contract=dense_dc,
        val_data_contract=dense_dc,
        optimizer=opt,
        scheduler=sched,
        global_step=1,
        epoch=0,
        batch_cursor=4,
        epoch_targets_seen=8,
        source_metadata=src_meta,
        device=torch.device("cpu"),
    )

    # Expected training contract for stream replay
    stream_tr_contract = copy.deepcopy(dense_tr_contract)
    stream_tr_contract["format"] = "native_stream_replay_v1"
    stream_tr_contract["stream_protocol"] = get_stream_protocol_contract()

    # Exact resume must reject because format and stream_protocol differ
    with pytest.raises(ValueError, match="Training contract mismatch"):
        load_native_training_checkpoint(
            checkpoint_path=ckpt_path,
            policy=policy,
            optimizer=opt,
            scheduler=sched,
            expected_source_metadata=src_meta,
            expected_training_contract=stream_tr_contract,
            expected_train_data_contract=dense_dc,
            expected_val_data_contract=dense_dc,
            device=torch.device("cpu"),
            mode="resume",
        )


# ============================================================================
# 3. Transition Validation and Contract Rejection Before Mutation
# ============================================================================

def test_transition_rejects_any_contract_change_before_mutation(tmp_path: Path):
    policy = TinyNamedPolicy()
    opt, sched = create_native_optimizer_and_scheduler(policy, total_steps=10)

    src_meta = {"checkpoint_sha256": "fake_sha_93k"}
    dense_tr_contract = {
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
        "warmup_updates": 2,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "total_scheduler_updates": 10,
        "gradient_checkpointing": True,
        "augmentation": True,
    }
    dense_dc = {
        "dataset_class": "NativeTrainingDataset",
        "context_mode": "native_causal_segments",
        "target_count": 18,
        "data_fingerprint": "base_dfp_train",
        "seed": 42,
        "val_fraction": 0.1,
        "history_frames": 16,
        "target_frames": 8,
        "split": "train",
    }
    val_dc = dict(dense_dc, split="val", data_fingerprint="base_dfp_val", target_count=8)

    ckpt_path = save_native_checkpoint(
        output_dir=tmp_path,
        filename="dense_parent.pt",
        policy=policy,
        config={},
        norm_stats={},
        training_contract=dense_tr_contract,
        train_data_contract=dense_dc,
        val_data_contract=val_dc,
        optimizer=opt,
        scheduler=sched,
        global_step=1,
        epoch=0,
        batch_cursor=4,
        epoch_targets_seen=8,
        source_metadata=src_meta,
        device=torch.device("cpu"),
    )

    exp_stream_tr = copy.deepcopy(dense_tr_contract)
    exp_stream_tr["format"] = "native_stream_replay_v1"
    exp_stream_tr["stream_protocol"] = get_stream_protocol_contract()

    # Clone target model weights to verify NO mutation occurs on failure
    fresh_policy = TinyNamedPolicy()
    for p in fresh_policy.parameters():
        p.data.fill_(99.0)
    initial_p_val = fresh_policy.embedder.model.mlp1.weight[0, 0].item()

    fresh_opt, fresh_sched = create_native_optimizer_and_scheduler(fresh_policy, total_steps=10)

    # 1. Reject if target lr is changed
    bad_stream_tr_lr = copy.deepcopy(exp_stream_tr)
    bad_stream_tr_lr["lr_head"] = 9e-3
    with pytest.raises(ValueError, match="lr_head"):
        transition_native_checkpoint(
            checkpoint_path=ckpt_path,
            policy=fresh_policy,
            optimizer=fresh_opt,
            scheduler=fresh_sched,
            expected_source_metadata=src_meta,
            expected_training_contract=bad_stream_tr_lr,
            expected_train_data_contract=dense_dc,
            expected_val_data_contract=val_dc,
            stream_protocol="stream_replay_v1",
            device=torch.device("cpu"),
        )
    assert fresh_policy.embedder.model.mlp1.weight[0, 0].item() == initial_p_val

    # 2. Reject if source SHA mismatch
    with pytest.raises(ValueError, match="Source checkpoint SHA256 mismatch"):
        transition_native_checkpoint(
            checkpoint_path=ckpt_path,
            policy=fresh_policy,
            optimizer=fresh_opt,
            scheduler=fresh_sched,
            expected_source_metadata={"checkpoint_sha256": "wrong_sha"},
            expected_training_contract=exp_stream_tr,
            expected_train_data_contract=dense_dc,
            expected_val_data_contract=val_dc,
            stream_protocol="stream_replay_v1",
            device=torch.device("cpu"),
        )
    assert fresh_policy.embedder.model.mlp1.weight[0, 0].item() == initial_p_val

    # 3. Reject if base train data contract changed
    bad_dc = copy.deepcopy(dense_dc)
    bad_dc["target_count"] = 999
    with pytest.raises(ValueError, match="train_data_contract mismatch"):
        transition_native_checkpoint(
            checkpoint_path=ckpt_path,
            policy=fresh_policy,
            optimizer=fresh_opt,
            scheduler=fresh_sched,
            expected_source_metadata=src_meta,
            expected_training_contract=exp_stream_tr,
            expected_train_data_contract=bad_dc,
            expected_val_data_contract=val_dc,
            stream_protocol="stream_replay_v1",
            device=torch.device("cpu"),
        )
    assert fresh_policy.embedder.model.mlp1.weight[0, 0].item() == initial_p_val

    # 4. Reject if base val data contract changed
    bad_val_dc = copy.deepcopy(val_dc)
    bad_val_dc["target_count"] = 999
    with pytest.raises(ValueError, match="val_data_contract mismatch"):
        transition_native_checkpoint(
            checkpoint_path=ckpt_path,
            policy=fresh_policy,
            optimizer=fresh_opt,
            scheduler=fresh_sched,
            expected_source_metadata=src_meta,
            expected_training_contract=exp_stream_tr,
            expected_train_data_contract=dense_dc,
            expected_val_data_contract=bad_val_dc,
            stream_protocol="stream_replay_v1",
            device=torch.device("cpu"),
        )
    assert fresh_policy.embedder.model.mlp1.weight[0, 0].item() == initial_p_val

    # 5. Reject if budget (total_scheduler_updates) changed
    bad_budget_tr = copy.deepcopy(exp_stream_tr)
    bad_budget_tr["total_scheduler_updates"] = 999
    with pytest.raises(ValueError, match="total_scheduler_updates"):
        transition_native_checkpoint(
            checkpoint_path=ckpt_path,
            policy=fresh_policy,
            optimizer=fresh_opt,
            scheduler=fresh_sched,
            expected_source_metadata=src_meta,
            expected_training_contract=bad_budget_tr,
            expected_train_data_contract=dense_dc,
            expected_val_data_contract=val_dc,
            stream_protocol="stream_replay_v1",
            device=torch.device("cpu"),
        )
    assert fresh_policy.embedder.model.mlp1.weight[0, 0].item() == initial_p_val


# ============================================================================
# 4. Stage Lineage Tracking and Preservation
# ============================================================================

def test_stage_lineage_tracking_and_preservation(tmp_path: Path):
    policy = TinyNamedPolicy()
    opt, sched = create_native_optimizer_and_scheduler(policy, total_steps=10)
    opt.step()
    sched.step()  # Align last_epoch=1 with global_step=1

    src_meta = {"checkpoint_sha256": "fake_sha_93k"}
    dense_tr_contract = {
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
        "warmup_updates": 2,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "total_scheduler_updates": 10,
        "gradient_checkpointing": True,
        "augmentation": True,
    }
    dense_dc = {
        "dataset_class": "NativeTrainingDataset",
        "context_mode": "native_causal_segments",
        "target_count": 18,
        "data_fingerprint": "base_dfp_train",
        "seed": 42,
        "val_fraction": 0.1,
        "history_frames": 16,
        "target_frames": 8,
        "split": "train",
    }
    val_dc = dict(dense_dc, split="val", data_fingerprint="base_dfp_val", target_count=8)

    parent_path = save_native_checkpoint(
        output_dir=tmp_path,
        filename="dense_parent.pt",
        policy=policy,
        config={"dim": 64},
        norm_stats={"mean": 0.0},
        training_contract=dense_tr_contract,
        train_data_contract=dense_dc,
        val_data_contract=val_dc,
        optimizer=opt,
        scheduler=sched,
        global_step=1,
        epoch=0,
        batch_cursor=4,
        epoch_targets_seen=8,
        source_metadata=src_meta,
        device=torch.device("cpu"),
    )

    parent_sha = compute_file_sha256(parent_path)

    exp_stream_tr = copy.deepcopy(dense_tr_contract)
    exp_stream_tr["format"] = "native_stream_replay_v1"
    exp_stream_tr["stream_protocol"] = get_stream_protocol_contract()

    # Transition
    target_policy = TinyNamedPolicy()
    target_opt, target_sched = create_native_optimizer_and_scheduler(target_policy, total_steps=10)

    res_info = transition_native_checkpoint(
        checkpoint_path=parent_path,
        policy=target_policy,
        optimizer=target_opt,
        scheduler=target_sched,
        expected_source_metadata=src_meta,
        expected_training_contract=exp_stream_tr,
        expected_train_data_contract=dense_dc,
        expected_val_data_contract=val_dc,
        stream_protocol="stream_replay_v1",
        device=torch.device("cpu"),
        parent_checkpoint_sha256=parent_sha,
    )

    # Lineage must be present in transition output
    lineage = res_info["stage_lineage"]
    assert lineage is not None
    assert lineage["parent_checkpoint_path"] == str(parent_path.resolve())
    assert lineage["parent_checkpoint_sha256"] == parent_sha
    assert lineage["from_protocol"] == "native_multiframe_v1"
    assert lineage["to_protocol"] == "stream_replay_v1"
    assert lineage["parent_global_step"] == 1
    assert lineage["parent_epoch"] == 0
    assert lineage["parent_batch_cursor"] == 4

    # Save newstage checkpoint
    newstage_dir = tmp_path / "newstage"
    newstage_dir.mkdir(parents=True, exist_ok=True)
    newstage_meta = dict(src_meta)
    newstage_meta["stage_lineage"] = lineage

    stream_dc = copy.deepcopy(dense_dc)
    stream_dc["stream_protocol"] = get_stream_protocol_contract()
    stream_val_dc = copy.deepcopy(val_dc)
    stream_val_dc["stream_protocol"] = get_stream_protocol_contract()

    target_sched.step()  # Align last_epoch=2 with global_step=2
    ckpt_step2 = save_native_checkpoint(
        output_dir=newstage_dir,
        filename="stream_step2.pt",
        policy=target_policy,
        config={"dim": 64},
        norm_stats={"mean": 0.0},
        training_contract=exp_stream_tr,
        train_data_contract=stream_dc,
        val_data_contract=stream_val_dc,
        optimizer=target_opt,
        scheduler=target_sched,
        global_step=2,
        epoch=0,
        batch_cursor=8,
        epoch_targets_seen=16,
        source_metadata=newstage_meta,
        device=torch.device("cpu"),
    )

    # Read back saved checkpoint and verify lineage is persisted
    saved_payload = torch.load(str(ckpt_step2), weights_only=False)
    assert saved_payload["source_metadata"]["checkpoint_sha256"] == "fake_sha_93k"
    assert saved_payload["stage_lineage"] == lineage
    assert saved_payload["source_metadata"]["stage_lineage"] == lineage

    # Resume from stream_step2.pt and verify lineage is recovered
    resume_policy = TinyNamedPolicy()
    resume_opt, resume_sched = create_native_optimizer_and_scheduler(resume_policy, total_steps=10)
    res_loaded = load_native_training_checkpoint(
        checkpoint_path=ckpt_step2,
        policy=resume_policy,
        optimizer=resume_opt,
        scheduler=resume_sched,
        expected_source_metadata=newstage_meta,
        expected_training_contract=exp_stream_tr,
        expected_train_data_contract=stream_dc,
        expected_val_data_contract=stream_val_dc,
        device=torch.device("cpu"),
        mode="resume",
    )
    assert res_loaded["stage_lineage"] == lineage


# ============================================================================
# 5. Direct Transition Optimizer, Scheduler, RNG, and Lineage Test
# ============================================================================

def test_direct_transition_optimizer_scheduler_rng_lineage(tmp_path: Path):
    """Verify transition restores model weights, populated AdamW moments for all groups,

    scheduler last_epoch, and per-rank RNG sequence using autograd loss on TinyNamedPolicy.
    """
    policy = TinyNamedPolicy()
    opt, sched = create_native_optimizer_and_scheduler(policy, total_steps=10, warmup_steps=2)

    # Run real backward to populate AdamW moments (exp_avg, exp_avg_sq, step) for every group
    loss = sum((p ** 2).sum() for p in policy.parameters())
    loss.backward()
    opt.step()
    sched.step()

    # Confirm all trainable parameters have populated moments in optimizer state
    assert len(opt.state) > 0
    for p in policy.parameters():
        if p.requires_grad:
            assert p in opt.state
            assert "exp_avg" in opt.state[p]
            assert "exp_avg_sq" in opt.state[p]
            assert opt.state[p]["step"] == 1

    # Record deterministic RNG seed states
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)

    src_meta = {"checkpoint_sha256": "source_sha_93000"}
    dense_tr_contract = {
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
        "warmup_updates": 2,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "total_scheduler_updates": 10,
        "gradient_checkpointing": True,
        "augmentation": True,
    }
    dense_dc = {
        "dataset_class": "NativeTrainingDataset",
        "context_mode": "native_causal_segments",
        "target_count": 18,
        "data_fingerprint": "base_dfp_train",
        "seed": 42,
        "val_fraction": 0.1,
        "history_frames": 16,
        "target_frames": 8,
        "split": "train",
    }
    val_dc = dict(dense_dc, split="val", data_fingerprint="base_dfp_val", target_count=8)

    parent_path = save_native_checkpoint(
        output_dir=tmp_path,
        filename="parent_ckpt_step1.pt",
        policy=policy,
        config={"dim": 64},
        norm_stats={"mean": 0.0},
        training_contract=dense_tr_contract,
        train_data_contract=dense_dc,
        val_data_contract=val_dc,
        optimizer=opt,
        scheduler=sched,
        global_step=1,
        epoch=0,
        batch_cursor=4,
        epoch_targets_seen=8,
        source_metadata=src_meta,
        device=torch.device("cpu"),
    )
    parent_sha = compute_file_sha256(parent_path)
    assert parent_sha is not None and len(parent_sha) == 64

    # Mutate in-memory RNG after saving to ensure transition actively restores saved state
    _ = random.random(), np.random.rand(), torch.randn(5)

    exp_stream_tr = copy.deepcopy(dense_tr_contract)
    exp_stream_tr["format"] = "native_stream_replay_v1"
    exp_stream_tr["stream_protocol"] = get_stream_protocol_contract()

    target_policy = TinyNamedPolicy()
    target_opt, target_sched = create_native_optimizer_and_scheduler(target_policy, total_steps=10, warmup_steps=2)

    # Transition without pre-supplying parent hash so it is actively computed
    res_info = transition_native_checkpoint(
        checkpoint_path=parent_path,
        policy=target_policy,
        optimizer=target_opt,
        scheduler=target_sched,
        expected_source_metadata=src_meta,
        expected_training_contract=exp_stream_tr,
        expected_train_data_contract=dense_dc,
        expected_val_data_contract=val_dc,
        stream_protocol="stream_replay_v1",
        device=torch.device("cpu"),
    )

    lineage = res_info["stage_lineage"]
    assert lineage is not None
    assert lineage["parent_checkpoint_sha256"] == parent_sha
    assert lineage["parent_checkpoint_path"] == str(parent_path.resolve())
    assert lineage["from_protocol"] == "native_multiframe_v1"
    assert lineage["to_protocol"] == "stream_replay_v1"
    assert lineage["parent_global_step"] == 1

    # Compare model state
    for k, v in policy.state_dict().items():
        assert torch.equal(target_policy.state_dict()[k], v)

    # Compare optimizer state & moments
    assert len(target_opt.param_groups) == len(opt.param_groups)
    for target_p, p in zip(target_policy.parameters(), policy.parameters()):
        if p.requires_grad:
            assert target_p in target_opt.state
            assert opt.state[p]["step"] == target_opt.state[target_p]["step"]
            assert torch.equal(opt.state[p]["exp_avg"], target_opt.state[target_p]["exp_avg"])
            assert torch.equal(opt.state[p]["exp_avg_sq"], target_opt.state[target_p]["exp_avg_sq"])

    # Compare scheduler state
    assert target_sched.last_epoch == sched.last_epoch == 1

    # Verify restored RNG reproduces exact expected next random outputs
    restored_py = random.random()
    restored_np = np.random.rand()
    restored_th = torch.randn(5)

    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    expected_py = random.random()
    expected_np = np.random.rand()
    expected_th = torch.randn(5)

    assert restored_py == expected_py
    assert restored_np == expected_np
    assert torch.equal(restored_th, expected_th)

    # Save newstage checkpoint at step 2 and resume
    stream_run_dir = tmp_path / "stream_run"
    stream_run_dir.mkdir(parents=True, exist_ok=True)
    stream_dc = copy.deepcopy(dense_dc)
    stream_dc["stream_protocol"] = get_stream_protocol_contract()
    stream_val_dc = copy.deepcopy(val_dc)
    stream_val_dc["stream_protocol"] = get_stream_protocol_contract()

    target_sched.step()
    ckpt_path_step2 = save_native_checkpoint(
        output_dir=stream_run_dir,
        filename="stream_step2.pt",
        policy=target_policy,
        config={"dim": 64},
        norm_stats={"mean": 0.0},
        training_contract=exp_stream_tr,
        train_data_contract=stream_dc,
        val_data_contract=stream_val_dc,
        optimizer=target_opt,
        scheduler=target_sched,
        global_step=2,
        epoch=0,
        batch_cursor=8,
        epoch_targets_seen=16,
        source_metadata=dict(src_meta, stage_lineage=lineage),
        device=torch.device("cpu"),
    )

    resume_policy = TinyNamedPolicy()
    resume_opt, resume_sched = create_native_optimizer_and_scheduler(resume_policy, total_steps=10, warmup_steps=2)
    res_loaded = load_native_training_checkpoint(
        checkpoint_path=ckpt_path_step2,
        policy=resume_policy,
        optimizer=resume_opt,
        scheduler=resume_sched,
        expected_source_metadata=dict(src_meta, stage_lineage=lineage),
        expected_training_contract=exp_stream_tr,
        expected_train_data_contract=stream_dc,
        expected_val_data_contract=stream_val_dc,
        device=torch.device("cpu"),
        mode="resume",
    )
    assert res_loaded["stage_lineage"] == lineage


# ============================================================================
# 6. Parameterized Contract Mutation Rejection Tests on Exact Resume
# ============================================================================

@pytest.mark.parametrize(
    "contract_type, key, altered_value, match_text",
    [
        ("training_contract", "format", "native_multiframe_v1", "format"),
        ("training_contract", "stream_protocol", {"protocol": "unknown"}, "stream_protocol"),
        ("training_contract", "total_scheduler_updates", 9999, "total_scheduler_updates"),
        ("training_contract", "epochs", 99, "epochs"),
        ("training_contract", "seed", 999, "seed"),
        ("training_contract", "target_frames", 99, "target_frames"),
        ("train_data_contract", "split", "val", "split"),
        ("train_data_contract", "data_fingerprint", "altered_dfp", "data_fingerprint"),
    ],
)
def test_exact_resume_contract_rejections_parameterized(
    tmp_path: Path, contract_type: str, key: str, altered_value: Any, match_text: str
):
    """Verify exact resume strictly rejects altered training and data contracts before mutating model/opt state."""
    policy = TinyNamedPolicy()
    opt, sched = create_native_optimizer_and_scheduler(policy, total_steps=10)
    loss = sum((p ** 2).sum() for p in policy.parameters())
    loss.backward()
    opt.step()
    sched.step()

    src_meta = {"checkpoint_sha256": "source_sha_exact"}
    tr_contract = {
        "format": "native_stream_replay_v1",
        "stream_protocol": get_stream_protocol_contract(),
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
        "warmup_updates": 2,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "total_scheduler_updates": 10,
        "gradient_checkpointing": True,
        "augmentation": True,
    }
    train_dc = {
        "dataset_class": "NativeTrainingDataset",
        "context_mode": "native_causal_segments",
        "target_count": 18,
        "data_fingerprint": "stream_dfp_train",
        "stream_protocol": get_stream_protocol_contract(),
        "seed": 42,
        "val_fraction": 0.1,
        "history_frames": 16,
        "target_frames": 8,
        "split": "train",
    }
    val_dc = dict(train_dc, split="val", data_fingerprint="stream_dfp_val", target_count=8)

    ckpt_path = save_native_checkpoint(
        output_dir=tmp_path,
        filename="resume_test_ckpt.pt",
        policy=policy,
        config={"dim": 64},
        norm_stats={"mean": 0.0},
        training_contract=tr_contract,
        train_data_contract=train_dc,
        val_data_contract=val_dc,
        optimizer=opt,
        scheduler=sched,
        global_step=1,
        epoch=0,
        batch_cursor=4,
        epoch_targets_seen=8,
        source_metadata=src_meta,
        device=torch.device("cpu"),
    )

    # Build expected contracts with the single altered field
    exp_tr = copy.deepcopy(tr_contract)
    exp_train_dc = copy.deepcopy(train_dc)
    exp_val_dc = copy.deepcopy(val_dc)

    if contract_type == "training_contract":
        exp_tr[key] = altered_value
    elif contract_type == "train_data_contract":
        exp_train_dc[key] = altered_value

    # Target candidate policy with known distinct weights
    candidate_policy = TinyNamedPolicy()
    for p in candidate_policy.parameters():
        p.data.fill_(123.456)
    candidate_opt, candidate_sched = create_native_optimizer_and_scheduler(candidate_policy, total_steps=10)

    with pytest.raises(ValueError, match=match_text):
        load_native_training_checkpoint(
            checkpoint_path=ckpt_path,
            policy=candidate_policy,
            optimizer=candidate_opt,
            scheduler=candidate_sched,
            expected_source_metadata=src_meta,
            expected_training_contract=exp_tr,
            expected_train_data_contract=exp_train_dc,
            expected_val_data_contract=exp_val_dc,
            device=torch.device("cpu"),
            mode="resume",
        )

    # Verify no mutation occurred on candidate policy
    for p in candidate_policy.parameters():
        assert torch.all(p.data == 123.456)


# ============================================================================
# 7. Matched History Diagnostics Isolation and Mismatch Validation
# ============================================================================

def test_matched_history_diagnostics_isolation_and_mismatch_validation(tmp_path: Path):
    """Test NativeSequencePolicy current-only single-group isolation vs isolated queries,

    and verify evaluate_matched_history_diagnostics fails on count/label mismatches or missing keys.
    """
    # 1. NativeSequencePolicy current-only replay groups equality with isolated single-target queries
    seq_policy = make_tiny_training_policy(shallow_layer=1, gradient_checkpointing=False)
    sample = make_sample(N=4, target_indices=(1, 3), horizon=2, action_dim=4)
    sample["state"] = torch.zeros(2, 4)
    sample["state_mask"] = torch.ones(2, 4)
    tgt_indices = sample["target_indices"]

    curr_sample = {
        "images_window": [sample["images_window"][i] for i in tgt_indices],
        "frame_ids": [sample["frame_ids"][i] for i in tgt_indices],
        "observation_times": [sample["observation_times"][i] for i in tgt_indices],
        "prompt": sample["prompt"],
        "target_indices": list(range(len(tgt_indices))),
        "actions": sample["actions"],
        "action_mask": sample["action_mask"],
        "state": sample["state"],
        "state_mask": sample["state_mask"],
        "replay_groups": [
            {"observation_indices": [i], "target_positions": [i]}
            for i in range(len(tgt_indices))
        ],
    }

    curr_deep, curr_shallow = seq_policy.features(curr_sample)

    for local_idx, orig_tidx in enumerate(tgt_indices):
        iso_sample = {
            "images_window": [sample["images_window"][orig_tidx]],
            "frame_ids": [sample["frame_ids"][orig_tidx]],
            "observation_times": [sample["observation_times"][orig_tidx]],
            "prompt": sample["prompt"],
            "target_indices": [0],
            "actions": sample["actions"][local_idx : local_idx + 1],
            "action_mask": sample["action_mask"][local_idx : local_idx + 1],
            "state": sample["state"][local_idx : local_idx + 1],
            "state_mask": sample["state_mask"][local_idx : local_idx + 1],
        }
        iso_deep, iso_shallow = seq_policy.features(iso_sample)
        assert torch.allclose(curr_deep[local_idx : local_idx + 1], iso_deep, atol=1e-5)
        assert torch.allclose(curr_shallow[local_idx : local_idx + 1], iso_shallow, atol=1e-5)

    # 2. Strict validation of evaluate_matched_history_diagnostics errors
    toy_policy = TinyNamedPolicy()
    base_val_ds = MockStreamTrainingDataset(split="val", stream_protocol="stream_replay_v1")
    dense_val_ds = MockStreamTrainingDataset(split="val", stream_protocol=None)

    # Missing state key in stream sample
    class MissingKeyDataset(MockStreamTrainingDataset):
        def __getitem__(self, idx: int):
            item = super().__getitem__(idx)
            del item["state"]
            return item

    with pytest.raises(KeyError, match="state"):
        evaluate_matched_history_diagnostics(
            model=toy_policy,
            val_dataset=MissingKeyDataset(split="val", stream_protocol="stream_replay_v1"),
            dense_val_dataset=dense_val_ds,
            eval_indices=[0],
            device=torch.device("cpu"),
            output_dir=tmp_path,
            global_step=1,
            epoch=0,
        )

    # Target frame IDs mismatch
    class MismatchedFrameDataset(MockStreamTrainingDataset):
        def __getitem__(self, idx: int):
            item = super().__getitem__(idx)
            item["frame_ids"] = [999, 1000]
            return item

    with pytest.raises(AssertionError, match="Target frame IDs mismatch"):
        evaluate_matched_history_diagnostics(
            model=toy_policy,
            val_dataset=MismatchedFrameDataset(split="val", stream_protocol="stream_replay_v1"),
            dense_val_dataset=dense_val_ds,
            eval_indices=[0],
            device=torch.device("cpu"),
            output_dir=tmp_path,
            global_step=1,
            epoch=0,
        )

    # Target count mismatch
    class MismatchedCountDataset(MockStreamTrainingDataset):
        def __getitem__(self, idx: int):
            item = super().__getitem__(idx)
            item["target_count"] = 99
            return item

    with pytest.raises(AssertionError, match="Invalid or mismatched target counts"):
        evaluate_matched_history_diagnostics(
            model=toy_policy,
            val_dataset=MismatchedCountDataset(split="val", stream_protocol="stream_replay_v1"),
            dense_val_dataset=dense_val_ds,
            eval_indices=[0],
            device=torch.device("cpu"),
            output_dir=tmp_path,
            global_step=1,
            epoch=0,
        )


# ============================================================================
# 8. Gloo 2-Rank Matched History Diagnostic Collective Test
# ============================================================================

def _gloo_matched_history_worker(
    rank: int,
    world_size: int,
    init_file: str,
    out_dir_str: str,
):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    dev = torch.device("cpu")
    model = TinyNamedPolicy()

    val_dataset = MockStreamTrainingDataset(
        root="", norm_stats=None, history_frames=16, target_frames=8,
        horizon=50, state_dim=24, action_dim=24, split="val", seed=42,
        stream_protocol="stream_replay_v1",
    )
    dense_val_dataset = MockStreamTrainingDataset(
        root="", norm_stats=None, history_frames=16, target_frames=8,
        horizon=50, state_dim=24, action_dim=24, split="val", seed=42,
        stream_protocol=None,
    )

    out_dir = Path(out_dir_str)
    diag_record = evaluate_matched_history_diagnostics(
        model=model,
        val_dataset=val_dataset,
        dense_val_dataset=dense_val_dataset,
        eval_indices=[0, 1, 2, 3],
        device=dev,
        output_dir=out_dir,
        global_step=1,
        epoch=0,
        process_group=None,
    )

    if rank == 0:
        assert diag_record is not None
        assert diag_record["total_targets"] == 8
        assert len(diag_record["segment_records"]) == 4
        # Verify segment records are sorted by segment_index across both ranks
        seg_indices = [r["segment_index"] for r in diag_record["segment_records"]]
        assert seg_indices == [0, 1, 2, 3]
    else:
        assert diag_record is None

    dist.barrier()
    dist.destroy_process_group()


def test_gloo_2rank_matched_history_diagnostics(tmp_path: Path):
    """Test 2-rank Gloo collective execution of evaluate_matched_history_diagnostics."""
    init_file = str(tmp_path / "gloo_matched_init")
    out_dir = tmp_path / "matched_diag_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    mp.spawn(
        _gloo_matched_history_worker,
        args=(2, init_file, str(out_dir)),
        nprocs=2,
        join=True,
    )

    diag_file = out_dir / "history_diagnostics.jsonl"
    assert diag_file.exists()
    lines = [json.loads(line) for line in diag_file.read_text().strip().split("\n")]
    assert len(lines) == 1
    assert lines[0]["total_targets"] == 8
    assert len(lines[0]["segment_records"]) == 4


# ============================================================================
# 9. Stop-File Functionality Tests
# ============================================================================

def test_check_stop_file_requested(tmp_path: Path):
    stop_file = tmp_path / "stop.txt"
    assert not check_stop_file_requested(stop_file, device=torch.device("cpu"))

    stop_file.write_text("STOP")
    assert check_stop_file_requested(stop_file, device=torch.device("cpu"))
    # File is NOT automatically deleted
    assert stop_file.exists()


def test_toy_training_stop_file_before_training(monkeypatch, tmp_path: Path):
    base_meta = {"checkpoint_sha256": "fake_sha_toy"}
    norm_stats = {}

    def make_base_policy():
        torch.manual_seed(9999)
        return TinyNamedPolicy()

    import fabri_moss.train_native as tn
    monkeypatch.setattr(tn, "require_training_device", lambda p, d: None)
    monkeypatch.setattr("fabri_moss.runtime.load_native_checkpoint", lambda **kw: (make_base_policy(), {"dim": 64}, norm_stats, base_meta))
    monkeypatch.setattr("fabri_moss.runtime.assert_native_fa2", lambda p: {"native_fa2": True})
    monkeypatch.setattr("fabri_moss.native_training.NativeSequencePolicy", lambda **kw: MockNativeSequencePolicy(kw["policy"]))
    monkeypatch.setattr("fabri_moss.native_data.NativeTrainingDataset", MockStreamTrainingDataset)

    stop_file = tmp_path / "stop_before.signal"
    stop_file.write_text("stop")

    out_dir = tmp_path / "run_stop_before"
    args = [
        "--output-dir", str(out_dir),
        "--epochs", "2",
        "--global-batch-size", "4",
        "--workers", "0",
        "--seed", "42",
        "--stop-file", str(stop_file),
    ]

    main(args, device=torch.device("cpu"))

    # Checkpoint created at step 0, no updates executed
    assert (out_dir / "last.pt").exists()
    ckpt = torch.load(str(out_dir / "last.pt"), weights_only=False)
    assert ckpt["global_step"] == 0
    assert ckpt["epoch"] == 0
    assert ckpt["batch_cursor"] == 0
    # Stop file preserved
    assert stop_file.exists()


# ============================================================================
# 10. End-to-End CPU Toy Transition, Diagnostics & Parity
# ============================================================================

def test_e2e_cpu_dense_to_stream_transition_diagnostics_and_parity(monkeypatch, tmp_path: Path):
    base_meta = {"checkpoint_sha256": "fake_sha_toy_93k"}
    norm_stats = {}

    def make_base_policy():
        torch.manual_seed(12345)
        return TinyNamedPolicy()

    import fabri_moss.train_native as tn
    monkeypatch.setattr(tn, "require_training_device", lambda p, d: None)
    monkeypatch.setattr("fabri_moss.runtime.load_native_checkpoint", lambda **kw: (make_base_policy(), {"dim": 64}, norm_stats, base_meta))
    monkeypatch.setattr("fabri_moss.runtime.assert_native_fa2", lambda p: {"native_fa2": True})
    monkeypatch.setattr("fabri_moss.native_training.NativeSequencePolicy", lambda **kw: MockNativeSequencePolicy(kw["policy"]))
    monkeypatch.setattr("fabri_moss.native_data.NativeTrainingDataset", MockStreamTrainingDataset)

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
        "--eval-every", "2",
        "--val-per-task", "1",
        "--grad-clip", "1.0",
    ]

    # Step 1: Run Stage 1 (dense multiframe) for 2 updates
    dense_dir = tmp_path / "stage1_dense"
    args_dense = base_args.copy()
    args_dense[1] = str(dense_dir)
    args_dense.extend(["--max-updates", "2"])
    main(args_dense, device=torch.device("cpu"))

    parent_ckpt_path = dense_dir / "last.pt"
    assert parent_ckpt_path.exists()
    dense_ckpt = torch.load(str(parent_ckpt_path), weights_only=False)
    assert dense_ckpt["global_step"] == 2
    assert dense_ckpt["training_contract"]["format"] == "native_multiframe_v1"

    # Step 2: Transition from dense to stream continuous for 2 updates (total 4)
    stream_cont_dir = tmp_path / "stage2_stream_continuous"
    args_cont = base_args.copy()
    args_cont[1] = str(stream_cont_dir)
    args_cont.extend([
        "--transition-from", str(parent_ckpt_path),
        "--stream-protocol", "stream_replay_v1",
        "--max-updates", "4",
    ])
    main(args_cont, device=torch.device("cpu"))

    # Verify first_step_diagnostics.json has ACTUAL global_step=3 (recorded after first step update), start_step=2
    diag_file = stream_cont_dir / "first_step_diagnostics.json"
    assert diag_file.exists()
    diag_data = json.loads(diag_file.read_text())
    assert diag_data["global_step"] == 3
    assert diag_data["start_step"] == 2
    assert diag_data["is_transition"] is True
    assert set(diag_data["chosen_representatives"].keys()) == {"vision", "projector", "llm", "head"}

    # Verify transition initial validation logged
    metrics_file = stream_cont_dir / "train_metrics.jsonl"
    assert metrics_file.exists()
    metric_lines = [json.loads(line) for line in metrics_file.read_text().strip().split("\n")]
    init_val = next(m for m in metric_lines if m.get("type") == "transition_initial_validation")
    assert init_val["global_step"] == 2

    # Verify history diagnostics persisted
    history_diag_file = stream_cont_dir / "history_diagnostics.jsonl"
    assert history_diag_file.exists()
    history_lines = [json.loads(line) for line in history_diag_file.read_text().strip().split("\n")]
    assert len(history_lines) >= 1
    assert "stream_mean_loss" in history_lines[0]
    assert "dense_mean_loss" in history_lines[0]
    assert "current_only_mean_loss" in history_lines[0]

    # Step 3: Run Transition with 1 update (step 3), then resume with exact contract to step 4
    stream_part_dir = tmp_path / "stage2_stream_interrupted"
    args_part = base_args.copy()
    args_part[1] = str(stream_part_dir)
    args_part.extend([
        "--transition-from", str(parent_ckpt_path),
        "--stream-protocol", "stream_replay_v1",
        "--max-updates", "3",
    ])
    main(args_part, device=torch.device("cpu"))

    # Verify checkpoint immediately saved at step 3 (first newstage update)
    ckpt_part = torch.load(str(stream_part_dir / "last.pt"), weights_only=False)
    assert ckpt_part["global_step"] == 3

    # Exact resume from step 3 to step 4
    args_resume = base_args.copy()
    args_resume[1] = str(stream_part_dir)
    args_resume.extend([
        "--resume", str(stream_part_dir / "last.pt"),
        "--stream-protocol", "stream_replay_v1",
        "--max-updates", "4",
    ])
    main(args_resume, device=torch.device("cpu"))

    # Verify parity between uninterrupted transition run and interrupted+resumed transition run
    ckpt_cont = torch.load(str(stream_cont_dir / "last.pt"), weights_only=False)
    ckpt_res = torch.load(str(stream_part_dir / "last.pt"), weights_only=False)

    assert ckpt_cont["global_step"] == 4
    assert ckpt_res["global_step"] == 4
    assert ckpt_cont["epoch"] == ckpt_res["epoch"]
    assert ckpt_cont["batch_cursor"] == ckpt_res["batch_cursor"]
    assert ckpt_cont["epoch_targets_seen"] == ckpt_res["epoch_targets_seen"]

    # 1. Model weights comparison
    for k in ckpt_cont["model"]:
        assert torch.allclose(ckpt_cont["model"][k], ckpt_res["model"][k], atol=1e-5), f"Model weight mismatch on {k}"

    # 2. Optimizer state comparison (every tensor and scalar)
    cont_opt_state = ckpt_cont["optimizer"]["state"]
    res_opt_state = ckpt_res["optimizer"]["state"]
    assert set(cont_opt_state.keys()) == set(res_opt_state.keys())
    for pid in cont_opt_state:
        for sk in cont_opt_state[pid]:
            v_cont = cont_opt_state[pid][sk]
            v_res = res_opt_state[pid][sk]
            if isinstance(v_cont, torch.Tensor):
                assert torch.allclose(v_cont, v_res, atol=1e-5), f"Optimizer state tensor mismatch on pid {pid}, key {sk}"
            else:
                assert v_cont == v_res, f"Optimizer state value mismatch on pid {pid}, key {sk}"

    # 3. Scheduler state comparison
    assert ckpt_cont["scheduler"]["last_epoch"] == ckpt_res["scheduler"]["last_epoch"]

    # 4. Per-rank RNG comparison
    for r_idx in range(len(ckpt_cont["rng_states_per_rank"])):
        cont_rng = ckpt_cont["rng_states_per_rank"][r_idx]
        res_rng = ckpt_res["rng_states_per_rank"][r_idx]
        assert cont_rng["python"] == res_rng["python"]
        np_cont, np_res = cont_rng["numpy"], res_rng["numpy"]
        assert np_cont[0] == np_res[0]
        assert np.array_equal(np_cont[1], np_res[1])
        assert np_cont[2:] == np_res[2:]
        assert torch.equal(cont_rng["torch_cpu"], res_rng["torch_cpu"])

    # 5. Lineage comparison
    lineage_keys = [
        "parent_checkpoint_path",
        "parent_checkpoint_sha256",
        "from_protocol",
        "to_protocol",
        "parent_global_step",
        "parent_epoch",
        "parent_batch_cursor",
        "parent_epoch_targets_seen",
    ]
    for lk in lineage_keys:
        assert ckpt_cont["stage_lineage"][lk] == ckpt_res["stage_lineage"][lk]


# ============================================================================
# 11. Stop-File Partial Epoch & Epoch Boundary Tests
# ============================================================================

def test_stop_file_partial_and_epoch_boundary_continuation(monkeypatch, tmp_path: Path):
    base_meta = {"checkpoint_sha256": "fake_sha_toy"}
    norm_stats = {}

    def make_base_policy():
        torch.manual_seed(9999)
        return TinyNamedPolicy()

    import fabri_moss.train_native as tn
    monkeypatch.setattr(tn, "require_training_device", lambda p, d: None)
    monkeypatch.setattr("fabri_moss.runtime.load_native_checkpoint", lambda **kw: (make_base_policy(), {"dim": 64}, norm_stats, base_meta))
    monkeypatch.setattr("fabri_moss.runtime.assert_native_fa2", lambda p: {"native_fa2": True})
    monkeypatch.setattr("fabri_moss.native_training.NativeSequencePolicy", lambda **kw: MockNativeSequencePolicy(kw["policy"]))

    out_dir = tmp_path / "stop_boundary_test"
    stop_file = tmp_path / "train.stop"

    base_args = [
        "--output-dir", str(out_dir),
        "--epochs", "2",
        "--global-batch-size", "4",
        "--workers", "0",
        "--seed", "42",
        "--stop-file", str(stop_file),
    ]

    # 1. Stop at partial epoch boundary (step 1, cursor 4 < 9)
    class StopTriggerDataset(MockStreamTrainingDataset):
        call_count = 0

        def __getitem__(self, idx: int):
            if self.split == "train":
                StopTriggerDataset.call_count += 1
                if StopTriggerDataset.call_count == 4:
                    # 4 samples in step 1 consumed -> create stop file before next step
                    stop_file.write_text("STOP_STEP1")
            return super().__getitem__(idx)

    monkeypatch.setattr("fabri_moss.native_data.NativeTrainingDataset", StopTriggerDataset)

    main(base_args, device=torch.device("cpu"))

    ckpt_part = torch.load(str(out_dir / "last.pt"), weights_only=False)
    assert ckpt_part["global_step"] == 1
    assert ckpt_part["epoch"] == 0
    assert ckpt_part["batch_cursor"] == 4
    assert stop_file.exists()  # Not deleted automatically
    stop_file.unlink()  # Remove manually for next continuation

    # 2. Resume from step 1, trigger stop file at epoch boundary (step 3, cursor 9 == total_segments)
    StopTriggerDataset.call_count = 0

    def stop_at_epoch_boundary(self, idx: int):
        if self.split == "train":
            StopTriggerDataset.call_count += 1
            if StopTriggerDataset.call_count == 5:
                stop_file.write_text("STOP_EPOCH_BOUNDARY")
        return MockStreamTrainingDataset.__getitem__(self, idx)

    StopTriggerDataset.__getitem__ = stop_at_epoch_boundary

    resume_args = base_args + ["--resume", str(out_dir / "last.pt")]
    main(resume_args, device=torch.device("cpu"))

    ckpt_epoch_stop = torch.load(str(out_dir / "last.pt"), weights_only=False)
    assert ckpt_epoch_stop["global_step"] == 3
    assert ckpt_epoch_stop["epoch"] == 0
    assert ckpt_epoch_stop["batch_cursor"] == 9  # total_segments
    assert ckpt_epoch_stop["epoch_targets_seen"] == 18
    # Epoch checkpoint NOT yet saved because full validation was deferred
    assert not (out_dir / "checkpoint_epoch_001.pt").exists()
    assert stop_file.exists()
    stop_file.unlink()  # Remove manually

    # 3. Resume from cursor=9 boundary to step 4: executes full val, advances to epoch 1, completes step 4
    StopTriggerDataset.call_count = 0
    StopTriggerDataset.__getitem__ = MockStreamTrainingDataset.__getitem__

    finish_args = base_args + ["--resume", str(out_dir / "last.pt"), "--max-updates", "4"]
    main(finish_args, device=torch.device("cpu"))

    assert (out_dir / "checkpoint_epoch_001.pt").exists()
    ckpt_step4 = torch.load(str(out_dir / "last.pt"), weights_only=False)
    assert ckpt_step4["global_step"] == 4
    assert ckpt_step4["epoch"] == 1
    assert ckpt_step4["batch_cursor"] == 4


# ============================================================================
# 12. Gloo 2-Rank Collective Stop-File Synchronization Test
# ============================================================================

def _gloo_stop_file_worker(rank: int, world_size: int, init_file: str, stop_file: str, result_file: str):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    dev = torch.device("cpu")
    # Case 1: no stop file
    res1 = check_stop_file_requested(stop_file, dev)
    dist.barrier()
    # Case 2: stop file created
    if rank == 0:
        Path(stop_file).write_text("STOP_NOW")
    dist.barrier()
    res2 = check_stop_file_requested(stop_file, dev)

    with open(f"{result_file}_{rank}.json", "w") as f:
        json.dump({"rank": rank, "res1": res1, "res2": res2}, f)

    dist.barrier()
    dist.destroy_process_group()


def test_gloo_2rank_stop_file_collective_broadcast(tmp_path: Path):
    init_file = str(tmp_path / "gloo_stop_init")
    stop_file = str(tmp_path / "signal.stop")
    result_prefix = str(tmp_path / "stop_res")

    mp.spawn(
        _gloo_stop_file_worker,
        args=(2, init_file, stop_file, result_prefix),
        nprocs=2,
        join=True,
    )

    with open(f"{result_prefix}_0.json") as f:
        r0 = json.load(f)
    with open(f"{result_prefix}_1.json") as f:
        r1 = json.load(f)

    assert r0["res1"] is False and r1["res1"] is False
    assert r0["res2"] is True and r1["res2"] is True


# ============================================================================
# 13. Launcher Script Wrapper Test
# ============================================================================

def test_run_stream_10epochs_script_wrapper():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_stream_10epochs.sh"
    assert script_path.exists()
    assert os.access(script_path, os.X_OK)

    content = script_path.read_text()
    assert "run_native_10epochs.sh" in content
    assert "--stream-protocol stream_replay_v1" in content
    assert "$@" in content
