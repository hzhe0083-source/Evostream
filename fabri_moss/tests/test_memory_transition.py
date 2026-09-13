"""Unit and integration tests for memory replay protocol transition and contract verification."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import random
from typing import Any, Dict

import numpy as np
import pytest
import torch

from fabri_moss.memory_protocol import get_memory_protocol_contract
from fabri_moss.stream_protocol import get_stream_protocol_contract
from fabri_moss.tests.test_train_native import TinyNamedPolicy
from fabri_moss.train_native import (
    compute_file_sha256,
    create_native_optimizer_and_scheduler,
    get_protocol_contract_and_format,
    load_native_training_checkpoint,
    parse_args,
    save_native_checkpoint,
    transition_native_checkpoint,
    validate_cli_arguments,
)


def _make_contracts() -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    src_meta = {"checkpoint_sha256": "source_sha_93000"}
    dense_tr_contract = {
        "format": "native_multiframe_v1",
        "epochs": 10,
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
        "total_scheduler_updates": 30,
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
    return src_meta, dense_tr_contract, dense_dc, val_dc


def test_transition_native_to_memory_replay_preserves_state_and_resumes(tmp_path: Path):
    """Test 1:

    - Train TinyNamedPolicy 1 real autograd step with AdamW (loss = sum(p**2)).
    - Save dense native_multiframe_v1 checkpoint.
    - Transition to memory_replay_v1 (verifying model, AdamW exp_avg/exp_avg_sq/step, scheduler, RNG, progress, non-empty parent hash).
    - Save memory checkpoint and strict resume with same contract succeeds.
    - Exact resume targeting stream_replay_v1 strictly rejected.
    """
    policy = TinyNamedPolicy()
    opt, sched = create_native_optimizer_and_scheduler(policy, total_steps=30, warmup_steps=2)

    # 1 real autograd step: sum of all squared parameters
    loss = sum((p ** 2).sum() for p in policy.parameters())
    loss.backward()
    opt.step()
    sched.step()

    # Confirm AdamW moments populated
    assert len(opt.state) > 0
    for p in policy.parameters():
        if p.requires_grad:
            assert p in opt.state
            assert "exp_avg" in opt.state[p]
            assert "exp_avg_sq" in opt.state[p]
            assert opt.state[p]["step"] == 1

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    src_meta, dense_tr_contract, dense_dc, val_dc = _make_contracts()

    # Save original dense checkpoint
    parent_path = save_native_checkpoint(
        output_dir=tmp_path,
        filename="dense_parent_step1.pt",
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

    # Prepare target memory training contract
    mem_contract, mem_format = get_protocol_contract_and_format("memory_replay_v1")
    assert mem_format == "native_memory_replay_v1"

    exp_mem_tr = copy.deepcopy(dense_tr_contract)
    exp_mem_tr["format"] = mem_format
    exp_mem_tr["stream_protocol"] = mem_contract

    target_policy = TinyNamedPolicy()
    target_opt, target_sched = create_native_optimizer_and_scheduler(target_policy, total_steps=30, warmup_steps=2)

    # Perturb RNG before transition
    _ = random.random(), np.random.rand(), torch.randn(5)

    # Perform transition
    res_info = transition_native_checkpoint(
        checkpoint_path=parent_path,
        policy=target_policy,
        optimizer=target_opt,
        scheduler=target_sched,
        expected_source_metadata=src_meta,
        expected_training_contract=exp_mem_tr,
        expected_train_data_contract=dense_dc,
        expected_val_data_contract=val_dc,
        stream_protocol="memory_replay_v1",
        device=torch.device("cpu"),
    )

    # Verify stage lineage & non-empty parent hash
    lineage = res_info["stage_lineage"]
    assert lineage is not None
    assert lineage["parent_checkpoint_sha256"] == parent_sha
    assert len(lineage["parent_checkpoint_sha256"]) == 64
    assert lineage["from_protocol"] == "native_multiframe_v1"
    assert lineage["to_protocol"] == "memory_replay_v1"
    assert lineage["parent_global_step"] == 1
    assert lineage["parent_epoch"] == 0
    assert lineage["parent_batch_cursor"] == 4

    # Verify model parameters match exactly
    for k, v in policy.state_dict().items():
        assert torch.equal(target_policy.state_dict()[k], v)

    # Verify AdamW moments, step match
    assert len(target_opt.param_groups) == len(opt.param_groups)
    for target_p, p in zip(target_policy.parameters(), policy.parameters()):
        if p.requires_grad:
            assert target_p in target_opt.state
            assert opt.state[p]["step"] == target_opt.state[target_p]["step"] == 1
            assert torch.equal(opt.state[p]["exp_avg"], target_opt.state[target_p]["exp_avg"])
            assert torch.equal(opt.state[p]["exp_avg_sq"], target_opt.state[target_p]["exp_avg_sq"])

    # Verify scheduler & progress
    assert target_sched.last_epoch == sched.last_epoch == 1
    assert res_info["global_step"] == 1
    assert res_info["epoch"] == 0
    assert res_info["batch_cursor"] == 4
    assert res_info["epoch_targets_seen"] == 8

    # Verify RNG restoration
    restored_py = random.random()
    restored_np = np.random.rand()
    restored_th = torch.randn(5)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    expected_py = random.random()
    expected_np = np.random.rand()
    expected_th = torch.randn(5)

    assert restored_py == expected_py
    assert restored_np == expected_np
    assert torch.equal(restored_th, expected_th)

    # Save newstage memory checkpoint at step 2
    mem_run_dir = tmp_path / "mem_run"
    mem_run_dir.mkdir(parents=True, exist_ok=True)
    mem_dc = copy.deepcopy(dense_dc)
    mem_dc["stream_protocol"] = mem_contract
    mem_val_dc = copy.deepcopy(val_dc)
    mem_val_dc["stream_protocol"] = mem_contract

    target_sched.step()
    mem_step2_path = save_native_checkpoint(
        output_dir=mem_run_dir,
        filename="mem_step2.pt",
        policy=target_policy,
        config={"dim": 64},
        norm_stats={"mean": 0.0},
        training_contract=exp_mem_tr,
        train_data_contract=mem_dc,
        val_data_contract=mem_val_dc,
        optimizer=target_opt,
        scheduler=target_sched,
        global_step=2,
        epoch=0,
        batch_cursor=8,
        epoch_targets_seen=16,
        source_metadata=dict(src_meta, stage_lineage=lineage),
        device=torch.device("cpu"),
    )

    # Strict resume with same contract succeeds
    resume_policy = TinyNamedPolicy()
    resume_opt, resume_sched = create_native_optimizer_and_scheduler(resume_policy, total_steps=30, warmup_steps=2)
    resume_info = load_native_training_checkpoint(
        checkpoint_path=mem_step2_path,
        policy=resume_policy,
        optimizer=resume_opt,
        scheduler=resume_sched,
        expected_source_metadata=dict(src_meta, stage_lineage=lineage),
        expected_training_contract=exp_mem_tr,
        expected_train_data_contract=mem_dc,
        expected_val_data_contract=mem_val_dc,
        device=torch.device("cpu"),
        mode="resume",
    )
    assert resume_info["global_step"] == 2
    assert resume_info["stage_lineage"] == lineage

    # Strict resume targeting stream_replay_v1 must reject
    stream_contract, stream_format = get_protocol_contract_and_format("stream_replay_v1")
    stream_tr_contract = copy.deepcopy(exp_mem_tr)
    stream_tr_contract["format"] = stream_format
    stream_tr_contract["stream_protocol"] = stream_contract

    with pytest.raises(ValueError, match="Training contract mismatch"):
        load_native_training_checkpoint(
            checkpoint_path=mem_step2_path,
            policy=resume_policy,
            optimizer=resume_opt,
            scheduler=resume_sched,
            expected_source_metadata=dict(src_meta, stage_lineage=lineage),
            expected_training_contract=stream_tr_contract,
            expected_train_data_contract=mem_dc,
            expected_val_data_contract=mem_val_dc,
            device=torch.device("cpu"),
            mode="resume",
        )


def test_parse_args_and_launcher_script_and_protocol_helper():
    """Test 2:

    - test parse_args accepts --stream-protocol memory_replay_v1.
    - validate_cli_arguments allows memory_replay_v1 for transition.
    - launcher script run_memory_10epochs.sh exists and is executable.
    - get_protocol_contract_and_format returns real dictionary and correct format.
    """
    # 1. parse_args accepts memory_replay_v1
    args = parse_args([
        "--output-dir", "/tmp/mem_out",
        "--stream-protocol", "memory_replay_v1",
    ])
    assert args.stream_protocol == "memory_replay_v1"

    # 2. validate_cli_arguments accepts transition with memory_replay_v1
    args_trans = parse_args([
        "--output-dir", "/tmp/mem_out",
        "--stream-protocol", "memory_replay_v1",
        "--transition-from", "/tmp/dense_ckpt.pt",
    ])
    validate_cli_arguments(args_trans)

    # 3. Launcher script exists and is executable
    launcher = Path(__file__).resolve().parent.parent / "scripts" / "run_memory_10epochs.sh"
    assert launcher.exists(), f"Launcher script does not exist: {launcher}"
    assert os.access(launcher, os.X_OK), f"Launcher script is not executable: {launcher}"
    content = launcher.read_text()
    assert "run_native_10epochs.sh" in content
    assert "--stream-protocol memory_replay_v1" in content
    assert "$@" in content

    # 4. get_protocol_contract_and_format returns real dictionary, not fake
    mem_contract, mem_format = get_protocol_contract_and_format("memory_replay_v1")
    assert isinstance(mem_contract, dict)
    assert mem_format == "native_memory_replay_v1"
    assert mem_contract["protocol_name"] == "memory_replay_v1"
    assert mem_contract["version"] == "1.2"
    assert mem_contract["anchor_policy"] == "snapshot_latest"
    assert mem_contract["recent_frames"] == 4
    assert mem_contract["memory_slots"] == 16
    assert mem_contract["consolidate_every"] == 4
    assert "algorithm_specification" in mem_contract
    assert isinstance(mem_contract["train_weights"], dict)


def test_transition_rejects_mutated_memory_contract_before_mutation(tmp_path: Path):
    """Test 3:

    Forcing alterations to memory recent_frames or memory_slots in expected contract causes transition to
    strictly reject before mutation (helper must match exact configuration).
    """
    policy = TinyNamedPolicy()
    opt, sched = create_native_optimizer_and_scheduler(policy, total_steps=30)
    src_meta, dense_tr_contract, dense_dc, val_dc = _make_contracts()

    parent_path = save_native_checkpoint(
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

    mem_contract, mem_format = get_protocol_contract_and_format("memory_replay_v1")

    # 1. Mutate recent_frames
    mutated_contract_1 = copy.deepcopy(mem_contract)
    mutated_contract_1["recent_frames"] = 8

    target_tr_1 = copy.deepcopy(dense_tr_contract)
    target_tr_1["format"] = mem_format
    target_tr_1["stream_protocol"] = mutated_contract_1

    target_policy = TinyNamedPolicy()
    orig_state_dict = copy.deepcopy(target_policy.state_dict())
    target_opt, target_sched = create_native_optimizer_and_scheduler(target_policy, total_steps=30)

    with pytest.raises(ValueError, match="does not match implemented contract"):
        transition_native_checkpoint(
            checkpoint_path=parent_path,
            policy=target_policy,
            optimizer=target_opt,
            scheduler=target_sched,
            expected_source_metadata=src_meta,
            expected_training_contract=target_tr_1,
            expected_train_data_contract=dense_dc,
            expected_val_data_contract=val_dc,
            stream_protocol="memory_replay_v1",
            device=torch.device("cpu"),
        )

    # Verify policy was NOT mutated
    for k, v in target_policy.state_dict().items():
        assert torch.equal(v, orig_state_dict[k])

    # 2. Mutate memory_slots
    mutated_contract_2 = copy.deepcopy(mem_contract)
    mutated_contract_2["memory_slots"] = 32

    target_tr_2 = copy.deepcopy(dense_tr_contract)
    target_tr_2["format"] = mem_format
    target_tr_2["stream_protocol"] = mutated_contract_2

    with pytest.raises(ValueError, match="does not match implemented contract"):
        transition_native_checkpoint(
            checkpoint_path=parent_path,
            policy=target_policy,
            optimizer=target_opt,
            scheduler=target_sched,
            expected_source_metadata=src_meta,
            expected_training_contract=target_tr_2,
            expected_train_data_contract=dense_dc,
            expected_val_data_contract=val_dc,
            stream_protocol="memory_replay_v1",
            device=torch.device("cpu"),
        )

    # Verify policy was NOT mutated
    for k, v in target_policy.state_dict().items():
        assert torch.equal(v, orig_state_dict[k])
